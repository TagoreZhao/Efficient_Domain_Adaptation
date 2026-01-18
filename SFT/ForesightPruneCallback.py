import math
import torch
import torch.distributed as dist

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from pruning.prune import foresight_prune


class ForesightPruneCallback(TrainerCallback):
    """
    Multi-GPU (DDP) safe epoch-level pruning.

    Semantics:
      - Runs at on_epoch_begin for epoch >= 1, which effectively means "after the previous epoch
        completed (including its eval/save)".

    PBS skipping:
      - skip_last_n_epochs does NOT skip pruning; it only forces PBS=False in the last N epochs.

    Sync:
      - If rank0_only=True: only rank 0 runs pruning, then parameters/buffers are broadcast to all ranks.
      - If rank0_only=False: all ranks run pruning; broadcasting is skipped by default.
        (Only safe if every rank uses identical calibration data and pruning is deterministic.)
    """

    def __init__(
        self,
        *,
        dataloader,
        prune_ratio: float,
        mask_lr: float = 0.5,
        nsamples: int = 128,
        merge_lora: bool = True,
        PBS: bool = False,
        device=None,
        skip_last_n_epochs: int = 0,
        rank0_only: bool = True,
        broadcast_buffers: bool = True,
    ):
        self.dataloader = dataloader
        self.prune_ratio = prune_ratio
        self.mask_lr = mask_lr
        self.nsamples = nsamples
        self.merge_lora = merge_lora
        self.PBS = PBS
        self.device = device
        self.skip_last_n_epochs = skip_last_n_epochs
        self.rank0_only = rank0_only
        self.broadcast_buffers = broadcast_buffers

        self._last_pruned_epoch_idx = None

    @staticmethod
    def _is_dist() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _rank() -> int:
        return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

    @staticmethod
    def _world_size() -> int:
        return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

    @staticmethod
    def _unwrap_model(m):
        # DDP unwrap. (If you are under FSDP/ZeRO-3, this is not enough.)
        return getattr(m, "module", m)

    @staticmethod
    def _broadcast_model_state(model, src: int = 0, broadcast_buffers: bool = True):
        # DDP-only: broadcast all parameters (and optionally buffers) from src to all ranks.
        for p in model.parameters():
            if p is None or p.data is None:
                continue
            dist.broadcast(p.data, src=src)
        if broadcast_buffers:
            for b in model.buffers():
                if b is None:
                    continue
                dist.broadcast(b, src=src)

    def _pbs_for_epoch(self, args: TrainingArguments, epoch_idx: int) -> bool:
        """
        Decide whether to enable PBS for this pruning call.

        We run pruning at on_epoch_begin(epoch_idx), which corresponds to "after epoch_idx-1 completed".
        For simplicity we apply the 'last N epochs' rule using epoch_idx against num_train_epochs.
        """
        pbs = self.PBS
        if args.num_train_epochs is not None and self.skip_last_n_epochs > 0:
            last_start = int(args.num_train_epochs) - self.skip_last_n_epochs
            if epoch_idx >= last_start:
                pbs = False
        return pbs

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model_kw = kwargs.get("model", None)
        if model_kw is None:
            return control

        if state.epoch is None:
            return control
        epoch_idx = int(math.floor(state.epoch + 1e-6))

        # epoch_idx == 0 means "starting epoch 0": no previous epoch yet
        if epoch_idx <= 0:
            return control

        # run exactly once per epoch index
        if self._last_pruned_epoch_idx == epoch_idx:
            return control
        self._last_pruned_epoch_idx = epoch_idx

        if self.dataloader is None:
            return control

        is_dist = self._is_dist()
        rank = self._rank()
        world_size = self._world_size()

        device = self.device if self.device is not None else args.device
        target_model = self._unwrap_model(model_kw)

        # Determine PBS for this pruning call (skip_last_n_epochs only affects PBS)
        pbs_this_epoch = self._pbs_for_epoch(args, epoch_idx)

        # Synchronize all ranks before pruning
        if is_dist and world_size > 1:
            dist.barrier()

        # Rank gating
        do_prune = (not self.rank0_only) or (rank == 0)
        if do_prune:
            was_training = target_model.training
            try:
                target_model.eval()
                with torch.no_grad():
                    foresight_prune(
                        target_model,
                        self.dataloader,
                        prune_ratio=self.prune_ratio,
                        mask_lr=self.mask_lr,
                        nsamples=self.nsamples,
                        merge_lora=self.merge_lora,
                        PBS=pbs_this_epoch,
                        device=device,
                    )
            finally:
                target_model.train(was_training)

        # Synchronize after pruning, then broadcast if rank0_only
        if is_dist and world_size > 1:
            dist.barrier()

            if self.rank0_only:
                self._broadcast_model_state(target_model, src=0, broadcast_buffers=self.broadcast_buffers)

            dist.barrier()

        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model_kw = kwargs.get("model", None)
        if model_kw is None or self.dataloader is None:
            return control

        is_dist = self._is_dist()
        rank = self._rank()
        world_size = self._world_size()

        device = self.device if self.device is not None else args.device
        target_model = self._unwrap_model(model_kw)

        # Optional: avoid double-running if train_end is triggered after an early stop mid-epoch
        if getattr(self, "_final_prune_done", False):
            return control
        self._final_prune_done = True

        if is_dist and world_size > 1:
            dist.barrier()

        do_prune = (not self.rank0_only) or (rank == 0)
        if do_prune:
            was_training = target_model.training
            try:
                target_model.eval()
                with torch.no_grad():
                    foresight_prune(
                        target_model,
                        self.dataloader,
                        prune_ratio=self.prune_ratio,
                        mask_lr=self.mask_lr,
                        nsamples=self.nsamples,
                        merge_lora=self.merge_lora,
                        PBS=False,              # force PBS off
                        device=device,
                    )
            finally:
                target_model.train(was_training)

        if is_dist and world_size > 1:
            dist.barrier()
            if self.rank0_only:
                self._broadcast_model_state(target_model, src=0, broadcast_buffers=self.broadcast_buffers)
            dist.barrier()

        return control
