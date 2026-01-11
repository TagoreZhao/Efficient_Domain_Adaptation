import torch
import os
import json, csv, platform, socket, subprocess
from datetime import datetime
from dataclasses import asdict
import torch.distributed as dist

def init_dist_if_needed():
    # torchrun sets these env vars
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        return True
    return False

def get_ranks():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size

@torch.no_grad()
def broadcast_model_state(model, src=0):
    """
    Broadcast parameters and buffers from src to all ranks.
    Works with NCCL if tensors are CUDA tensors.
    """
    for p in model.parameters():
        dist.broadcast(p.data, src=src)
    for b in model.buffers():
        dist.broadcast(b.data, src=src)

def rank0_prune_then_sync(model, prune_fn, *prune_args, **prune_kwargs):
    """
    Run prune_fn(model=...) only on rank 0, then broadcast pruned weights to all ranks.
    """
    is_dist = init_dist_if_needed()
    rank, local_rank, world_size = get_ranks()

    # Ensure each process uses its own GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if (not is_dist) or (rank == 0):
        prune_fn(*prune_args, model=model, **prune_kwargs)

    if is_dist:
        dist.barrier()
        broadcast_model_state(model, src=0)
        dist.barrier()

    return model

def _to_dict(cfg):
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    try:
        return asdict(cfg)
    except Exception:
        return {k: getattr(cfg, k) for k in dir(cfg)
                if not k.startswith("_") and isinstance(getattr(cfg, k), (int, float, bool, str, list, dict, tuple))}

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "trainable_ratio": float(trainable/total)}

def save_run_manifest(
    out_dir,
    model_name,
    peft_config,
    sft_config,
    tokenizer_name,
    train_len,
    val_len,
    pre_metrics,
    post_metrics,
    extra=None,
    model_for_count=None,
):
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "host": {"node": socket.gethostname(), "platform": platform.platform()},
        "cuda": {
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "peft_config": _to_dict(peft_config),
        "sft_config": _to_dict(sft_config),
        "dataset_sizes": {"train": int(train_len), "validation": int(val_len)},
        "param_counts": count_params(model_for_count) if model_for_count is not None else None,
        "metrics": {
            "pre": pre_metrics,     # dict you fill (PubMedQA/MedNLI/HQS before FT)
            "post": post_metrics,   # dict you fill (after FT)
        },
        "extra": extra or {},
    }
    path = os.path.join(out_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path