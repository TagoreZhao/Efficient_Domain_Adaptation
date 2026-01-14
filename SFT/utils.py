import torch
import os
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

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "trainable_ratio": float(trainable/total)}
