import torch
import os
import json, csv, platform, socket, subprocess
from datetime import datetime
from dataclasses import asdict

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