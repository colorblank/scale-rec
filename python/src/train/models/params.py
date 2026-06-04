from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    emb_trainable = 0
    dense_trainable = 0
    non_trainable = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "emb_" in name:
                emb_trainable += p.numel()
            else:
                dense_trainable += p.numel()
        else:
            non_trainable += p.numel()
    return {
        "emb_trainable": emb_trainable,
        "dense_trainable": dense_trainable,
        "non_trainable": non_trainable,
        "total": emb_trainable + dense_trainable + non_trainable,
    }


def format_parameter_summary(model: nn.Module) -> str:
    counts = count_parameters(model)
    total_mb = (counts["total"] * 4) / (1024 * 1024)
    return (
        f"total={counts['total']:,} ({total_mb:.2f} MB), "
        f"emb_trainable={counts['emb_trainable']:,}, "
        f"dense_trainable={counts['dense_trainable']:,}, "
        f"non_trainable={counts['non_trainable']:,}"
    )
