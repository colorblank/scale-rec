from __future__ import annotations

"""safetensors 权重导出 — Candle VarMap 可直接加载。"""

import logging
from pathlib import Path
from typing import Any

import torch.nn as nn
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


def export_to_safetensors(model: nn.Module, path: str) -> None:
    """Export model state_dict to safetensors; keys must match Candle VarBuilder::pp() paths."""
    state_dict = model.state_dict()
    save_file(state_dict, path)
    logger.info("saved %d tensors to %s", len(state_dict), path)


def replace_inactive_embedding_rows(path: str | Path, report: dict[str, Any]) -> None:
    """Replace inactive DictMapper/hash rows with the active-row mean in serving weights."""
    path = Path(path)
    state_dict = {name: tensor.clone() for name, tensor in load_file(str(path)).items()}
    changed = 0
    supported_ops = {"DictMapper", "FeatureHash", "ParsedFeatureHash", "ConcatHash"}

    for feature_name, stat in report.get("features", {}).items():
        if stat.get("operator_type") not in supported_ops:
            continue
        inactive = list(stat.get("inactive_bucket_ids", []))
        if not inactive:
            continue
        active = [i for i, hits in enumerate(stat.get("bucket_hits", [])) if hits > 0]
        if not active:
            raise ValueError(
                f"embedding feature '{feature_name}' has no active bucket; refusing to publish"
            )

        suffix = f"emb_{feature_name}.weight"
        matched = [name for name in state_dict if name.endswith(suffix)]
        if not matched:
            raise ValueError(f"no embedding weight found for feature '{feature_name}'")
        for name in matched:
            weight = state_dict[name].clone()
            mean = weight[active].mean(dim=0)
            weight[inactive] = mean
            state_dict[name] = weight
            changed += len(inactive)

    if changed:
        tmp_path = path.with_name(f".{path.name}.tmp")
        save_file(state_dict, str(tmp_path))
        tmp_path.replace(path)
        logger.info("replaced %d inactive embedding rows in %s", changed, path)


def print_state_dict_keys(model: nn.Module) -> None:
    """Print state_dict keys and shapes for debugging Candle naming alignment."""
    for key, tensor in model.state_dict().items():
        print(f"  {key:<60} {list(tensor.shape)}")
