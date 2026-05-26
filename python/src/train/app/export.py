from __future__ import annotations

"""safetensors 权重导出 — Candle VarMap 可直接加载。"""

import logging

import torch.nn as nn
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def export_to_safetensors(model: nn.Module, path: str) -> None:
    """Export model state_dict to safetensors; keys must match Candle VarBuilder::pp() paths."""
    state_dict = model.state_dict()
    save_file(state_dict, path)
    logger.info("saved %d tensors to %s", len(state_dict), path)


def print_state_dict_keys(model: nn.Module) -> None:
    """Print state_dict keys and shapes for debugging Candle naming alignment."""
    for key, tensor in model.state_dict().items():
        print(f"  {key:<60} {list(tensor.shape)}")
