from __future__ import annotations

"""safetensors 权重导出 — Candle VarMap 可直接加载。"""

import torch.nn as nn
from safetensors.torch import save_file


def export_to_safetensors(model: nn.Module, path: str):
    """Export model state_dict to safetensors; keys must match Candle VarBuilder::pp() paths."""
    state_dict = model.state_dict()
    save_file(state_dict, path)
    print(f"[Export] Saved {len(state_dict)} tensors to {path}")


def print_state_dict_keys(model: nn.Module):
    """Print state_dict keys and shapes for debugging Candle naming alignment."""
    for key, tensor in model.state_dict().items():
        print(f"  {key:<60} {list(tensor.shape)}")
