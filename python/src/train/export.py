from __future__ import annotations

"""权重导出：safetensors 格式，Candle VarMap 可直接加载。"""
"""Weight export to safetensors format, loadable by Candle VarMap."""

import torch.nn as nn
from safetensors.torch import save_file


def export_to_safetensors(model: nn.Module, path: str):
    state_dict = model.state_dict()
    save_file(state_dict, path)
    print(f"[Export] Saved {len(state_dict)} tensors to {path}")


def print_state_dict_keys(model: nn.Module):
    for key, tensor in model.state_dict().items():
        print(f"  {key:<60} {list(tensor.shape)}")
