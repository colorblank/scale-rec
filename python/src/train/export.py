"""Weight export to safetensors format, loadable by Candle VarMap."""
from safetensors.torch import save_file
import torch.nn as nn

def export_to_safetensors(model: nn.Module, path: str):
    state_dict = model.state_dict()
    save_file(state_dict, path)
    print(f"[Export] Saved {len(state_dict)} tensors to {path}")

def print_state_dict_keys(model: nn.Module):
    for key, tensor in model.state_dict().items():
        print(f"  {key:<60} {list(tensor.shape)}")
