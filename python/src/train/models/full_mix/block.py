"""Full-Mix block: parameterized full token mixing + GLU-improved P-FFNs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerTokenGluFfn(nn.Module):
    """Dedicated GLU-style FFN per token."""

    def __init__(self, num_tokens: int, token_dim: int, hidden_factor: float) -> None:
        super().__init__()
        if hidden_factor <= 0:
            raise ValueError("hidden_factor must be > 0")
        hidden_dim = max(1, int(token_dim * hidden_factor))
        self.up = nn.ModuleList(nn.Linear(token_dim, hidden_dim) for _ in range(num_tokens))
        self.gate = nn.ModuleList(nn.Linear(token_dim, hidden_dim) for _ in range(num_tokens))
        self.down = nn.ModuleList(nn.Linear(hidden_dim, token_dim) for _ in range(num_tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token_idx, (up, gate, down) in enumerate(
            zip(self.up, self.gate, self.down, strict=False)
        ):
            token = x[:, token_idx, :]
            hidden = F.gelu(up(token)) * torch.sigmoid(gate(token))
            outputs.append(down(hidden).unsqueeze(1))
        return torch.cat(outputs, dim=1)


class FullMixBlock(nn.Module):
    """RankElastor-style full token mixing block."""

    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        hidden_factor: float,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be > 0")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        if hidden_factor <= 0:
            raise ValueError("hidden_factor must be > 0")
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        flat_dim = token_dim * num_tokens
        self.full_mixing = nn.Linear(flat_dim, flat_dim)
        self.norm_mixing = nn.LayerNorm(token_dim)
        self.pffn = PerTokenGluFfn(num_tokens, token_dim, hidden_factor)
        self.norm_ffn = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        mixed = self.full_mixing(x.reshape(batch_size, -1))
        mixed = mixed.reshape(batch_size, self.num_tokens, self.token_dim)
        s = self.norm_mixing(mixed + x)
        return self.norm_ffn(self.pffn(s) + s)
