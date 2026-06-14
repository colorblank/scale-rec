"""RankMixer block: token mixing + per-token FFN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerTokenFfn(nn.Module):
    """Dedicated two-layer GELU FFN per token."""

    def __init__(self, num_tokens: int, token_dim: int, hidden_factor: float) -> None:
        super().__init__()
        hidden_dim = max(1, int(token_dim * hidden_factor))
        self.up = nn.ModuleList(nn.Linear(token_dim, hidden_dim) for _ in range(num_tokens))
        self.down = nn.ModuleList(nn.Linear(hidden_dim, token_dim) for _ in range(num_tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token_idx, (up, down) in enumerate(zip(self.up, self.down, strict=False)):
            token = x[:, token_idx, :]
            outputs.append(down(F.gelu(up(token))).unsqueeze(1))
        return torch.cat(outputs, dim=1)


class RankMixerBlock(nn.Module):
    """Dense RankMixer block from the paper."""

    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        num_heads: int,
        hidden_factor: float,
    ) -> None:
        super().__init__()
        if num_heads != num_tokens:
            raise ValueError("RankMixer requires num_heads == num_tokens for residual shape")
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.norm_mixing = nn.LayerNorm(token_dim)
        self.pffn = PerTokenFfn(num_tokens, token_dim, hidden_factor)
        self.norm_ffn = nn.LayerNorm(token_dim)

    def token_mixing(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        head_dim = self.token_dim // self.num_heads
        x_heads = x.view(batch_size, self.num_tokens, self.num_heads, head_dim)
        mixed = x_heads.permute(0, 2, 1, 3).contiguous()
        return mixed.view(batch_size, self.num_heads, self.num_tokens * head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.token_mixing(x)
        s = self.norm_mixing(mixed + x)
        return self.norm_ffn(self.pffn(s) + s)
