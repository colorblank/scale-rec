"""TokenMixer-Large: Mixing & Reverting + PerTokenSwiGLU 模块。"""

import torch
import torch.nn as nn

from ..unimixer.swiglu import PerTokenSwiGlu


class TokenMixerLargeBlock(nn.Module):
    """TokenMixer-Large block with Mixing & Reverting paradigm.

    1) Mixing: split tokens by head → concat across tokens → per-head SwiGLU
    2) Reverting: split back by token → concat across heads → per-token SwiGLU
    3) Residual on original input.
    """

    def __init__(
        self,
        embed_dim: int,
        token_dim: int,
        num_tokens: int,
        num_heads: int,
        hidden_factor: float,
        down_init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        head_token_dim = num_tokens * token_dim // num_heads
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.head_pswiglu = PerTokenSwiGlu(
            num_heads, head_token_dim, hidden_factor, down_init_scale
        )
        self.token_pswiglu = PerTokenSwiGlu(num_tokens, token_dim, hidden_factor, down_init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        x_2d = x.view(bs, self.num_tokens, self.token_dim)

        x_heads = x_2d.view(bs, self.num_tokens, self.num_heads, self.token_dim // self.num_heads)
        x_hm = x_heads.permute(2, 0, 1, 3)
        head_input = x_hm.reshape(self.num_heads, bs, -1).permute(1, 0, 2)
        head_mixed = self.head_pswiglu(head_input)

        head_mixed_2d = head_mixed.permute(1, 0, 2)
        x_revert = head_mixed_2d.reshape(self.num_heads, bs, self.num_tokens, -1).permute(
            2, 1, 0, 3
        )
        x_revert = x_revert.reshape(bs, self.num_tokens, self.token_dim)

        token_mixed = self.token_pswiglu(x_revert)
        return (token_mixed + x_2d).reshape(bs, self.embed_dim)
