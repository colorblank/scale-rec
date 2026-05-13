from __future__ import annotations

"""UniMixerBlock：交互 + 激活 + 归一 + 残差。"""
import torch.nn as nn

from .norm import SiameseNorm
from .swiglu import PerTokenSwiGlu
from .unimixing import UniMixing, UniMixingLite


class UniMixerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        block_size,
        token_dim,
        num_tokens,
        use_lite,
        hidden_factor,
        num_basis,
        rank,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self._use_lite = use_lite
        if use_lite:
            self.unimixing_lite = UniMixingLite(embed_dim, block_size, num_basis, rank)
        else:
            self.unimixing = UniMixing(embed_dim, block_size)
        self.pswiglu = PerTokenSwiGlu(num_tokens, token_dim, hidden_factor)
        self.siamese_norm = SiameseNorm(embed_dim)

    def _apply_unimixing(self, x, temperature):
        return (
            self.unimixing_lite(x, temperature)
            if self._use_lite
            else self.unimixing(x, temperature)
        )

    def forward(self, x, temperature, x_bar_opt=None, y_bar_opt=None, use_siamese=False):
        bs = x.shape[0]
        if use_siamese:
            ybn = self.siamese_norm.forward_rmsnorm(y_bar_opt)
            mixed = self._apply_unimixing(x_bar_opt + ybn, temperature)
            mt = mixed.view(bs, self.num_tokens, self.token_dim)
            ps = self.pswiglu(mt)
            psf = ps.reshape(bs, self.embed_dim)
            bo = psf + mixed
            xbn = self.siamese_norm.forward_rmsnorm(x_bar_opt + bo)
            ybn2 = y_bar_opt + bo
            return bo, xbn, ybn2
        else:
            mixed = self._apply_unimixing(x, temperature)
            mt = mixed.view(bs, self.num_tokens, self.token_dim)
            return mixed + self.pswiglu(mt).reshape(bs, self.embed_dim)
