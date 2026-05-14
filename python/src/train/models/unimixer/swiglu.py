from __future__ import annotations

"""PerTokenSwiGLU：Token 维度 SwiGLU。"""
import math

import torch
import torch.nn as nn


class PerTokenSwiGlu(nn.Module):
    def __init__(self, num_tokens, token_dim, hidden_factor):
        super().__init__()
        h = int(token_dim * hidden_factor)
        self.token_dim = token_dim
        self.hidden_dim = h
        ub = math.sqrt(6.0 / (token_dim + h))
        gb = math.sqrt(6.0 / (token_dim + h))
        db = math.sqrt(6.0 / (h + token_dim))
        self.w_up = nn.Parameter(torch.empty(num_tokens, h, token_dim).uniform_(-ub, ub))
        self.b_up = nn.Parameter(torch.zeros(1, num_tokens, h))
        self.w_gate = nn.Parameter(torch.empty(num_tokens, h, token_dim).uniform_(-gb, gb))
        self.b_gate = nn.Parameter(torch.zeros(1, num_tokens, h))
        self.w_down = nn.Parameter(torch.empty(num_tokens, token_dim, h).uniform_(-db, db))
        self.b_down = nn.Parameter(torch.zeros(1, num_tokens, token_dim))

    @staticmethod
    def _einsum(x, w):
        # x: [B, T, D], w: [T, H, D] -> output: [B, T, H]
        return torch.einsum("btd,thd->bth", x, w)

    def forward(self, x):
        up = self._einsum(x, self.w_up) + self.b_up
        gate = self._einsum(x, self.w_gate) + self.b_gate
        hidden = up * (gate * torch.sigmoid(gate))
        wdt = self.w_down.transpose(1, 2)
        bt, t, h = hidden.shape
        _, d, _ = self.w_down.shape
        out = torch.matmul(hidden.unsqueeze(2), wdt.unsqueeze(0).expand(bt, t, h, d)).squeeze(2)
        return out + self.b_down
