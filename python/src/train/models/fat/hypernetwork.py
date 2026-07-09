"""Basis-Composed Hypernetwork for FAT (Field-Aware Transformer).

Paper §3.4: Synthesizes field-specific projection matrices from shared bases,
reducing parameter complexity from O(F·d²) to O(M·d² + F·k).

Each parameter type Φ ∈ {Q, K, V, FFN₁, FFN₂} maintains:
  - M base matrices  B_{m,Φ}  with the same dims as the target W_Φ^{(f)}
  - A lightweight single-layer MLP router g_ψ: ℝ^k → ℝ^M

At forward time, the router maps a field meta-embedding φ_f to top-K sparse
coefficients; the field-specific projection is the weighted sum of selected bases:

    W_Φ^{(f)} = Σ_{m∈π_f}  α_m · B_{m,Φ}

Reference:
    https://arxiv.org/abs/2511.12081
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasisHypernetwork(nn.Module):
    """Basis-Composed Hypernetwork for synthesizing field-specific projections.

    Args:
        num_fields: Number of fields (features) F.
        d: Model (hidden) dimension.
        d_ff: FFN intermediate dimension.
        M: Number of base matrices per parameter type (default 64).
        k: Field meta-embedding dimension (default 64).
        K: Number of top-activated bases per field (default 3).
    """

    def __init__(
        self,
        num_fields: int,
        d: int,
        d_ff: int,
        M: int = 64,
        k: int = 64,
        K: int = 3,
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.d = d
        self.d_ff = d_ff
        self.M = M
        self.k = k
        self.K = K

        # Field meta-embeddings:  φ_f ∈ ℝ^k, one per field
        self.field_meta = nn.Parameter(torch.randn(num_fields, k) * 0.1)

        # ── Base matrices (shared across fields) ──
        # Attention: Q, K, V  ∈ ℝ^{M × d × d}
        self.bases_q = nn.Parameter(torch.randn(M, d, d) * 0.02)
        self.bases_k = nn.Parameter(torch.randn(M, d, d) * 0.02)
        self.bases_v = nn.Parameter(torch.randn(M, d, d) * 0.02)

        # FFN:  W₁ ∈ ℝ^{M × d × d_ff},  W₂ ∈ ℝ^{M × d_ff × d}
        self.bases_ffn1 = nn.Parameter(torch.randn(M, d, d_ff) * 0.02)
        self.bases_ffn2 = nn.Parameter(torch.randn(M, d_ff, d) * 0.02)

        # ── Lightweight single-layer routers ──
        self.router_q = nn.Linear(k, M, bias=False)
        self.router_k = nn.Linear(k, M, bias=False)
        self.router_v = nn.Linear(k, M, bias=False)
        self.router_ffn1 = nn.Linear(k, M, bias=False)
        self.router_ffn2 = nn.Linear(k, M, bias=False)

        # Field-pair interaction modulation:  w_{f_i, f_j} ∈ ℝ
        # Initialized from 𝒩(0, 0.01) per paper §5.1.3
        self.field_pair_w = nn.Parameter(torch.randn(num_fields, num_fields) * 0.01)

    def _route(self, router: nn.Linear, field_ids: torch.Tensor) -> torch.Tensor:
        """Compute top-K sparse selection coefficients.

        Args:
            router: Single-layer MLP mapping k → M.
            field_ids: [N] field indices.

        Returns:
            coefficients: [N, M]  (sparse: only K non-zeros per row).
        """
        meta = self.field_meta[field_ids]  # [N, k]
        scores = router(meta)  # [N, M]
        topk_vals, topk_idx = torch.topk(scores, self.K, dim=-1)  # [N, K]
        alpha = F.softmax(topk_vals, dim=-1)  # [N, K]

        coeffs = torch.zeros(scores.shape[0], self.M, device=scores.device)
        coeffs.scatter_(1, topk_idx, alpha)
        return coeffs

    def compose(self, bases: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """Weighted sum of bases: W = Σ_m α_m · B_m.

        Args:
            bases: [M, d_in, d_out]
            coefficients: [F, M]

        Returns:
            projections: [F, d_in, d_out]
        """
        return torch.einsum("fm,mio->fio", coefficients, bases)

    def compose_all_qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compose Q, K, V projections for all fields.

        Returns:
            W_q, W_k, W_v: each [num_fields, d, d]
        """
        field_ids = torch.arange(self.num_fields, device=self.field_meta.device)
        coeffs_q = self._route(self.router_q, field_ids)
        coeffs_k = self._route(self.router_k, field_ids)
        coeffs_v = self._route(self.router_v, field_ids)
        return (
            self.compose(self.bases_q, coeffs_q),
            self.compose(self.bases_k, coeffs_k),
            self.compose(self.bases_v, coeffs_v),
        )

    def compose_all_ffn(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose FFN projections W₁, W₂ for all fields.

        Returns:
            W_1: [num_fields, d, d_ff],  W_2: [num_fields, d_ff, d]
        """
        field_ids = torch.arange(self.num_fields, device=self.field_meta.device)
        coeffs_1 = self._route(self.router_ffn1, field_ids)
        coeffs_2 = self._route(self.router_ffn2, field_ids)
        return (
            self.compose(self.bases_ffn1, coeffs_1),
            self.compose(self.bases_ffn2, coeffs_2),
        )
