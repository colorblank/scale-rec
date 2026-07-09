"""FAT (Field-Aware Transformer) — main model implementation.

Paper: From Scaling to Structured Expressivity: Rethinking Transformers
       for CTR Prediction.  arXiv:2511.12081  (KDD 2026)

Architecture overview:
  1. FeatureEmbeddings  →  per-field token embeddings
  2. Field-aware bias + L × FAT blocks:
       - Field-Decomposed Attention (§3.2)
       - Field-Aware FFN (§3.3)
  3. Sum pooling over fields  →  global vector
  4. OutputHead / task towers  →  per-task logits

The field-specific projections (Q, K, V, FFN₁, FFN₂) are synthesised by the
Basis-Composed Hypernetwork (§3.4) and cached per forward pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ...core.config import PoolingMode

from ...core.model_output import ModelExecution, ModelOutput
from ...core.output_contract import NormalizedOutputContract
from ...layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ...layers.mlp import Mlp
from ...layers.towers import Activation, MultiTaskConfig, TaskTower
from ..esmm import _probability_for_relation
from ..output_head import OutputHead
from .attention import FieldDecomposedAttention
from .ffn import FieldAwareFFN
from .hypernetwork import BasisHypernetwork


class FATBlock(nn.Module):
    """One FAT block: Field-Decomposed Attention + Field-Aware FFN + residual.

    Paper §2 (Figure 1):
        Z^{(l+1)} = FAT-FFN(LayerNorm(FAT-Attn(Z^{(l)}))) + Z^{(l)}
    """

    def __init__(self, d: int, d_ff: int, n_heads: int, num_fields: int, dropout: float = 0.0):
        super().__init__()
        self.attn = FieldDecomposedAttention(d, n_heads, num_fields, dropout)
        self.ffn = FieldAwareFFN(d, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        W_v: torch.Tensor,
        W_1: torch.Tensor,
        W_2: torch.Tensor,
        field_pair_w: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attn(x, W_q, W_k, W_v, field_pair_w)
        ffn_out = self.ffn(attn_out, W_1, W_2)
        return ffn_out + x


class FATModel(nn.Module):
    """Field-Aware Transformer for CTR prediction.

    Args:
        features: List of (name, vocab_size, embed_dim) feature specs.
        d: Model (hidden) dimension.
        d_ff: FFN intermediate dimension.
        num_layers: Number of FAT blocks (L).
        n_heads: Number of attention heads.
        M: Number of base matrices per parameter type.
        k: Field meta-embedding dimension.
        K: Top-K activated bases per field.
        dropout: Dropout rate.
        deep_hidden_dims: Optional deep MLP hidden dims after sum pooling.
        shared_bottom_dims: Optional shared bottom MLP dims.
        task_config: Legacy MultiTaskConfig (mutually exclusive with output_contract).
        pooling_map: Per-feature pooling strategy override.
        total_dim: Total embedding dimension override.
        output_contract: NormalizedOutputContract (mutually exclusive with task_config).
    """

    def __init__(
        self,
        features: list[FeatureTuple],
        d: int = 128,
        d_ff: int = 512,
        num_layers: int = 2,
        n_heads: int = 8,
        M: int = 64,
        k: int = 64,
        K: int = 3,
        dropout: float = 0.0,
        deep_hidden_dims: list[int] | None = None,
        shared_bottom_dims: list[int] | None = None,
        task_config: MultiTaskConfig | None = None,
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
        super().__init__()
        deep_hidden_dims = deep_hidden_dims or []
        shared_bottom_dims = shared_bottom_dims or []

        # Features → per-field dense tokens
        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        num_fields = len(features)
        embed_dim = features[0][2] if features else d
        # Use consistent model dimension unless features specify a single shared dim
        model_dim = d

        # Project heterogeneous feature embeddings to uniform model dimension
        if any(f[2] != model_dim for f in features):
            self.input_proj = nn.Linear(embed_dim, model_dim)
        else:
            self.input_proj = None

        # Basis-Composed Hypernetwork
        self.hypernetwork = BasisHypernetwork(
            num_fields=num_fields,
            d=model_dim,
            d_ff=d_ff,
            M=M,
            k=k,
            K=K,
        )

        # Stacked FAT blocks
        self.blocks = nn.ModuleList(
            [FATBlock(model_dim, d_ff, n_heads, num_fields, dropout) for _ in range(num_layers)]
        )

        # Sum pooling → global vector, then optional projection
        self.output_proj = nn.Linear(model_dim, model_dim)

        # Optional deep MLP after pooling
        self.has_deep = bool(deep_hidden_dims)
        if self.has_deep:
            self.deep = Mlp(model_dim, deep_hidden_dims[:-1], deep_hidden_dims[-1], Activation.RELU)
            fusion_dim = deep_hidden_dims[-1]
        else:
            fusion_dim = model_dim

        # Shared bottom MLP (before task towers)
        if shared_bottom_dims:
            self.shared_bottom = Mlp(
                fusion_dim,
                shared_bottom_dims[:-1],
                shared_bottom_dims[-1],
                Activation.RELU,
            )
            shared_dim = shared_bottom_dims[-1]
        else:
            shared_dim = fusion_dim

        if output_contract is not None:
            self.output_contract = output_contract
            self.output_head = OutputHead(
                output_contract,
                {"shared": shared_dim},
            )
            self.task_names = [tower.name for tower in output_contract.towers]
            self.relation_names = list(output_contract.relation_order)
            return

        self.task_config = task_config
        if task_config is None:
            raise ValueError("FATModel requires task_config or output_contract")
        self.task_names = [tower.name for tower in self.task_config.towers]
        self.relation_names = [relation.target for relation in self.task_config.relations]
        for tower in self.task_config.towers:
            setattr(self, f"{tower.name}_tower", TaskTower(tower, shared_dim))

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        if hasattr(self, "output_contract"):
            return self.forward_execution(x_inputs).outputs
        shared = self._shared(x_inputs)
        return self._forward_legacy(shared)

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        shared = self._shared(x_inputs)
        if hasattr(self, "output_contract"):
            return self.output_head.forward({"shared": shared})
        outputs = self._forward_legacy(shared)
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _shared(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        """Compute the shared representation through FAT backbone.

        Returns: [batch, shared_dim]
        """
        stacked = self.embeddings.forward_stacked(x_inputs)
        x = torch.cat(stacked, dim=1)

        if self.input_proj is not None:
            x = self.input_proj(x)

        # Compose all field-specific projections from hypernetwork
        W_q, W_k, W_v = self.hypernetwork.compose_all_qkv()
        W_1, W_2 = self.hypernetwork.compose_all_ffn()
        field_pair_w = self.hypernetwork.field_pair_w

        # FAT blocks
        for block in self.blocks:
            x = block(x, W_q, W_k, W_v, W_1, W_2, field_pair_w)

        # Sum pooling over fields: global vector
        pooled = x.sum(dim=1)  # [B, D]
        pooled = self.output_proj(pooled)

        # Optional deep MLP
        if self.has_deep:
            pooled = self.deep(pooled)

        # Optional shared bottom
        if hasattr(self, "shared_bottom"):
            pooled = self.shared_bottom(pooled)

        return pooled

    def _forward_legacy(self, shared: torch.Tensor) -> ModelOutput:
        outputs = ModelOutput()
        for name in self.task_names:
            tower = getattr(self, f"{name}_tower")
            outputs.insert(name, tower(shared), tower.output_kind)
        for relation in self.task_config.relations:
            outputs.insert_probability(relation.target, self._apply_relation(relation, outputs))
        return outputs

    @staticmethod
    def _apply_relation(relation: object, outputs: ModelOutput) -> torch.Tensor:
        if not relation.sources:
            raise ValueError(f"Relation '{relation.target}' has no sources")
        probs = [
            _probability_for_relation(relation, outputs, source) for source in relation.sources
        ]
        if relation.op == "multiply":
            result = probs[0]
            for value in probs[1:]:
                result = result * value
            return result
        if relation.op == "add":
            result = probs[0]
            for value in probs[1:]:
                result = result + value
            return result
        if relation.op == "subtract":
            if len(probs) != 2:
                raise ValueError(f"Relation '{relation.target}' subtract requires 2 sources")
            return probs[0] - probs[1]
        if relation.op == "divide":
            if len(probs) != 2:
                raise ValueError(f"Relation '{relation.target}' divide requires 2 sources")
            return probs[0] / (probs[1] + 1e-8)
        raise ValueError(f"Unknown relation op: {relation.op}")
