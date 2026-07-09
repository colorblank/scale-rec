"""Deep Interest Network (DIN, arXiv:1706.06978)."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..core.config import PoolingMode

from ..core.model_output import ModelExecution, ModelOutput
from ..core.output_contract import NormalizedOutputContract
from ..layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ..layers.mlp import Mlp
from ..layers.towers import Activation
from .output_head import OutputHead


class DIN(nn.Module):
    """Deep Interest Network.

    Shared item embedding for behavior sequence + candidate ad.
    Local activation unit computes adaptive, unnormalized user interest weights.
    """

    def __init__(
        self,
        features: list[FeatureTuple],
        item_vocab_size: int = 10000,
        embed_dim: int = 16,
        activation_hidden_dims: list[int] | None = None,
        mlp_hidden_dims: list[int] | None = None,
        behavior_feature: str = "hist_item_ids",
        candidate_feature: str = "item_id",
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.behavior_feature = behavior_feature
        self.candidate_feature = candidate_feature

        self.item_embedding = nn.Embedding(item_vocab_size, embed_dim)

        other_features = [
            feature
            for feature in features
            if feature[0] not in {behavior_feature, candidate_feature}
        ]
        self.embeddings = FeatureEmbeddings(other_features, pooling_map) if other_features else None
        other_dim = self.embeddings.total_dim if self.embeddings is not None else 0

        self.activation_unit = Mlp(
            4 * embed_dim,
            activation_hidden_dims or [],
            1,
            Activation.RELU,
        )

        mlp_input_dim = embed_dim + embed_dim + other_dim
        self.mlp = Mlp(
            mlp_input_dim,
            mlp_hidden_dims or [],
            1,
            Activation.RELU,
        )

        self.output_head = (
            OutputHead(output_contract, {"shared": 1}) if output_contract is not None else None
        )

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        if self.output_head is not None:
            return self.forward_execution(x_inputs).outputs
        return ModelOutput.binary_logits({"pred": self._shared(x_inputs)})

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        if self.output_head is not None:
            return self.output_head({"shared": self._shared(x_inputs)})
        outputs = self.forward(x_inputs)
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _shared(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        behavior_idx = x_inputs[self.behavior_feature]
        candidate_idx = x_inputs[self.candidate_feature]

        if behavior_idx.dim() == 1:
            behavior_idx = behavior_idx.unsqueeze(1)
        behavior_embs = self.item_embedding(behavior_idx)
        candidate_emb = self.item_embedding(candidate_idx)

        batch, seq_len, embed_dim = behavior_embs.shape

        candidate_emb_3d = candidate_emb.unsqueeze(1).expand(batch, seq_len, embed_dim)
        prod = behavior_embs * candidate_emb_3d
        diff = behavior_embs - candidate_emb_3d

        au_input = torch.cat([behavior_embs, candidate_emb_3d, prod, diff], dim=2)
        au_input_flat = au_input.reshape(batch * seq_len, 4 * embed_dim)
        attn_weights = self.activation_unit(au_input_flat).reshape(batch, seq_len, 1)

        interest_emb = (behavior_embs * attn_weights).sum(dim=1)

        if self.embeddings is not None:
            other_emb = self.embeddings(x_inputs)
            combined = torch.cat([interest_emb, candidate_emb, other_emb], dim=1)
        else:
            combined = torch.cat([interest_emb, candidate_emb], dim=1)
        return self.mlp(combined)
