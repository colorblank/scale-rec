from __future__ import annotations

"""ESMM：全量空间多任务，CTR·CVR 乘积链消除 SSB。"""
import torch
import torch.nn as nn

from ..layers.embedding import FeatureEmbeddings
from ..layers.mlp import Mlp
from ..layers.towers import Activation, TaskTower, TowerConfig


class ESMM(nn.Module):
    def __init__(self, features, shared_bottom_dims, ctr_hidden_dims, cvr_hidden_dims):
        super().__init__()
        self.embeddings = FeatureEmbeddings(features)
        if shared_bottom_dims:
            self.shared_bottom = Mlp(
                self.embeddings.total_dim,
                shared_bottom_dims[:-1],
                shared_bottom_dims[-1],
                Activation.RELU,
            )
            sd = shared_bottom_dims[-1]
        else:
            sd = self.embeddings.total_dim
        self.ctr_tower = TaskTower(TowerConfig("ctr", ctr_hidden_dims, 1, Activation.RELU), sd)
        self.cvr_tower = TaskTower(TowerConfig("cvr", cvr_hidden_dims, 1, Activation.RELU), sd)

    def forward(self, x_inputs):
        concat = self.embeddings(x_inputs)
        shared = self.shared_bottom(concat) if hasattr(self, "shared_bottom") else concat
        ctr_logits = self.ctr_tower(shared)
        cvr_logits = self.cvr_tower(shared)
        ctcvr = torch.sigmoid(ctr_logits) * torch.sigmoid(cvr_logits)
        return {"ctr": ctr_logits, "cvr": cvr_logits, "ctcvr": ctcvr}
