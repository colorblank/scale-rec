from __future__ import annotations

"""ESMM：全量空间多任务（5 塔），点击条件乘积消除 SSB。"""

import torch
import torch.nn as nn

from ..layers.embedding import FeatureEmbeddings
from ..layers.mlp import Mlp
from ..layers.towers import Activation, TaskTower, TowerConfig


class ESMM(nn.Module):
    """Entire-space multi-task with 5 towers.

    概率关系:
      is_click = is_click_detail ∨ is_click_stock (点击包含详情点击和股票点击)
      is_cvr ← is_click (转化在点击后发生)
      stay_time ← is_click_detail (阅读仅在点详情后发生)

      P(detail)  = σ(click) × σ(detail)
      P(stock)   = σ(click) × σ(stock)
      P(cvr)     = σ(click) × σ(cvr)
      P(stay)    = σ(detail) × σ(stay)
    """

    def __init__(
        self,
        features,
        shared_bottom_dims,
        click_hidden_dims,
        cvr_hidden_dims,
        detail_hidden_dims,
        stock_hidden_dims,
        stay_hidden_dims,
    ):
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

        self.click_tower = TaskTower(
            TowerConfig("click", click_hidden_dims, 1, Activation.RELU), sd
        )
        self.cvr_tower = TaskTower(TowerConfig("cvr", cvr_hidden_dims, 1, Activation.RELU), sd)
        self.detail_tower = TaskTower(
            TowerConfig("detail", detail_hidden_dims, 1, Activation.RELU), sd
        )
        self.stock_tower = TaskTower(
            TowerConfig("stock", stock_hidden_dims, 1, Activation.RELU), sd
        )
        self.stay_tower = TaskTower(TowerConfig("stay", stay_hidden_dims, 1, Activation.RELU), sd)

    def forward(self, x_inputs):
        concat = self.embeddings(x_inputs)
        shared = self.shared_bottom(concat) if hasattr(self, "shared_bottom") else concat

        click_logits = self.click_tower(shared)
        cvr_logits = self.cvr_tower(shared)
        detail_logits = self.detail_tower(shared)
        stock_logits = self.stock_tower(shared)
        stay_logits = self.stay_tower(shared)

        click_prob = torch.sigmoid(click_logits)
        detail_prob = torch.sigmoid(detail_logits)

        return {
            "click": click_logits,
            "cvr": cvr_logits,
            "detail": detail_logits,
            "stock": stock_logits,
            "stay": stay_logits,
            # ESMM 条件概率乘积
            "ctcvr": click_prob * torch.sigmoid(cvr_logits),
            "ctdetail": click_prob * detail_prob,
            "ctstock": click_prob * torch.sigmoid(stock_logits),
            "ctstay": detail_prob * torch.sigmoid(stay_logits),  # stay 条件于 detail
        }
