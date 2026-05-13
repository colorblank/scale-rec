"""模型注册与配置 — 对应 src/models/mod.rs。"""
from dataclasses import dataclass, field
import torch.nn as nn, yaml
from ..layers.towers import Activation, MultiTaskConfig, TaskRelation, TowerConfig
from .lr import LogisticRegression
from .deepfm import DeepFM
from .mmoe import MMoE
from .esmm import ESMM
from .unimixer.model import UniMixerModel
from .unimixer.tokenizer import FeatureTokenizer


@dataclass
class TaskConfigEntry:
    name: str
    tower_dims: list[int] = field(default_factory=list)


def _parse_task_config(raw):
    towers = [
        TowerConfig(
            t["name"],
            t.get("hidden_dims", []),
            t.get("output_dim", 1),
            Activation.from_str(t.get("activation", "relu")),
        )
        for t in raw.get("towers", [])
    ]
    relations = [
        TaskRelation(r["target"], r["sources"], r["op"])
        for r in raw.get("relations", [])
    ]
    return MultiTaskConfig(towers=towers, relations=relations)


@dataclass
class ModelConfig:
    """YAML model config. Feature specs come from FeatureDag, not duplicated here."""

    type: str
    fm_k: int = 16
    deep_hidden_dims: list[int] = field(default_factory=list)
    shared_bottom_dims: list[int] = field(default_factory=list)
    num_experts: int = 4
    expert_hidden_dims: list[int] = field(default_factory=list)
    expert_output_dim: int = 32
    task_configs: list[TaskConfigEntry] = field(default_factory=list)
    ctr_hidden_dims: list[int] = field(default_factory=list)
    cvr_hidden_dims: list[int] = field(default_factory=list)
    token_dim: int = 64
    num_tokens: int = 8
    num_blocks: int = 2
    block_size: int | None = None
    use_lite: bool = False
    hidden_factor: float = 1.0
    num_basis: int = 4
    rank: int = 16
    use_siamese: bool = False
    task_config: MultiTaskConfig | None = None

    @classmethod
    def from_yaml(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, raw):
        task_configs = [
            TaskConfigEntry(t["name"], t.get("tower_dims", []))
            for t in raw.get("task_configs", [])
        ]
        tc = _parse_task_config(raw["task_config"]) if "task_config" in raw else None
        return cls(
            type=raw["type"],
            fm_k=raw.get("fm_k", 16),
            deep_hidden_dims=raw.get("deep_hidden_dims", []),
            shared_bottom_dims=raw.get("shared_bottom_dims", []),
            num_experts=raw.get("num_experts", 4),
            expert_hidden_dims=raw.get("expert_hidden_dims", []),
            expert_output_dim=raw.get("expert_output_dim", 32),
            task_configs=task_configs,
            ctr_hidden_dims=raw.get("ctr_hidden_dims", []),
            cvr_hidden_dims=raw.get("cvr_hidden_dims", []),
            token_dim=raw.get("token_dim", 64),
            num_tokens=raw.get("num_tokens", 8),
            num_blocks=raw.get("num_blocks", 2),
            block_size=raw.get("block_size"),
            use_lite=raw.get("use_lite", False),
            hidden_factor=raw.get("hidden_factor", 1.0),
            num_basis=raw.get("num_basis", 4),
            rank=raw.get("rank", 16),
            use_siamese=raw.get("use_siamese", False),
            task_config=tc,
        )

    def build(
        self,
        features: list[tuple[str, int, int]],
        tokenizer: FeatureTokenizer | None = None,
    ):
        m = self.type
        if m == "lr":
            return LogisticRegression(features)
        elif m == "deepfm":
            return DeepFM(features, self.fm_k, self.deep_hidden_dims)
        elif m == "mmoe":
            return MMoE(
                features,
                self.shared_bottom_dims,
                self.num_experts,
                self.expert_hidden_dims,
                self.expert_output_dim,
                [(t.name, t.tower_dims) for t in self.task_configs],
            )
        elif m == "esmm":
            return ESMM(
                features,
                self.shared_bottom_dims,
                self.ctr_hidden_dims,
                self.cvr_hidden_dims,
            )
        elif m == "unimixer":
            if tokenizer is None:
                raise ValueError("UniMixer requires external FeatureTokenizer")
            if self.task_config is None:
                raise ValueError("UniMixer requires task_config")
            return UniMixerModel(
                tokenizer=tokenizer,
                token_dim=self.token_dim,
                num_tokens=self.num_tokens,
                num_blocks=self.num_blocks,
                block_size_opt=self.block_size,
                use_lite=self.use_lite,
                hidden_factor=self.hidden_factor,
                num_basis=self.num_basis,
                rank=self.rank,
                task_config=self.task_config,
                use_siamese=self.use_siamese,
            )
        raise ValueError(f"Unknown model type: {m}")
