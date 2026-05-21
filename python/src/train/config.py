from __future__ import annotations

"""特征配置类型 — mirrors src/feats/config.rs."""

from dataclasses import dataclass, field
from typing import Optional

import yaml


class Role:
    """列角色常量 — mirrors src/feats/config.rs Role enum."""

    FEATURE = "feature"
    LABEL = "label"
    DISCARD = "discard"


@dataclass
class DType:
    tag: str
    inner: Optional["DType"] = None
    length: int = 0

    @classmethod
    def from_dict(cls, raw: str | dict) -> "DType":
        if isinstance(raw, str):
            return cls(tag=raw)
        if isinstance(raw, dict) and "list" in raw:
            inner_raw = raw["list"]["dtype"]
            return cls(tag="list", inner=cls.from_dict(inner_raw), length=raw["list"]["length"])
        raise ValueError(f"Invalid DType: {raw}")


@dataclass
class EmbedConfig:
    vocab_size: int
    embed_dim: int
    pooling: str = "first"  # first | flatten | mean | sum | max


@dataclass
class SourceDef:
    name: str
    dtype: DType
    default_val: str
    source: Optional[str] = None
    embed: Optional[EmbedConfig] = None
    role: str = Role.FEATURE
    column_index: Optional[int] = None


@dataclass
class OperatorDef:
    name: str
    op_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    embed: Optional[EmbedConfig] = None


@dataclass
class FlowConfig:
    version: str
    sources: list[SourceDef]
    operators: list[OperatorDef]

    @classmethod
    def from_yaml(cls, path: str) -> "FlowConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "FlowConfig":
        sources = []
        for s in raw.get("sources", []):
            embed = EmbedConfig(**s["embed"]) if "embed" in s else None
            sources.append(
                SourceDef(
                    name=s["name"],
                    source=s.get("source"),
                    dtype=DType.from_dict(s["dtype"]),
                    default_val=s["default_val"],
                    embed=embed,
                    role=s.get("role", Role.FEATURE),
                    column_index=s.get("column_index"),
                )
            )
        operators = []
        for o in raw.get("operators", []):
            embed = EmbedConfig(**o["embed"]) if "embed" in o else None
            operators.append(
                OperatorDef(
                    name=o["name"],
                    op_type=o["op_type"],
                    inputs=o.get("inputs", []),
                    outputs=o.get("outputs", []),
                    params=o.get("params", {}),
                    embed=embed,
                )
            )
        return cls(version=raw["version"], sources=sources, operators=operators)

    @property
    def feature_sources(self) -> list[SourceDef]:
        """返回 role==feature 的 source 列表。"""
        return [s for s in self.sources if s.role == Role.FEATURE]

    @property
    def label_sources(self) -> list[SourceDef]:
        """返回 role==label 的 source 列表。"""
        return [s for s in self.sources if s.role == Role.LABEL]

    @property
    def discard_sources(self) -> list[SourceDef]:
        """返回 role==discard 的 source 列表。"""
        return [s for s in self.sources if s.role == Role.DISCARD]
