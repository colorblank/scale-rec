"""Feature pipeline config types — mirrors src/feats/config.rs."""

from dataclasses import dataclass, field
from typing import Optional
import yaml


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
            return cls(
                tag="list", inner=cls.from_dict(inner_raw), length=raw["list"]["length"]
            )
        raise ValueError(f"Invalid DType: {raw}")


@dataclass
class EmbedConfig:
    vocab_size: int
    embed_dim: int


@dataclass
class SourceDef:
    name: str
    source: str
    dtype: DType
    default_val: str
    embed: Optional[EmbedConfig] = None


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
                    source=s["source"],
                    dtype=DType.from_dict(s["dtype"]),
                    default_val=s["default_val"],
                    embed=embed,
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
