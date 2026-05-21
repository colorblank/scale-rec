from __future__ import annotations

"""Debug tracer: 逐阶段记录特征预处理管道的输入/输出值和异常。"""
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StageType(Enum):
    DEFAULT_INIT = "DEFAULT_INIT"
    RAW_OVERRIDE = "RAW_OVERRIDE"
    OPERATOR = "OPERATOR"


@dataclass
class DebugConfig:
    feature_filter: str = "all"  # "all" | "embed_only" | ["name1","name2"]
    max_trace_samples: int = 100
    output_dir: str = ""


@dataclass
class ValueSnapshot:
    value: Any
    type_name: str

    @classmethod
    def of(cls, val: Any) -> "ValueSnapshot":
        t = type(val).__name__
        if isinstance(val, list):
            inner = type(val[0]).__name__ if val else "empty"
            t = f"list[{inner}]"
        elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            t = "float(anomaly)"
        return cls(value=val, type_name=t)


@dataclass
class Anomaly:
    feature: str
    reason: str
    value: Any = None


@dataclass
class StageTrace:
    stage_type: StageType
    stage_name: str
    inputs: dict[str, ValueSnapshot] = field(default_factory=dict)
    outputs: dict[str, ValueSnapshot] = field(default_factory=dict)
    overridden: list[str] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {"type": self.stage_type.value}
        if self.stage_name:
            d["name"] = self.stage_name
        if self.inputs:
            d["inputs"] = {k: {"v": v.value, "t": v.type_name} for k, v in self.inputs.items()}
        if self.outputs:
            d["outputs"] = {k: {"v": v.value, "t": v.type_name} for k, v in self.outputs.items()}
        if self.overridden:
            d["overridden"] = self.overridden
        if self.anomalies:
            d["anomalies"] = [{"feature": a.feature, "reason": a.reason} for a in self.anomalies]
        return d


class DebugTracer:
    """Tracer attached to FeatureDag — records per-sample stage traces."""

    def __init__(self, config: DebugConfig):
        self.config = config
        self.traces: list[SampleTrace] = []
        self._current: SampleTrace | None = None
        self._total_seen = 0  # global counter across all batches

    # ── called by FeatureDag.execute() ──

    def begin_sample(self, _batch_sample_id: int = 0) -> None:
        if self._total_seen >= self.config.max_trace_samples:
            self._current = None
            self._total_seen += 1
            return
        self._current = SampleTrace(self._total_seen)
        self._total_seen += 1

    def end_sample(self) -> None:
        if self._current is not None:
            self.traces.append(self._current)
            self._current = None

    def trace_defaults(self, context: dict[str, Any]) -> None:
        if self._current is None:
            return
        out = {n: ValueSnapshot.of(v) for n, v in context.items()}
        self._current.stages.append(StageTrace(StageType.DEFAULT_INIT, "", outputs=out))

    def trace_overrides(self, context: dict[str, Any], overridden: list[str]) -> None:
        if self._current is None:
            return
        inp = {n: ValueSnapshot.of(v) for n, v in context.items() if n not in overridden}
        out = {n: ValueSnapshot.of(v) for n, v in context.items()}
        self._current.stages.append(
            StageTrace(StageType.RAW_OVERRIDE, "", inputs=inp, outputs=out, overridden=overridden)
        )

    def trace_operator(
        self,
        op_name: str,
        input_names: list[str],
        input_vals: list[Any],
        output_names: list[str],
        output_val: Any,
    ) -> None:
        """Record one operator execution."""
        if self._current is None:
            return
        inp = {n: ValueSnapshot.of(v) for n, v in zip(input_names, input_vals)}
        out = {n: ValueSnapshot.of(output_val) for n in output_names}
        stage = StageTrace(StageType.OPERATOR, op_name, inputs=inp, outputs=out)
        # Anomaly detection
        for n, vs in inp.items():
            self._check_anomalies(n, vs, op_name, stage.anomalies)
        for n, vs in out.items():
            self._check_anomalies(n, vs, op_name, stage.anomalies)
        self._current.stages.append(stage)

    # ── anomaly detection ──

    def _check_anomalies(self, name: str, vs: ValueSnapshot, op: str, out: list[Anomaly]) -> None:
        v = vs.value
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out.append(Anomaly(name, "NaN/Inf", v))
        elif isinstance(v, list) and len(v) == 0:
            out.append(Anomaly(name, "empty_list"))

    # ── summary / output ──

    def build_summary(self) -> dict:
        per_feature: dict[str, dict] = {}
        per_operator: dict[str, dict] = {}
        total = len(self.traces)
        for t in self.traces:
            for stage in t.stages:
                op = stage.stage_name or stage.stage_type.value
                e = per_operator.setdefault(op, {"executions": 0, "anomalies": 0})
                e["executions"] += 1
                e["anomalies"] += len(stage.anomalies)
                for name, vs in stage.outputs.items():
                    f = per_feature.setdefault(name, {"missing": 0, "anomalies": 0})
                    if (
                        vs.value is None
                        or vs.value == ""
                        or (isinstance(vs.value, float) and math.isnan(vs.value))
                    ):
                        f["missing"] += 1
                    f["anomalies"] += sum(1 for a in stage.anomalies if a.feature == name)
        return {"total_samples": total, "per_feature": per_feature, "per_operator": per_operator}

    def save(self) -> None:
        if not self.config.output_dir:
            return
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1_000_000)
        # Summary
        summary = self.build_summary()
        sp = Path(self.config.output_dir) / f"summary_{ts}.json"
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        # Traces (JSONL)
        tp = Path(self.config.output_dir) / f"traces_{ts}.jsonl"
        with open(tp, "w", encoding="utf-8") as f:
            for t in self.traces:
                f.write(json.dumps(t.to_dict(), default=str) + "\n")
        print(f"[Debug] summary → {sp}")
        print(f"[Debug] traces → {tp} ({len(self.traces)} samples)")


@dataclass
class SampleTrace:
    sample_id: int
    stages: list[StageTrace] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"sample": self.sample_id, "stages": [s.to_dict() for s in self.stages]}
