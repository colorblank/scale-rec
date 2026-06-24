from __future__ import annotations

"""独立评估模块：可配置指标 + 日志文件输出。"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...core.config import EvalConfig
from ...core.model_output import ensure_model_output
from ...core.output_contract import NormalizedOutputContract
from ...core.preprocessor import TrainingPreprocessor
from ..loss.multi_task import _pick_labels, _to_device
from ..loss.objective import evaluate_mask
from ..metrics import compute_metrics, get_available_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: EvalConfig) -> None:
        self.cfg = config
        unknown = sorted(set(config.metrics) - set(get_available_metrics()))
        if unknown:
            raise ValueError(f"Unknown eval metrics: {unknown}")

    def evaluate(
        self,
        model: torch.nn.Module,
        preprocessor: TrainingPreprocessor,
        batches: list[dict[str, Any]],
        task_names: list[str],
        label_map: dict[str, str],
        device: torch.device | None = None,
        task_metrics: dict[str, list[str]] | None = None,
        output_kinds: dict[str, str] | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> dict[str, dict[str, float]]:
        """评估所有 task × metric。

        Returns: {task_name: {metric_name: value}}
        """
        if device is None:
            device = next(model.parameters()).device

        was_training = model.training
        model.eval()

        if output_contract is not None:
            results = self._evaluate_contract(
                model,
                preprocessor,
                batches,
                output_contract,
                device,
            )
            if was_training:
                model.train()
            if self.cfg.log_path:
                self._write_log(results, task_names)
            return results

        # 收集预测
        logits_buf: dict[str, list[np.ndarray]] = {t: [] for t in task_names}
        labels_buf: dict[str, list[np.ndarray]] = {t: [] for t in task_names}
        groups_buf: dict[str, list[np.ndarray]] = {t: [] for t in task_names}

        with torch.no_grad():
            for batch in batches:
                features = batch["features"]
                if isinstance(features, dict):
                    rows = features
                else:
                    rows = [{k: v for k, v in r.items() if v is not None} for r in features]
                features_on_device = _to_device(preprocessor.preprocess_batch(rows), device)
                if hasattr(model, "forward_execution"):
                    outputs = model.forward_execution(features_on_device).nodes
                else:
                    outputs = ensure_model_output(model(features_on_device))

                # 提取分组特征
                group_ids: np.ndarray | None = None
                has_gauc = "gauc" in self.cfg.metrics
                if has_gauc:
                    gf = self.cfg.gauc_group_feature
                    if isinstance(features, dict):
                        gids = features.get(gf, [0] * len(next(iter(features.values()))))
                    else:
                        gids = [r.get(gf, 0) for r in features]
                    group_ids = np.array([g if g is not None else 0 for g in gids], dtype=object)

                for t in task_names:
                    if t not in outputs:
                        continue
                    col = label_map.get(t, t)
                    raw = _pick_labels(batch["labels"], col, t)
                    arr = np.array(
                        [float(v) if v is not None else np.nan for v in raw], dtype=np.float32
                    )
                    mask = ~np.isnan(arr)
                    if not mask.any():
                        continue
                    logits_buf[t].append(outputs.tensor(t).cpu().numpy().flatten()[mask])
                    labels_buf[t].append(arr[mask])
                    if has_gauc and group_ids is not None:
                        groups_buf[t].append(group_ids[mask])

        if was_training:
            model.train()

        # 按 task 计算
        results: dict[str, dict[str, float]] = {}
        for t in task_names:
            if not logits_buf[t] or not labels_buf[t]:
                continue
            y = np.concatenate(labels_buf[t])
            p = np.concatenate(logits_buf[t])
            if len(y) == 0:
                continue
            g = np.concatenate(groups_buf[t]) if groups_buf.get(t) else None
            metrics = (task_metrics or {}).get(t, self.cfg.metrics)
            if not metrics:
                continue
            results[t] = compute_metrics(
                y,
                p,
                metrics,
                group_ids=g,
                output_kind=(output_kinds or {}).get(t, "binary_logit"),
            )

        # 日志文件
        if self.cfg.log_path:
            self._write_log(results, task_names)

        return results

    def _evaluate_contract(
        self,
        model: torch.nn.Module,
        preprocessor: TrainingPreprocessor,
        batches: list[dict[str, Any]],
        contract: NormalizedOutputContract,
        device: torch.device,
    ) -> dict[str, dict[str, float]]:
        prediction_buf: dict[str, list[np.ndarray]] = {
            metric.name: [] for metric in contract.metrics
        }
        label_buf: dict[str, list[np.ndarray]] = {metric.name: [] for metric in contract.metrics}
        with torch.no_grad():
            for batch in batches:
                features = batch["features"]
                rows = (
                    features
                    if isinstance(features, dict)
                    else [
                        {key: value for key, value in row.items() if value is not None}
                        for row in features
                    ]
                )
                tensors = _to_device(preprocessor.preprocess_batch(rows), device)
                nodes = model.forward_execution(tensors).nodes
                batch_values = _raw_batch_values(features, batch["labels"])
                for metric in contract.metrics:
                    output = nodes.get(metric.source)
                    if output is None:
                        raise ValueError(
                            f"metric '{metric.name}' source '{metric.source}' is missing"
                        )
                    raw = _pick_labels(batch["labels"], metric.label)
                    labels = np.array(
                        [float(value) if value is not None else np.nan for value in raw],
                        dtype=np.float32,
                    )
                    valid = ~np.isnan(labels)
                    if metric.mask is not None:
                        valid &= evaluate_mask(
                            metric.mask,
                            batch_values,
                            len(labels),
                        )
                    if not valid.any():
                        continue
                    prediction_buf[metric.name].append(output.tensor.cpu().numpy().flatten()[valid])
                    label_buf[metric.name].append(labels[valid])

        results: dict[str, dict[str, float]] = {}
        for metric in contract.metrics:
            if not prediction_buf[metric.name]:
                continue
            predictions = np.concatenate(prediction_buf[metric.name])
            labels = np.concatenate(label_buf[metric.name])
            value = compute_metrics(
                labels,
                predictions,
                [metric.type],
                output_kind=contract.node_kinds[metric.source],
            )[metric.type]
            results.setdefault(metric.source, {})[metric.type] = value
        return results

    def _write_log(self, results: dict[str, dict[str, float]], task_names: list[str]) -> None:
        path = Path(self.cfg.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}]")
            for t in sorted(results):
                for m, v in results[t].items():
                    f.write(f" {t}_{m}={v:.6f}")
            f.write("\n")
        logger.info("eval log saved to %s", path)


def _raw_batch_values(features: Any, labels: dict[str, list[Any]]) -> dict[str, Any]:
    if isinstance(features, dict):
        values = dict(features)
    else:
        names = {name for row in features for name in row}
        values = {name: [row.get(name) for row in features] for name in names}
    values.update(labels)
    return values
