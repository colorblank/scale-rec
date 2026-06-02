from __future__ import annotations

"""独立评估模块：可配置指标 + 日志文件输出。"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ...core.config import EvalConfig
from ...core.dag import FeatureDag
from ..loss.multi_task import _pick_labels, _to_device
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
        dag: FeatureDag,
        batches: list[dict[str, Any]],
        task_names: list[str],
        label_map: dict[str, str],
        device: Optional[torch.device] = None,
    ) -> dict[str, dict[str, float]]:
        """评估所有 task × metric。

        Returns: {task_name: {metric_name: value}}
        """
        if device is None:
            device = next(model.parameters()).device

        was_training = model.training
        model.eval()

        # 收集预测
        logits_buf: dict[str, list[np.ndarray]] = {t: [] for t in task_names}
        labels_buf: dict[str, list[np.ndarray]] = {t: [] for t in task_names}
        groups_buf: dict[str, list[np.ndarray]] = {t: [] for t in task_names}

        with torch.no_grad():
            for batch in batches:
                rows = [{k: v for k, v in r.items() if v is not None} for r in batch["features"]]
                outputs = model(_to_device(dag.preprocess_batch(rows), device))

                # 提取分组特征
                group_ids: Optional[np.ndarray] = None
                has_gauc = "gauc" in self.cfg.metrics
                if has_gauc:
                    gf = self.cfg.gauc_group_feature
                    gids = [r.get(gf, 0) for r in batch["features"]]
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
                    logits_buf[t].append(outputs[t].cpu().numpy().flatten()[mask])
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
            results[t] = compute_metrics(y, p, self.cfg.metrics, group_ids=g)

        # 日志文件
        if self.cfg.log_path:
            self._write_log(results, task_names)

        return results

    def _write_log(self, results: dict[str, dict[str, float]], task_names: list[str]) -> None:
        path = Path(self.cfg.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}]")
            for t in sorted(results):
                for m, v in results[t].items():
                    f.write(f" {t}_{m}={v:.6f}")
            f.write("\n")
        logger.info("eval log saved to %s", path)
