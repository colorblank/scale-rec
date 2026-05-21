from __future__ import annotations

"""训练编排器：验证集收集、多 epoch 训练、checkpoint、耗时统计。"""

import logging
import time
from dataclasses import dataclass
from typing import Iterator

import torch

from .config import FlowConfig
from .dag import FeatureDag
from .data import stream_file_batches
from .export import export_to_safetensors
from .metrics import Batch, compute_aucs, compute_loss, _to_device

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """训练超参数。"""

    epochs: int = 30
    batch_size: int = 64
    lr: float = 0.005
    weight_decay: float = 1e-4
    eval_samples: int = 2000
    eval_interval: int = 50
    log_interval: int = 10
    export_path: str = ""


def _collect_batches(batches: Iterator[Batch], max_samples: int) -> list[Batch]:
    result: list[Batch] = []
    collected = 0
    for b in batches:
        result.append(b)
        collected += len(b["features"])
        if collected >= max_samples:
            break
    return result


class Trainer:
    """训练编排器，封装验证集收集、epoch 循环、日志、checkpoint。"""

    def __init__(
        self,
        model: torch.nn.Module,
        dag: FeatureDag,
        task_names: list[str],
        label_map: dict[str, str],
        device: torch.device,
        config: TrainConfig,
        *,
        data_path: str,
        flow_config: FlowConfig,
        has_header: bool = True,
        sep: str = "\t",
        null_markers: set[str] | None = None,
    ):
        self.model = model
        self.dag = dag
        self.task_names = task_names
        self.label_map = label_map
        self.device = device
        self.cfg = config

        self._data_path = data_path
        self._flow_config = flow_config
        self._has_header = has_header
        self._sep = sep
        self._null_markers = null_markers

        self.eval_batches: list[Batch] = []
        self._n_eval_batches = 0

    # ── 公开 API ──

    def fit(self) -> float:
        """完整训练流程：收集验证集 → 多 epoch 训练 → 返回 best AUC。"""
        self._collect_eval()

        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.optimizer = opt
        best = 0.0

        for epoch in range(1, self.cfg.epochs + 1):
            avg_loss = self._train_epoch(epoch)
            aucs = self._validate()
            self._log_epoch(epoch, avg_loss, aucs)

            cur = max(aucs.values(), default=0)
            if cur > best:
                best = cur
                self._save_checkpoint(best)

        return best

    # ── 内部 ──

    def _collect_eval(self) -> None:
        reader = stream_file_batches(
            self._data_path, self._flow_config, self.cfg.batch_size,
            has_header=self._has_header, sep=self._sep,
            null_markers=self._null_markers,
        )
        self.eval_batches = _collect_batches(reader, self.cfg.eval_samples)
        self._n_eval_batches = len(self.eval_batches)
        n = sum(len(b["features"]) for b in self.eval_batches)
        logger.info("validation: %d samples (%d batches)", n, self._n_eval_batches)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        t_data = t_preproc = t_forward = t_loss = t_backward = 0.0
        t0_epoch = time.perf_counter()

        stream = enumerate(
            stream_file_batches(
                self._data_path, self._flow_config, self.cfg.batch_size,
                has_header=self._has_header, sep=self._sep,
                null_markers=self._null_markers,
            )
        )
        t0_iter = time.perf_counter()
        for i, batch in stream:
            t_data += time.perf_counter() - t0_iter
            if i < self._n_eval_batches:
                t0_iter = time.perf_counter()
                continue

            t0 = time.perf_counter()
            feat = _to_device(self.dag.preprocess_batch(batch["features"]), self.device)
            t_preproc += time.perf_counter() - t0

            t0 = time.perf_counter()
            outputs = self.model(feat)
            t_forward += time.perf_counter() - t0

            t0 = time.perf_counter()
            loss = compute_loss(outputs, batch["labels"], self.label_map)
            t_loss += time.perf_counter() - t0

            if loss is None:
                continue

            t0 = time.perf_counter()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            t_backward += time.perf_counter() - t0

            total_loss += loss.item()
            n_batches += 1

            if n_batches % self.cfg.log_interval == 0:
                logger.info(
                    "  batch %4d  avg_loss=%.6f  cur_loss=%.6f",
                    n_batches, total_loss / n_batches, loss.item(),
                )

            if n_batches % self.cfg.eval_interval == 0:
                self._eval_during_training(n_batches, total_loss)

            t0_iter = time.perf_counter()

        avg_loss = total_loss / max(n_batches, 1)
        self._log_timing(epoch, time.perf_counter() - t0_epoch, n_batches,
                         t_data, t_preproc, t_forward, t_loss, t_backward)
        return avg_loss

    def _validate(self) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            aucs = compute_aucs(
                self.model, self.dag, self.eval_batches,
                self.task_names, self.label_map, self.device,
            )
        return aucs

    def _eval_during_training(self, n_batches: int, total_loss: float) -> None:
        t0 = time.perf_counter()
        aucs = self._validate()
        t_eval = (time.perf_counter() - t0) * 1000
        self.model.train()
        parts = [f"batch {n_batches:4d}  loss={total_loss / n_batches:.6f}  eval={t_eval:.0f}ms"]
        for t in sorted(self.task_names):
            if t in aucs and t != "stay":
                parts.append(f"{t}: auc={aucs[t]:.4f}")
        logger.info("  " + "  ".join(parts))

    def _save_checkpoint(self, auc: float) -> None:
        export_to_safetensors(self.model, self.cfg.export_path)
        logger.info("checkpoint saved to %s (best auc=%.4f)", self.cfg.export_path, auc)

    # ── 日志 ──

    def _log_epoch(self, epoch: int, avg_loss: float, aucs: dict[str, float]) -> None:
        parts = [f"epoch {epoch:3d}/{self.cfg.epochs}  loss={avg_loss:.6f}"]
        for t in sorted(self.task_names):
            if t in aucs and t != "stay":
                parts.append(f"{t}: auc={aucs[t]:.4f}")
        logger.info("  ".join(parts))

    def _log_timing(
        self, epoch: int, t_epoch: float, n: int,
        t_data: float, t_preproc: float, t_forward: float,
        t_loss: float, t_backward: float,
    ) -> None:
        if n == 0:
            return
        t_total = t_data + t_preproc + t_forward + t_loss + t_backward
        ms = lambda s: s * 1000 / n  # noqa: E731
        pct = lambda s: s / t_total * 100 if t_total > 0 else 0  # noqa: E731
        logger.info(
            "  [timing epoch %d] total=%.1fs  batches=%d | "
            "per_batch: data=%.1fms(%.0f%%) preproc=%.1fms(%.0f%%) "
            "forward=%.1fms(%.0f%%) loss=%.1fms(%.0f%%) backward=%.1fms(%.0f%%)",
            epoch, t_epoch, n,
            ms(t_data), pct(t_data), ms(t_preproc), pct(t_preproc),
            ms(t_forward), pct(t_forward), ms(t_loss), pct(t_loss),
            ms(t_backward), pct(t_backward),
        )
