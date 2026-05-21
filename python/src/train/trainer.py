from __future__ import annotations

"""训练编排器：验证集收集、多 epoch 训练、checkpoint、耗时统计。"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Iterator

import torch

from .config import FlowConfig
from .dag import FeatureDag
from .data import stream_file_batches
from .export import export_to_safetensors
from .metrics import Batch, MultiTaskLoss, _to_device, compute_aucs

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

    # ── LR warmup + cosine decay ──
    warmup_epochs: int = 2
    min_lr_ratio: float = 0.01  # 最终 lr = lr * min_lr_ratio

    # ── Gradient clipping ──
    grad_max_norm: float = 1.0

    # ── Early stopping ──
    early_stopping_patience: int = 5

    # ── EMA ──
    ema_decay: float = 0.999  # θ_ema = ema_decay * θ_ema + (1 - ema_decay) * θ
    ema_enabled: bool = True

    # ── Multi-task loss ──
    loss_weighting: str = "static"  # "static" | "equal" | "uncertainty"

    # ── TensorBoard ──
    tb_dir: str = ""  # TensorBoard 日志目录，空字符串=禁用
    tb_grad_interval: int = 100  # 每隔 N batch 记录梯度直方图


def _collect_batches(batches: Iterator[Batch], max_samples: int) -> list[Batch]:
    result: list[Batch] = []
    collected = 0
    for b in batches:
        result.append(b)
        collected += len(b["features"])
        if collected >= max_samples:
            break
    return result


class _LRScheduler:
    """Warmup → Cosine decay learning rate scheduler (step-wise)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.min_lr = base_lr * min_lr_ratio

    def step(self, epoch: int) -> None:
        if epoch <= self.warmup:
            lr = self.base_lr * epoch / max(self.warmup, 1)
        else:
            progress = (epoch - self.warmup) / max(self.total - self.warmup, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class _EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}
        self._model = model

    def update(self) -> None:
        with torch.no_grad():
            for name, p in self._model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(p, alpha=1 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> None:
        """Copy EMA weights into target model."""
        sd = model.state_dict()
        for name, p in self.shadow.items():
            if name in sd:
                sd[name].copy_(p)


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

        self.loss_fn = MultiTaskLoss(task_names, label_map, mode=self.cfg.loss_weighting)
        self.lr_scheduler: _LRScheduler | None = None
        self.ema: _EMA | None = None
        self._best_auc = 0.0
        self._stale_epochs = 0
        self._tb_writer = None
        self._global_step = 0

    # ── 公开 API ──

    def fit(self) -> float:
        self._collect_eval()

        self.loss_fn = self.loss_fn.to(self.device)
        opt = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.optimizer = opt
        self.lr_scheduler = _LRScheduler(
            opt,
            self.cfg.lr,
            self.cfg.warmup_epochs,
            self.cfg.epochs,
            self.cfg.min_lr_ratio,
        )
        if self.cfg.ema_enabled:
            self.ema = _EMA(self.model, self.cfg.ema_decay)

        if self.cfg.tb_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(self.cfg.tb_dir)
                logger.info("tensorboard: %s", self.cfg.tb_dir)
            except Exception as e:
                logger.warning("tensorboard 初始化失败 (pip install tensorboard): %s", e)
                self._tb_writer = None

        self._best_auc = 0.0
        self._stale_epochs = 0
        self._global_step = 0
        best_epoch = 0

        for epoch in range(1, self.cfg.epochs + 1):
            self.lr_scheduler.step(epoch)
            avg_loss = self._train_epoch(epoch)
            aucs = self._validate()
            self._log_epoch(epoch, avg_loss, aucs)

            # TensorBoard scalars
            if self._tb_writer is not None:
                self._tb_writer.add_scalar("train/loss", avg_loss, epoch)
                self._tb_writer.add_scalar("train/lr", self.lr_scheduler.current_lr(), epoch)
                for t, v in aucs.items():
                    self._tb_writer.add_scalar(f"val/auc_{t}", v, epoch)

            cur = max(aucs.values(), default=0)
            if cur > self._best_auc:
                self._best_auc = cur
                self._stale_epochs = 0
                best_epoch = epoch
                self._save_checkpoint(cur)
            else:
                self._stale_epochs += 1

            if self._stale_epochs >= self.cfg.early_stopping_patience:
                logger.info(
                    "early stopping at epoch %d (patience=%d, best=%.4f@epoch%d)",
                    epoch,
                    self.cfg.early_stopping_patience,
                    self._best_auc,
                    best_epoch,
                )
                break

        if self._tb_writer is not None:
            self._tb_writer.close()

        # 最终导出 EMA 权重
        if self.ema is not None:
            self.ema.apply_to(self.model)
            export_to_safetensors(self.model, self.cfg.export_path)
            logger.info("EMA weights exported to %s", self.cfg.export_path)

        return self._best_auc

    # ── 内部 ──

    def _collect_eval(self) -> None:
        reader = stream_file_batches(
            self._data_path,
            self._flow_config,
            self.cfg.batch_size,
            has_header=self._has_header,
            sep=self._sep,
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
                self._data_path,
                self._flow_config,
                self.cfg.batch_size,
                has_header=self._has_header,
                sep=self._sep,
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
            loss = self.loss_fn(outputs, batch["labels"])
            t_loss += time.perf_counter() - t0

            if loss is None:
                continue

            t0 = time.perf_counter()
            self.optimizer.zero_grad()
            loss.backward()
            grad_pre = self._grad_global_norm() if self._tb_writer is not None else 0.0
            if self.cfg.grad_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_max_norm)
            if self._tb_writer is not None:
                self._tb_writer.add_scalar("grad/pre_clip_norm", grad_pre, self._global_step)
            self.optimizer.step()
            t_backward += time.perf_counter() - t0

            if self.ema is not None:
                self.ema.update()

            total_loss += loss.item()
            n_batches += 1
            self._global_step += 1

            # TensorBoard: 梯度直方图（post-clip, 仅 weight/bias）
            if self._tb_writer is not None and n_batches % self.cfg.tb_grad_interval == 0:
                self._tb_writer.add_scalar(
                    "grad/post_clip_norm", self._grad_global_norm(), self._global_step
                )
                for name, p in self.model.named_parameters():
                    if p.grad is not None and ("weight" in name or "bias" in name):
                        self._tb_writer.add_histogram(f"grad/{name}", p.grad, self._global_step)

            if n_batches % self.cfg.log_interval == 0:
                logger.info(
                    "  batch %4d  avg_loss=%.6f  cur_loss=%.6f  lr=%.2e",
                    n_batches,
                    total_loss / n_batches,
                    loss.item(),
                    self.lr_scheduler.current_lr() if self.lr_scheduler else self.cfg.lr,
                )

            if n_batches % self.cfg.eval_interval == 0:
                self._eval_during_training(n_batches, total_loss)

            t0_iter = time.perf_counter()

        avg_loss = total_loss / max(n_batches, 1)
        self._log_timing(
            epoch,
            time.perf_counter() - t0_epoch,
            n_batches,
            t_data,
            t_preproc,
            t_forward,
            t_loss,
            t_backward,
        )
        return avg_loss

    def _validate(self) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            aucs = compute_aucs(
                self.model,
                self.dag,
                self.eval_batches,
                self.task_names,
                self.label_map,
                self.device,
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

    def _grad_global_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total**0.5

    def _save_checkpoint(self, auc: float) -> None:
        export_to_safetensors(self.model, self.cfg.export_path)
        logger.info("checkpoint saved to %s (best auc=%.4f)", self.cfg.export_path, auc)

    # ── 日志 ──

    def _log_epoch(self, epoch: int, avg_loss: float, aucs: dict[str, float]) -> None:
        lr = self.lr_scheduler.current_lr() if self.lr_scheduler else self.cfg.lr
        parts = [f"epoch {epoch:3d}/{self.cfg.epochs}  loss={avg_loss:.6f}  lr={lr:.2e}"]
        for t in sorted(self.task_names):
            if t in aucs and t != "stay":
                parts.append(f"{t}: auc={aucs[t]:.4f}")
        logger.info("  ".join(parts))
        weights = self.loss_fn.task_weights_info()
        if self.cfg.loss_weighting != "equal":
            w_str = "  ".join(f"w({t})={weights.get(t, 1):.3f}" for t in sorted(weights))
            logger.info("  [%s] %s", self.cfg.loss_weighting, w_str)

    def _log_timing(
        self,
        epoch: int,
        t_epoch: float,
        n: int,
        t_data: float,
        t_preproc: float,
        t_forward: float,
        t_loss: float,
        t_backward: float,
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
            epoch,
            t_epoch,
            n,
            ms(t_data),
            pct(t_data),
            ms(t_preproc),
            pct(t_preproc),
            ms(t_forward),
            pct(t_forward),
            ms(t_loss),
            pct(t_loss),
            ms(t_backward),
            pct(t_backward),
        )
