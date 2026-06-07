from __future__ import annotations

"""训练编排器。"""

import logging
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import torch

from ..app.artifacts import TrainingArtifactManager
from ..app.data import (
    estimate_files_batches,
    estimate_files_rows,
    stream_file_batches,
    stream_files_batches,
)
from ..app.export import export_to_safetensors
from ..core.config import FlowConfig, TrainConfig
from ..core.dag import FeatureDag
from ..core.task import TaskSpec, legacy_task_specs
from .eval.evaluator import Evaluator
from .loss.multi_task import MultiTaskLoss, _to_device
from .optim.scheduler import LRScheduler, build_optimizer
from .quality import FeatureQualityReport, summarize_feature_quality

logger = logging.getLogger(__name__)

Batch = dict[str, Any]

_MINIMIZE_METRICS = {"logloss", "mae", "mse"}


def _estimate_rows(path: str, has_header: bool = True) -> int:
    return estimate_files_rows([path], has_header=has_header)


def _estimate_batches(path: str, batch_size: int, has_header: bool = True) -> int:
    return estimate_files_batches([path], batch_size, has_header=has_header)


def _collect_batches(batches: Iterator[Batch], max_samples: int) -> list[Batch]:
    result: list[Batch] = []
    collected = 0
    for batch in batches:
        result.append(batch)
        features = batch["features"]
        size = len(next(iter(features.values()))) if isinstance(features, dict) else len(features)
        collected += size
        if collected >= max_samples:
            break
    return result


class _EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}
        self._model = model

    def update(self) -> None:
        with torch.no_grad():
            for name, p in self._model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(p, alpha=1 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for name, p in self.shadow.items():
            if name in state:
                state[name].copy_(p)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dag: FeatureDag,
        task_names: list[str],
        label_map: dict[str, str],
        device: torch.device,
        config: TrainConfig,
        *,
        model_type: str = "",
        flow_config: FlowConfig,
        data_path: Optional[str] = None,
        data_paths: Optional[list[str]] = None,
        has_header: bool = True,
        sep: str = "\t",
        null_markers: Optional[set[str]] = None,
        read_chunk_rows: Optional[int] = None,
        fast_no_na: bool = False,
        memory_map: bool = False,
        task_specs: Optional[list[TaskSpec]] = None,
        artifact_manager: Optional[TrainingArtifactManager] = None,
        repo_root: Union[str, Path, None] = None,
    ) -> None:
        self.model = model
        self.dag = dag
        self.cfg = config
        self.task_specs = (
            task_specs
            or config.tasks
            or legacy_task_specs(
                task_names,
                label_map,
                config.task_weights,
                default_metrics=list(config.eval.metrics),
            )
        )
        self.task_names = [spec.name for spec in self.task_specs]
        self.label_map = {spec.name: spec.label for spec in self.task_specs}
        self._validate_task_contract(flow_config)
        missing_metrics = [spec.name for spec in self.task_specs if not spec.metrics]
        if missing_metrics:
            raise ValueError(
                "Task specs must define metrics in configuration: " + ", ".join(missing_metrics)
            )
        self.task_metrics = {spec.name: list(spec.metrics) for spec in self.task_specs}
        self._monitor_task, self._monitor_metric, self._monitor_mode = self._resolve_monitor()
        self.device = device
        self.model_type = model_type

        if data_paths is None:
            if data_path is None:
                raise ValueError("Trainer requires data_path or data_paths")
            data_paths = [data_path]
        if not data_paths:
            raise ValueError("Trainer data_paths must not be empty")
        self._data_paths = [str(path) for path in data_paths]
        self._flow_config = flow_config
        self._has_header = has_header
        self._sep = sep
        self._null_markers = null_markers
        self._read_chunk_rows = read_chunk_rows
        self._fast_no_na = fast_no_na
        self._memory_map = memory_map
        self.artifacts = artifact_manager
        self.repo_root = repo_root

        self.eval_batches: list[Batch] = []
        self._n_eval_batches = 0
        self.feature_quality: Optional[FeatureQualityReport] = None
        self.loss_fn = MultiTaskLoss(
            self.task_names,
            self.label_map,
            mode=config.loss_weighting,
            task_weights=config.task_weights,
            task_specs=self.task_specs,
        )
        self.lr_scheduler: Optional[LRScheduler] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.ema: Optional[_EMA] = None
        self.evaluator = Evaluator(config.eval)
        self._best_score = self._initial_best_score()
        self._stale_epochs = 0
        self._global_step = 0

        from ..models.params import format_parameter_summary

        logger.info("Model parameters: %s", format_parameter_summary(model))

    def fit(self) -> float:
        self._collect_eval()
        self.loss_fn = self.loss_fn.to(self.device)

        total_rows = estimate_files_rows(self._data_paths, has_header=self._has_header)
        eval_rows = sum(
            len(next(iter(b["features"].values())))
            if isinstance(b["features"], dict)
            else len(b["features"])
            for b in self.eval_batches
        )
        train_rows = max(total_rows - eval_rows, 0)
        total_batches = estimate_files_batches(
            self._data_paths, self.cfg.batch_size, has_header=self._has_header
        )
        eval_batches = self._n_eval_batches
        train_batches = max(total_batches - eval_batches, 1)

        logger.info(
            "data: files=%d rows(total=%d train=%d eval=%d) batch_size=%d batches(total~%d train~%d eval=%d) tasks=%s labels=%s",
            len(self._data_paths),
            total_rows,
            train_rows,
            eval_rows,
            self.cfg.batch_size,
            total_batches,
            train_batches,
            eval_batches,
            self.task_names,
            self.label_map,
        )
        logger.info(
            "reader: pandas read_chunk_rows=%s fast_no_na=%s memory_map=%s",
            self._effective_read_chunk_rows(),
            self._fast_no_na,
            self._memory_map,
        )

        emb_params: list[torch.nn.Parameter] = []
        dense_params: list[torch.nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                (emb_params if "emb_" in name else dense_params).append(param)

        cfg = self.cfg.optim
        emb_lr = cfg.emb_lr if cfg.emb_lr is not None else cfg.lr
        emb_wd = cfg.emb_weight_decay if cfg.emb_weight_decay is not None else cfg.weight_decay
        param_groups = [
            {
                "params": emb_params,
                "lr": emb_lr,
                "weight_decay": emb_wd,
                "momentum": cfg.momentum,
            },
            {
                "params": dense_params + list(self.loss_fn.parameters()),
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "momentum": cfg.momentum,
            },
        ]
        self.optimizer = build_optimizer(param_groups, cfg.name)

        batches_per_epoch = train_batches
        total_steps = self.cfg.epochs * batches_per_epoch
        warmup = min(self.cfg.lr_schedule.warmup_steps, max(1, total_steps // 10))
        self.lr_scheduler = LRScheduler(
            [self.optimizer],
            warmup,
            total_steps,
            self.cfg.lr_schedule.min_lr_ratio,
        )
        logger.info(
            "optimizer: %s, emb_lr=%.0e lr=%.0e, warmup=%d/%d steps",
            cfg.name,
            emb_lr,
            cfg.lr,
            warmup,
            total_steps,
        )

        if self.cfg.ema_decay > 0:
            self.ema = _EMA(self.model, self.cfg.ema_decay)

        self._best_score = self._initial_best_score()
        self._stale_epochs = 0
        self._global_step = 0
        best_epoch = 0
        for epoch in range(1, self.cfg.epochs + 1):
            avg_loss = self._train_epoch(epoch)
            eval_results = self.evaluator.evaluate(
                self.model,
                self.dag,
                self.eval_batches,
                self.task_names,
                self.label_map,
                self.device,
                task_metrics=self.task_metrics,
            )
            if not eval_results:
                raise ValueError(
                    "No evaluation labels were collected. Check the dataset and the "
                    "configured label columns for this model."
                )
            self._log_epoch(epoch, avg_loss, eval_results)

            cur = self._monitor_score(eval_results)
            is_best = self._is_better(cur, self._best_score)
            if is_best:
                self._best_score = cur
                self._stale_epochs = 0
                best_epoch = epoch
            if self.artifacts is not None:
                self.artifacts.save_checkpoint(
                    self.model,
                    epoch=epoch,
                    step=self._global_step,
                    score=cur,
                    metric_name=self._monitor_name(),
                    is_best=is_best,
                )
            elif is_best:
                self._save_checkpoint(cur)
            else:
                self._stale_epochs += 1

            if (
                self.cfg.early_stopping_patience > 0
                and self._stale_epochs >= self.cfg.early_stopping_patience
            ):
                logger.info(
                    "early stopping at epoch %d (best=%.4f@%d)",
                    epoch,
                    self._best_score,
                    best_epoch,
                )
                break

        published_version = None
        if self.ema is not None:
            self.ema.apply_to(self.model)
            published_version = (
                f"{self.artifacts.paths.run_version}/ema-final" if self.artifacts else "ema-final"
            )
            export_to_safetensors(self.model, self.cfg.export_path)
            logger.info("EMA weights exported to %s", self.cfg.export_path)
        elif self.artifacts is not None and self.artifacts.best is not None:
            published_version = self.artifacts.best.version

        if self.artifacts is not None:
            published_source = (
                None if self.ema is not None else self.artifacts.paths.best_alias_path
            )
            self.artifacts.finalize(
                model=self.model if self.ema is not None else None,
                model_type=self.model_type,
                tasks=self.task_names,
                label_col_map=self.label_map,
                metrics=self.feature_quality_metrics(),
                repo_root=self.repo_root,
                published_version=published_version,
                best_score=self._best_score,
                published_source=published_source,
            )

        return self._best_score

    def _collect_eval(self) -> None:
        self.eval_batches = _collect_batches(self._iter_eval_batches(), self.cfg.eval_samples)
        self._n_eval_batches = len(self.eval_batches)
        n_samples = sum(
            len(next(iter(b["features"].values())))
            if isinstance(b["features"], dict)
            else len(b["features"])
            for b in self.eval_batches
        )
        logger.info(
            "validation: %d samples (%d batches) from %s",
            n_samples,
            self._n_eval_batches,
            self._data_paths[-1],
        )
        self.feature_quality = summarize_feature_quality(self.dag, self.eval_batches)
        logger.info("feature quality: rows=%d", self.feature_quality.rows)

    def feature_quality_metrics(self) -> dict[str, float]:
        if self.feature_quality is None:
            return {}
        return self.feature_quality.to_metrics()

    def _train_epoch(self, epoch: int) -> float:
        if self.optimizer is None or self.lr_scheduler is None:
            raise RuntimeError("Trainer.fit() must initialize optimizer before training")

        self.model.train()
        total_loss = 0.0
        n_batches = 0
        t_data = t_preproc = t_forward = t_loss = t_backward = 0.0
        t0_epoch = time.perf_counter()

        t0_iter = time.perf_counter()
        for batch in self._iter_batches():
            t_data += time.perf_counter() - t0_iter

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
                t0_iter = time.perf_counter()
                continue

            t0 = time.perf_counter()
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_max_norm)
            self.lr_scheduler.step(self._global_step + 1)
            self.optimizer.step()
            t_backward += time.perf_counter() - t0

            if self.ema is not None:
                self.ema.update()

            total_loss += loss.item()
            n_batches += 1
            self._global_step += 1

            if self.cfg.log_interval > 0 and n_batches % self.cfg.log_interval == 0:
                self._log_batch(n_batches, total_loss, loss.item())

            if self.cfg.eval_interval > 0 and n_batches % self.cfg.eval_interval == 0:
                self._eval_during_training(n_batches, total_loss)

            t0_iter = time.perf_counter()

        if n_batches == 0:
            expected_labels = ", ".join(sorted(set(self.label_map.values())))
            label_sources = ", ".join(s.name for s in self._flow_config.label_sources)
            raise ValueError(
                "No supervised batches were processed. Check that the dataset exposes "
                "the configured label columns and that the task config matches them. "
                f"expected labels=[{expected_labels}] label_sources=[{label_sources}] "
                f"batch_size={self.cfg.batch_size} eval_rows={self._n_eval_batches}"
            )

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

    def _iter_batches(self) -> Iterator[Batch]:
        if len(self._data_paths) > 1:
            yield from stream_files_batches(
                self._data_paths[:-1],
                self._flow_config,
                self.cfg.batch_size,
                **self._reader_kwargs(),
            )
        for idx, batch in enumerate(self._iter_last_file_batches()):
            if idx < self._n_eval_batches:
                continue
            yield batch

    def _iter_eval_batches(self) -> Iterator[Batch]:
        return self._iter_last_file_batches()

    def _iter_last_file_batches(self) -> Iterator[Batch]:
        return stream_file_batches(
            self._data_paths[-1],
            self._flow_config,
            self.cfg.batch_size,
            **self._reader_kwargs(),
        )

    def _reader_kwargs(self) -> dict[str, Any]:
        return {
            "has_header": self._has_header,
            "sep": self._sep,
            "null_markers": self._null_markers,
            "read_chunk_rows": self._read_chunk_rows,
            "fast_no_na": self._fast_no_na,
            "memory_map": self._memory_map,
        }

    def _iter_all_batches(self) -> Iterator[Batch]:
        return stream_files_batches(
            self._data_paths,
            self._flow_config,
            self.cfg.batch_size,
            **self._reader_kwargs(),
        )

    def _effective_read_chunk_rows(self) -> Union[int, str]:
        if self._read_chunk_rows is not None and self._read_chunk_rows > 0:
            return max(self._read_chunk_rows, self.cfg.batch_size)
        return "auto"

    def _eval_during_training(self, n_batches: int, total_loss: float) -> None:
        t0 = time.perf_counter()
        self.model.eval()
        results = self.evaluator.evaluate(
            self.model,
            self.dag,
            self.eval_batches,
            self.task_names,
            self.label_map,
            self.device,
            task_metrics=self.task_metrics,
        )
        self.model.train()
        t_eval = (time.perf_counter() - t0) * 1000
        parts = [f"batch {n_batches:4d}  loss={total_loss / n_batches:.6f}  eval={t_eval:.0f}ms"]
        for task in sorted(self.task_names):
            task_metrics = results.get(task, {})
            for metric, value in sorted(task_metrics.items()):
                parts.append(f"{task}_{metric}={value:.4f}")
        logger.info("  " + "  ".join(parts))

    def _validate_task_contract(self, flow_config: FlowConfig) -> None:
        if not self.task_specs:
            raise ValueError("Trainer requires at least one supervised task")
        duplicates = sorted({name for name in self.task_names if self.task_names.count(name) > 1})
        if duplicates:
            raise ValueError("Duplicate task specs: " + ", ".join(duplicates))
        label_sources = {source.name for source in flow_config.label_sources}
        missing_labels = sorted(
            {spec.label for spec in self.task_specs if spec.label not in label_sources}
        )
        if missing_labels:
            raise ValueError(
                "Task labels must be declared as feature config label sources: "
                + ", ".join(missing_labels)
            )

    def _resolve_monitor(self) -> tuple[str, str, str]:
        task = self.cfg.eval.monitor_task or self.task_names[0]
        if task not in self.task_names:
            raise ValueError(
                f"eval.monitor_task '{task}' is not a configured task: {self.task_names}"
            )
        metric = self.cfg.eval.monitor_metric
        task_metrics = self.task_metrics.get(task, [])
        if metric not in task_metrics:
            raise ValueError(
                f"eval.monitor_metric '{metric}' is not configured for task '{task}': "
                f"{task_metrics}"
            )
        mode = self.cfg.eval.monitor_mode
        if mode == "auto":
            mode = "min" if metric in _MINIMIZE_METRICS else "max"
        logger.info("monitor: %s.%s mode=%s", task, metric, mode)
        return task, metric, mode

    def _monitor_score(self, results: dict[str, dict[str, float]]) -> float:
        task_results = results.get(self._monitor_task)
        if not task_results or self._monitor_metric not in task_results:
            raise ValueError(
                f"Monitor metric '{self._monitor_name()}' was not produced by evaluation. "
                f"available={results}"
            )
        return task_results[self._monitor_metric]

    def _monitor_name(self) -> str:
        return f"{self._monitor_task}.{self._monitor_metric}"

    def _initial_best_score(self) -> float:
        return float("inf") if self._monitor_mode == "min" else float("-inf")

    def _is_better(self, current: float, best: float) -> bool:
        if self._monitor_mode == "min":
            return current < best
        return current > best

    def _log_batch(self, n_batches: int, total_loss: float, current_loss: float) -> None:
        if self.lr_scheduler is None:
            lr = self.cfg.optim.lr
        else:
            lr = self.lr_scheduler.current_lr()
        parts = [
            f"batch {n_batches:4d}  avg_loss={total_loss / n_batches:.6f}  cur_loss={current_loss:.6f}",
            f"lr={lr:.2e}",
        ]
        task_losses = self.loss_fn.last_losses()
        pos_rates = self.loss_fn.last_pos_rates()
        for task in sorted(task_losses):
            parts.append(f"{task}={task_losses[task]:.4f}(pr={pos_rates.get(task, 0):.2f})")
        logger.info("  " + "  ".join(parts))

    def _save_checkpoint(self, score: float) -> None:
        export_to_safetensors(self.model, self.cfg.export_path)
        logger.info("checkpoint: %s (%s=%.4f)", self.cfg.export_path, self._monitor_name(), score)

    def _log_epoch(
        self,
        epoch: int,
        avg_loss: float,
        results: dict[str, dict[str, float]],
    ) -> None:
        lr = self.lr_scheduler.current_lr() if self.lr_scheduler else self.cfg.optim.lr
        parts = [f"epoch {epoch:3d}/{self.cfg.epochs}  loss={avg_loss:.6f}  lr={lr:.2e}"]
        for task in sorted(self.task_names):
            for metric, value in sorted(results.get(task, {}).items()):
                parts.append(f"{task}_{metric}={value:.4f}")
        logger.info("  ".join(parts))

        weights = self.loss_fn.task_weights_info()
        if self.cfg.loss_weighting != "equal":
            weight_str = "  ".join(
                f"w({task})={weights.get(task, 1):.3f}" for task in sorted(weights)
            )
            logger.info("  [%s] %s", self.cfg.loss_weighting, weight_str)

    def _log_timing(
        self,
        epoch: int,
        t_epoch: float,
        n_batches: int,
        t_data: float,
        t_preproc: float,
        t_forward: float,
        t_loss: float,
        t_backward: float,
    ) -> None:
        if n_batches == 0:
            return
        t_total = t_data + t_preproc + t_forward + t_loss + t_backward
        ms = lambda seconds: seconds * 1000 / n_batches  # noqa: E731
        pct = lambda seconds: seconds / t_total * 100 if t_total > 0 else 0  # noqa: E731
        logger.info(
            "  [timing epoch %d] total=%.1fs batches=%d | per_batch: "
            "data=%.1fms(%.0f%%) preproc=%.1fms(%.0f%%) forward=%.1fms(%.0f%%) "
            "loss=%.1fms(%.0f%%) backward=%.1fms(%.0f%%)",
            epoch,
            t_epoch,
            n_batches,
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
