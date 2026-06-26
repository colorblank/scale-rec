from __future__ import annotations

"""DAG 预处理：从执行结果构建模型输入 tensor。"""


import torch

from .config import PoolingMode, TruncationSide
from .feature_info import FeatureInfo


class DagPreprocessor:
    def __init__(self, feat_info: FeatureInfo) -> None:
        self._embed_infos = feat_info.embed_infos()
        self._embed_names = feat_info.embed_names()

    def preprocess(self, context: dict[str, list]) -> dict[str, torch.Tensor]:
        n_rows = len(next(iter(context.values()))) if context else 0
        embed_names = self._embed_names
        embed_infos = self._embed_infos

        tensors: dict[str, torch.Tensor] = {}
        for name in embed_names:
            col = context.get(name)
            pooling = embed_infos[name].pooling
            vals = _feature_values(col, n_rows)
            if pooling is not PoolingMode.FIRST:
                if not vals:
                    tensors[name] = torch.tensor([], dtype=torch.long)
                    continue
                if not isinstance(vals[0], list):
                    raise ValueError(
                        f"feature '{name}' pooling '{pooling.value}' requires list-valued inputs"
                    )
                if pooling is PoolingMode.FLATTEN:
                    seq_len = embed_infos[name].seq_len
                    if not seq_len or seq_len <= 0:
                        raise ValueError(f"feature '{name}' pooling flatten requires seq_len > 0")
                else:
                    seq_len = embed_infos[name].seq_len
                    if not seq_len or seq_len <= 0:
                        raise ValueError(
                            f"feature '{name}' pooling '{pooling.value}' requires fixed max_len > 0"
                        )
                trunc = embed_infos[name].truncation
                if trunc is TruncationSide.TAIL:
                    padded = [v[-seq_len:] + [0] * max(seq_len - len(v), 0) for v in vals]
                else:
                    padded = [v[:seq_len] + [0] * max(seq_len - len(v), 0) for v in vals]
                tensors[name] = torch.tensor(padded, dtype=torch.long)
            else:
                if vals and isinstance(vals[0], list):
                    vals = [v[0] if isinstance(v, list) and v else v for v in vals]
                tensors[name] = torch.tensor(vals, dtype=torch.long)
        return tensors

    def embed_names(self) -> tuple[str, ...]:
        return self._embed_names


def _feature_values(col: list | None, n_rows: int) -> list:
    if n_rows <= 0:
        return []
    if col is None:
        return [0] * n_rows
    if len(col) >= n_rows:
        return col
    return [*col, *([0] * (n_rows - len(col)))]


class TrainingPreprocessor:
    """训练预处理管线：包装 FeatureDag 供训练/评估使用。"""

    def __init__(self, dag: object) -> None:
        self._dag = dag

    def preprocess_batch(self, rows: list[dict] | dict[str, list]) -> dict[str, torch.Tensor]:
        return self._dag.preprocess_batch(rows)  # type: ignore

    @property
    def dag(self) -> object:
        return self._dag
