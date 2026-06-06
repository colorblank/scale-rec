"""数据加载模块：多日物品特征索引 + 训练文件流式读取。"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

from ..core.config import FlowConfig, parse_float_strict, parse_int_strict

logger = logging.getLogger(__name__)

NULL_MARKERS = {"NULL", "\\N", "null", "None", ""}
DTYPE_PANDAS = {"int": "Int64", "float": "float64", "string": "str", "enum": "str"}


def _parse_default(val_str: str, dtype_tag: str) -> Any:
    """按 dtype 解析配置中的 default_val。"""
    if dtype_tag == "int":
        return parse_int_strict(val_str) if val_str else 0
    elif dtype_tag == "float":
        return parse_float_strict(val_str) if val_str else 0.0
    return str(val_str)


def _build_reader_params(
    sources: list[dict],
    has_header: bool,
    separator: str,
    na_vals: list[str],
) -> dict[str, Any]:
    """从 source 配置构建 pandas read_csv 参数。

    返回: {names, dtype, na_values, default_vals}
      - names: 无 header 时的列名列表
      - dtype: 列名→pandas dtype 映射
      - na_values: NULL 字符串标记
      - default_vals: {列名: 缺失填充值}（来自 config 的 default_val）
    """
    names = [s["name"] for s in sources]
    dtype = {}
    default_vals: dict[str, Any] = {}
    for s in sources:
        n = s["name"]
        dt = s.get("dtype", "string")
        dtype[n] = DTYPE_PANDAS.get(dt, "str")
        default_val = s.get("default_val", "")
        if dt == "int":
            default_vals[n] = parse_int_strict(default_val) if default_val else 0
        elif dt == "float":
            default_vals[n] = parse_float_strict(default_val) if default_val else 0.0
        else:
            default_vals[n] = str(default_val) if default_val else ""

    params: dict[str, Any] = {
        "sep": separator,
        "na_values": na_vals,
        "keep_default_na": False,
        "dtype": dtype,
        "on_bad_lines": "skip",
    }
    if has_header:
        params["header"] = 0
    else:
        params["header"] = None
        params["names"] = names
    return params, names, dtype, default_vals


def _read_file_compat(path: str, params: dict, names: list[str]) -> pd.DataFrame:
    """兼容读取：如果列数不匹配（ragged lines），用宽松模式重试。"""
    try:
        df = pd.read_csv(path, **params)
    except (pd.errors.ParserError, ValueError):
        logger.exception("strict csv read failed for %s; retrying relaxed ragged-line mode", path)
        # 宽松模式：读为单列再 split
        df = pd.read_csv(
            path,
            sep="\n",
            header=None if params.get("header") == 0 else 0,
            na_values=params.get("na_values", []),
            keep_default_na=False,
        )
        df = df.iloc[:, 0].str.split(params["sep"], regex=False, expand=True)
        df = df.iloc[:, : len(names)]
        if params.get("header") is None:
            df.columns = names
        else:
            df = df.iloc[1:]
            df.columns = names[: len(df.columns)]
    return df


def build_item_index(
    item_files: list[str],
    item_sources: list[dict],
    has_header: bool = True,
    separator: str = "\t",
    null_markers: Optional[set[str]] = None,
) -> dict[str, dict[str, str]]:
    """用 pandas 读取多日物品文件，按 item_id 去重后构建索引。

    列名、类型、缺失填充值全部来自 item_sources 配置。
    后读文件覆盖先读文件中同 item_id 的记录（keep="last"）。

    Args:
        item_files: 物品文件列表，从旧到新排列。
        item_sources: 物品侧 source 定义列表 [{name, dtype, default_val}, ...]。
        has_header: 文件是否含 header 行。
        separator: 字段分隔符。
        null_markers: NULL 字符串集合。

    Returns:
        item_id → {feature_name: value} 映射表。
    """
    if null_markers is None:
        null_markers = NULL_MARKERS
    # 只保留 feature-role 的 source（物品文件不应含标签）
    feature_only = [s for s in item_sources if s.get("role", "feature") == "feature"]
    na_vals = list(null_markers)
    params, names, dtype, default_vals = _build_reader_params(
        feature_only, has_header, separator, na_vals
    )

    dfs = []
    for path in item_files:
        if not Path(path).exists():
            logger.warning("skip missing item file: %s", path)
            continue
        df = _read_file_compat(path, params, names)
        # 按配置填充缺失值
        for col, default in default_vals.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        keep = [c for c in names if c in df.columns]
        dfs.append(df[keep])

    if not dfs:
        return {}

    merged = pd.concat(dfs).drop_duplicates(subset="item_id", keep="last")
    logger.info("%d item files → %d unique items", len(dfs), len(merged))

    index: dict[str, dict[str, str]] = {}
    for row in merged.itertuples(index=False, name=None):
        d = {
            col: str(val) if not pd.isna(val) else ""
            for col, val in zip(merged.columns, row, strict=False)
        }
        item_id = d.pop("item_id", "")
        if item_id:
            index[item_id] = d
    return index


# ═══════════════════════════════════════════════════════════════════
# 训练文件流式读取
# ═══════════════════════════════════════════════════════════════════


def estimate_rows(path: str, has_header: bool = True) -> int:
    if not os.path.exists(path):
        return 1
    file_size = os.path.getsize(path)
    if file_size < 1024 * 1024 * 50:
        with open(path, "rb") as f:
            lines = sum(1 for _ in f)
        if has_header and lines > 0:
            lines -= 1
        return max(1, lines)
    with open(path, "rb") as f:
        if has_header:
            next(f, None)
        sample = f.read(1024 * 500)
        lines = sample.count(b"\n")
        if lines == 0 or len(sample) == 0:
            return 1
        avg_line_size = len(sample) / lines
        estimated = int(file_size / avg_line_size)
        if has_header:
            estimated = max(1, estimated - 1)
        return max(1, estimated)


def estimate_files_rows(paths: list[str], has_header: bool = True) -> int:
    return sum(estimate_rows(path, has_header=has_header) for path in paths)


def estimate_files_batches(paths: list[str], batch_size: int, has_header: bool = True) -> int:
    size = max(batch_size, 1)
    batches = sum(math.ceil(estimate_rows(path, has_header=has_header) / size) for path in paths)
    return max(1, batches)


def stream_file_batches(
    path: str,
    flow_config: FlowConfig,
    batch_size: int,
    *,
    has_header: bool = True,
    sep: str = "\t",
    null_markers: Optional[set[str]] = None,
    read_chunk_rows: Optional[int] = None,
    fast_no_na: bool = False,
    memory_map: bool = False,
) -> Iterator[dict[str, Any]]:
    """pandas chunk read 流式读取单文件，按 role 分离 feature/label/discard。

    列名、类型、缺失填充值全部来自 flow_config.sources 配置。

    Yields:
        {"features": [row_dict, ...], "labels": {label_name: [value, ...]}}
    """
    if null_markers is None:
        null_markers = NULL_MARKERS
    na_vals = list(null_markers)

    feature_sources = flow_config.feature_sources
    label_sources = flow_config.label_sources

    seen: set[str] = set()
    names: list[str] = []
    dtype: dict[str, str] = {}
    defaults: dict[str, Any] = {}
    for s in flow_config.sources:
        if s.name in seen:
            continue
        seen.add(s.name)
        names.append(s.name)
        dt = s.dtype.tag if hasattr(s.dtype, "tag") else str(s.dtype)
        dtype[s.name] = DTYPE_PANDAS.get(dt, "str")
        defaults[s.name] = _parse_default(s.default_val, dt)

    feature_names = _unique_source_names(feature_sources)
    label_names = _unique_source_names(label_sources)
    usecols = [name for name in [*feature_names, *label_names] if name in seen]
    dtype = {name: dtype[name] for name in usecols if name in dtype}
    defaults = {name: defaults[name] for name in usecols if name in defaults}

    # Read in larger chunks (e.g. 32k lines) to minimize Pandas overhead.
    read_chunk_size = (
        max(read_chunk_rows, batch_size)
        if read_chunk_rows is not None and read_chunk_rows > 0
        else max(batch_size * 256, 32768)
    )

    params: dict[str, Any] = {
        "sep": sep,
        "dtype": dtype,
        "usecols": usecols,
        "chunksize": read_chunk_size,
        "memory_map": memory_map,
    }
    if fast_no_na:
        params["na_filter"] = False
        params["keep_default_na"] = False
    else:
        params["na_values"] = na_vals
        params["keep_default_na"] = False
    if has_header:
        params["header"] = 0
    else:
        params["header"] = None
        params["names"] = names

    for chunk in pd.read_csv(path, **params):
        for col, default in defaults.items():
            if col in chunk.columns:
                chunk[col] = chunk[col].fillna(default)

        # Convert feature columns directly to dict of lists
        present_features = [col for col in feature_names if col in chunk.columns]
        features_dict = {col: chunk[col].tolist() for col in present_features}

        # Convert label columns to lists
        labels_dict = {
            ln: [None if pd.isna(v) else v for v in chunk[ln].tolist()]
            for ln in label_names
            if ln in chunk.columns
        }

        # Yield mini-batches from the chunk
        n_rows = len(chunk)
        for start in range(0, n_rows, batch_size):
            end = start + batch_size
            batch_features = {col: vals[start:end] for col, vals in features_dict.items()}
            batch_labels = {ln: vals[start:end] for ln, vals in labels_dict.items()}
            yield {"features": batch_features, "labels": batch_labels}


def _unique_source_names(sources: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for source in sources:
        name = source.name
        if name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def stream_files_batches(
    paths: list[str],
    flow_config: FlowConfig,
    batch_size: int,
    *,
    has_header: bool = True,
    sep: str = "\t",
    null_markers: Optional[set[str]] = None,
    read_chunk_rows: Optional[int] = None,
    fast_no_na: bool = False,
    memory_map: bool = False,
) -> Iterator[dict[str, Any]]:
    """Sequentially stream mini-batches from multiple files."""

    for path in paths:
        yield from stream_file_batches(
            path,
            flow_config,
            batch_size,
            has_header=has_header,
            sep=sep,
            null_markers=null_markers,
            read_chunk_rows=read_chunk_rows,
            fast_no_na=fast_no_na,
            memory_map=memory_map,
        )
