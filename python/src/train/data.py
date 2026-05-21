"""生产数据加载模块：多日物品特征索引 + 流式 Join + 批处理迭代。

数据分为两类文件：
  - 用户行为文件：日级，~50GB，Tab 分隔，含 user/item/context/标签
  - 物品特征文件：日级，~100MB，Tab 分隔，含 item 所有特征字段

物品有最长 7 天有效期，需按 item_id 跨文件合并。

用法：
  item_index = build_item_index(item_files, item_source_names, null_markers)
  for batch in stream_join(user_file, item_index, source_names, label_names, ...):
      tensors = dag.preprocess_batch(batch["features"])
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import pandas as pd

NULL_MARKERS = {"NULL", "\\N", "null", "None", ""}


def build_item_index(
    item_files: list[str],
    item_source_names: list[str],
    has_header: bool = True,
    separator: str = "\t",
    null_markers: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    """用 pandas 读取多日物品文件，按 item_id 去重后构建索引。

    后读文件覆盖先读文件中同 item_id 的记录（keep="last"）。
    仅提取 item_source_names 中声明的列。

    Args:
        item_files: 物品特征文件路径列表，从旧到新排列。
        item_source_names: FlowConfig 中 source="Item" 的特征名列表。
        has_header: 文件是否含 header 行。
        separator: 字段分隔符。
        null_markers: NULL 字符串集合。

    Returns:
        item_id → {feature_name: value} 映射表。
    """
    if null_markers is None:
        null_markers = NULL_MARKERS
    na_vals = list(null_markers)

    dfs = []
    for path in item_files:
        if not os.path.exists(path):
            print(f"[ItemIndex] skip missing: {path}")
            continue
        if has_header:
            df = pd.read_csv(path, sep=separator, na_values=na_vals,
                             dtype=str, keep_default_na=False)
        else:
            df = pd.read_csv(path, sep=separator, header=None, na_values=na_vals,
                             dtype=str, keep_default_na=False)
            df.columns = item_source_names[:len(df.columns)]
        # 仅保留 item_source_names 中存在的列
        keep = [c for c in item_source_names if c in df.columns]
        dfs.append(df[keep])

    if not dfs:
        return {}

    merged = pd.concat(dfs).drop_duplicates(subset="item_id", keep="last")
    print(f"[ItemIndex] {len(dfs)} files → {len(merged)} unique items")

    merged = merged.fillna("")
    index: dict[str, dict[str, str]] = {}
    for _, row in merged.iterrows():
        d = {col: str(row[col]) for col in merged.columns}
        item_id = d.pop("item_id")
        if item_id:
            index[item_id] = d
    return index


def _parse_val(raw: str, dtype_tag: str) -> Any:
    """将 TSV 原始字符串按 dtype 解析为 Python 原生类型。"""
    if dtype_tag == "int":
        try:
            return int(float(raw))
        except (ValueError, TypeError):
            return 0
    elif dtype_tag == "float":
        try:
            return float(raw)
        except (ValueError, TypeError):
            return 0.0
    return raw


def stream_join(
    user_file: str,
    item_index: dict[str, dict[str, str]],
    source_names: list[str],
    source_dtypes: dict[str, str],
    label_names: list[str],
    batch_size: int = 1024,
    separator: str = "\t",
    null_markers: set[str] | None = None,
    skip_missing_item: bool = False,
) -> Iterator[dict[str, Any]]:
    """pandas chunk read 流式读取用户行为文件，按 item_id 关联物品特征。

    内存中仅保留当前 chunk，50GB 文件可安全处理。

    Args:
        user_file: 用户行为文件路径。
        item_index: build_item_index 构建的物品特征索引。
        source_names: FlowConfig 中所有 source name 列表。
        source_dtypes: {source_name: dtype_tag}。
        label_names: 标签列名列表。
        batch_size: 批大小（chunk size）。
        separator: 字段分隔符。
        null_markers: NULL 字符串集合。
        skip_missing_item: True 时跳过 item_id 不在索引中的行。

    Yields:
        {"features": [dict, ...], "labels": {label_name: [value, ...]}}
    """
    if null_markers is None:
        null_markers = NULL_MARKERS
    na_vals = list(null_markers)

    # 构建 dtype 映射（所有列读为 str，后续按需解析）
    all_cols = source_names + [ln for ln in label_names if ln not in source_names]
    dtype_map = {c: "str" for c in all_cols}

    n_joined, n_missed, n_total = 0, 0, 0
    for chunk in pd.read_csv(
        user_file, sep=separator, dtype=dtype_map, na_values=na_vals,
        keep_default_na=False, chunksize=batch_size,
    ):
        n_total += len(chunk)
        feature_rows: list[dict[str, Any]] = []
        labels: dict[str, list[Any]] = {ln: [] for ln in label_names}

        for _, row in chunk.iterrows():
            item_id = str(row.get("item_id", "")) if not pd.isna(row.get("item_id")) else ""

            # 物品特征查找
            item_features = item_index.get(item_id)
            if item_features is None:
                if skip_missing_item:
                    n_missed += 1
                    continue
                item_features = {}
                n_missed += 1
            else:
                n_joined += 1

            # 合并行：物品特征优先，用户/上下文从行提取
            frow: dict[str, Any] = {}
            for name in source_names:
                dtype_tag = source_dtypes.get(name, "string")
                if name in item_features:
                    raw = item_features[name]
                elif name in chunk.columns:
                    raw = str(row[name]) if not pd.isna(row.get(name)) else ""
                else:
                    continue

                if raw in null_markers:
                    continue
                frow[name] = _parse_val(raw, dtype_tag)

            feature_rows.append(frow)

            # 标签列
            for ln in label_names:
                if ln in chunk.columns:
                    raw = row[ln]
                    if pd.isna(raw):
                        labels[ln].append(None)
                    else:
                        labels[ln].append(raw)
                else:
                    labels[ln].append(None)

        yield {"features": feature_rows, "labels": labels}

    print(
        f"[StreamJoin] done: total={n_total} joined={n_joined} "
        f"missed={n_missed} ({100 * n_missed / max(n_total, 1):.1f}%)"
    )
