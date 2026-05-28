"""生成 discover-main-sort 特征配置文件。

默认策略：生产测试样本统一使用 FeatureHash（无状态哈希），避免维护离线词表状态。

字段分类：
  一、低基数枚举/ID → FeatureHash
  二、JSON 对象数组 [{"score","tag"}] → JsonExtractList → FeatureHash
  三、JSON 纯值数组 ["a","b"] → JsonExtractList → FeatureHash
  四、JSON 数组含二次切分 → JsonExtractList → ListStringParser → FeatureHash → SequenceOp
  五、两级结构化 "K1#V1|K2#V2|..." → StringParser → FeatureHash
  六、单级分隔符 "K1,K2|K3,K4|..." → StringParser → FeatureHash
  七、RQ-VAE 语义 ID 序列 → StringParser → FlatSplit → FeatureHash
  八、高基数 ID/文本 → StringConcat → FeatureHash
  九、数值 → Bucketing / ExpressionOp → Bucketing
  十、交互特征 → ListOverlap / FeatureHash
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

VERSION = "1.0.0"
FULL_CONFIG_FILE = "feature_config_discover.yaml"

ConfigDict = dict[str, Any]
SourceDef = ConfigDict
OperatorDef = ConfigDict
EmbedDef = ConfigDict

LABEL_SOURCES: list[SourceDef] = [
    {"name": "is_click", "dtype": "int", "default_val": "0"},
    {"name": "is_cvr", "dtype": "int", "default_val": "0"},
    {"name": "is_click_detail", "dtype": "int", "default_val": "0"},
    {"name": "is_click_stock", "dtype": "int", "default_val": "0"},
    {"name": "stay_time", "dtype": "int", "default_val": "-1"},
    {"name": "ctr", "dtype": "int", "default_val": "0"},
    {"name": "cvr", "dtype": "int", "default_val": "0"},
]


def _source(name: str, source: str, dtype: str, default_val: str) -> SourceDef:
    return {"name": name, "source": source, "dtype": dtype, "default_val": default_val}


def _embed(vocab_size: int, embed_dim: int, **extra: Any) -> EmbedDef:
    embed: EmbedDef = {"vocab_size": vocab_size, "embed_dim": embed_dim}
    embed.update(extra)
    return embed


def _op(
    name: str,
    op_type: str,
    inputs: list[str],
    outputs: list[str],
    params: ConfigDict | None = None,
    embed: EmbedDef | None = None,
) -> OperatorDef:
    op: OperatorDef = {
        "name": name,
        "op_type": op_type,
        "inputs": inputs,
        "outputs": outputs,
        "params": params or {},
    }
    if embed is not None:
        op["embed"] = embed
    return op


def feature_hash(
    name: str,
    inputs: list[str],
    output: str,
    vocab_size: int,
    *,
    num_hashes: int = 1,
    embed: EmbedDef | None = None,
) -> OperatorDef:
    return _op(
        name,
        "FeatureHash",
        inputs,
        [output],
        {"vocab_size": vocab_size, "num_hashes": num_hashes},
        embed,
    )


def single_feature_hash(
    name: str,
    input_name: str,
    output: str,
    vocab_size: int,
    *,
    embed: EmbedDef | None = None,
) -> OperatorDef:
    return feature_hash(name, [input_name], output, vocab_size, embed=embed)


def bucket(
    name: str,
    input_name: str,
    output: str,
    boundaries: list[float],
    *,
    embed: EmbedDef | None = None,
) -> OperatorDef:
    return _op(
        name, "Bucketing", [input_name], [output], {"boundaries": boundaries}, embed
    )


def expression(name: str, input_name: str, output: str, script: str) -> OperatorDef:
    return _op(name, "ExpressionOp", [input_name], [output], {"script": script})


def json_extract_list(
    name: str,
    input_name: str,
    output: str,
    *,
    key: str | None,
    pad_len: int,
    pad_val: str = "",
) -> OperatorDef:
    return _op(
        name,
        "JsonExtractList",
        [input_name],
        [output],
        {"key": key, "pad_len": pad_len, "pad_val": pad_val},
    )


def json_tags(name: str, input_name: str, output: str, pad_len: int = 3) -> OperatorDef:
    return json_extract_list(name, input_name, output, key="tag", pad_len=pad_len)


def json_values(
    name: str, input_name: str, output: str, pad_len: int = 5
) -> OperatorDef:
    return json_extract_list(name, input_name, output, key=None, pad_len=pad_len)


def string_parser(
    name: str,
    input_name: str,
    output: str,
    *,
    sep1: str,
    sep2: str,
    key_index: int,
    pad_len: int,
    pad_val: str = "",
) -> OperatorDef:
    return _op(
        name,
        "StringParser",
        [input_name],
        [output],
        {
            "sep1": sep1,
            "sep2": sep2,
            "key_index": key_index,
            "pad_len": pad_len,
            "pad_val": pad_val,
        },
    )


def list_string_parser(
    name: str, input_name: str, output: str, *, sep: str, key_index: int
) -> OperatorDef:
    return _op(
        name,
        "ListStringParser",
        [input_name],
        [output],
        {"sep": sep, "key_index": key_index},
    )


def flat_split(
    name: str,
    input_name: str,
    output: str,
    *,
    sep: str = ",",
    max_len: int = 0,
    pad_val: str = "",
) -> OperatorDef:
    return _op(
        name,
        "FlatSplit",
        [input_name],
        [output],
        {"sep": sep, "max_len": max_len, "pad_val": pad_val},
    )


def list_overlap(
    name: str,
    left: str,
    right: str,
    output: str,
    *,
    embed: EmbedDef | None = None,
) -> OperatorDef:
    return _op(name, "ListOverlap", [left, right], [output], {}, embed)


def string_concat(
    name: str, inputs: list[str], output: str, *, separator: str = "_"
) -> OperatorDef:
    return _op(name, "StringConcat", inputs, [output], {"separator": separator})


def generate_config() -> ConfigDict:
    return {
        "version": VERSION,
        "sources": _build_sources(),
        "operators": _build_operators(),
    }


def generate_item_config() -> ConfigDict:
    """生成仅含 Item 侧 source 定义的配置（用于 Polars 读取物品文件）。"""
    item_sources = [s for s in _build_sources() if s["source"] == "Item"]
    return {"version": VERSION, "sources": item_sources}


def generate_user_config() -> ConfigDict:
    """生成仅含 User/Context 侧 source + label 定义的配置（用于 Polars 读取用户文件）。"""
    user_sources = [s for s in _build_sources() if s["source"] in ("User", "Context")]
    # item_id 作为 Join 键也需出现在用户侧配置
    item_id_src = _source("item_id", "Item", "int", "0")
    return {"version": VERSION, "sources": [item_id_src] + user_sources + LABEL_SOURCES}


# ═══════════════════════════════════════════════════════
# Sources
# ═══════════════════════════════════════════════════════


def _build_sources() -> list[SourceDef]:
    return [
        # ── Item (18 字段) ──
        {"name": "item_id", "source": "Item", "dtype": "int", "default_val": "0"},
        {
            "name": "item_type",
            "source": "Item",
            "dtype": "string",
            "default_val": "unknown",
        },
        {"name": "title", "source": "Item", "dtype": "string", "default_val": ""},
        {"name": "content", "source": "Item", "dtype": "string", "default_val": ""},
        {"name": "insight", "source": "Item", "dtype": "string", "default_val": ""},
        {
            "name": "roleneeds_first_label",
            "source": "Item",
            "dtype": "string",
            "default_val": "[]",
        },
        {
            "name": "roleneeds_second_label",
            "source": "Item",
            "dtype": "string",
            "default_val": "[]",
        },
        {
            "name": "invest_label",
            "source": "Item",
            "dtype": "string",
            "default_val": "[]",
        },
        {
            "name": "invest_label_second",
            "source": "Item",
            "dtype": "string",
            "default_val": "[]",
        },
        {
            "name": "invest_label_third",
            "source": "Item",
            "dtype": "string",
            "default_val": "[]",
        },
        {
            "name": "quality_score_label",
            "source": "Item",
            "dtype": "float",
            "default_val": "0.0",
        },
        {
            "name": "stock_list",
            "source": "Item",
            "dtype": "string",
            "default_val": "[]",
        },
        {
            "name": "entity_words_label",
            "source": "Item",
            "dtype": "string",
            "default_val": "[]",
        },
        {
            "name": "item_entities_v3",
            "source": "Item",
            "dtype": "string",
            "default_val": "[]",
        },
        {"name": "author_id", "source": "Item", "dtype": "int", "default_val": "0"},
        {"name": "author", "source": "Item", "dtype": "string", "default_val": ""},
        {
            "name": "source_name",
            "source": "Item",
            "dtype": "string",
            "default_val": "unknown",
        },
        {"name": "emb_id", "source": "Item", "dtype": "string", "default_val": "[]"},
        # ── User/Context（非标签字段）──
        {"name": "user_id", "source": "User", "dtype": "int", "default_val": "0"},
        {
            "name": "rec_algo",
            "source": "Context",
            "dtype": "string",
            "default_val": "unknown",
        },
        {"name": "scene", "source": "Context", "dtype": "int", "default_val": "0"},
        {"name": "stay_time", "source": "Context", "dtype": "int", "default_val": "0"},
        {
            "name": "p_date",
            "source": "Context",
            "dtype": "string",
            "default_val": "20260331",
        },
        {
            "name": "fav_securities",
            "source": "User",
            "dtype": "string",
            "default_val": "",
        },
        {
            "name": "recent_stocks",
            "source": "User",
            "dtype": "string",
            "default_val": "",
        },
        {
            "name": "interest_keywords",
            "source": "User",
            "dtype": "string",
            "default_val": "",
        },
        {
            "name": "follow_authors",
            "source": "User",
            "dtype": "string",
            "default_val": "",
        },
        {
            "name": "is_new_user",
            "source": "User",
            "dtype": "string",
            "default_val": "老用户",
        },
        {"name": "hold_stocks", "source": "User", "dtype": "string", "default_val": ""},
        {
            "name": "hist_hold_stocks",
            "source": "User",
            "dtype": "string",
            "default_val": "",
        },
        {
            "name": "historical_click_items",
            "source": "User",
            "dtype": "string",
            "default_val": "",
        },
        {
            "name": "asset_level",
            "source": "User",
            "dtype": "string",
            "default_val": "未知",
        },
        {
            "name": "last_trade_date",
            "source": "User",
            "dtype": "string",
            "default_val": "",
        },
        {"name": "city", "source": "User", "dtype": "string", "default_val": "未知"},
        {
            "name": "investment_horizon",
            "source": "User",
            "dtype": "string",
            "default_val": "",
        },
        {
            "name": "invest_style",
            "source": "User",
            "dtype": "string",
            "default_val": "",
        },
        {
            "name": "theme_interest",
            "source": "User",
            "dtype": "string",
            "default_val": "",
        },
        {
            "name": "industry_interest",
            "source": "User",
            "dtype": "string",
            "default_val": "",
        },
    ]


# ═══════════════════════════════════════════════════════
# Operators
# ═══════════════════════════════════════════════════════


def _build_operators() -> list[OperatorDef]:
    ops: list[OperatorDef] = []

    def add(op: OperatorDef) -> None:
        ops.append(op)

    def fh(
        name: str,
        inps: list[str],
        out: str,
        vocab_size: int,
        num_hashes: int = 1,
        embed: EmbedDef | None = None,
    ) -> None:
        add(
            feature_hash(
                name, inps, out, vocab_size, num_hashes=num_hashes, embed=embed
            )
        )

    def single_fh(
        name: str, inp: str, out: str, vocab_size: int, embed: EmbedDef | None = None
    ) -> None:
        add(single_feature_hash(name, inp, out, vocab_size, embed=embed))

    def bk(
        name: str,
        inp: str,
        out: str,
        boundaries: list[float],
        embed: EmbedDef | None = None,
    ) -> None:
        add(bucket(name, inp, out, boundaries, embed=embed))

    def ex(name: str, inp: str, out: str, script: str) -> None:
        add(expression(name, inp, out, script))

    def jt(name: str, inp: str, out: str, pad_len: int = 3) -> None:
        add(json_tags(name, inp, out, pad_len))

    def jl(name: str, inp: str, out: str, pad_len: int = 5) -> None:
        add(json_values(name, inp, out, pad_len))

    def sp(
        name: str,
        inp: str,
        out: str,
        sep1: str,
        sep2: str,
        key_index: int,
        pad_len: int,
        pad_val: str = "",
    ) -> None:
        add(
            string_parser(
                name,
                inp,
                out,
                sep1=sep1,
                sep2=sep2,
                key_index=key_index,
                pad_len=pad_len,
                pad_val=pad_val,
            )
        )

    def lsp(name: str, inp: str, out: str, sep: str, key_index: int) -> None:
        add(list_string_parser(name, inp, out, sep=sep, key_index=key_index))

    def fls(
        name: str,
        inp: str,
        out: str,
        sep: str = ",",
        max_len: int = 0,
        pad_val: str = "",
    ) -> None:
        add(flat_split(name, inp, out, sep=sep, max_len=max_len, pad_val=pad_val))

    def lo(
        name: str, inp1: str, inp2: str, out: str, embed: EmbedDef | None = None
    ) -> None:
        add(list_overlap(name, inp1, inp2, out, embed=embed))

    # ═══════════════════════════════════════════════════
    # Section 1: 单值特征 → FeatureHash（全量 FeatureHash，sources 无 embed）
    # ═══════════════════════════════════════════════════
    single_hash_specs = [
        ("item_id_hash", "item_id", "item_id_idx", 5000, 16),
        ("user_id_hash", "user_id", "user_id_idx", 5000, 16),
        ("scene_hash", "scene", "scene_idx", 10, 4),
        ("item_type_hash", "item_type", "item_type_idx", 20, 4),
        ("source_name_hash", "source_name", "source_name_idx", 20, 4),
        ("rec_algo_hash", "rec_algo", "rec_algo_idx", 20, 4),
        ("is_new_user_hash", "is_new_user", "is_new_user_idx", 10, 4),
        ("asset_level_hash", "asset_level", "asset_level_idx", 20, 4),
        ("city_hash", "city", "city_idx", 50, 4),
        (
            "investment_horizon_hash",
            "investment_horizon",
            "investment_horizon_idx",
            20,
            4,
        ),
    ]
    for name, input_name, output, vocab_size, embed_dim in single_hash_specs:
        single_fh(
            name, input_name, output, vocab_size, embed=_embed(vocab_size, embed_dim)
        )

    # ═══════════════════════════════════════════════════
    # Section 2: 数值 → Bucketing / ExpressionOp
    # ═══════════════════════════════════════════════════
    bk(
        "quality_score_bucket",
        "quality_score_label",
        "quality_score_bucket",
        [0.2, 0.4, 0.6, 0.8],
        embed={"vocab_size": 5, "embed_dim": 4},
    )
    ex(
        "quality_score_log",
        "quality_score_label",
        "quality_score_log",
        "log(v0 + 0.01)",
    )
    bk(
        "quality_score_log_bucket",
        "quality_score_log",
        "quality_score_log_bucket",
        [-4.0, -2.0, -1.0, 0.0],
        embed={"vocab_size": 5, "embed_dim": 4},
    )
    bk(
        "stay_time_bucket",
        "stay_time",
        "stay_time_bucket",
        [60, 300, 900, 3600],
        embed={"vocab_size": 5, "embed_dim": 4},
    )

    # ═══════════════════════════════════════════════════
    # Section 3: JSON 对象数组 → JsonExtractList → FeatureHash
    # ═══════════════════════════════════════════════════
    json_tag_specs = [
        ("roleneeds_first", "roleneeds_first_label", 3, 100),
        ("roleneeds_second", "roleneeds_second_label", 3, 100),
        ("invest_label", "invest_label", 3, 100),
        ("invest_label_second", "invest_label_second", 3, 100),
        ("invest_label_third", "invest_label_third", 3, 100),
        ("entity_words", "entity_words_label", 5, 200),
    ]
    for prefix, input_name, pad_len, vocab_size in json_tag_specs:
        tags = f"{prefix}_tags"
        jt(f"{prefix}_parse", input_name, tags, pad_len)
        fh(
            f"{prefix}_hash",
            [tags],
            f"{prefix}_ids",
            vocab_size,
            embed=_embed(vocab_size, 4),
        )

    # ═══════════════════════════════════════════════════
    # Section 4: JSON 纯值数组 → JsonExtractList → FeatureHash
    # ═══════════════════════════════════════════════════
    json_value_specs = [
        ("entities_v3", "item_entities_v3", 4, 1000),
        ("emb_id", "emb_id", 4, 1000),
    ]
    for prefix, input_name, pad_len, vocab_size in json_value_specs:
        values = f"{prefix}_list"
        jl(f"{prefix}_parse", input_name, values, pad_len)
        fh(
            f"{prefix}_hash",
            [values],
            f"{prefix}_ids",
            vocab_size,
            embed=_embed(vocab_size, 4),
        )

    # ═══════════════════════════════════════════════════
    # Section 5: JSON 数组含二次切分 → JsonExtractList → ListStringParser → FeatureHash → SequenceOp
    # ═══════════════════════════════════════════════════
    jl("stock_list_parse", "stock_list", "stock_list_raw", 5)
    lsp("stock_code_extract", "stock_list_raw", "stock_codes", ",", 0)
    fh(
        "stock_code_hash",
        ["stock_codes"],
        "stock_code_ids",
        500,
        embed={"vocab_size": 500, "embed_dim": 4},
    )

    # ═══════════════════════════════════════════════════
    # Section 6: 高基数 ID/文本 → StringConcat → FeatureHash
    # ═══════════════════════════════════════════════════
    high_cardinality_specs = [
        ("title_hash", "title", "title_idx", 2000, 8),
        ("content_hash", "content", "content_idx", 8000, 8),
        ("insight_hash", "insight", "insight_idx", 500, 4),
        ("author_id_hash", "author_id", "author_id_idx", 1000, 8),
        ("author_hash", "author", "author_idx", 1000, 8),
        ("last_trade_date_hash", "last_trade_date", "last_trade_date_idx", 500, 4),
        ("p_date_hash", "p_date", "p_date_idx", 100, 4),
    ]
    for name, input_name, output, vocab_size, embed_dim in high_cardinality_specs:
        single_fh(
            name, input_name, output, vocab_size, embed=_embed(vocab_size, embed_dim)
        )

    # ═══════════════════════════════════════════════════
    # Section 7: User — 结构化字符串 → StringParser → FeatureHash
    # ═══════════════════════════════════════════════════

    # fav_securities: "代码,市场#权重|..." → 提取代码+市场 → 二次切分取代码 → FeatureHash
    sp("fav_stock_parse", "fav_securities", "fav_stock_raw", "|", "#", 0, 5)
    lsp("fav_stock_code_extract", "fav_stock_raw", "fav_stock_codes", ",", 0)
    fh(
        "fav_stock_hash",
        ["fav_stock_codes"],
        "fav_stock_ids",
        500,
        embed={"vocab_size": 500, "embed_dim": 4},
    )

    # interest_keywords: "词#权重|..." → 提取词 → FeatureHash
    sp("interest_kw_parse", "interest_keywords", "interest_kw_list", "|", "#", 0, 10)
    fh(
        "interest_kw_hash",
        ["interest_kw_list"],
        "interest_kw_ids",
        500,
        embed={"vocab_size": 500, "embed_dim": 4},
    )

    # follow_authors: "作者ID#2.0|..." → FeatureHash
    sp("follow_authors_parse", "follow_authors", "follow_authors_list", "|", "#", 0, 10)
    fh(
        "follow_authors_hash",
        ["follow_authors_list"],
        "follow_authors_ids",
        500,
        embed={"vocab_size": 500, "embed_dim": 4},
    )

    # hold_stocks: "代码#市场#权重#均值|..." → FeatureHash
    sp("hold_stocks_parse", "hold_stocks", "hold_stocks_raw", "|", "#", 0, 5)
    fh(
        "hold_stocks_hash",
        ["hold_stocks_raw"],
        "hold_stocks_ids",
        500,
        embed={"vocab_size": 500, "embed_dim": 4},
    )

    # hist_hold_stocks
    sp(
        "hist_hold_stocks_parse",
        "hist_hold_stocks",
        "hist_hold_stocks_raw",
        "|",
        "#",
        0,
        5,
    )
    fh(
        "hist_hold_stocks_hash",
        ["hist_hold_stocks_raw"],
        "hist_hold_stocks_ids",
        500,
        embed={"vocab_size": 500, "embed_dim": 4},
    )

    # invest_style: "风格名#权重#标志|..." → FeatureHash
    sp("invest_style_parse", "invest_style", "invest_style_list", "|", "#", 0, 5)
    fh(
        "invest_style_hash",
        ["invest_style_list"],
        "invest_style_ids",
        100,
        embed={"vocab_size": 100, "embed_dim": 4},
    )

    # theme_interest: "题材名#关注程度#权重|..." → FeatureHash
    sp("theme_interest_parse", "theme_interest", "theme_interest_list", "|", "#", 0, 5)
    fh(
        "theme_interest_hash",
        ["theme_interest_list"],
        "theme_interest_ids",
        100,
        embed={"vocab_size": 100, "embed_dim": 4},
    )

    # industry_interest: "行业名(市场)#关注程度#权重|..." → FeatureHash
    sp(
        "industry_interest_parse",
        "industry_interest",
        "industry_interest_list",
        "|",
        "#",
        0,
        5,
    )
    fh(
        "industry_interest_hash",
        ["industry_interest_list"],
        "industry_interest_ids",
        100,
        embed={"vocab_size": 100, "embed_dim": 4},
    )

    # recent_stocks: "代码,市场|..." → FeatureHash
    sp("recent_stocks_parse", "recent_stocks", "recent_stocks_raw", "|", ",", 0, 10)
    fh(
        "recent_stocks_hash",
        ["recent_stocks_raw"],
        "recent_stocks_ids",
        500,
        embed={"vocab_size": 500, "embed_dim": 4},
    )

    # ═══════════════════════════════════════════════════
    # Section 8: RQ-VAE 语义 ID 序列 → StringParser → FlatSplit → FeatureHash
    # ═══════════════════════════════════════════════════
    sp("hist_items_parse", "historical_click_items", "hist_vectors", "|", "#", 0, 10)
    fls("hist_semantic_flat", "hist_vectors", "hist_semantic_ids", ",", max_len=40)
    fh(
        "hist_semantic_hash",
        ["hist_semantic_ids"],
        "hist_semantic_mapped",
        2000,
        embed={"vocab_size": 2000, "embed_dim": 4},
    )

    # ═══════════════════════════════════════════════════
    # Section 9: 交互特征
    # ═══════════════════════════════════════════════════
    lo(
        "entity_overlap",
        "entities_v3_list",
        "hist_semantic_ids",
        "entity_overlap_flag",
        embed={"vocab_size": 2, "embed_dim": 4},
    )
    lo(
        "stock_overlap",
        "stock_codes",
        "fav_stock_codes",
        "stock_overlap_flag",
        embed={"vocab_size": 2, "embed_dim": 4},
    )
    lo(
        "stock_overlap_recent",
        "stock_codes",
        "recent_stocks_raw",
        "stock_overlap_recent_flag",
        embed={"vocab_size": 2, "embed_dim": 4},
    )

    # 用户-作者交叉哈希
    add(
        string_concat("user_author_concat", ["user_id", "author_id"], "user_author_str")
    )
    fh(
        "user_author_hash",
        ["user_author_str"],
        "user_author_hash_idx",
        5000,
        embed={"vocab_size": 5000, "embed_dim": 8},
    )

    # 内容类型-来源交叉哈希
    add(
        string_concat(
            "type_source_concat", ["item_type", "source_name"], "type_source_str"
        )
    )
    fh(
        "type_source_hash",
        ["type_source_str"],
        "type_source_hash_idx",
        100,
        embed={"vocab_size": 100, "embed_dim": 4},
    )

    return ops


def _write_yaml(data: ConfigDict, name: str) -> str:
    path = Path(__file__).resolve().parent / name
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )
    return str(path)


def main() -> None:
    # 统一配置（DAG + 模型）
    full = generate_config()
    path = _write_yaml(full, FULL_CONFIG_FILE)
    n_feat = sum(
        len(op.get("outputs", [])) for op in full["operators"] if "embed" in op
    )
    print(
        f"[Config] {len(full['sources'])} sources, {len(full['operators'])} ops"
        f" → {n_feat} features → {path}"
    )


if __name__ == "__main__":
    main()
