"""生成 discover-main-sort 完整特征配置文件。"""

from __future__ import annotations

import os

import yaml

# ── 受控词汇表 (数据生成器将仅使用这些值) ──

STOCK_CODES = [f"60{i:04d}" for i in range(50)]  # 600000-600049
ENTITY_CODES_A = [f"a_{i:02d}" for i in range(50)]
ENTITY_CODES_B = [f"b_{i:02d}" for i in range(50)]
ENTITY_CODES_C = [f"c_{i:02d}" for i in range(50)]
ENTITY_CODES_D = [f"d_{i:02d}" for i in range(50)]

INTEREST_KEYWORDS = [
    "人工智能",
    "新能源",
    "半导体",
    "医药",
    "消费",
    "金融",
    "房地产",
    "汽车",
    "互联网",
    "5G",
    "区块链",
    "云计算",
    "大数据",
    "物联网",
    "芯片",
    "光伏",
    "锂电池",
    "军工",
    "农业",
    "传媒",
]

FOLLOW_AUTHORS = [str(100000 + i) for i in range(20)]

INVEST_STYLES = [
    "主力追踪",
    "事件驱动",
    "成长股投资",
    "价值投资",
    "趋势交易",
    "量化对冲",
]

THEME_INTERESTS = [
    "同花顺新质50",
    "车联网(车路协同)",
    "低空经济",
    "人工智能",
    "新能源",
    "半导体",
    "机器人",
    "数字经济",
]

INDUSTRY_INTERESTS = [
    "系统软件(A股)",
    "煤炭开采加工",
    "银行",
    "医药生物",
    "食品饮料",
    "电力设备",
    "汽车制造",
    "房地产",
]

ROLE_NEEDS_FIRST = ["投顾", "量化", "个人投资者", "机构投资者"]
ROLE_NEEDS_SECOND = ["投资研究", "风险管理", "资产配置", "交易执行"]
INVEST_LABELS = ["大势研判", "行业研究", "公司研究", "技术分析"]
INVEST_LABELS_SECOND = ["大盘技术面", "行业基本面", "公司财务", "资金流向"]
INVEST_LABELS_THIRD = ["行业新闻", "公司公告", "政策解读", "市场数据"]
ENTITY_WORDS = [
    "反弹",
    "上涨",
    "下跌",
    "突破",
    "支撑",
    "压力",
    "成交量",
    "MACD",
    "KDJ",
    "RSI",
]

SOURCE_NAMES = ["社区", "同花顺", "东方财富", "雪球"]
REC_ALGOS = ["favSecuritiesV1", "stockFenshiKxian-hot", "favEntitiesV1"]
CITIES = ["上海市", "北京市", "深圳市", "杭州市", "广州市", "成都市"]
ASSET_LEVELS = ["1万以下", "1-10万", "10-30万", "30-100万", "100万以上"]
ITEM_TYPES = [
    "state",
    "ask_answer",
    "iwc_dialogue",
    "news",
    "report",
    "snslivepost",
    "snsview",
]


def _make_mapping(values: list[str], start_idx: int = 1) -> dict:
    return {v: start_idx + i for i, v in enumerate(values)}


def generate_config() -> dict:
    return {
        "version": "1.0.0",
        "sources": _build_sources(),
        "operators": _build_operators(),
    }


def _build_sources() -> list[dict]:
    return [
        # ═══════════ Item 特征 (18 字段) ═══════════
        {
            "name": "item_id",
            "source": "Item",
            "dtype": "int",
            "default_val": "0",
            "embed": {"vocab_size": 5000, "embed_dim": 16},
        },
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
        # ═══════════ User 特征 (非标签字段, 21 个) ═══════════
        # 标签字段 is_click/is_cvr/is_click_detail/is_click_stock 放在数据中, 不作为 source
        {
            "name": "user_id",
            "source": "User",
            "dtype": "int",
            "default_val": "0",
            "embed": {"vocab_size": 5000, "embed_dim": 16},
        },
        {
            "name": "rec_algo",
            "source": "Context",
            "dtype": "string",
            "default_val": "unknown",
        },
        {
            "name": "scene",
            "source": "Context",
            "dtype": "int",
            "default_val": "0",
            "embed": {"vocab_size": 2, "embed_dim": 4},
        },
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


def _build_operators() -> list[dict]:
    ops = []

    # ── 辅助函数 ──
    def dm(name, inp, out, mapping, default_idx=0, embed=None):
        op = {
            "name": name,
            "op_type": "DictMapper",
            "inputs": [inp],
            "outputs": [out],
            "params": {"mapping": mapping, "default_idx": default_idx},
        }
        if embed:
            op["embed"] = embed
        ops.append(op)

    def bk(name, inp, out, boundaries, embed=None):
        op = {
            "name": name,
            "op_type": "Bucketing",
            "inputs": [inp],
            "outputs": [out],
            "params": {"boundaries": boundaries},
        }
        if embed:
            op["embed"] = embed
        ops.append(op)

    def jl(name, inp, out, key, pad_len, pad_val=""):
        op = {
            "name": name,
            "op_type": "JsonExtractList",
            "inputs": [inp],
            "outputs": [out],
            "params": {"key": key, "pad_len": pad_len, "pad_val": pad_val},
        }
        ops.append(op)

    def lsp(name, inp, out, sep, key_index):
        op = {
            "name": name,
            "op_type": "ListStringParser",
            "inputs": [inp],
            "outputs": [out],
            "params": {"sep": sep, "key_index": key_index},
        }
        ops.append(op)

    def fls(name, inp, out, sep=",", max_len=0, pad_val=""):
        op = {
            "name": name,
            "op_type": "FlatSplit",
            "inputs": [inp],
            "outputs": [out],
            "params": {"sep": sep, "max_len": max_len, "pad_val": pad_val},
        }
        ops.append(op)

    def sp(name, inp, out, sep1, sep2, key_index, pad_len, pad_val=""):
        op = {
            "name": name,
            "op_type": "StringParser",
            "inputs": [inp],
            "outputs": [out],
            "params": {
                "sep1": sep1,
                "sep2": sep2,
                "key_index": key_index,
                "pad_len": pad_len,
                "pad_val": pad_val,
            },
        }
        ops.append(op)

    def sch(
        name,
        inp,
        out,
        vocab_size,
        num_hashes=1,
        embed=None,
    ):
        """Generate StringConcat + FeatureHash chain."""
        concat_out = f"{out}_str"
        ops.append({
            "name": f"{name}_concat",
            "op_type": "StringConcat",
            "inputs": [inp],
            "outputs": [concat_out],
            "params": {"separator": "_"},
        })
        op = {
            "name": name,
            "op_type": "FeatureHash",
            "inputs": [concat_out],
            "outputs": [out],
            "params": {
                "vocab_size": vocab_size,
                "num_hashes": num_hashes,
            },
        }
        if embed:
            op["embed"] = embed
        ops.append(op)

    def lo(name, inp1, inp2, out, embed=None):
        op = {
            "name": name,
            "op_type": "ListOverlap",
            "inputs": [inp1, inp2],
            "outputs": [out],
            "params": {},
        }
        if embed:
            op["embed"] = embed
        ops.append(op)

    def sq(name, inp, out, max_len, pad_val=0, embed=None):
        op = {
            "name": name,
            "op_type": "SequenceOp",
            "inputs": [inp],
            "outputs": [out],
            "params": {"max_len": max_len, "pad_val": pad_val},
        }
        if embed:
            op["embed"] = embed
        ops.append(op)

    def ex(name, inp, out, script):
        op = {
            "name": name,
            "op_type": "ExpressionOp",
            "inputs": [inp],
            "outputs": [out],
            "params": {"script": script},
        }
        ops.append(op)

    def fh(name, inps, out, vocab_size, num_hashes=1, separator="|", embed=None):
        op = {
            "name": name,
            "op_type": "FeatureHash",
            "inputs": inps,
            "outputs": [out],
            "params": {
                "vocab_size": vocab_size,
                "num_hashes": num_hashes,
                "separator": separator,
            },
        }
        if embed:
            op["embed"] = embed
        ops.append(op)

    # ═══════════════════════════════════════════
    # Section 1: Item — 枚举/分类型 → DictMapper
    # ═══════════════════════════════════════════
    dm(
        "item_type_map",
        "item_type",
        "item_type_idx",
        _make_mapping(ITEM_TYPES),
        embed={"vocab_size": 8, "embed_dim": 4},
    )
    dm(
        "source_name_map",
        "source_name",
        "source_name_idx",
        _make_mapping(SOURCE_NAMES),
        embed={"vocab_size": 5, "embed_dim": 4},
    )

    # ═══════════════════════════════════════════
    # Section 2: Item — 数值 → Bucketing / ExpressionOp
    # ═══════════════════════════════════════════
    bk(
        "quality_score_bucket",
        "quality_score_label",
        "quality_score_bucket",
        [0.2, 0.4, 0.6, 0.8],
        embed={"vocab_size": 5, "embed_dim": 4},
    )
    ex(
        "quality_score_log_op",
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

    # ═══════════════════════════════════════════
    # Section 3: Item — JSON 对象数组 [{score,tag}] → JsonExtractList(key="tag") → DictMapper
    # ═══════════════════════════════════════════
    # roleneeds_first_label
    jl(
        "roleneeds_first_parse",
        "roleneeds_first_label",
        "roleneeds_first_tags",
        "tag",
        3,
        "",
    )
    dm(
        "roleneeds_first_map",
        "roleneeds_first_tags",
        "roleneeds_first_ids",
        _make_mapping(ROLE_NEEDS_FIRST),
        embed={"vocab_size": 5, "embed_dim": 4},
    )

    # roleneeds_second_label
    jl(
        "roleneeds_second_parse",
        "roleneeds_second_label",
        "roleneeds_second_tags",
        "tag",
        3,
        "",
    )
    dm(
        "roleneeds_second_map",
        "roleneeds_second_tags",
        "roleneeds_second_ids",
        _make_mapping(ROLE_NEEDS_SECOND),
        embed={"vocab_size": 5, "embed_dim": 4},
    )

    # invest_label
    jl("invest_label_parse", "invest_label", "invest_label_tags", "tag", 3, "")
    dm(
        "invest_label_map",
        "invest_label_tags",
        "invest_label_ids",
        _make_mapping(INVEST_LABELS),
        embed={"vocab_size": 5, "embed_dim": 4},
    )

    # invest_label_second
    jl(
        "invest_label_second_parse",
        "invest_label_second",
        "invest_label_second_tags",
        "tag",
        3,
        "",
    )
    dm(
        "invest_label_second_map",
        "invest_label_second_tags",
        "invest_label_second_ids",
        _make_mapping(INVEST_LABELS_SECOND),
        embed={"vocab_size": 5, "embed_dim": 4},
    )

    # invest_label_third
    jl(
        "invest_label_third_parse",
        "invest_label_third",
        "invest_label_third_tags",
        "tag",
        3,
        "",
    )
    dm(
        "invest_label_third_map",
        "invest_label_third_tags",
        "invest_label_third_ids",
        _make_mapping(INVEST_LABELS_THIRD),
        embed={"vocab_size": 5, "embed_dim": 4},
    )

    # entity_words_label
    jl("entity_words_parse", "entity_words_label", "entity_words_tags", "tag", 5, "")
    dm(
        "entity_words_map",
        "entity_words_tags",
        "entity_words_ids",
        _make_mapping(ENTITY_WORDS),
        embed={"vocab_size": 11, "embed_dim": 4},
    )

    # ═══════════════════════════════════════════
    # Section 4: Item — JSON 字符串数组 → JsonExtractList → (ListStringParser) → DictMapper
    # ═══════════════════════════════════════════
    # stock_list: ["603538,17"] → split by "," 取代码
    jl("stock_list_parse", "stock_list", "stock_list_raw", None, 5, "")
    lsp("stock_code_extract", "stock_list_raw", "stock_codes", ",", 0)
    dm(
        "stock_code_map",
        "stock_codes",
        "stock_code_ids",
        _make_mapping(STOCK_CODES),
        embed={"vocab_size": 51, "embed_dim": 4},
    )
    sq(
        "stock_code_pad",
        "stock_code_ids",
        "stock_code_padded",
        5,
        0,
        embed={"vocab_size": 51, "embed_dim": 4},
    )

    # item_entities_v3: ["a_93","b_139","c_140","d_41"] → 直接映射
    jl("entities_v3_parse", "item_entities_v3", "entities_v3_list", None, 4, "")
    dm(
        "entities_v3_map",
        "entities_v3_list",
        "entities_v3_ids",
        _make_mapping(
            ENTITY_CODES_A + ENTITY_CODES_B + ENTITY_CODES_C + ENTITY_CODES_D,
            start_idx=1,
        ),
        embed={"vocab_size": 201, "embed_dim": 4},
    )

    # emb_id: 同 item_entities_v3 格式
    jl("emb_id_parse", "emb_id", "emb_id_list", None, 4, "")
    dm(
        "emb_id_map",
        "emb_id_list",
        "emb_id_ids",
        _make_mapping(
            ENTITY_CODES_A + ENTITY_CODES_B + ENTITY_CODES_C + ENTITY_CODES_D,
            start_idx=1,
        ),
        embed={"vocab_size": 201, "embed_dim": 4},
    )

    # ═══════════════════════════════════════════
    # Section 5: Item — 高基数文本 → StringConcat → FeatureHash
    # ═══════════════════════════════════════════
    sch(
        "author_id_hash",
        "author_id",
        "author_id_idx",
        500,
        embed={"vocab_size": 500, "embed_dim": 8},
    )
    sch(
        "title_hash",
        "title",
        "title_idx",
        500,
        embed={"vocab_size": 500, "embed_dim": 8},
    )
    sch(
        "author_hash",
        "author",
        "author_idx",
        200,
        embed={"vocab_size": 200, "embed_dim": 8},
    )

    # ═══════════════════════════════════════════
    # Section 6: User — 枚举/分类型 → DictMapper
    # ═══════════════════════════════════════════
    dm(
        "rec_algo_map",
        "rec_algo",
        "rec_algo_idx",
        _make_mapping(REC_ALGOS),
        embed={"vocab_size": 4, "embed_dim": 4},
    )
    dm(
        "is_new_user_map",
        "is_new_user",
        "is_new_user_idx",
        {"老用户": 1, "新用户": 2},
        embed={"vocab_size": 3, "embed_dim": 4},
    )
    dm(
        "asset_level_map",
        "asset_level",
        "asset_level_idx",
        _make_mapping(ASSET_LEVELS),
        embed={"vocab_size": 6, "embed_dim": 4},
    )
    dm(
        "city_map",
        "city",
        "city_idx",
        _make_mapping(CITIES),
        embed={"vocab_size": 7, "embed_dim": 4},
    )
    dm(
        "investment_horizon_map",
        "investment_horizon",
        "investment_horizon_idx",
        {"短期": 1, "中期": 2, "长期": 3},
        embed={"vocab_size": 4, "embed_dim": 4},
    )

    # ═══════════════════════════════════════════
    # Section 7: User — 数值 → Bucketing
    # ═══════════════════════════════════════════
    bk(
        "stay_time_bucket",
        "stay_time",
        "stay_time_bucket",
        [60, 300, 900, 3600],
        embed={"vocab_size": 5, "embed_dim": 4},
    )

    # ═══════════════════════════════════════════
    # Section 8: User — 结构化字符串 "a#b|c#d" → StringParser → (ListStringParser) → DictMapper
    # ═══════════════════════════════════════════
    # fav_securities: "code,market#weight|..."
    sp("fav_stock_parse", "fav_securities", "fav_stock_raw", "|", "#", 0, 5, "")
    lsp("fav_stock_code_extract", "fav_stock_raw", "fav_stock_codes", ",", 0)
    dm(
        "fav_stock_map",
        "fav_stock_codes",
        "fav_stock_ids",
        _make_mapping(STOCK_CODES),
        embed={"vocab_size": 51, "embed_dim": 4},
    )

    # recent_stocks: "code,market|code,market|..." (无 # 分隔)
    sp("recent_stocks_parse", "recent_stocks", "recent_stocks_raw", "|", ",", 0, 10, "")
    dm(
        "recent_stocks_map",
        "recent_stocks_raw",
        "recent_stocks_ids",
        _make_mapping(STOCK_CODES),
        embed={"vocab_size": 51, "embed_dim": 4},
    )

    # interest_keywords: "词#权重|..."
    sp(
        "interest_kw_parse",
        "interest_keywords",
        "interest_kw_list",
        "|",
        "#",
        0,
        10,
        "",
    )
    dm(
        "interest_kw_map",
        "interest_kw_list",
        "interest_kw_ids",
        _make_mapping(INTEREST_KEYWORDS),
        embed={"vocab_size": 21, "embed_dim": 4},
    )

    # follow_authors: "authorId#2.0|..."
    sp(
        "follow_authors_parse",
        "follow_authors",
        "follow_authors_list",
        "|",
        "#",
        0,
        10,
        "",
    )
    dm(
        "follow_authors_map",
        "follow_authors_list",
        "follow_authors_ids",
        _make_mapping(FOLLOW_AUTHORS),
        embed={"vocab_size": 21, "embed_dim": 4},
    )

    # hold_stocks: "code#market#weight#mean|..."
    sp("hold_stocks_parse", "hold_stocks", "hold_stocks_raw", "|", "#", 0, 5, "")
    dm(
        "hold_stocks_map",
        "hold_stocks_raw",
        "hold_stocks_ids",
        _make_mapping(STOCK_CODES),
        embed={"vocab_size": 51, "embed_dim": 4},
    )

    # hist_hold_stocks: "code#market#days#mean|..."
    sp(
        "hist_hold_stocks_parse",
        "hist_hold_stocks",
        "hist_hold_stocks_raw",
        "|",
        "#",
        0,
        5,
        "",
    )
    dm(
        "hist_hold_stocks_map",
        "hist_hold_stocks_raw",
        "hist_hold_stocks_ids",
        _make_mapping(STOCK_CODES),
        embed={"vocab_size": 51, "embed_dim": 4},
    )

    # historical_click_items: "a,b,c,d#timestamp|..."
    sp(
        "hist_items_parse",
        "historical_click_items",
        "hist_items_raw",
        "|",
        "#",
        0,
        10,
        "",
    )
    fls("hist_a_extract", "hist_items_raw", "hist_semantic_ids", ",", max_len=40)
    dm(
        "hist_a_map",
        "hist_semantic_ids",
        "hist_a_ids",
        _make_mapping(ENTITY_CODES_A),
        embed={"vocab_size": 51, "embed_dim": 4},
    )

    # invest_style: "风格名#权重#类型标志|..."
    sp("invest_style_parse", "invest_style", "invest_style_list", "|", "#", 0, 5, "")
    dm(
        "invest_style_map",
        "invest_style_list",
        "invest_style_ids",
        _make_mapping(INVEST_STYLES),
        embed={"vocab_size": 7, "embed_dim": 4},
    )

    # theme_interest: "题材名#关注程度#权重|..."
    sp(
        "theme_interest_parse",
        "theme_interest",
        "theme_interest_list",
        "|",
        "#",
        0,
        5,
        "",
    )
    dm(
        "theme_interest_map",
        "theme_interest_list",
        "theme_interest_ids",
        _make_mapping(THEME_INTERESTS),
        embed={"vocab_size": 9, "embed_dim": 4},
    )

    # industry_interest: "行业名(市场)#关注程度#权重|..."
    sp(
        "industry_interest_parse",
        "industry_interest",
        "industry_interest_list",
        "|",
        "#",
        0,
        5,
        "",
    )
    dm(
        "industry_interest_map",
        "industry_interest_list",
        "industry_interest_ids",
        _make_mapping(INDUSTRY_INTERESTS),
        embed={"vocab_size": 9, "embed_dim": 4},
    )

    # ═══════════════════════════════════════════
    # Section 9: 交互特征 → ListOverlap
    # ═══════════════════════════════════════════
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

    # ═══════════════════════════════════════════
    # Section 10: 特征哈希 — 无状态多种子 DJB2, Python/Rust 一致
    # ═══════════════════════════════════════════
    # 单哈希: user_id + author_id → 1 个索引
    fh(
        "user_author_hash",
        ["user_id", "author_id"],
        "user_author_hash_idx",
        vocab_size=5000,
        num_hashes=1,
        separator="|",
        embed={"vocab_size": 5000, "embed_dim": 8},
    )
    # 多哈希 (k=4 降低碰撞): item_type + source_name → 4 个索引
    fh(
        "item_type_source_hash",
        ["item_type", "source_name"],
        "item_type_source_hash_ids",
        vocab_size=1000,
        num_hashes=4,
        separator="|",
        embed={"vocab_size": 1000, "embed_dim": 4},
    )

    return ops


def main():
    config = generate_config()
    out_path = os.path.join(os.path.dirname(__file__), "feature_config_discover.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )
    print(
        f"[Config] {len(config['sources'])} sources, {len(config['operators'])} operators → {out_path}"
    )


if __name__ == "__main__":
    main()
