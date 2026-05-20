"""生成 discover-main-sort 特征配置文件。

基于数据格式逐一分析，为每个字段设计正确的算子链。

字段分类：
  一、单值枚举/分类型 → DictMapper
  二、JSON 对象数组 [{"score","tag"}] → JsonExtractList(key="tag") → DictMapper
  三、JSON 纯值数组 ["a","b"] → JsonExtractList(key=null) → DictMapper
  四、JSON 数组含二次切分 → JsonExtractList → ListStringParser → DictMapper → SequenceOp
  五、两级结构化 "K1#V1|K2#V2|..." → StringParser → [ListStringParser] → DictMapper
  六、单级分隔符 "K1,K2|K3,K4|..." → StringParser(sep2=",") → DictMapper
  七、RQ-VAE 语义 ID 序列 → StringParser → FlatSplit → DictMapper
  八、高基数 ID/文本 → StringConcat → FeatureHash
  九、数值 → Bucketing / ExpressionOp → Bucketing
  十、交互特征 → ListOverlap / FeatureHash
"""

from __future__ import annotations

import os

import yaml

# ═══════════════════════════════════════════════════════
# 受控词汇表（与数据生成器对齐）
# ═══════════════════════════════════════════════════════

STOCK_CODES = [f"60{i:04d}" for i in range(50)]  # 600000-600049
ENTITY_CODES_A = [f"a_{i:02d}" for i in range(50)]
ENTITY_CODES_B = [f"b_{i:02d}" for i in range(50)]
ENTITY_CODES_C = [f"c_{i:02d}" for i in range(50)]
ENTITY_CODES_D = [f"d_{i:02d}" for i in range(50)]
ALL_ENTITY_CODES = ENTITY_CODES_A + ENTITY_CODES_B + ENTITY_CODES_C + ENTITY_CODES_D

INTEREST_KEYWORDS = [
    "人工智能", "新能源", "半导体", "医药", "消费", "金融", "房地产",
    "汽车", "互联网", "5G", "区块链", "云计算", "大数据", "物联网",
    "芯片", "光伏", "锂电池", "军工", "农业", "传媒",
]
FOLLOW_AUTHORS = [str(100000 + i) for i in range(20)]
INVEST_STYLES = ["主力追踪", "事件驱动", "成长股投资", "价值投资", "趋势交易", "量化对冲"]
THEME_INTERESTS = [
    "同花顺新质50", "车联网(车路协同)", "低空经济", "人工智能",
    "新能源", "半导体", "机器人", "数字经济",
]
INDUSTRY_INTERESTS = [
    "系统软件(A股)", "煤炭开采加工", "银行", "医药生物",
    "食品饮料", "电力设备", "汽车制造", "房地产",
]
ROLE_NEEDS_FIRST = ["投顾", "量化", "个人投资者", "机构投资者"]
ROLE_NEEDS_SECOND = ["投资研究", "风险管理", "资产配置", "交易执行"]
INVEST_LABELS = ["大势研判", "行业研究", "公司研究", "技术分析"]
INVEST_LABELS_SECOND = ["大盘技术面", "行业基本面", "公司财务", "资金流向"]
INVEST_LABELS_THIRD = ["行业新闻", "公司公告", "政策解读", "市场数据"]
ENTITY_WORDS = ["反弹", "上涨", "下跌", "突破", "支撑", "压力", "成交量", "MACD", "KDJ", "RSI"]
SOURCE_NAMES = ["社区", "同花顺", "东方财富", "雪球"]
REC_ALGOS = ["favSecuritiesV1", "stockFenshiKxian-hot", "favEntitiesV1"]
CITIES = ["上海市", "北京市", "深圳市", "杭州市", "广州市", "成都市"]
ASSET_LEVELS = ["1万以下", "1-10万", "10-30万", "30-100万", "100万以上"]
ITEM_TYPES = ["state", "ask_answer", "iwc_dialogue", "news", "report", "snslivepost", "snsview"]


def _m(vals: list[str], start: int = 1) -> dict:
    """生成 DictMapper 映射表，值从 start 起始（0 保留为 default_idx）。"""
    return {v: start + i for i, v in enumerate(vals)}


def generate_config() -> dict:
    return {
        "version": "1.0.0",
        "sources": _build_sources(),
        "operators": _build_operators(),
    }


# ═══════════════════════════════════════════════════════
# Sources
# ═══════════════════════════════════════════════════════

def _build_sources() -> list[dict]:
    return [
        # ── Item (18 字段) ──
        # 八：高基数 ID → embed
        {"name": "item_id", "source": "Item", "dtype": "int", "default_val": "0",
         "embed": {"vocab_size": 5000, "embed_dim": 16}},
        # 一：枚举 → DictMapper
        {"name": "item_type", "source": "Item", "dtype": "string", "default_val": "unknown"},
        # 八：文本 → StringConcat → FeatureHash
        {"name": "title", "source": "Item", "dtype": "string", "default_val": ""},
        {"name": "content", "source": "Item", "dtype": "string", "default_val": ""},
        {"name": "insight", "source": "Item", "dtype": "string", "default_val": ""},
        # 二：JSON 对象数组 → JsonExtractList → DictMapper
        {"name": "roleneeds_first_label", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "roleneeds_second_label", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "invest_label", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "invest_label_second", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "invest_label_third", "source": "Item", "dtype": "string", "default_val": "[]"},
        # 九：数值 → Bucketing
        {"name": "quality_score_label", "source": "Item", "dtype": "float", "default_val": "0.0"},
        # 四：JSON 数组二次切分 → JsonExtractList → ListStringParser → DictMapper
        {"name": "stock_list", "source": "Item", "dtype": "string", "default_val": "[]"},
        # 二：JSON 对象数组 → JsonExtractList(key="tag") → DictMapper
        {"name": "entity_words_label", "source": "Item", "dtype": "string", "default_val": "[]"},
        # 三：JSON 纯值数组 → JsonExtractList(key=null) → DictMapper
        {"name": "item_entities_v3", "source": "Item", "dtype": "string", "default_val": "[]"},
        # 八：高基数 ID → StringConcat → FeatureHash
        {"name": "author_id", "source": "Item", "dtype": "int", "default_val": "0"},
        {"name": "author", "source": "Item", "dtype": "string", "default_val": ""},
        # 一：枚举 → DictMapper
        {"name": "source_name", "source": "Item", "dtype": "string", "default_val": "unknown"},
        # 三：JSON 纯值数组 → JsonExtractList(key=null) → DictMapper
        {"name": "emb_id", "source": "Item", "dtype": "string", "default_val": "[]"},

        # ── User（非标签字段）──
        # 八：高基数 ID → embed
        {"name": "user_id", "source": "User", "dtype": "int", "default_val": "0",
         "embed": {"vocab_size": 5000, "embed_dim": 16}},
        # 一：枚举 → DictMapper
        {"name": "rec_algo", "source": "Context", "dtype": "string", "default_val": "unknown"},
        # 一：小基数 int → 直接 embed
        {"name": "scene", "source": "Context", "dtype": "int", "default_val": "0",
         "embed": {"vocab_size": 2, "embed_dim": 4}},
        # 九：数值 → Bucketing
        {"name": "stay_time", "source": "Context", "dtype": "int", "default_val": "0"},
        # 八：日期 → StringConcat → FeatureHash
        {"name": "p_date", "source": "Context", "dtype": "string", "default_val": "20260331"},
        # 五：两级结构化 "code,market#weight|..." → StringParser → ListStringParser → DictMapper
        {"name": "fav_securities", "source": "User", "dtype": "string", "default_val": ""},
        # 六：单级分隔 "code,market|..." → StringParser(sep2=",") → DictMapper
        {"name": "recent_stocks", "source": "User", "dtype": "string", "default_val": ""},
        # 五：两级结构化 "词#权重|..." → StringParser → DictMapper
        {"name": "interest_keywords", "source": "User", "dtype": "string", "default_val": ""},
        # 五：两级结构化 "作者ID#2.0|..." → StringParser → DictMapper
        {"name": "follow_authors", "source": "User", "dtype": "string", "default_val": ""},
        # 一：枚举 → DictMapper
        {"name": "is_new_user", "source": "User", "dtype": "string", "default_val": "老用户"},
        # 五：两级结构化 "code#market#weight#mean|..." → StringParser → DictMapper
        {"name": "hold_stocks", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "hist_hold_stocks", "source": "User", "dtype": "string", "default_val": ""},
        # 七：RQ-VAE 序列 → StringParser → FlatSplit → DictMapper
        {"name": "historical_click_items", "source": "User", "dtype": "string", "default_val": ""},
        # 一：枚举 → DictMapper
        {"name": "asset_level", "source": "User", "dtype": "string", "default_val": "未知"},
        {"name": "last_trade_date", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "city", "source": "User", "dtype": "string", "default_val": "未知"},
        {"name": "investment_horizon", "source": "User", "dtype": "string", "default_val": ""},
        # 五：两级结构化 "风格名#权重#标志|..." → StringParser → DictMapper
        {"name": "invest_style", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "theme_interest", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "industry_interest", "source": "User", "dtype": "string", "default_val": ""},
    ]


# ═══════════════════════════════════════════════════════
# Operators — 按格式分类组织
# ═══════════════════════════════════════════════════════

def _build_operators() -> list[dict]:
    ops = []

    # ── 辅助函数 ──

    def add(op: dict):
        ops.append(op)

    def dm(name, inp, out, mapping, default_idx=0, embed=None):
        """一、单值枚举/分类型 → DictMapper"""
        op = {"name": name, "op_type": "DictMapper", "inputs": [inp], "outputs": [out],
              "params": {"mapping": mapping, "default_idx": default_idx}}
        if embed:
            op["embed"] = embed
        add(op)

    def bk(name, inp, out, boundaries, embed=None):
        """九、数值 → Bucketing"""
        op = {"name": name, "op_type": "Bucketing", "inputs": [inp], "outputs": [out],
              "params": {"boundaries": boundaries}}
        if embed:
            op["embed"] = embed
        add(op)

    def ex(name, inp, out, script):
        """九、数值 → ExpressionOp"""
        add({"name": name, "op_type": "ExpressionOp", "inputs": [inp], "outputs": [out],
             "params": {"script": script}})

    # 解析 JSON 对象数组 [{"score","tag"}] → 提取 tag 字段
    def json_tags(name, inp, out, pad_len=3, pad_val=""):
        """二、JSON 对象数组 → JsonExtractList(key="tag")"""
        add({"name": name, "op_type": "JsonExtractList", "inputs": [inp], "outputs": [out],
             "params": {"key": "tag", "pad_len": pad_len, "pad_val": pad_val}})

    # 解析 JSON 纯值数组 ["a","b"] → 直接提取
    def json_list(name, inp, out, pad_len=5, pad_val=""):
        """三、JSON 纯值数组 → JsonExtractList(key=null)"""
        add({"name": name, "op_type": "JsonExtractList", "inputs": [inp], "outputs": [out],
             "params": {"key": None, "pad_len": pad_len, "pad_val": pad_val}})

    def lsp(name, inp, out, sep, key_index):
        """四、列表元素二次切分 → ListStringParser"""
        add({"name": name, "op_type": "ListStringParser", "inputs": [inp], "outputs": [out],
             "params": {"sep": sep, "key_index": key_index}})

    def sp(name, inp, out, sep1, sep2, key_index, pad_len, pad_val=""):
        """五/六、两级/单级分隔符字符串 → StringParser"""
        add({"name": name, "op_type": "StringParser", "inputs": [inp], "outputs": [out],
             "params": {"sep1": sep1, "sep2": sep2, "key_index": key_index,
                        "pad_len": pad_len, "pad_val": pad_val}})

    def fls(name, inp, out, sep=",", max_len=0, pad_val=""):
        """七、列表打平分割 → FlatSplit"""
        add({"name": name, "op_type": "FlatSplit", "inputs": [inp], "outputs": [out],
             "params": {"sep": sep, "max_len": max_len, "pad_val": pad_val}})

    def sq(name, inp, out, max_len, pad_val=0, embed=None):
        """列表定长 → SequenceOp"""
        op = {"name": name, "op_type": "SequenceOp", "inputs": [inp], "outputs": [out],
              "params": {"max_len": max_len, "pad_val": pad_val}}
        if embed:
            op["embed"] = embed
        add(op)

    def concat(name, inps, out, separator=""):
        """八、字符串拼接 → StringConcat"""
        add({"name": name, "op_type": "StringConcat", "inputs": inps, "outputs": [out],
             "params": {"separator": separator}})

    def fh(name, inps, out, vocab_size, num_hashes=1, embed=None):
        """八、特征哈希 → FeatureHash"""
        op = {"name": name, "op_type": "FeatureHash", "inputs": inps, "outputs": [out],
              "params": {"vocab_size": vocab_size, "num_hashes": num_hashes}}
        if embed:
            op["embed"] = embed
        add(op)

    def lo(name, inp1, inp2, out, embed=None):
        """十、列表交集检测 → ListOverlap"""
        op = {"name": name, "op_type": "ListOverlap", "inputs": [inp1, inp2], "outputs": [out],
              "params": {}}
        if embed:
            op["embed"] = embed
        add(op)

    # ═══════════════════════════════════════════════════
    # Section 1: Item — 枚举/分类型 → DictMapper (类别一)
    # ═══════════════════════════════════════════════════
    dm("item_type_map", "item_type", "item_type_idx", _m(ITEM_TYPES),
       embed={"vocab_size": 8, "embed_dim": 4})
    dm("source_name_map", "source_name", "source_name_idx", _m(SOURCE_NAMES),
       embed={"vocab_size": 5, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 2: Item — 数值 → Bucketing + ExpressionOp → Bucketing (类别九)
    # ═══════════════════════════════════════════════════
    # quality_score_label ∈ [0,1]: 直接分桶
    bk("quality_score_bucket", "quality_score_label", "quality_score_bucket",
       [0.2, 0.4, 0.6, 0.8],
       embed={"vocab_size": 5, "embed_dim": 4})
    # log 变换后分桶，捕捉低分段的细粒度差异
    ex("quality_score_log", "quality_score_label", "quality_score_log",
       "log(v0 + 0.01)")
    bk("quality_score_log_bucket", "quality_score_log", "quality_score_log_bucket",
       [-4.0, -2.0, -1.0, 0.0],
       embed={"vocab_size": 5, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 3: Item — JSON 对象数组 [{"score","tag"}] (类别二)
    # JsonExtractList(key="tag") → DictMapper
    # ═══════════════════════════════════════════════════
    json_tags("roleneeds_first_parse", "roleneeds_first_label", "roleneeds_first_tags", 3)
    dm("roleneeds_first_map", "roleneeds_first_tags", "roleneeds_first_ids",
       _m(ROLE_NEEDS_FIRST), embed={"vocab_size": 5, "embed_dim": 4})

    json_tags("roleneeds_second_parse", "roleneeds_second_label", "roleneeds_second_tags", 3)
    dm("roleneeds_second_map", "roleneeds_second_tags", "roleneeds_second_ids",
       _m(ROLE_NEEDS_SECOND), embed={"vocab_size": 5, "embed_dim": 4})

    json_tags("invest_label_parse", "invest_label", "invest_label_tags", 3)
    dm("invest_label_map", "invest_label_tags", "invest_label_ids",
       _m(INVEST_LABELS), embed={"vocab_size": 5, "embed_dim": 4})

    json_tags("invest_label_second_parse", "invest_label_second", "invest_label_second_tags", 3)
    dm("invest_label_second_map", "invest_label_second_tags", "invest_label_second_ids",
       _m(INVEST_LABELS_SECOND), embed={"vocab_size": 5, "embed_dim": 4})

    json_tags("invest_label_third_parse", "invest_label_third", "invest_label_third_tags", 3)
    dm("invest_label_third_map", "invest_label_third_tags", "invest_label_third_ids",
       _m(INVEST_LABELS_THIRD), embed={"vocab_size": 5, "embed_dim": 4})

    json_tags("entity_words_parse", "entity_words_label", "entity_words_tags", 5)
    dm("entity_words_map", "entity_words_tags", "entity_words_ids",
       _m(ENTITY_WORDS), embed={"vocab_size": 11, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 4: Item — JSON 纯值数组 ["a","b"] (类别三)
    # item_entities_v3: RQ-VAE 实体编码
    # emb_id: 向量召回 ID
    # JsonExtractList(key=null) → DictMapper
    # ═══════════════════════════════════════════════════
    json_list("entities_v3_parse", "item_entities_v3", "entities_v3_list", 4)
    dm("entities_v3_map", "entities_v3_list", "entities_v3_ids",
       _m(ALL_ENTITY_CODES, start=1), embed={"vocab_size": 201, "embed_dim": 4})

    json_list("emb_id_parse", "emb_id", "emb_id_list", 4)
    dm("emb_id_map", "emb_id_list", "emb_id_ids",
       _m(ALL_ENTITY_CODES, start=1), embed={"vocab_size": 201, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 5: Item — JSON 数组含二次切分 (类别四)
    # stock_list: ["603538,17","002130,33"] → 提取纯股票代码
    # JsonExtractList(key=null, pad_len=5) → ListStringParser(sep=",") → DictMapper → SequenceOp
    # ═══════════════════════════════════════════════════
    json_list("stock_list_parse", "stock_list", "stock_list_raw", 5)
    lsp("stock_code_extract", "stock_list_raw", "stock_codes", ",", 0)
    dm("stock_code_map", "stock_codes", "stock_code_ids",
       _m(STOCK_CODES), embed={"vocab_size": 51, "embed_dim": 4})
    sq("stock_code_pad", "stock_code_ids", "stock_code_padded", 5,
       embed={"vocab_size": 51, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 6: Item — 高基数 ID/文本 → StringConcat → FeatureHash (类别八)
    # title → hash
    # content → hash
    # insight → hash (小词表，常为空)
    # author_id → hash
    # author → hash
    # ═══════════════════════════════════════════════════
    concat("title_concat", ["title"], "title_str")
    fh("title_hash", ["title_str"], "title_idx", 2000,
       embed={"vocab_size": 2000, "embed_dim": 8})

    concat("content_concat", ["content"], "content_str")
    fh("content_hash", ["content_str"], "content_idx", 8000,
       embed={"vocab_size": 8000, "embed_dim": 8})

    concat("insight_concat", ["insight"], "insight_str")
    fh("insight_hash", ["insight_str"], "insight_idx", 500,
       embed={"vocab_size": 500, "embed_dim": 4})

    concat("author_id_concat", ["author_id"], "author_id_str")
    fh("author_id_hash", ["author_id_str"], "author_id_idx", 1000,
       embed={"vocab_size": 1000, "embed_dim": 8})

    concat("author_concat", ["author"], "author_str")
    fh("author_hash", ["author_str"], "author_idx", 1000,
       embed={"vocab_size": 1000, "embed_dim": 8})

    # ═══════════════════════════════════════════════════
    # Section 7: User — 枚举/分类型 → DictMapper (类别一)
    # ═══════════════════════════════════════════════════
    dm("rec_algo_map", "rec_algo", "rec_algo_idx", _m(REC_ALGOS),
       embed={"vocab_size": 4, "embed_dim": 4})
    dm("is_new_user_map", "is_new_user", "is_new_user_idx",
       {"老用户": 1, "新用户": 2}, embed={"vocab_size": 3, "embed_dim": 4})
    dm("asset_level_map", "asset_level", "asset_level_idx", _m(ASSET_LEVELS),
       embed={"vocab_size": 6, "embed_dim": 4})
    dm("city_map", "city", "city_idx", _m(CITIES),
       embed={"vocab_size": 7, "embed_dim": 4})
    dm("investment_horizon_map", "investment_horizon", "investment_horizon_idx",
       {"短期": 1, "中期": 2, "长期": 3}, embed={"vocab_size": 4, "embed_dim": 4})

    # last_trade_date: 高基数日期 → FeatureHash
    concat("last_trade_date_concat", ["last_trade_date"], "last_trade_date_str")
    fh("last_trade_date_hash", ["last_trade_date_str"], "last_trade_date_idx", 500,
       embed={"vocab_size": 500, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 8: User — 数值 → Bucketing (类别九)
    # ═══════════════════════════════════════════════════
    bk("stay_time_bucket", "stay_time", "stay_time_bucket",
       [60, 300, 900, 3600],
       embed={"vocab_size": 5, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 9: User — 两级结构化字符串 (类别五)
    # "K1#V1|K2#V2|..." → StringParser(sep1="|", sep2="#") → DictMapper
    #
    # fav_securities: "代码,市场#权重|..." → 二次切分 ListStringParser(sep=",")
    # interest_keywords: "词#权重|..." → 直接 DictMapper
    # follow_authors: "作者ID#2.0|..." → 直接 DictMapper
    # hold_stocks: "代码#市场#权重#均值|..." → DictMapper(代码)
    # hist_hold_stocks: "代码#市场#天数#均值|..." → DictMapper(代码)
    # invest_style: "风格名#权重#标志|..." → DictMapper(风格名)
    # theme_interest: "题材名#关注程度#权重|..." → DictMapper(题材名)
    # industry_interest: "行业名(市场)#关注程度#权重|..." → DictMapper(行业名)
    # ═══════════════════════════════════════════════════

    # fav_securities: 提取代码部分 "代码,市场" → 二次切分取纯代码
    sp("fav_stock_parse", "fav_securities", "fav_stock_raw", "|", "#", 0, 5)
    lsp("fav_stock_code_extract", "fav_stock_raw", "fav_stock_codes", ",", 0)
    dm("fav_stock_map", "fav_stock_codes", "fav_stock_ids",
       _m(STOCK_CODES), embed={"vocab_size": 51, "embed_dim": 4})

    # interest_keywords: 提取关键词部分
    sp("interest_kw_parse", "interest_keywords", "interest_kw_list", "|", "#", 0, 10)
    dm("interest_kw_map", "interest_kw_list", "interest_kw_ids",
       _m(INTEREST_KEYWORDS), embed={"vocab_size": 21, "embed_dim": 4})

    # follow_authors: 提取作者 ID
    sp("follow_authors_parse", "follow_authors", "follow_authors_list", "|", "#", 0, 10)
    dm("follow_authors_map", "follow_authors_list", "follow_authors_ids",
       _m(FOLLOW_AUTHORS), embed={"vocab_size": 21, "embed_dim": 4})

    # hold_stocks: 提取股票代码
    sp("hold_stocks_parse", "hold_stocks", "hold_stocks_raw", "|", "#", 0, 5)
    dm("hold_stocks_map", "hold_stocks_raw", "hold_stocks_ids",
       _m(STOCK_CODES), embed={"vocab_size": 51, "embed_dim": 4})

    # hist_hold_stocks: 提取股票代码
    sp("hist_hold_stocks_parse", "hist_hold_stocks", "hist_hold_stocks_raw", "|", "#", 0, 5)
    dm("hist_hold_stocks_map", "hist_hold_stocks_raw", "hist_hold_stocks_ids",
       _m(STOCK_CODES), embed={"vocab_size": 51, "embed_dim": 4})

    # invest_style: 提取风格名
    sp("invest_style_parse", "invest_style", "invest_style_list", "|", "#", 0, 5)
    dm("invest_style_map", "invest_style_list", "invest_style_ids",
       _m(INVEST_STYLES), embed={"vocab_size": 7, "embed_dim": 4})

    # theme_interest: 提取题材名
    sp("theme_interest_parse", "theme_interest", "theme_interest_list", "|", "#", 0, 5)
    dm("theme_interest_map", "theme_interest_list", "theme_interest_ids",
       _m(THEME_INTERESTS), embed={"vocab_size": 9, "embed_dim": 4})

    # industry_interest: 提取行业名
    sp("industry_interest_parse", "industry_interest", "industry_interest_list", "|", "#", 0, 5)
    dm("industry_interest_map", "industry_interest_list", "industry_interest_ids",
       _m(INDUSTRY_INTERESTS), embed={"vocab_size": 9, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 10: User — 单级分隔符字符串 (类别六)
    # recent_stocks: "代码,市场|..." → StringParser(sep2=",") → DictMapper
    # ═══════════════════════════════════════════════════
    sp("recent_stocks_parse", "recent_stocks", "recent_stocks_raw", "|", ",", 0, 10)
    dm("recent_stocks_map", "recent_stocks_raw", "recent_stocks_ids",
       _m(STOCK_CODES), embed={"vocab_size": 51, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 11: User — RQ-VAE 语义 ID 序列 (类别七)
    # historical_click_items: "a_xx,b_xx,c_xx,d_xx#ts|..."
    # StringParser → FlatSplit → DictMapper
    # ═══════════════════════════════════════════════════
    sp("hist_items_parse", "historical_click_items", "hist_vectors", "|", "#", 0, 10)
    fls("hist_semantic_flat", "hist_vectors", "hist_semantic_ids", ",", max_len=40)
    dm("hist_semantic_map", "hist_semantic_ids", "hist_semantic_mapped",
       _m(ALL_ENTITY_CODES, start=1), embed={"vocab_size": 201, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 12: 高基数 ID — StringConcat → FeatureHash (类别八)
    # p_date → hash
    # ═══════════════════════════════════════════════════
    concat("p_date_concat", ["p_date"], "p_date_str")
    fh("p_date_hash", ["p_date_str"], "p_date_idx", 100,
       embed={"vocab_size": 100, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 13: 交互特征 (类别十)
    # ═══════════════════════════════════════════════════

    # 实体重叠：物品实体 vs 用户历史点击实体
    lo("entity_overlap", "entities_v3_list", "hist_semantic_ids", "entity_overlap_flag",
       embed={"vocab_size": 2, "embed_dim": 4})

    # 股票重叠：物品关联股票 vs 用户自选股
    lo("stock_overlap", "stock_codes", "fav_stock_codes", "stock_overlap_flag",
       embed={"vocab_size": 2, "embed_dim": 4})

    # 股票重叠：物品关联股票 vs 用户近期关注股票
    lo("stock_overlap_recent", "stock_codes", "recent_stocks_raw", "stock_overlap_recent_flag",
       embed={"vocab_size": 2, "embed_dim": 4})

    # 用户-作者交叉哈希
    concat("user_author_concat", ["user_id", "author_id"], "user_author_str")
    fh("user_author_hash", ["user_author_str"], "user_author_hash_idx", 5000,
       embed={"vocab_size": 5000, "embed_dim": 8})

    # 内容类型-来源交叉哈希
    concat("type_source_concat", ["item_type", "source_name"], "type_source_str")
    fh("type_source_hash", ["type_source_str"], "type_source_hash_idx", 100,
       embed={"vocab_size": 100, "embed_dim": 4})

    return ops


def main():
    config = generate_config()
    out_path = os.path.join(os.path.dirname(__file__), "feature_config_discover.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False,
                  sort_keys=False, width=120)
    print(f"[Config] {len(config['sources'])} sources, {len(config['operators'])} operators → {out_path}")


if __name__ == "__main__":
    main()
