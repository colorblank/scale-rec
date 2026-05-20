"""生成 discover-main-sort 特征配置文件。

默认策略：高基数特征使用 FeatureHash（无状态哈希），低基数枚举保留 DictMapper。

字段分类：
  一、低基数枚举 → DictMapper（仅 item_type, source_name, rec_algo, is_new_user, asset_level, city, investment_horizon, scene）
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

import os

import yaml

# ═══════════════════════════════════════════════════════
# 受控词汇表（仅用于低基数枚举类 DictMapper）
# ═══════════════════════════════════════════════════════

ITEM_TYPES = ["state", "ask_answer", "iwc_dialogue", "news", "report", "snslivepost", "snsview"]
SOURCE_NAMES = ["社区", "同花顺", "东方财富", "雪球"]
REC_ALGOS = ["favSecuritiesV1", "stockFenshiKxian-hot", "favEntitiesV1"]
CITIES = ["上海市", "北京市", "深圳市", "杭州市", "广州市", "成都市"]
ASSET_LEVELS = ["1万以下", "1-10万", "10-30万", "30-100万", "100万以上"]


def _m(vals: list[str], start: int = 1) -> dict:
    return {v: start + i for i, v in enumerate(vals)}


def generate_config() -> dict:
    return {"version": "1.0.0", "sources": _build_sources(), "operators": _build_operators()}


# ═══════════════════════════════════════════════════════
# Sources
# ═══════════════════════════════════════════════════════

def _build_sources() -> list[dict]:
    return [
        # ── Item (18 字段) ──
        {"name": "item_id", "source": "Item", "dtype": "int", "default_val": "0"},
        {"name": "item_type", "source": "Item", "dtype": "string", "default_val": "unknown"},
        {"name": "title", "source": "Item", "dtype": "string", "default_val": ""},
        {"name": "content", "source": "Item", "dtype": "string", "default_val": ""},
        {"name": "insight", "source": "Item", "dtype": "string", "default_val": ""},
        {"name": "roleneeds_first_label", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "roleneeds_second_label", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "invest_label", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "invest_label_second", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "invest_label_third", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "quality_score_label", "source": "Item", "dtype": "float", "default_val": "0.0"},
        {"name": "stock_list", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "entity_words_label", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "item_entities_v3", "source": "Item", "dtype": "string", "default_val": "[]"},
        {"name": "author_id", "source": "Item", "dtype": "int", "default_val": "0"},
        {"name": "author", "source": "Item", "dtype": "string", "default_val": ""},
        {"name": "source_name", "source": "Item", "dtype": "string", "default_val": "unknown"},
        {"name": "emb_id", "source": "Item", "dtype": "string", "default_val": "[]"},

        # ── User/Context（非标签字段）──
        {"name": "user_id", "source": "User", "dtype": "int", "default_val": "0"},
        {"name": "rec_algo", "source": "Context", "dtype": "string", "default_val": "unknown"},
        {"name": "scene", "source": "Context", "dtype": "int", "default_val": "0"},
        {"name": "stay_time", "source": "Context", "dtype": "int", "default_val": "0"},
        {"name": "p_date", "source": "Context", "dtype": "string", "default_val": "20260331"},
        {"name": "fav_securities", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "recent_stocks", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "interest_keywords", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "follow_authors", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "is_new_user", "source": "User", "dtype": "string", "default_val": "老用户"},
        {"name": "hold_stocks", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "hist_hold_stocks", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "historical_click_items", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "asset_level", "source": "User", "dtype": "string", "default_val": "未知"},
        {"name": "last_trade_date", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "city", "source": "User", "dtype": "string", "default_val": "未知"},
        {"name": "investment_horizon", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "invest_style", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "theme_interest", "source": "User", "dtype": "string", "default_val": ""},
        {"name": "industry_interest", "source": "User", "dtype": "string", "default_val": ""},
    ]


# ═══════════════════════════════════════════════════════
# Operators
# ═══════════════════════════════════════════════════════

def _build_operators() -> list[dict]:
    ops = []

    def add(op: dict):
        ops.append(op)

    # ── FeatureHash 辅助：列表特征直接哈希（FeatureHash 内部拼接所有输入）──
    def fh(name, inps, out, vocab_size, num_hashes=1, embed=None):
        op = {"name": name, "op_type": "FeatureHash", "inputs": inps, "outputs": [out],
              "params": {"vocab_size": vocab_size, "num_hashes": num_hashes}}
        if embed:
            op["embed"] = embed
        add(op)

    # ── StringConcat → FeatureHash（单值文本先拼接再哈希）──
    def concat_hash(name, inp, out, vocab_size, embed=None):
        concat_out = f"{out}_str"
        add({"name": f"{name}_concat", "op_type": "StringConcat",
             "inputs": [inp], "outputs": [concat_out], "params": {"separator": ""}})
        fh(name, [concat_out], out, vocab_size, embed=embed)

    # ── 数值 ──
    def bk(name, inp, out, boundaries, embed=None):
        add({"name": name, "op_type": "Bucketing", "inputs": [inp], "outputs": [out],
             "params": {"boundaries": boundaries},
             **(embed or {})})
        if embed:
            ops[-1]["embed"] = embed

    def ex(name, inp, out, script):
        add({"name": name, "op_type": "ExpressionOp", "inputs": [inp], "outputs": [out],
             "params": {"script": script}})

    # ── JSON 解析 ──
    def json_tags(name, inp, out, pad_len=3, pad_val=""):
        add({"name": name, "op_type": "JsonExtractList", "inputs": [inp], "outputs": [out],
             "params": {"key": "tag", "pad_len": pad_len, "pad_val": pad_val}})

    def json_list(name, inp, out, pad_len=5, pad_val=""):
        add({"name": name, "op_type": "JsonExtractList", "inputs": [inp], "outputs": [out],
             "params": {"key": None, "pad_len": pad_len, "pad_val": pad_val}})

    # ── 字符串解析 ──
    def sp(name, inp, out, sep1, sep2, key_index, pad_len, pad_val=""):
        add({"name": name, "op_type": "StringParser", "inputs": [inp], "outputs": [out],
             "params": {"sep1": sep1, "sep2": sep2, "key_index": key_index,
                        "pad_len": pad_len, "pad_val": pad_val}})

    def lsp(name, inp, out, sep, key_index):
        add({"name": name, "op_type": "ListStringParser", "inputs": [inp], "outputs": [out],
             "params": {"sep": sep, "key_index": key_index}})

    def fls(name, inp, out, sep=",", max_len=0, pad_val=""):
        add({"name": name, "op_type": "FlatSplit", "inputs": [inp], "outputs": [out],
             "params": {"sep": sep, "max_len": max_len, "pad_val": pad_val}})

    def sq(name, inp, out, max_len, pad_val=0):
        add({"name": name, "op_type": "SequenceOp", "inputs": [inp], "outputs": [out],
             "params": {"max_len": max_len, "pad_val": pad_val}})

    def lo(name, inp1, inp2, out, embed=None):
        op = {"name": name, "op_type": "ListOverlap", "inputs": [inp1, inp2], "outputs": [out],
              "params": {}}
        if embed:
            op["embed"] = embed
        add(op)

    # ═══════════════════════════════════════════════════
    # Section 1: 单值特征 → FeatureHash（全量 FeatureHash，sources 无 embed）
    # ═══════════════════════════════════════════════════
    concat_hash("item_id_hash", "item_id", "item_id_idx", 5000,
                embed={"vocab_size": 5000, "embed_dim": 16})
    concat_hash("user_id_hash", "user_id", "user_id_idx", 5000,
                embed={"vocab_size": 5000, "embed_dim": 16})
    concat_hash("scene_hash", "scene", "scene_idx", 10,
                embed={"vocab_size": 10, "embed_dim": 4})
    concat_hash("item_type_hash", "item_type", "item_type_idx", 20,
                embed={"vocab_size": 20, "embed_dim": 4})
    concat_hash("source_name_hash", "source_name", "source_name_idx", 20,
                embed={"vocab_size": 20, "embed_dim": 4})
    concat_hash("rec_algo_hash", "rec_algo", "rec_algo_idx", 20,
                embed={"vocab_size": 20, "embed_dim": 4})
    concat_hash("is_new_user_hash", "is_new_user", "is_new_user_idx", 10,
                embed={"vocab_size": 10, "embed_dim": 4})
    concat_hash("asset_level_hash", "asset_level", "asset_level_idx", 20,
                embed={"vocab_size": 20, "embed_dim": 4})
    concat_hash("city_hash", "city", "city_idx", 50,
                embed={"vocab_size": 50, "embed_dim": 4})
    concat_hash("investment_horizon_hash", "investment_horizon", "investment_horizon_idx", 20,
                embed={"vocab_size": 20, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 2: 数值 → Bucketing / ExpressionOp
    # ═══════════════════════════════════════════════════
    bk("quality_score_bucket", "quality_score_label", "quality_score_bucket",
       [0.2, 0.4, 0.6, 0.8], embed={"vocab_size": 5, "embed_dim": 4})
    ex("quality_score_log", "quality_score_label", "quality_score_log", "log(v0 + 0.01)")
    bk("quality_score_log_bucket", "quality_score_log", "quality_score_log_bucket",
       [-4.0, -2.0, -1.0, 0.0], embed={"vocab_size": 5, "embed_dim": 4})
    bk("stay_time_bucket", "stay_time", "stay_time_bucket",
       [60, 300, 900, 3600], embed={"vocab_size": 5, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 3: JSON 对象数组 → JsonExtractList → FeatureHash
    # ═══════════════════════════════════════════════════
    json_tags("roleneeds_first_parse", "roleneeds_first_label", "roleneeds_first_tags", 3)
    fh("roleneeds_first_hash", ["roleneeds_first_tags"], "roleneeds_first_ids", 100,
       embed={"vocab_size": 100, "embed_dim": 4})

    json_tags("roleneeds_second_parse", "roleneeds_second_label", "roleneeds_second_tags", 3)
    fh("roleneeds_second_hash", ["roleneeds_second_tags"], "roleneeds_second_ids", 100,
       embed={"vocab_size": 100, "embed_dim": 4})

    json_tags("invest_label_parse", "invest_label", "invest_label_tags", 3)
    fh("invest_label_hash", ["invest_label_tags"], "invest_label_ids", 100,
       embed={"vocab_size": 100, "embed_dim": 4})

    json_tags("invest_label_second_parse", "invest_label_second", "invest_label_second_tags", 3)
    fh("invest_label_second_hash", ["invest_label_second_tags"], "invest_label_second_ids", 100,
       embed={"vocab_size": 100, "embed_dim": 4})

    json_tags("invest_label_third_parse", "invest_label_third", "invest_label_third_tags", 3)
    fh("invest_label_third_hash", ["invest_label_third_tags"], "invest_label_third_ids", 100,
       embed={"vocab_size": 100, "embed_dim": 4})

    json_tags("entity_words_parse", "entity_words_label", "entity_words_tags", 5)
    fh("entity_words_hash", ["entity_words_tags"], "entity_words_ids", 200,
       embed={"vocab_size": 200, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 4: JSON 纯值数组 → JsonExtractList → FeatureHash
    # ═══════════════════════════════════════════════════
    json_list("entities_v3_parse", "item_entities_v3", "entities_v3_list", 4)
    fh("entities_v3_hash", ["entities_v3_list"], "entities_v3_ids", 1000,
       embed={"vocab_size": 1000, "embed_dim": 4})

    json_list("emb_id_parse", "emb_id", "emb_id_list", 4)
    fh("emb_id_hash", ["emb_id_list"], "emb_id_ids", 1000,
       embed={"vocab_size": 1000, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 5: JSON 数组含二次切分 → JsonExtractList → ListStringParser → FeatureHash → SequenceOp
    # ═══════════════════════════════════════════════════
    json_list("stock_list_parse", "stock_list", "stock_list_raw", 5)
    lsp("stock_code_extract", "stock_list_raw", "stock_codes", ",", 0)
    fh("stock_code_hash", ["stock_codes"], "stock_code_ids", 500,
       embed={"vocab_size": 500, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 6: 高基数 ID/文本 → StringConcat → FeatureHash
    # ═══════════════════════════════════════════════════
    concat_hash("title_hash", "title", "title_idx", 2000, embed={"vocab_size": 2000, "embed_dim": 8})
    concat_hash("content_hash", "content", "content_idx", 8000, embed={"vocab_size": 8000, "embed_dim": 8})
    concat_hash("insight_hash", "insight", "insight_idx", 500, embed={"vocab_size": 500, "embed_dim": 4})
    concat_hash("author_id_hash", "author_id", "author_id_idx", 1000, embed={"vocab_size": 1000, "embed_dim": 8})
    concat_hash("author_hash", "author", "author_idx", 1000, embed={"vocab_size": 1000, "embed_dim": 8})
    concat_hash("last_trade_date_hash", "last_trade_date", "last_trade_date_idx", 500,
                embed={"vocab_size": 500, "embed_dim": 4})
    concat_hash("p_date_hash", "p_date", "p_date_idx", 100, embed={"vocab_size": 100, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 7: User — 结构化字符串 → StringParser → FeatureHash
    # ═══════════════════════════════════════════════════

    # fav_securities: "代码,市场#权重|..." → 提取代码+市场 → 二次切分取代码 → FeatureHash
    sp("fav_stock_parse", "fav_securities", "fav_stock_raw", "|", "#", 0, 5)
    lsp("fav_stock_code_extract", "fav_stock_raw", "fav_stock_codes", ",", 0)
    fh("fav_stock_hash", ["fav_stock_codes"], "fav_stock_ids", 500,
       embed={"vocab_size": 500, "embed_dim": 4})

    # interest_keywords: "词#权重|..." → 提取词 → FeatureHash
    sp("interest_kw_parse", "interest_keywords", "interest_kw_list", "|", "#", 0, 10)
    fh("interest_kw_hash", ["interest_kw_list"], "interest_kw_ids", 500,
       embed={"vocab_size": 500, "embed_dim": 4})

    # follow_authors: "作者ID#2.0|..." → FeatureHash
    sp("follow_authors_parse", "follow_authors", "follow_authors_list", "|", "#", 0, 10)
    fh("follow_authors_hash", ["follow_authors_list"], "follow_authors_ids", 500,
       embed={"vocab_size": 500, "embed_dim": 4})

    # hold_stocks: "代码#市场#权重#均值|..." → FeatureHash
    sp("hold_stocks_parse", "hold_stocks", "hold_stocks_raw", "|", "#", 0, 5)
    fh("hold_stocks_hash", ["hold_stocks_raw"], "hold_stocks_ids", 500,
       embed={"vocab_size": 500, "embed_dim": 4})

    # hist_hold_stocks
    sp("hist_hold_stocks_parse", "hist_hold_stocks", "hist_hold_stocks_raw", "|", "#", 0, 5)
    fh("hist_hold_stocks_hash", ["hist_hold_stocks_raw"], "hist_hold_stocks_ids", 500,
       embed={"vocab_size": 500, "embed_dim": 4})

    # invest_style: "风格名#权重#标志|..." → FeatureHash
    sp("invest_style_parse", "invest_style", "invest_style_list", "|", "#", 0, 5)
    fh("invest_style_hash", ["invest_style_list"], "invest_style_ids", 100,
       embed={"vocab_size": 100, "embed_dim": 4})

    # theme_interest: "题材名#关注程度#权重|..." → FeatureHash
    sp("theme_interest_parse", "theme_interest", "theme_interest_list", "|", "#", 0, 5)
    fh("theme_interest_hash", ["theme_interest_list"], "theme_interest_ids", 100,
       embed={"vocab_size": 100, "embed_dim": 4})

    # industry_interest: "行业名(市场)#关注程度#权重|..." → FeatureHash
    sp("industry_interest_parse", "industry_interest", "industry_interest_list", "|", "#", 0, 5)
    fh("industry_interest_hash", ["industry_interest_list"], "industry_interest_ids", 100,
       embed={"vocab_size": 100, "embed_dim": 4})

    # recent_stocks: "代码,市场|..." → FeatureHash
    sp("recent_stocks_parse", "recent_stocks", "recent_stocks_raw", "|", ",", 0, 10)
    fh("recent_stocks_hash", ["recent_stocks_raw"], "recent_stocks_ids", 500,
       embed={"vocab_size": 500, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 8: RQ-VAE 语义 ID 序列 → StringParser → FlatSplit → FeatureHash
    # ═══════════════════════════════════════════════════
    sp("hist_items_parse", "historical_click_items", "hist_vectors", "|", "#", 0, 10)
    fls("hist_semantic_flat", "hist_vectors", "hist_semantic_ids", ",", max_len=40)
    fh("hist_semantic_hash", ["hist_semantic_ids"], "hist_semantic_mapped", 2000,
       embed={"vocab_size": 2000, "embed_dim": 4})

    # ═══════════════════════════════════════════════════
    # Section 9: 交互特征
    # ═══════════════════════════════════════════════════
    lo("entity_overlap", "entities_v3_list", "hist_semantic_ids", "entity_overlap_flag",
       embed={"vocab_size": 2, "embed_dim": 4})
    lo("stock_overlap", "stock_codes", "fav_stock_codes", "stock_overlap_flag",
       embed={"vocab_size": 2, "embed_dim": 4})
    lo("stock_overlap_recent", "stock_codes", "recent_stocks_raw", "stock_overlap_recent_flag",
       embed={"vocab_size": 2, "embed_dim": 4})

    # 用户-作者交叉哈希
    add({"name": "user_author_concat", "op_type": "StringConcat",
         "inputs": ["user_id", "author_id"], "outputs": ["user_author_str"],
         "params": {"separator": "_"}})
    fh("user_author_hash", ["user_author_str"], "user_author_hash_idx", 5000,
       embed={"vocab_size": 5000, "embed_dim": 8})

    # 内容类型-来源交叉哈希
    add({"name": "type_source_concat", "op_type": "StringConcat",
         "inputs": ["item_type", "source_name"], "outputs": ["type_source_str"],
         "params": {"separator": "_"}})
    fh("type_source_hash", ["type_source_str"], "type_source_hash_idx", 100,
       embed={"vocab_size": 100, "embed_dim": 4})

    return ops


def main():
    config = generate_config()
    out_path = os.path.join(os.path.dirname(__file__), "feature_config_discover.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False,
                  sort_keys=False, width=120)
    ops_n = len(config["operators"])
    feats = len([s for s in config["sources"] if "embed" in s])
    for op in config["operators"]:
        if "embed" in op:
            feats += len(op.get("outputs", []))
    print(
        f"[Config] {len(config['sources'])} sources, {ops_n} operators"
        f" → {feats} embeddable features → {out_path}"
    )


if __name__ == "__main__":
    main()
