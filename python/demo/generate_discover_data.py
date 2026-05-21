from __future__ import annotations

"""discover-main-sort v2 合成训练数据生成。

43 字段 Tab 分隔无 header，匹配真实数据格式。
"""

import csv
import json
import math
import random
from pathlib import Path

# ═══════════════════════════════════════════
# 受控词汇表
# ═══════════════════════════════════════════

STOCK_CODES = [f"60{i:04d}" for i in range(50)]
ENTITY_CODES = {
    "a": [f"a_{i:02d}" for i in range(50)],
    "b": [f"b_{i:02d}" for i in range(50)],
    "c": [f"c_{i:02d}" for i in range(50)],
    "d": [f"d_{i:02d}" for i in range(50)],
}

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
INVEST_STYLES = ["主力追踪", "事件驱动", "成长股投资", "价值投资", "趋势交易", "量化对冲"]
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
ENTITY_WORDS = ["反弹", "上涨", "下跌", "突破", "支撑", "压力", "成交量", "MACD", "KDJ", "RSI"]
SOURCE_NAMES = ["社区", "同花顺", "东方财富", "雪球"]
REC_ALGOS = ["favSecuritiesV1", "stockFenshiKxian-hot", "favEntitiesV1"]
CITIES = ["上海市", "北京市", "深圳市", "杭州市", "广州市", "成都市"]
ASSET_LEVELS = ["1万以下", "1-10万", "10-30万", "30-100万", "100万以上"]
ITEM_TYPES = ["state", "ask_answer", "iwc_dialogue", "news", "report", "snslivepost", "snsview"]
AUTHOR_NAMES = [
    "诗予拾",
    "股市老手",
    "价值猎人",
    "趋势追踪者",
    "金融分析师李",
    "小散日记",
    "涨停板猎手",
    "波段王者",
]
INVESTMENT_HORIZONS = ["超短", "短", "中", "长", "超长线"]
FUND_CODES = [
    "000001",
    "001632",
    "002011",
    "003003",
    "004685",
    "005827",
    "006228",
    "007119",
    "008087",
    "009548",
]
FUND_NAMES = [
    "华夏成长混合",
    "天弘食品饮料",
    "易方达蓝筹",
    "南方中证500",
    "嘉实新兴产业",
    "广发科技先锋",
    "富国天惠",
    "招商中证白酒",
    "博时主题行业",
    "汇添富消费行业",
]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _json_obj_array(tags: list[dict]) -> str:
    return json.dumps(tags, ensure_ascii=False)


def _json_str_array(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def _pick_with_scores(
    pool: list[str], rng: random.Random, min_n: int = 1, max_n: int = 3
) -> list[dict]:
    count = rng.randint(min_n, max_n)
    chosen = rng.sample(pool, min(count, len(pool)))
    return [{"score": round(rng.uniform(0.5, 1.0), 4), "tag": t} for t in chosen]


def _raw_overlap(a: list[str], b: list[str]) -> int:
    return 1 if set(a) & set(b) else 0


def _compute_labels(
    rng: random.Random,
    user_id: int,
    quality: float,
    stock_overlap: int,
    entity_overlap: int,
) -> tuple[int, int, int, int, int]:
    """计算标签: is_click, is_cvr, is_click_detail, is_click_stock, stay_time."""
    user_bias = math.sin(user_id * 0.17) * 0.8
    ctr_logit = (
        user_bias
        + quality * 2.5
        + stock_overlap * 0.6
        + entity_overlap * 0.4
        + rng.uniform(-1.0, 1.0)
        - 2.5
    )
    is_click = 1 if sigmoid(ctr_logit) > 0.5 else 0

    is_cvr = 0
    if is_click == 1:
        cvr_logit = (
            user_bias * 0.5 + quality * 1.5 + entity_overlap * 0.5 + rng.uniform(-0.5, 0.5) - 1.0
        )
        is_cvr = 1 if sigmoid(cvr_logit) > 0.5 else 0

    is_click_detail = 1 if is_click == 1 and rng.random() < 0.7 else 0
    is_click_stock = 1 if is_click == 1 and stock_overlap == 1 and rng.random() < 0.5 else 0

    stay_time = -1
    if is_click_detail == 1:
        stay_time = int(rng.expovariate(1.0 / 300) + 10)
    elif is_click == 1:
        stay_time = int(rng.expovariate(1.0 / 60) + 5)

    return is_click, is_cvr, is_click_detail, is_click_stock, stay_time


# ═══════════════════════════════════════════
# Item 特征 (19 字段, v2)
# ═══════════════════════════════════════════


def make_item(item_id: int, rng: random.Random) -> dict:
    stock_codes = rng.sample(STOCK_CODES, rng.randint(1, 4))
    a_code = rng.choice(ENTITY_CODES["a"])
    b_code = rng.choice(ENTITY_CODES["b"])
    c_code = rng.choice(ENTITY_CODES["c"])
    d_code = rng.choice(ENTITY_CODES["d"])
    entities = [a_code, b_code, c_code, d_code]

    quality = round(rng.uniform(0.1, 1.0), 4)
    item_type = rng.choice(ITEM_TYPES)

    wordnum = rng.randint(30, 1200)
    answerscore = rng.randint(20, 100) if item_type == "ask_answer" else 0
    has_picture = "1" if rng.random() < 0.3 else "0"
    has_video = "1" if rng.random() < 0.05 else "0"

    return {
        "item_id": item_id,
        "item_type": item_type,
        "roleneeds_first_label": _json_obj_array(_pick_with_scores(ROLE_NEEDS_FIRST, rng, 1, 2)),
        "roleneeds_second_label": _json_obj_array(_pick_with_scores(ROLE_NEEDS_SECOND, rng, 1, 2)),
        "invest_label": _json_obj_array(_pick_with_scores(INVEST_LABELS, rng, 1, 2)),
        "invest_label_second": _json_obj_array(_pick_with_scores(INVEST_LABELS_SECOND, rng, 1, 2)),
        "invest_label_third": _json_obj_array(_pick_with_scores(INVEST_LABELS_THIRD, rng, 1, 2)),
        "quality_score_label": quality,
        "stock_list": _json_str_array([f"{c},17" for c in stock_codes]),
        "entity_words_label": _json_obj_array(_pick_with_scores(ENTITY_WORDS, rng, 2, 5)),
        "item_entities_v3": _json_str_array(entities),
        "author_id": rng.randint(1000, 3000),
        "author": rng.choice(AUTHOR_NAMES),
        "source_name": rng.choice(SOURCE_NAMES),
        "emb_id": _json_str_array(entities),
        "wordnum": wordnum,
        "answerscore": answerscore,
        "has_picture": has_picture,
        "has_video": has_video,
        "_stock_codes": stock_codes,
        "_entities": entities,
    }


# ═══════════════════════════════════════════
# User 特征 (v2)
# ═══════════════════════════════════════════

THEME_LEVELS = ["高度关注", "一般关注", "偶尔关注", "极少关注"]


def make_user(user_id: int, rng: random.Random) -> dict:
    fav_codes = rng.sample(STOCK_CODES, rng.randint(3, 8))
    recent_codes = rng.sample(STOCK_CODES, rng.randint(5, 15))
    hold_codes = rng.sample(STOCK_CODES, rng.randint(2, 5))
    hist_hold_codes_data = rng.sample(STOCK_CODES, rng.randint(2, 5))

    fav_str = "|".join(
        f"{c},{rng.choice(['17', '33'])}#{rng.uniform(1, 15):.2f}" for c in fav_codes
    )
    recent_str = "|".join(f"{c},{rng.choice(['17', '33'])}" for c in recent_codes)

    kws = rng.sample(INTEREST_KEYWORDS, rng.randint(5, 10))
    kw_str = "|".join(f"{kw}#{rng.uniform(1, 10):.2f}" for kw in kws)

    auths = rng.sample(FOLLOW_AUTHORS, rng.randint(3, 8))
    auth_str = "|".join(f"{a}#2.0" for a in auths)

    hold_str = "|".join(
        f"{c}#{rng.choice(['17', '33'])}#{rng.uniform(0.01, 0.2):.3f}#{rng.uniform(0.01, 0.1):.3f}"
        for c in hold_codes
    )
    hist_hold_str = "|".join(
        f"{c}#{rng.choice(['17', '33'])}#{rng.uniform(1, 10):.1f}#{rng.uniform(0.5, 5):.2f}"
        for c in hist_hold_codes_data
    )

    hist_items = []
    hist_a_codes = []
    for _ in range(rng.randint(5, 15)):
        a = rng.choice(ENTITY_CODES["a"])
        b = rng.choice(ENTITY_CODES["b"])
        c = rng.choice(ENTITY_CODES["c"])
        d = rng.choice(ENTITY_CODES["d"])
        ts = rng.randint(1773800000, 1774000000)
        hist_items.append(f"{a},{b},{c},{d}#{ts}")
        hist_a_codes.append(a)
    hist_click_str = "|".join(hist_items)

    styles = rng.sample(INVEST_STYLES, rng.randint(1, 3))
    style_str = "|".join(f"{s}#{rng.uniform(0.01, 0.6):.2f}#0" for s in styles)

    themes = rng.sample(THEME_INTERESTS, rng.randint(2, 5))
    theme_str = "|".join(
        f"{t}#{rng.choice(THEME_LEVELS)}#{rng.uniform(0.3, 1):.2f}" for t in themes
    )
    inds = rng.sample(INDUSTRY_INTERESTS, rng.randint(2, 5))
    ind_str = "|".join(f"{i}#{rng.choice(THEME_LEVELS)}#{rng.uniform(0.3, 1):.2f}" for i in inds)

    # fund_favorites: "code#name|..." (大部分为空)
    fund_str = ""
    if rng.random() < 0.15:
        funds = rng.sample(range(len(FUND_CODES)), rng.randint(1, 3))
        fund_str = "|".join(f"{FUND_CODES[i]}#{FUND_NAMES[i]}" for i in funds)

    return {
        "user_id": user_id,
        "rec_algo": rng.choice(REC_ALGOS),
        "p_date": "20260331",
        "fav_securities": fav_str,
        "recent_stocks": recent_str,
        "interest_keywords": kw_str,
        "follow_authors": auth_str,
        "is_new_user": "老用户" if rng.random() < 0.85 else "新用户",
        "hold_stocks": hold_str,
        "hist_hold_stocks": hist_hold_str,
        "historical_click_items": hist_click_str,
        "asset_level": rng.choice(ASSET_LEVELS),
        "last_login_date": f"20260{3 + rng.randint(0, 2):01d}{rng.randint(10, 30):02d}",
        "city": rng.choice(CITIES),
        "investment_horizon": rng.choice(INVESTMENT_HORIZONS) if rng.random() < 0.7 else "",
        "invest_style": style_str,
        "theme_interest": theme_str,
        "industry_interest": ind_str,
        "fund_favorites": fund_str,
        "_fav_codes": fav_codes,
        "_recent_codes": recent_codes,
        "_hist_a_codes": hist_a_codes,
    }


# ═══════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════


def main():
    out_dir = Path(__file__).resolve().parent / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "discover_train_data.txt"

    # v2 字段顺序 (Tab 分隔, 无 header, 43 列) — 严格对应文档字段序号
    all_cols = [
        # 1-3: 基础标识
        "user_id",
        "item_id",
        "rec_algo",
        # 4-8: 标签列 (is_click/is_cvr/is_click_detail/is_click_stock/stay_time)
        "is_click",
        "is_cvr",
        "is_click_detail",
        "is_click_stock",
        "stay_time",
        # 9: 日期分区
        "p_date",
        # 10-19: User 标签特征 (A3..K001950)
        "fav_securities",
        "recent_stocks",
        "interest_keywords",
        "follow_authors",
        "is_new_user",
        "hold_stocks",
        "hist_hold_stocks",
        "historical_click_items",
        "asset_level",
        "last_login_date",
        # 20-25: User 画像特征
        "city",
        "investment_horizon",
        "invest_style",
        "theme_interest",
        "industry_interest",
        "fund_favorites",
        # 26: 内容类型
        "item_type",
        # 27-32: Item 标签特征
        "roleneeds_first_label",
        "roleneeds_second_label",
        "invest_label",
        "invest_label_second",
        "invest_label_third",
        "quality_score_label",
        # 33-39: Item 关联特征
        "stock_list",
        "entity_words_label",
        "item_entities_v3",
        "author_id",
        "author",
        "source_name",
        "emb_id",
        # 40-43: 新增 Item 属性
        "wordnum",
        "answerscore",
        "has_picture",
        "has_video",
    ]

    rng = random.Random(42)
    n_users = 200
    n_items = 2000
    rows_per_user = 10

    item_pool: dict[int, dict] = {}
    for iid in range(n_items):
        item_pool[iid] = make_item(iid, rng)

    total_rows = n_users * rows_per_user
    click_sum = 0
    cvr_sum = 0

    # 每列的值来源: item 优先, 回退 user, 标签列通过 _compute_labels 计算
    label_names = {"is_click", "is_cvr", "is_click_detail", "is_click_stock", "stay_time"}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")

        for uid in range(n_users):
            uf = make_user(uid, rng)
            # user 字典中的 stay_time 仅作特征默认值，标签值由 _compute_labels 覆盖
            for _ in range(rows_per_user):
                iid = rng.randint(0, n_items - 1)
                it = item_pool[iid]

                stock_overlap = _raw_overlap(it["_stock_codes"], uf["_fav_codes"])
                entity_overlap = _raw_overlap(it["_entities"], uf["_hist_a_codes"])

                is_click, is_cvr, is_click_detail, is_click_stock, stay_time_label = (
                    _compute_labels(
                        rng,
                        uid,
                        it["quality_score_label"],
                        stock_overlap,
                        entity_overlap,
                    )
                )
                click_sum += is_click
                cvr_sum += is_cvr

                label_vals = {
                    "is_click": is_click,
                    "is_cvr": is_cvr,
                    "is_click_detail": is_click_detail,
                    "is_click_stock": is_click_stock,
                    "stay_time": stay_time_label,
                }

                row = []
                for col in all_cols:
                    if col in label_names:
                        row.append(label_vals[col])
                    else:
                        row.append(it.get(col, uf.get(col, "")))
                w.writerow(row)

    click_rate = click_sum / total_rows
    cvr_rate = cvr_sum / total_rows
    print(f"[Generate v2] {total_rows} rows, {len(all_cols)} cols (no header)")
    print(f"  is_click rate={click_rate:.3f}  is_cvr rate={cvr_rate:.3f}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
