from __future__ import annotations

"""discover-main-sort 合成训练数据生成。

数据列顺序直接来自 `examples/feature_config_discover.yaml`，确保训练、验证和文档
使用同一份 source 契约。
"""

import csv
import json
import random
from pathlib import Path

from train.core.config import FlowConfig

from .paths import DEMO_ARTIFACT_DIR, DISCOVER_FEATURE_CONFIG

FLOW_CONFIG = FlowConfig.from_yaml(str(DISCOVER_FEATURE_CONFIG))
SOURCE_NAMES = [source.name for source in FLOW_CONFIG.sources]

STOCK_CODES = [f"60{i:04d}" for i in range(200)]
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
FOLLOW_AUTHORS = [str(100000 + i) for i in range(30)]
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
SOURCE_NAMES_POOL = ["社区", "同花顺", "东方财富", "雪球"]
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
THEME_LEVELS = ["高度关注", "一般关注", "偶尔关注", "极少关注"]


def _json_list(items: list[object]) -> str:
    return json.dumps(items, ensure_ascii=False)


def _pick_with_scores(
    pool: list[str], rng: random.Random, min_n: int = 1, max_n: int = 3
) -> list[dict[str, object]]:
    count = rng.randint(min_n, max_n)
    chosen = rng.sample(pool, min(count, len(pool)))
    return [{"score": round(rng.uniform(0.5, 1.0), 4), "tag": t} for t in chosen]


def _join_tagged(items: list[str], rng: random.Random, sep: str = "#") -> str:
    return "|".join(f"{item}{sep}{rng.uniform(0.5, 1.0):.2f}" for item in items)


def _make_item(item_id: int, rng: random.Random) -> dict[str, object]:
    item_type = rng.choice(ITEM_TYPES)
    stock_codes = rng.sample(STOCK_CODES, rng.randint(1, 4))
    entities = [
        rng.choice(ENTITY_CODES["a"]),
        rng.choice(ENTITY_CODES["b"]),
        rng.choice(ENTITY_CODES["c"]),
        rng.choice(ENTITY_CODES["d"]),
    ]
    quality = round(rng.uniform(0.1, 1.0), 4)
    title_hint = rng.choice(["热点", "解读", "复盘", "快讯", "观察", "精选"])

    return {
        "item_id": item_id,
        "item_type": item_type,
        "title": f"{title_hint} {item_type} {item_id}",
        "content": f"{title_hint}内容 {rng.choice(INTEREST_KEYWORDS)} {item_id}",
        "insight": f"{rng.choice(['看多', '中性', '谨慎'])} {rng.choice(INTEREST_KEYWORDS)}",
        "roleneeds_first_label": _json_list(_pick_with_scores(ROLE_NEEDS_FIRST, rng, 1, 2)),
        "roleneeds_second_label": _json_list(_pick_with_scores(ROLE_NEEDS_SECOND, rng, 1, 2)),
        "invest_label": _json_list(_pick_with_scores(INVEST_LABELS, rng, 1, 2)),
        "invest_label_second": _json_list(_pick_with_scores(INVEST_LABELS_SECOND, rng, 1, 2)),
        "invest_label_third": _json_list(_pick_with_scores(INVEST_LABELS_THIRD, rng, 1, 2)),
        "quality_score_label": quality,
        "stock_list": _json_list([f"{code},17" for code in stock_codes]),
        "entity_words_label": _json_list(_pick_with_scores(ENTITY_WORDS, rng, 2, 5)),
        "item_entities_v3": _json_list(entities),
        "author_id": rng.randint(1000, 3000),
        "author": rng.choice(AUTHOR_NAMES),
        "source_name": rng.choice(SOURCE_NAMES_POOL),
        "emb_id": _json_list(entities),
    }


def _make_user(user_id: int, rng: random.Random) -> dict[str, object]:
    fav_codes = rng.sample(STOCK_CODES, rng.randint(3, 8))
    recent_codes = rng.sample(STOCK_CODES, rng.randint(5, 15))
    hold_codes = rng.sample(STOCK_CODES, rng.randint(2, 5))
    hist_hold_codes = rng.sample(STOCK_CODES, rng.randint(2, 5))
    hist_items = []
    for _ in range(rng.randint(5, 15)):
        a = rng.choice(ENTITY_CODES["a"])
        b = rng.choice(ENTITY_CODES["b"])
        c = rng.choice(ENTITY_CODES["c"])
        d = rng.choice(ENTITY_CODES["d"])
        ts = rng.randint(1773800000, 1774000000)
        hist_items.append(f"{a},{b},{c},{d}#{ts}")

    return {
        "user_id": user_id,
        "fav_securities": "|".join(
            f"{code},{rng.choice(['17', '33'])}#{rng.uniform(1, 15):.2f}" for code in fav_codes
        ),
        "recent_stocks": "|".join(
            f"{code},{rng.choice(['17', '33'])}" for code in recent_codes
        ),
        "interest_keywords": _join_tagged(rng.sample(INTEREST_KEYWORDS, rng.randint(5, 10)), rng),
        "follow_authors": _join_tagged(rng.sample(FOLLOW_AUTHORS, rng.randint(3, 8)), rng),
        "is_new_user": "老用户" if rng.random() < 0.85 else "新用户",
        "hold_stocks": "|".join(
            f"{code}#{rng.choice(['17', '33'])}#{rng.uniform(0.01, 0.2):.3f}#{rng.uniform(0.01, 0.1):.3f}"
            for code in hold_codes
        ),
        "hist_hold_stocks": "|".join(
            f"{code}#{rng.choice(['17', '33'])}#{rng.uniform(1, 10):.1f}#{rng.uniform(0.5, 5):.2f}"
            for code in hist_hold_codes
        ),
        "historical_click_items": "|".join(hist_items),
        "asset_level": rng.choice(ASSET_LEVELS),
        "last_trade_date": f"202603{rng.randint(1, 31):02d}",
        "city": rng.choice(CITIES),
        "investment_horizon": rng.choice(INVESTMENT_HORIZONS) if rng.random() < 0.7 else "",
        "invest_style": "|".join(
            f"{style}#{rng.uniform(0.01, 0.6):.2f}#0"
            for style in rng.sample(INVEST_STYLES, rng.randint(1, 3))
        ),
        "theme_interest": "|".join(
            f"{theme}#{rng.choice(THEME_LEVELS)}#{rng.uniform(0.3, 1.0):.2f}"
            for theme in rng.sample(THEME_INTERESTS, rng.randint(2, 5))
        ),
        "industry_interest": "|".join(
            f"{industry}#{rng.choice(THEME_LEVELS)}#{rng.uniform(0.3, 1.0):.2f}"
            for industry in rng.sample(INDUSTRY_INTERESTS, rng.randint(2, 5))
        ),
    }


def _make_context(rng: random.Random) -> dict[str, object]:
    return {
        "rec_algo": rng.choice(REC_ALGOS),
        "scene": rng.randint(0, 9),
        "stay_time": max(0, int(rng.expovariate(1.0 / 300) + 10)),
        "p_date": "20260331",
    }


def _make_row(item: dict[str, object], user_id: int, rng: random.Random) -> dict[str, object]:
    record = {}
    record.update(item)
    record.update(_make_user(user_id, rng))
    record.update(_make_context(rng))

    missing = [name for name in SOURCE_NAMES if name not in record]
    if missing:
        raise KeyError(f"missing discover sources: {missing}")
    return record


def _output_path() -> Path:
    return DEMO_ARTIFACT_DIR / "discover_train_data.txt"


def main() -> None:
    out_path = _output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    n_users = 200
    n_items = 2000
    rows_per_user = 10

    item_pool = {iid: _make_item(iid, rng) for iid in range(n_items)}

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for uid in range(n_users):
            for _ in range(rows_per_user):
                item_id = rng.randint(0, n_items - 1)
                row = _make_row(item_pool[item_id], uid, rng)
                writer.writerow([row[name] for name in SOURCE_NAMES])

    print(f"[Generate discover] {n_users * rows_per_user} rows, {len(SOURCE_NAMES)} cols -> {out_path}")


if __name__ == "__main__":
    main()
