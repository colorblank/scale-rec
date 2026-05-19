from __future__ import annotations

"""合成训练数据生成：82 特征，2000 条 CTR+CVR 数据。"""
import csv
import json
import math
import os
import random


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def bucket(val, boundaries):
    b = 0
    for v in boundaries:
        if val < v:
            break
        b += 1
    return b


def user_ctr_pref(uid):
    return 1.5 * math.sin(uid * 0.15)


def user_cvr_pref(uid):
    return 1.2 * math.cos(uid * 0.12) - 0.3


TAGS = [
    "sports",
    "music",
    "gaming",
    "reading",
    "travel",
    "food",
    "fashion",
    "tech",
    "fitness",
    "art",
    "movie",
    "pet",
    "car",
    "photo",
    "diy",
]
CAT_VALS = ["val_0", "val_1", "val_2", "val_3", "val_4"]


def generate_row(uid: int, rng: random.Random) -> list:
    row = [uid]  # user_id

    # User stat features (15)
    for _ in range(15):
        row.append(round(rng.uniform(0.0, 1.0), 3))

    # User cat features (15)
    for _ in range(15):
        row.append(rng.choice(CAT_VALS))

    # User tags (5 groups)
    user_tag_lists = []
    for _ in range(5):
        n = rng.randint(3, 8)
        tags = "|".join(f"{rng.choice(TAGS)}#{rng.randint(0, 5)}" for _ in range(n))
        row.append(tags)
        user_tag_lists.append(tags)

    # user_history
    hist = ",".join(
        f"{rng.randint(100, 900)}:{rng.randint(1, 5)}" for _ in range(rng.randint(10, 20))
    )
    row.append(hist or "0:0")

    # Context features (5) — request-level, source=Context
    row.append(rng.randint(0, 23))  # ctx_hour
    row.append(rng.choice(["phone", "pad", "pc"]))  # ctx_device
    row.append(rng.choice(["ios", "android", "web"]))  # ctx_platform
    row.append(rng.choice(["wifi", "4g", "5g"]))  # ctx_network
    row.append(rng.choice(["home", "detail", "search", "cart"]))  # ctx_page

    # item_id
    item_id = rng.randint(0, 1999)
    row.append(item_id)

    # Item stat features (15)
    for _ in range(15):
        row.append(round(rng.uniform(0.0, 1.0), 3))

    # Item cat features (15)
    for _ in range(15):
        row.append(rng.choice(CAT_VALS))

    # Item tags (5 groups)
    item_tag_lists = []
    for _ in range(5):
        n = rng.randint(3, 8)
        tags = "|".join(f"{rng.choice(TAGS)}#1" for _ in range(n))
        row.append(tags)
        item_tag_lists.append(tags)

    # ItemStats features (5) — source=ItemStats, pre-computed offline
    for _ in range(5):
        row.append(round(rng.uniform(0.0, 0.5), 4))

    # stock_list: ["code,market", ...]
    stocks = [f"{rng.randint(600000, 609999)},17" for _ in range(rng.randint(1, 3))]
    row.append(json.dumps(stocks))

    # fav_securities: code,market#weight|...
    favs = "|".join(
        f"{rng.randint(600000, 609999)},17#{rng.uniform(0, 10):.2f}"
        for _ in range(rng.randint(2, 5))
    )
    row.append(favs)

    # Labels: CTR + CVR
    ctr_logit = user_ctr_pref(uid) + sum(rng.uniform(-0.1, 0.1) for _ in range(5))
    ctr = 1 if sigmoid(ctr_logit) > 0.5 else 0
    cvr = 0
    if ctr == 1:
        cvr_logit = user_cvr_pref(uid) - 0.5 + sum(rng.uniform(-0.05, 0.05) for _ in range(3))
        cvr = 1 if sigmoid(cvr_logit) > 0.5 else 0
    row.append(ctr)
    row.append(cvr)
    return row


def main():
    out = os.path.join(os.path.dirname(__file__), "temp", "train_data.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    headers = ["user_id"]
    for i in range(15):
        headers.append(f"user_stat_{i}")
    for i in range(15):
        headers.append(f"user_cat_{i}")
    for i in range(5):
        headers.append(f"user_tags_{i}")
    headers.append("user_history")
    headers.extend(["ctx_hour", "ctx_device", "ctx_platform", "ctx_network", "ctx_page"])
    headers.append("item_id")
    for i in range(15):
        headers.append(f"item_stat_{i}")
    for i in range(15):
        headers.append(f"item_cat_{i}")
    for i in range(5):
        headers.append(f"item_tags_{i}")
    for name in ["item_ctr_7d", "item_cvr_7d", "item_click_24h", "item_order_30d", "item_expo_7d"]:
        headers.append(name)
    headers.extend(["stock_list", "fav_securities"])
    headers.extend(["ctr", "cvr"])

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        rng = random.Random(42)
        for uid in range(200):
            for _ in range(10):
                w.writerow(generate_row(uid, rng))
    print(f"[Generate] {200 * 10} rows, {len(headers)} cols -> {out}")


if __name__ == "__main__":
    main()
