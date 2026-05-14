from __future__ import annotations

"""合成训练数据生成：2000 条 CTR+CVR 数据，cvr 条件依赖于 ctr。"""
import csv
import math
import os
import random


def bucket_age(age: float) -> int:
    boundaries = [18, 25, 35, 50]
    bucket = 0
    for b in boundaries:
        if age < b:
            break
        bucket += 1
    return bucket


def map_category(category: str) -> int:
    mapping = {"electronics": 1, "fashion": 2, "books": 3}
    return mapping.get(category, 0)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def user_ctr_pref(uid: int) -> float:
    return 1.5 * math.sin(uid * 0.15)


def user_cvr_pref(uid: int) -> float:
    return 1.2 * math.cos(uid * 0.12) - 0.3


def generate_row(uid: int, rng: random.Random) -> list:
    user_age = round(rng.uniform(10.0, 70.0), 1)
    categories = ["electronics", "fashion", "books", "unknown"]
    item_category = rng.choices(
        categories, weights=[0.35, 0.30, 0.20, 0.15], k=1
    )[0]
    tag1 = rng.choice(["sports", "music", "gaming", "reading", "travel"])
    tag2 = rng.choice(["sports", "music", "gaming", "reading", "travel"])
    user_tags = f"{tag1}#{rng.randint(0, 5)}|{tag2}#{rng.randint(0, 5)}"

    # Item tags: biased by category for meaningful overlap signal
    cat_tag_bias = {
        "electronics": ["gaming", "music", "travel"],
        "fashion": ["sports", "travel", "music"],
        "books": ["reading", "travel", "music"],
        "unknown": ["sports", "reading", "gaming"],
    }
    item_tag_choices = cat_tag_bias.get(item_category, ["sports", "music"])
    item_tag1 = rng.choice(item_tag_choices)
    item_tag2 = rng.choice(item_tag_choices)
    item_tags = f"{item_tag1}#1|{item_tag2}#1"

    item_price = round(rng.uniform(10.0, 1000.0), 2)

    age_bucket = bucket_age(user_age)
    cat_idx = map_category(item_category)

    # CTR logit: user preference + age + category + price
    ctr_logit = user_ctr_pref(uid)
    if age_bucket == 1:
        ctr_logit += 1.0
    elif age_bucket == 2:
        ctr_logit += 0.5
    if cat_idx == 1:
        ctr_logit += 1.5
    elif cat_idx == 2:
        ctr_logit += 0.7
    elif cat_idx == 3:
        ctr_logit -= 0.6
    ctr_logit += 0.4 * math.log(item_price + 1.0) / math.log(1001.0)
    ctr = 1 if sigmoid(ctr_logit) > 0.5 else 0

    # CVR logit: different user preference + category + age (no price); only when CTR=1
    cvr = 0
    if ctr == 1:
        cvr_logit = user_cvr_pref(uid) - 0.5
        if age_bucket in (1, 2):
            cvr_logit += 0.8
        elif age_bucket == 0:
            cvr_logit -= 0.3
        if cat_idx == 1:
            cvr_logit += 1.0
        elif cat_idx == 2:
            cvr_logit += 0.4
        elif cat_idx == 3:
            cvr_logit -= 0.8
        cvr = 1 if sigmoid(cvr_logit) > 0.5 else 0

    return [uid, user_age, item_category, user_tags, item_tags, item_price, ctr, cvr]


def main() -> None:
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    output = os.path.join(temp_dir, "train_data.csv")
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["user_id", "user_age", "item_category", "user_tags", "item_tags", "item_price", "ctr", "cvr"]
        )
        rng = random.Random(42)
        for uid in range(200):
            for _ in range(10):
                writer.writerow(generate_row(uid, rng))
    print(f"[Generate] {200 * 10} rows -> {output}")


if __name__ == "__main__":
    main()
