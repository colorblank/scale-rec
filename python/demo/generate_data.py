from __future__ import annotations

"""合成训练数据生成：基于 feature_config 生成 2000 条 CTR 数据，标签由确定性特征交互产生。"""
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


def user_preference(uid: int) -> float:
    """Fixed user preference: each user gets a stable score from their ID."""
    return 1.5 * math.sin(uid * 0.15) - 0.5


def generate_row(uid: int, rng: random.Random) -> list:
    """Generate one row; label is deterministic given features (no Bernoulli noise)."""
    user_age = round(rng.uniform(10.0, 70.0), 1)
    categories = ["electronics", "fashion", "books", "unknown"]
    item_category = rng.choices(
        categories, weights=[0.35, 0.30, 0.20, 0.15], k=1
    )[0]
    tag1 = rng.choice(["sports", "music", "gaming", "reading", "travel"])
    tag2 = rng.choice(["sports", "music", "gaming", "reading", "travel"])
    user_tags = f"{tag1}#{rng.randint(0, 5)}|{tag2}#{rng.randint(0, 5)}"
    item_price = round(rng.uniform(10.0, 1000.0), 2)

    age_bucket = bucket_age(user_age)
    cat_idx = map_category(item_category)

    logit = user_preference(uid)
    if age_bucket == 1:
        logit += 1.0
    elif age_bucket == 2:
        logit += 0.5
    if cat_idx == 1:
        logit += 1.5
    elif cat_idx == 2:
        logit += 0.7
    elif cat_idx == 3:
        logit -= 0.6
    logit += 0.4 * math.log(item_price + 1.0) / math.log(1001.0)

    ctr = 1 if sigmoid(logit) > 0.5 else 0
    return [uid, user_age, item_category, user_tags, item_price, ctr]


def main() -> None:
    os.makedirs(os.path.dirname(__file__) or ".", exist_ok=True)
    output = os.path.join(os.path.dirname(__file__), "train_data.csv")
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["user_id", "user_age", "item_category", "user_tags", "item_price", "ctr"]
        )
        # 200 users, 10 rows each = 2000 total; each user has consistent preference
        rng = random.Random(42)
        for uid in range(200):
            for _ in range(10):
                writer.writerow(generate_row(uid, rng))
    print(f"[Generate] {200 * 10} rows → {output}")


if __name__ == "__main__":
    main()
