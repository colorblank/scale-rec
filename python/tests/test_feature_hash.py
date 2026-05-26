from __future__ import annotations

"""FeatureHash 测试：Python/Rust 一致性 + 碰撞率 + 速度 benchmark."""

import time

import pytest

from train.ops.feature_hash import FeatureHash, _djb2_seeded


class TestDjb2Seeded:
    """验证 DJB2 多种子哈希与 Rust 逐位一致。"""

    def test_known_vectors(self):
        # 与 Rust djb2_seeded 输出一致 (已验证)
        assert _djb2_seeded("hello_world", 0) == 1442432207
        assert _djb2_seeded("hello_world", 1) == 626576976
        assert _djb2_seeded("hello_world", 4) == 326494931
        assert _djb2_seeded("author_617931428", 0) == 1395292063
        assert _djb2_seeded("", 0) == 5861588
        assert _djb2_seeded("", 99) == 193441046

    def test_deterministic(self):
        for s in range(20):
            assert _djb2_seeded("same_key", s) == _djb2_seeded("same_key", s)

    def test_different_seeds_different_outputs(self):
        hashes = {_djb2_seeded("key", s) for s in range(100)}
        assert len(hashes) == 100  # 100 seeds → 100 distinct hashes

    def test_different_keys_different_outputs(self):
        h1 = _djb2_seeded("abc", 0)
        h2 = _djb2_seeded("abd", 0)
        assert h1 != h2


class TestFeatureHashSingle:
    """单哈希 (num_hashes=1) 测试。"""

    def test_output_range(self):
        op = FeatureHash(500, 1)
        for _ in range(1000):
            assert 0 <= op.process(["test"]) < 500

    def test_deterministic(self):
        op = FeatureHash(1000)
        a = op.process(["hello", "world"])
        b = op.process(["hello", "world"])
        assert a == b

    def test_multiple_inputs(self):
        op = FeatureHash(10000, 1, "_")
        # 拼接 "user" + "_" + "123" 与单输入 "user_123" 是不同的键
        r1 = op.process(["user", "123"])
        r2 = op.process(["user123"])
        assert r1 != r2  # "user_123" != "user123"

    def test_none_input(self):
        op = FeatureHash(100, 1)
        r = op.process([None, "hello"])
        assert isinstance(r, int)
        assert 0 <= r < 100

    def test_empty_string(self):
        op = FeatureHash(100, 1)
        r = op.process([""])
        assert isinstance(r, int)
        assert 0 <= r < 100

    def test_batch(self):
        op = FeatureHash(1000, 1, "|")
        cols = [["a", "b", "c"], ["1", "2", "3"]]
        batch = op.process_batch(cols)
        assert len(batch) == 3
        assert batch[0] == op.process(["a", "1"])
        assert batch[1] == op.process(["b", "2"])
        assert batch[2] == op.process(["c", "3"])

    def test_batch_rejects_mixed_scalar_and_list_rows(self):
        op = FeatureHash(1000, 1, "|")
        cols = [["a", "b", ["c", "d"]], ["1", "2", "3"]]

        with pytest.raises(ValueError, match="mixed scalar/list rows"):
            op.process_batch(cols)

    def test_hash_scope_changes_output_without_affecting_default(self):
        default_op = FeatureHash(1_000_000, 1)
        scoped_op = FeatureHash(1_000_000, 1, namespace="user_id", salt="salt", version="v2")
        assert default_op.process(["abc"]) == FeatureHash(1_000_000, 1).process(["abc"])
        assert default_op.process(["abc"]) != scoped_op.process(["abc"])


class TestFeatureHashMulti:
    """多哈希 (num_hashes>1) 测试。"""

    def test_output_shape(self):
        op = FeatureHash(1000, 4)
        r = op.process(["test"])
        assert isinstance(r, list)
        assert len(r) == 4
        assert all(0 <= x < 1000 for x in r)

    def test_all_distinct(self):
        """高概率: k=4 个哈希值互不相同。"""
        op = FeatureHash(100000, 8)
        for _ in range(50):
            r = op.process(["test_key"])
            assert len(set(r)) == 8, f"Hash collision at {r}"

    def test_batch_multi(self):
        op = FeatureHash(500, 3, "|")
        cols = [["x", "y"], ["1", "2"]]
        batch = op.process_batch(cols)
        assert len(batch) == 2
        assert isinstance(batch[0], list) and len(batch[0]) == 3
        assert isinstance(batch[1], list) and len(batch[1]) == 3
        assert batch[0] == op.process(["x", "1"])
        assert batch[1] == op.process(["y", "2"])

    def test_single_vs_multi_first(self):
        """num_hashes=1 的输出等于 num_hashes=4 的第一个值。"""
        op1 = FeatureHash(1000, 1)
        op4 = FeatureHash(1000, 4)
        for key in ["test", "hello", "abc|def"]:
            assert op1.process([key]) == op4.process([key])[0]


class TestCollisionRate:
    """碰撞率测试。"""

    def test_collision_rate(self):
        vocab = 10000
        n_keys = 5000
        op = FeatureHash(vocab, 1)
        seen = set()
        collisions = 0
        for i in range(n_keys):
            h = op.process([f"key_{i}"])
            if h in seen:
                collisions += 1
            else:
                seen.add(h)
        rate = collisions / n_keys
        # 期望碰撞率 ≈ n_keys / vocab / 2 ≈ 0.25
        assert rate < 0.5, f"Collision rate {rate:.3f} too high"

    def test_multi_hash_reduces_effective_collisions(self):
        """k 个哈希的 AND 碰撞 (k 个位置全碰) 概率远低于单哈希。"""
        vocab = 5000
        n_keys = 2000
        k = 4
        op = FeatureHash(vocab, k)
        seen_sets: list[set] = [set() for _ in range(k)]
        full_collisions = 0
        for i in range(n_keys):
            hashes = op.process([f"item_{i}"])
            all_match = True
            for j, h in enumerate(hashes):
                if h not in seen_sets[j]:
                    all_match = False
                seen_sets[j].add(h)
            if all_match:
                full_collisions += 1
        rate = full_collisions / n_keys
        # 期望: (n_keys/vocab)^k ≈ (2000/5000)^4 ≈ 0.0256
        assert rate < 0.15, f"Multi-hash full-collision rate {rate:.3f} too high"


class TestBenchmark:
    """速度 benchmark。"""

    def test_hash_throughput(self):
        op = FeatureHash(100000, 4)
        n = 50000
        keys = [f"user_{i}_item_{i * 7 % 1000}" for i in range(n)]
        start = time.perf_counter()
        for key in keys:
            op.process([key])
        elapsed = time.perf_counter() - start
        rate = n / elapsed
        print(f"\n[Benchmark] FeatureHash k=4: {n:,} hashes in {elapsed:.3f}s = {rate:,.0f} ops/s")
        # 期望 > 50,000 ops/s (Python)
        assert rate > 10000, f"Too slow: {rate:,.0f} ops/s"

    def test_batch_throughput(self):
        op = FeatureHash(100000, 4)
        n = 10000
        cols = [[f"val_{i}" for i in range(n)], [f"ctx_{i % 100}" for i in range(n)]]
        start = time.perf_counter()
        op.process_batch(cols)
        elapsed = time.perf_counter() - start
        rate = n / elapsed
        print(f"[Benchmark] Batch k=4: {n:,} rows in {elapsed:.3f}s = {rate:,.0f} rows/s")
        assert rate > 5000
