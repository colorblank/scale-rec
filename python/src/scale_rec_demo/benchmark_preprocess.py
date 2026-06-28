from __future__ import annotations

"""Benchmark Python and Rust feature preprocessing throughput.

The benchmark intentionally mirrors the training hot path: pandas batch slicing,
column-to-list materialization, FeatureDag.preprocess_batch(), and torch tensor
construction inside FeatureDag.
"""

import argparse
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from train.core.config import FlowConfig, Role
from train.core.dag import FeatureDag


@dataclass
class BenchResult:
    mode: str
    batch_size: int
    rows: int
    batches: int
    repeats: int
    warmup_batches: int
    prepare_s_median: float
    preprocess_s_median: float
    total_s_median: float
    rows_per_s_median: float
    features: int
    checksum: int
    profile_s_median: dict[str, float] = field(default_factory=dict)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="python/artifacts/demo/demo_train_data.txt",
        help="CSV/TSV data file to benchmark",
    )
    parser.add_argument(
        "--feature-config",
        default="examples/shared/feature_config_demo.yaml",
        help="shared feature config YAML",
    )
    parser.add_argument(
        "--batch-sizes",
        default="128,512,1024",
        help="comma-separated batch sizes to test",
    )
    parser.add_argument("--rows", type=int, default=0, help="limit rows; 0 means all rows")
    parser.add_argument("--repeat", type=int, default=3, help="timed repeats per mode/batch size")
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=2,
        help="warmup batches before timed repeats",
    )
    parser.add_argument(
        "--mode",
        choices=("python", "rust", "both"),
        default="both",
        help="preprocessing implementation to benchmark",
    )
    parser.add_argument(
        "--require-rust",
        action="store_true",
        help="fail if Rust feat_engine is unavailable",
    )
    parser.add_argument("--no-header", action="store_true", help="input file has no header row")
    parser.add_argument("--separator", default="\\t", help="field separator; use '\\t' for tab")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print phase timings for preprocessing internals",
    )
    return parser.parse_args()


def _decode_separator(raw: str) -> str:
    if raw == "\\t":
        return "\t"
    if raw == "\\n":
        return "\n"
    return raw


def _load_dataframe(
    path: str,
    flow_config: FlowConfig,
    *,
    no_header: bool,
    separator: str,
    rows: int,
) -> pd.DataFrame:
    source_names = [source.name for source in flow_config.sources]
    nrows = rows if rows > 0 else None
    if no_header:
        return pd.read_csv(
            path,
            sep=separator,
            header=None,
            names=source_names,
            nrows=nrows,
            keep_default_na=False,
        )
    return pd.read_csv(path, sep=separator, nrows=nrows, keep_default_na=False)


def _feature_source_names(flow_config: FlowConfig, df: pd.DataFrame) -> list[str]:
    names = [
        source.name
        for source in flow_config.sources
        if source.role not in {Role.LABEL, Role.DISCARD} and source.name in df.columns
    ]
    if not names:
        raise ValueError("no feature source columns found in data")
    return names


def _iter_batches(
    df: pd.DataFrame,
    feature_names: list[str],
    batch_size: int,
) -> tuple[dict[str, list[Any]], float]:
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size]
        t0 = time.perf_counter()
        columns = {name: batch_df[name].tolist() for name in feature_names}
        yield columns, time.perf_counter() - t0


def _run_warmup(
    dag: FeatureDag,
    df: pd.DataFrame,
    feature_names: list[str],
    batch_size: int,
    warmup_batches: int,
) -> None:
    if warmup_batches <= 0:
        return
    for idx, (columns, _prepare_s) in enumerate(_iter_batches(df, feature_names, batch_size)):
        if idx >= warmup_batches:
            break
        dag.preprocess_batch(columns)


def _run_once(
    dag: FeatureDag,
    df: pd.DataFrame,
    feature_names: list[str],
    batch_size: int,
) -> tuple[float, float, int, int]:
    prepare_s = 0.0
    preprocess_s = 0.0
    batches = 0
    checksum = 0
    for columns, batch_prepare_s in _iter_batches(df, feature_names, batch_size):
        prepare_s += batch_prepare_s
        t0 = time.perf_counter()
        tensors = dag.preprocess_batch(columns)
        preprocess_s += time.perf_counter() - t0
        batches += 1
        checksum += sum(int(tensor.numel()) for tensor in tensors.values())
    return prepare_s, preprocess_s, batches, checksum


def _run_python_profile_once(
    dag: FeatureDag,
    df: pd.DataFrame,
    feature_names: list[str],
    batch_size: int,
) -> dict[str, float]:
    timings = {"python_execute_s": 0.0, "python_tensor_s": 0.0}
    for columns, _batch_prepare_s in _iter_batches(df, feature_names, batch_size):
        t0 = time.perf_counter()
        result = dag.execute_batch(columns)
        timings["python_execute_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        dag.preprocessor.preprocess(result)
        timings["python_tensor_s"] += time.perf_counter() - t0
    return timings


def _run_rust_profile_once(
    dag: FeatureDag,
    df: pd.DataFrame,
    feature_names: list[str],
    batch_size: int,
) -> dict[str, float]:
    session = getattr(dag, "_rust_session", None)
    if session is None:
        raise RuntimeError("rust profile requested but FeatureDag has no Rust session")
    timings = {
        "python_str_s": 0.0,
        "python_tensor_s": 0.0,
        "rust_parse_s": 0.0,
        "rust_execute_s": 0.0,
        "rust_extract_s": 0.0,
        "rust_total_s": 0.0,
    }
    for columns, _batch_prepare_s in _iter_batches(df, feature_names, batch_size):
        t0 = time.perf_counter()
        str_columns = {
            name: [str(value) if value is not None else None for value in col]
            for name, col in columns.items()
        }
        timings["python_str_s"] += time.perf_counter() - t0

        rust_result, rust_timings = session.preprocess_batch_profile(str_columns)
        for name, value in rust_timings.items():
            timings[name] = timings.get(name, 0.0) + float(value)

        t0 = time.perf_counter()
        {name: torch.tensor(vals, dtype=torch.long) for name, vals in rust_result.items()}
        timings["python_tensor_s"] += time.perf_counter() - t0
    return timings


def _median_profile(profiles: list[dict[str, float]]) -> dict[str, float]:
    if not profiles:
        return {}
    keys = sorted({key for profile in profiles for key in profile})
    return {key: statistics.median(profile.get(key, 0.0) for profile in profiles) for key in keys}


def _build_dag(
    flow_config: FlowConfig,
    config_path: str,
    mode: str,
    *,
    require_rust: bool,
) -> FeatureDag | None:
    if mode == "python":
        return FeatureDag(flow_config)
    try:
        return FeatureDag(
            flow_config,
            use_rust=True,
            config_path=config_path,
            require_rust=require_rust,
        )
    except ImportError:
        if require_rust:
            raise
        return None


def _benchmark_mode(
    *,
    mode: str,
    flow_config: FlowConfig,
    config_path: str,
    df: pd.DataFrame,
    feature_names: list[str],
    batch_size: int,
    repeat: int,
    warmup_batches: int,
    require_rust: bool,
    profile: bool,
) -> BenchResult | None:
    dag = _build_dag(flow_config, config_path, mode, require_rust=require_rust)
    if dag is None:
        return None

    _run_warmup(dag, df, feature_names, batch_size, warmup_batches)

    prepare_times = []
    preprocess_times = []
    total_times = []
    rows_per_s = []
    batch_counts = []
    profile_runs = []
    checksum = 0
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeat):
            t0 = time.perf_counter()
            prepare_s, preprocess_s, batches, checksum = _run_once(
                dag, df, feature_names, batch_size
            )
            total_s = time.perf_counter() - t0
            prepare_times.append(prepare_s)
            preprocess_times.append(preprocess_s)
            total_times.append(total_s)
            rows_per_s.append(len(df) / total_s if total_s > 0 else 0.0)
            batch_counts.append(batches)
            if profile:
                if mode == "python":
                    profile_runs.append(
                        _run_python_profile_once(dag, df, feature_names, batch_size)
                    )
                else:
                    profile_runs.append(_run_rust_profile_once(dag, df, feature_names, batch_size))
    finally:
        if gc_was_enabled:
            gc.enable()

    return BenchResult(
        mode=mode,
        batch_size=batch_size,
        rows=len(df),
        batches=max(batch_counts) if batch_counts else 0,
        repeats=repeat,
        warmup_batches=warmup_batches,
        prepare_s_median=statistics.median(prepare_times),
        preprocess_s_median=statistics.median(preprocess_times),
        total_s_median=statistics.median(total_times),
        rows_per_s_median=statistics.median(rows_per_s),
        features=len(dag.embeddable_features()),
        checksum=checksum,
        profile_s_median=_median_profile(profile_runs),
    )


def _print_table(results: list[BenchResult]) -> None:
    if not results:
        print("no benchmark results")
        return
    header = (
        "mode    batch  rows    batches  rows/s    total_s  prepare_s  preprocess_s  features"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.mode:<7} "
            f"{result.batch_size:>5} "
            f"{result.rows:>7} "
            f"{result.batches:>8} "
            f"{result.rows_per_s_median:>9.1f} "
            f"{result.total_s_median:>8.4f} "
            f"{result.prepare_s_median:>9.4f} "
            f"{result.preprocess_s_median:>12.4f} "
            f"{result.features:>8}"
        )
    profiled = [result for result in results if result.profile_s_median]
    if profiled:
        print("\nphase timings, median seconds per full pass")
        for result in profiled:
            phase_items = {
                name: value
                for name, value in result.profile_s_median.items()
                if not name.startswith(("op:", "op_type:"))
            }
            parts = ", ".join(f"{name}={value:.4f}" for name, value in phase_items.items())
            print(f"{result.mode} batch={result.batch_size}: {parts}")

            op_type_items = sorted(
                (
                    (name.removeprefix("op_type:"), value)
                    for name, value in result.profile_s_median.items()
                    if name.startswith("op_type:")
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if op_type_items:
                top = ", ".join(f"{name}={value:.4f}" for name, value in op_type_items[:8])
                print(f"  top op types: {top}")

            op_items = sorted(
                (
                    (name.removeprefix("op:"), value)
                    for name, value in result.profile_s_median.items()
                    if name.startswith("op:")
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if op_items:
                top = ", ".join(f"{name}={value:.4f}" for name, value in op_items[:8])
                print(f"  top ops: {top}")


def main() -> None:
    args = _parse_args()
    config_path = str(Path(args.feature_config))
    flow_config = FlowConfig.from_yaml(config_path)
    separator = _decode_separator(args.separator)
    df = _load_dataframe(
        args.data,
        flow_config,
        no_header=args.no_header,
        separator=separator,
        rows=args.rows,
    )
    feature_names = _feature_source_names(flow_config, df)
    batch_sizes = [int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()]
    modes = ["python", "rust"] if args.mode == "both" else [args.mode]

    results = []
    for batch_size in batch_sizes:
        for mode in modes:
            result = _benchmark_mode(
                mode=mode,
                flow_config=flow_config,
                config_path=config_path,
                df=df,
                feature_names=feature_names,
                batch_size=batch_size,
                repeat=args.repeat,
                warmup_batches=args.warmup_batches,
                require_rust=args.require_rust,
                profile=args.profile,
            )
            if result is None:
                print("rust mode skipped: feat_engine is unavailable")
                continue
            results.append(result)

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
    else:
        _print_table(results)


if __name__ == "__main__":
    main()
