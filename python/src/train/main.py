from __future__ import annotations

"""训练入口：Polars → DAG → Model → BCE → Adam → safetensors。"""

import argparse
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from .config import FlowConfig
from .dag import FeatureDag
from .export import export_to_safetensors, print_state_dict_keys
from .models import ModelConfig, get_output_spec


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dag: FeatureDag,
    df: pd.DataFrame,
    batch_size: int,
    label_col_map: dict[str, str] | None = None,
) -> float:
    """Train one epoch: slice df -> preprocess -> forward -> BCE loss -> backward.

    Args:
        label_col_map: dict mapping model output keys to DataFrame column names.
                       Defaults to identity map (output key == column name).
    """
    if label_col_map is None:
        label_col_map = {}
    model.train()
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size]
        actual_bs = len(batch_df)
        feature_tensors = dag.preprocess_batch(batch_df.to_dict("records"))
        outputs = model(feature_tensors)
        loss = None
        for task_name, logits in outputs.items():
            label_col = label_col_map.get(task_name, task_name)
            if label_col in batch_df.columns:
                labels = torch.tensor(batch_df[label_col].to_numpy(), dtype=torch.float32).view(
                    actual_bs, 1
                )
                task_loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss = task_loss if loss is None else loss + task_loss
        if loss is None:
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    """CLI: load configs -> build DAG + model -> train -> export safetensors."""
    from pathlib import Path

    _pkg_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-config",
        default=str(_pkg_root.parent / "examples" / "feature_config.yaml"),
    )
    parser.add_argument("--model-config", default=str(_pkg_root / "config" / "model_lr.yaml"))
    parser.add_argument("--data", default=str(_pkg_root / "data" / "train.parquet"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--export-path", default=str(_pkg_root / "model.safetensors"))
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    flow_config = FlowConfig.from_yaml(args.feature_config)
    print(
        f"[Config] feature version={flow_config.version}, sources={len(flow_config.sources)}, ops={len(flow_config.operators)}"
    )
    model_config = ModelConfig.from_yaml(args.model_config)
    print(f"[Config] model type={model_config.type}")

    dag = FeatureDag(flow_config, debug_mode=args.debug > 0)
    features = dag.feature_tuples()
    print(f"[DAG] {len(features)} embeddable features:")
    for name, vocab, dim in features:
        print(f"  {name:<25} vocab={vocab:<8} dim={dim}")

    tokenizer = None
    if model_config.type == "unimixer":
        from .models.unimixer.tokenizer import FeatureTokenizer

        p = model_config.params
        td = p.get("token_dim", 64)
        nt = p.get("num_tokens", 8)
        tokenizer = FeatureTokenizer(features, td, nt)
        print(
            f"[Tokenizer] {len(features)} features -> {nt} tokens x {td}d = {tokenizer.total_embed_dim}"
        )

    model = model_config.build(features, tokenizer=tokenizer)

    # UniMixer: wrap to align state_dict with Rust vb.pp("unimixer") naming
    if model_config.type == "unimixer":
        import torch.nn as nn

        # Detach submodules from original model, then nest under "unimixer" prefix
        blocks = model.blocks
        task_towers = model.task_towers
        final_norm = model.final_norm
        tokenizer_mod = model.tokenizer

        wrapper = nn.Module()
        wrapper.add_module("tokenizer", tokenizer_mod)
        inner = nn.Module()
        inner.add_module("blocks", blocks)
        inner.add_module("task_towers", task_towers)
        if final_norm is not None:
            inner.add_module("final_norm", final_norm)
        wrapper.add_module("unimixer", inner)

        # Hook up forward — closure over original model for its config/forward logic
        _raw = model

        def _forward(
            self: torch.nn.Module,
            x_inputs: dict[str, torch.Tensor],
            temperature: float | None = None,
        ) -> dict[str, Any]:
            return _raw(x_inputs, temperature)

        import types

        wrapper.forward = types.MethodType(_forward, wrapper)
        model = wrapper

    print(f"[Model] params: {sum(p.numel() for p in model.parameters()):,}")
    if args.debug:
        print_state_dict_keys(model)

    # Build label column mapping — model tells us via output_spec()
    spec = get_output_spec(model_config.type, model)
    label_col_map = spec.get("label_col_map", {})

    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    # Ensure label columns are int
    for col in ["ctr", "cvr", "pred"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    print(f"[Data] {len(df)} rows, cols={df.columns}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[Train] {args.epochs} epochs, batch_size={args.batch_size}")
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, optimizer, dag, df, args.batch_size, label_col_map)
        print(f"  epoch {epoch + 1:3d}/{args.epochs}  loss={avg_loss:.6f}")

    print(f"[Export] {args.export_path}")
    export_to_safetensors(model, args.export_path)
    print("Done.")


if __name__ == "__main__":
    main()
