"""Training loop: Polars data → DAG → Model → BCE → Adam → export."""
import argparse
import polars as pl, torch, torch.nn.functional as F
from .config import FlowConfig
from .dag import FeatureDag
from .export import export_to_safetensors, print_state_dict_keys
from .models import ModelConfig

def preprocess_batch(dag, batch_df, embed_features):
    feature_lists = {name: [] for name, _ in embed_features}
    for row in batch_df.to_dicts():
        raw = {k: v for k, v in row.items() if k in dag.sources}
        result = dag.execute(raw)
        for name, _ in embed_features:
            feature_lists[name].append(result.features.get(name, 0))
    return {name: torch.tensor(vals, dtype=torch.long) for name, vals in feature_lists.items()}

def train_epoch(model, optimizer, dag, df, embed_features, task_names, batch_size):
    model.train(); total_loss = 0.0; n_batches = 0
    for start in range(0, len(df), batch_size):
        batch_df = df.slice(start, batch_size); actual_bs = len(batch_df)
        feature_tensors = preprocess_batch(dag, batch_df, embed_features)
        outputs = model(feature_tensors)
        loss = None
        for task_name, logits in outputs.items():
            if task_name in batch_df.columns:
                labels = torch.tensor(batch_df[task_name].to_numpy(), dtype=torch.float32).view(actual_bs, 1)
                task_loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss = task_loss if loss is None else loss + task_loss
        if loss is None: continue
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item(); n_batches += 1
    return total_loss / max(n_batches, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-config", default="examples/feature_config.yaml")
    parser.add_argument("--model-config", default="config/model_lr.yaml")
    parser.add_argument("--data", default="data/train.parquet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--export-path", default="model.safetensors")
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    flow_config = FlowConfig.from_yaml(args.feature_config)
    print(f"[Config] version={flow_config.version}, sources={len(flow_config.sources)}, ops={len(flow_config.operators)}")
    model_config = ModelConfig.from_yaml(args.model_config)
    print(f"[Config] type={model_config.type}")

    dag = FeatureDag(flow_config, debug_mode=args.debug > 0)
    embed_features = dag.embeddable_features()
    print(f"[DAG] {len(embed_features)} embeddable features:")
    for name, emb in embed_features: print(f"  {name:<25} vocab={emb.vocab_size:<8} dim={emb.embed_dim}")

    tokenizer = None
    if model_config.type == "unimixer":
        from .models.unimixer.tokenizer import FeatureTokenizer
        tf = [(name, emb.vocab_size, emb.embed_dim) for name, emb in embed_features]
        tokenizer = FeatureTokenizer(tf, model_config.token_dim, model_config.num_tokens)
        print(f"[Tokenizer] {len(tf)} features -> {model_config.num_tokens} tokens x {model_config.token_dim}d = {tokenizer.total_embed_dim}")

    model = model_config.build(tokenizer=tokenizer)
    print(f"[Model] params: {sum(p.numel() for p in model.parameters()):,}")
    if args.debug: print_state_dict_keys(model)

    df = pl.read_parquet(args.data)
    print(f"[Data] {len(df)} rows, cols={df.columns}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[Train] {args.epochs} epochs, batch_size={args.batch_size}")
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, optimizer, dag, df, embed_features, [], args.batch_size)
        print(f"  epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.6f}")

    print(f"[Export] Saving to {args.export_path}")
    export_to_safetensors(model, args.export_path)
    print("Done.")

if __name__ == "__main__":
    main()
