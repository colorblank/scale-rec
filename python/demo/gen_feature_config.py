"""生成大规模特征配置文件（80 个可嵌入特征）。"""
import yaml
import os

sources = []
operators = []

# ── 用户ID特征 (1) ──
sources.append({"name": "user_id", "source": "User", "dtype": "int", "default_val": "0",
                 "embed": {"vocab_size": 1000, "embed_dim": 16}})

# ── 用户统计特征：15 个浮点 + 分桶 (15) ──
for i in range(15):
    name = f"user_stat_{i}"
    sources.append({"name": name, "source": "User", "dtype": "float", "default_val": "0.0"})
    op_name = f"user_stat_{i}_bucket"
    boundaries = [0.2, 0.4, 0.6, 0.8]
    operators.append({"name": op_name, "op_type": "Bucketing", "inputs": [name],
                       "outputs": [f"{name}_bucket"],
                       "params": {"boundaries": boundaries},
                       "embed": {"vocab_size": len(boundaries)+1, "embed_dim": 4}})

# ── 用户分类特征：15 个 string + DictMapper (15) ──
for i in range(15):
    name = f"user_cat_{i}"
    sources.append({"name": name, "source": "User", "dtype": "string", "default_val": "unknown"})
    op_name = f"user_cat_{i}_map"
    mapping = {f"val_{j}": j+1 for j in range(5)}
    mapping["unknown"] = 0
    operators.append({"name": op_name, "op_type": "DictMapper", "inputs": [name],
                       "outputs": [f"{name}_idx"],
                       "params": {"mapping": mapping, "default_idx": 0},
                       "embed": {"vocab_size": 6, "embed_dim": 4}})

# ── 用户标签序列特征：5 个 StringParser + DictMapper (5×2=10) ──
for i in range(5):
    src_name = f"user_tags_{i}"
    sources.append({"name": src_name, "source": "User", "dtype": "string", "default_val": ""})
    parse_op = f"user_tags_{i}_parse"
    operators.append({"name": parse_op, "op_type": "StringParser", "inputs": [src_name],
                       "outputs": [f"{src_name}_list"],
                       "params": {"sep1": "|", "sep2": "#", "key_index": 0, "pad_len": 10, "pad_val": "none"}})
    tag_mapping = {t: j+1 for j, t in enumerate(
        ["sports","music","gaming","reading","travel","food","fashion","tech","fitness","art",
         "movie","pet","car","photo","diy"])}
    map_op = f"user_tags_{i}_map"
    operators.append({"name": map_op, "op_type": "DictMapper", "inputs": [f"{src_name}_list"],
                       "outputs": [f"{src_name}_ids"],
                       "params": {"mapping": tag_mapping, "default_idx": 0},
                       "embed": {"vocab_size": len(tag_mapping)+1, "embed_dim": 4}})

# ── 物品ID特征 (1) ──
sources.append({"name": "item_id", "source": "Item", "dtype": "int", "default_val": "0",
                 "embed": {"vocab_size": 2000, "embed_dim": 16}})

# ── 物品统计特征：15 个浮点 + 分桶 (15) ──
for i in range(15):
    name = f"item_stat_{i}"
    sources.append({"name": name, "source": "Item", "dtype": "float", "default_val": "0.0"})
    boundaries = [0.2, 0.4, 0.6, 0.8]
    operators.append({"name": f"item_stat_{i}_bucket", "op_type": "Bucketing", "inputs": [name],
                       "outputs": [f"{name}_bucket"],
                       "params": {"boundaries": boundaries},
                       "embed": {"vocab_size": len(boundaries)+1, "embed_dim": 4}})

# ── 物品分类特征：15 个 string + DictMapper (15) ──
for i in range(15):
    name = f"item_cat_{i}"
    sources.append({"name": name, "source": "Item", "dtype": "string", "default_val": "unknown"})
    mapping = {f"val_{j}": j+1 for j in range(5)}
    mapping["unknown"] = 0
    operators.append({"name": f"item_cat_{i}_map", "op_type": "DictMapper", "inputs": [name],
                       "outputs": [f"{name}_idx"],
                       "params": {"mapping": mapping, "default_idx": 0},
                       "embed": {"vocab_size": 6, "embed_dim": 4}})

# ── 物品标签序列特征：5 个 StringParser + DictMapper (5×2=10) ──
for i in range(5):
    src_name = f"item_tags_{i}"
    sources.append({"name": src_name, "source": "Item", "dtype": "string", "default_val": ""})
    operators.append({"name": f"item_tags_{i}_parse", "op_type": "StringParser", "inputs": [src_name],
                       "outputs": [f"{src_name}_list"],
                       "params": {"sep1": "|", "sep2": "#", "key_index": 0, "pad_len": 10, "pad_val": "none"}})
    tag_mapping = {t: j+1 for j, t in enumerate(
        ["sports","music","gaming","reading","travel","food","fashion","tech","fitness","art",
         "movie","pet","car","photo","diy"])}
    operators.append({"name": f"item_tags_{i}_map", "op_type": "DictMapper", "inputs": [f"{src_name}_list"],
                       "outputs": [f"{src_name}_ids"],
                       "params": {"mapping": tag_mapping, "default_idx": 0},
                       "embed": {"vocab_size": len(tag_mapping)+1, "embed_dim": 4}})

# ── ListOverlap: user tags vs item tags (5, non-embeddable) ──
for i in range(5):
    operators.append({"name": f"overlap_{i}", "op_type": "ListOverlap",
                       "inputs": [f"user_tags_{i}_list", f"item_tags_{i}_list"],
                       "outputs": [f"overlap_{i}_flag"], "params": {}})

config = {"version": "1.0.0", "sources": sources, "operators": operators}

out = os.path.join(os.path.dirname(__file__), "feature_config_demo.yaml")
with open(out, "w", encoding="utf-8") as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, width=200)

# Count embeddable
emb_count = sum(1 for s in sources if "embed" in s)
emb_count += sum(1 for o in operators if "embed" in o)
print(f"Generated {out}")
print(f"Sources: {len(sources)}, Operators: {len(operators)}")
print(f"Embeddable features: {emb_count}")
