# 特征预处理系统文档

本文档完整描述 scale-rec 特征预处理系统的架构、配置格式、全部 10 个算子及执行模式，面向算法工程师和系统开发者。

---

## 1. 架构概览

特征预处理管线由三层组成：

```
FeatureConfig (YAML)
  ├── sources[]      — 原始输入特征定义
  └── operators[]    — 算子 DAG 节点定义
        │
        ▼
  FeatureDag (Rust: src/feats/dag.rs  Python: python/src/train/dag.py)
  ├── 拓扑排序 → execution_order
  ├── 预编译执行计划 (ExecutionPlan)
  └── 特征类型推导 (user / item / cross)
        │
        ▼
  execute(raw_inputs) → FeatureResult
  execute_batch(columns) → HashMap<String, Vec<Fv>>
  execute_plan(columns) → Vec<Vec<Fv>>   (零 HashMap 热路径)
```

**核心设计原则**：
- **双语言一致性**：Rust 推理引擎与 Python 训练管线共享同一份 YAML 配置，算子行为完全一致。
- **特征定义唯一来源**：所有特征的 vocab_size、embed_dim 由 `FeatureDag.embeddable_features()` 统一导出，模型配置中不重复定义。
- **强类型特征值**：`Fv` 枚举替代 `Arc<dyn Any>`，消除 vtable/downcast 运行时开销。

---

## 2. YAML 配置格式

### 2.1 顶层结构

```yaml
version: "1.0.0"
sources:
  - name: user_id
    source: User          # 来源分组: User | Item | Context | ItemStats
    dtype: int            # 数据类型
    default_val: "0"      # 缺失默认值（字符串形式）
    embed:                # 可选：直接送入 Embedding
      vocab_size: 1000
      embed_dim: 16
operators:
  - name: user_id_map     # 算子名称（全局唯一）
    op_type: DictMapper   # 算子类型
    inputs: [user_id]     # 输入特征名列表
    outputs: [user_id_idx] # 输出特征名列表
    params:               # 算子参数（自由格式 YAML）
      mapping: {ios: 1, android: 2}
      default_idx: 0
    embed:                # 可选：输出送入 Embedding
      vocab_size: 4
      embed_dim: 8
```

### 2.2 SourceDef — 原始输入源

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 是 | 全局唯一特征名 |
| `source` | string | 是 | 来源分组，用于 broadcast 模式优化 |
| `dtype` | enum | 是 | `int` / `float` / `string` / `list` |
| `default_val` | string | 是 | 默认值（字符串形式，按 dtype 解析） |
| `embed` | object | 否 | 直接嵌入配置 `{vocab_size, embed_dim}` |

**List 类型**：`dtype` 支持嵌套列表声明。
```yaml
dtype: { dtype: string, length: 10 }
```

**Source 分组的语义**（详见第 5 节 broadcast 模式）：

| source 值 | 分类 | 含义 |
|---|---|---|
| `User` | user | 用户画像特征，一次请求内不变 |
| `Context` | user | 请求上下文特征，一次请求内不变 |
| `Item` | item | 候选物品特征，每个候选不同 |
| `ItemStats` | item | 物品离线统计特征，每个候选不同 |

### 2.3 OperatorDef — 算子节点

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 是 | 全局唯一算子名 |
| `op_type` | string | 是 | 算子类型标识 |
| `inputs` | string[] | 是 | 输入特征名列表 |
| `outputs` | string[] | 是 | 输出特征名列表 |
| `params` | object | 否 | 算子参数（各算子自行解析） |
| `embed` | object | 否 | 输出嵌入配置 `{vocab_size, embed_dim}` |

---

## 3. 类型系统

### 3.1 Fv — 强类型特征值

```rust
pub enum Fv {
    Int(i32),           // type_name: "int"
    Float(f32),         // type_name: "float"
    Str(String),        // type_name: "str"
    IntList(Vec<i32>),  // type_name: "list[int]"
    StrList(Vec<String>), // type_name: "list[str]"
}
```

所有算子的输入输出均为 `Fv`。`IntList` 用于序列特征（ID 序列），`StrList` 用于标签/字符串列表特征。

### 3.2 类型转换规则

DAG 在初始化阶段根据 SourceDef.dtype 生成默认值：
- `Int` → `Fv::Int(parsed)`
- `Float` → `Fv::Float(parsed)`
- `String` → `Fv::Str(string)`
- `List { dtype: String, length: N }` → `Fv::StrList(vec![string; N])`

---

## 4. 全部 10 个算子

### 4.1 Bucketing — 数值分桶

将连续数值离散化为整数桶索引。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `boundaries` | float[] | — | 有序桶边界，N 个边界产生 N+1 个桶 |

**输入**：`Int` 或 `Float`（单值）
**输出**：`Int`（桶索引 0..N）

**行为**：桶 0 = `(-inf, boundaries[0])`，桶 i = `[boundaries[i-1], boundaries[i])`，桶 N = `[boundaries[N-1], +inf)`。

**示例**：
```yaml
- name: hour_bucket
  op_type: Bucketing
  inputs: [ctx_hour]
  outputs: [hour_bucket]
  params: { boundaries: [6, 12, 18, 22] }
  embed: { vocab_size: 5, embed_dim: 4 }
# 输入 15 → 输出 2（落在 [12, 18) 区间）
```

---

### 4.2 DictMapper — 字典映射

将字符串/数值 key 映射为整数索引，支持单值和列表输入。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `mapping` | object | — | key → index 映射表 |
| `default_idx` | int | 0 | 未命中时的默认索引 |

**输入**：`Int` / `Float` / `Str` / `StrList` / `IntList`
**输出**：`Int` / `IntList`（与输入结构对应）

**行为**：
- 单值输入 → `mapping.get(str(val), default_idx)`
- 列表输入 → 逐元素映射，保持列表长度
- 非 string/int/float 类型元素 → `default_idx`

**示例**：
```yaml
- name: device_map
  op_type: DictMapper
  inputs: [ctx_device]
  outputs: [device_idx]
  params:
    mapping: { phone: 1, pad: 2, pc: 3 }
    default_idx: 0
  embed: { vocab_size: 4, embed_dim: 4 }
# 输入 "phone" → 输出 1
```

---

### 4.3 StringParser — 两级字符串分词

解析 `K1#V1|K2#V2` 格式的拼接字符串，支持固定长度填充。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `sep1` | string | `"#"` | 一级分隔符（分隔键值对） |
| `sep2` | string | `"\|"` | 二级分隔符（分隔 Key 和 Value） |
| `key_index` | int | 0 | 提取索引：0=Key, 1=Value |
| `pad_len` | int | 0 | 结果列表固定长度，不足则填充 |
| `pad_val` | string | `"unknown"` | 填充值 |

**输入**：`Str`
**输出**：`StrList`

**示例**：
```yaml
- name: tags_parse
  op_type: StringParser
  inputs: [user_tags_raw]
  outputs: [tag_list]
  params:
    sep1: "|"
    sep2: "#"
    key_index: 0
    pad_len: 10
    pad_val: "none"
# 输入 "sports#1|music#2|gaming#3"
# 输出 ["sports", "music", "gaming", "none", "none", ...]（补齐至 10 个）
```

---

### 4.4 JsonExtractList — JSON 数组解析

解析 JSON 字符串中的数组，提取指定 key 的内容。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `key` | string | null | 对象数组的提取字段名，null 表示数组元素为纯值 |
| `pad_len` | int | 0 | 结果列表固定长度 |
| `pad_val` | string | `""` | 填充值 |

**输入**：`Str`（JSON 格式字符串）
**输出**：`StrList`

**行为**：
- `key` 非空：从对象数组中提取指定字段
- `key` 为空：数组元素直接转为字符串
- 解析失败或空字符串 → 全填充值

**示例**：
```yaml
# 场景一：对象数组提取字段
- name: extract_tags
  op_type: JsonExtractList
  inputs: [json_tags]
  outputs: [tag_list]
  params: { key: "tag", pad_len: 3, pad_val: "none" }
# 输入 '[{"tag":"科技"},{"tag":"数码"}]'
# 输出 ["科技", "数码", "none"]

# 场景二：简单字符串数组
- name: extract_codes
  op_type: JsonExtractList
  inputs: [stock_list]
  outputs: [codes]
  params: { pad_len: 5, pad_val: "" }
# 输入 '["600519,17","000001,33"]'
# 输出 ["600519,17", "000001,33", "", "", ""]
```

---

### 4.5 ListStringParser — 列表二次切分

对 `StrList` 中每个元素进行分隔符切分，提取指定索引。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `sep` | string | `","` | 元素内分隔符 |
| `key_index` | int | 0 | 提取切分后的第几个字段 |

**输入**：`StrList`
**输出**：`StrList`

**示例**：
```yaml
- name: extract_code
  op_type: ListStringParser
  inputs: [code_with_market]
  outputs: [pure_codes]
  params: { sep: ",", key_index: 0 }
# 输入 ["600519,17", "000001,33"]
# 输出 ["600519", "000001"]
```

---

### 4.6 ExpressionOp — 脚本表达式

使用 Rhai 脚本执行数学计算。变量 `v0, v1, ...` 对应输入列表的索引。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `script` | string | — | Rhai 表达式，支持 `log`, `abs`, `max`, `min`, `sqrt` |

**输入**：任意 `Fv`（在脚本中自动转为数值）
**输出**：`Float`（计算结果）

**内置函数**：`log(x)`, `abs(x)`, `max(x,y)`, `min(x,y)`, `sqrt(x)`

**示例**：
```yaml
# CTR 平滑计算
- name: calc_smooth_ctr
  op_type: ExpressionOp
  inputs: [click_cnt, expo_cnt]
  outputs: [smooth_ctr]
  params: { script: "v0 / (v1 + 1.0)" }
# 输入 [3, 10] → 输出 0.272727...

# 对数变换
- name: log_transform
  op_type: ExpressionOp
  inputs: [raw_value]
  outputs: [log_value]
  params: { script: "log(v0 + 1.0)" }
```

---

### 4.7 CrossFeature — 特征交叉

两个列表特征的笛卡尔积或内积。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `cross_type` | string | `"cartesian"` | `"cartesian"` 或 `"inner_product"` |

**输入**：两个 `StrList`（cartesian）或两个 `IntList`（inner_product）
**输出**：`StrList`（cartesian）或 `Float`（inner_product）

**示例**：
```yaml
# 笛卡尔积：用户标签 × 物品标签
- name: tag_cross
  op_type: CrossFeature
  inputs: [user_tags, item_tags]
  outputs: [cross_tags]
  params: { cross_type: cartesian }
# 输入 ["a","b"] × ["x","y"]
# 输出 ["a_x", "a_y", "b_x", "b_y"]

# 内积：两个嵌入向量的点积
- name: vector_dot
  op_type: CrossFeature
  inputs: [user_vec, item_vec]
  outputs: [dot_score]
  params: { cross_type: inner_product }
# 输入 [1,2,3] × [4,5,6] → 输出 1*4 + 2*5 + 3*6 = 32.0
```

---

### 4.8 ListOverlap — 列表交集检测

判断两个字符串列表是否有交集。

| 参数 | 无 |
|---|---|

**输入**：两个 `StrList`
**输出**：`Int`（1=有交集，0=无交集）

**示例**：
```yaml
- name: tag_overlap
  op_type: ListOverlap
  inputs: [user_tag_list, item_tag_list]
  outputs: [overlap_flag]
# 输入 ["sports","music"] 和 ["music","travel"]
# 输出 1（交集: "music"）
```

---

### 4.9 SequenceOp — 序列截断填充

将整数序列裁剪或填充至固定长度。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `max_len` | int | 10 | 序列最大长度 |
| `pad_val` | int | 0 | 填充值 |

**输入**：`IntList`
**输出**：`IntList`（长度固定为 max_len）

**示例**：
```yaml
- name: pad_history
  op_type: SequenceOp
  inputs: [click_seq]
  outputs: [padded_seq]
  params: { max_len: 20, pad_val: 0 }
# 输入 [3, 7, 15]（长度 3）
# 输出 [3, 7, 15, 0, 0, ..., 0]（长度 20）
```

---

### 4.10 StringConcatHash — 哈希交叉

拼接两个字段并通过哈希映射到固定词表。支持**训练模式**（自动构建映射）和**推理模式**（使用持久化映射文件）。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `vocab_size` | int | 1000 | 总词表大小 |
| `oov_reserve` | int | 0 | OOV 保留空间（vocab_size 尾部） |
| `separator` | string | `"\|"` | 拼接分隔符 |
| `mode` | string | `"train"` | `"train"` 自动建表 / `"inference"` 读取映射文件 |
| `hash_map_path` | string | `""` | 推理模式下映射文件路径 |

**输入**：两个 `Str`（或 `Int` + `Str`）
**输出**：`Int`（词表内索引）

**行为**：
- 拼接：`s1 + separator + s2`
- 训练模式：首次遇到的 key 分配递增索引，词表满后使用 djb2 哈希落入 OOV 区
- 推理模式：查映射表，未命中则 djb2 哈希落入 OOV 区
- 哈希算法：djb2（`h = 5381; h = h*33 + byte`），对 `0x7FFFFFFF` 取模

**示例**：
```yaml
- name: user_item_cross
  op_type: StringConcatHash
  inputs: [user_id, item_id]
  outputs: [cross_id]
  params:
    vocab_size: 10000
    oov_reserve: 1000
    separator: "_"
    mode: train
    hash_map_path: ""
  embed: { vocab_size: 10000, embed_dim: 16 }
# 输入 user_id="u1", item_id="i5" → key="u1_i5" → 映射索引或哈希值
```

**插件算子**：`PluginOp` 通过 `cdylib` 动态加载外部算子。`params` 需包含 `path`（动态库路径）和 `op_name`（算子标识）。外部库必须导出 `process_custom` 符号，签名为 `fn(&[&(dyn Any)]) -> Result<Box<dyn Any>, String>`。不建议在生产环境依赖，主要用于实验性特征快速验证。

---

## 5. 执行模式

### 5.1 单样本执行 `execute()`

```rust
let result: FeatureResult = dag.execute(&raw_inputs)?;
// result.features: HashMap<String, Fv>  — 所有特征值（源 + 算子输出）
// result.source_names: HashSet<String>  — 来自源的名称
// result.computed_names: HashSet<String> — 算子计算的名称
```

执行流程：
1. **Stage 1**：注入原始输入值
2. **Stage 2**：为缺失的源填充默认值
3. **Stage 3**：按拓扑序执行所有算子

用于 Python 训练管线的 `preprocess_batch()`，逐行执行后 stack 为 tensor。

### 5.2 批量执行 `execute_batch()`

```rust
let columns: HashMap<String, Vec<Fv>> = ...;  // 列式数据
let result = dag.execute_batch(&columns, &skip_ops)?;
```

将多个样本按列组织，算子调用 `process_batch()` 一次处理整列。消除逐行的 trait object 分发开销。每个算子可选择性实现 `process_batch()`，默认回退到逐行 `process()`。

### 5.3 预编译计划执行 `execute_plan()`（热路径）

```rust
let context: Vec<Vec<Fv>> = dag.plan.execute_plan(&columns, &skip_op_idx, &precomputed)?;
```

**零 HashMap 运行时查找**。DAG 构建时预解析所有输入/输出列为整数索引，运行时用 `Vec<Vec<Fv>>` + 索引访问替代 `HashMap<String, Vec<Fv>>`。这是 `InferenceEngine` 的默认执行路径。

### 5.4 Broadcast 模式

用于推荐系统的典型场景：一次请求中用户特征不变，仅物品特征变化。

```
┌──────────────────────────────┐
│ Step 1: Precompute           │
│ user + 1 item → 提取 user    │
│ 算子输出（user_op_indices）  │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│ Step 2: Batch                │
│ user 特征广播 N 份           │
│ item 特征 N 个样本各不同     │
│ skip user ops, 注入 precomputed │
│ → 避免重复计算 user 侧特征   │
└──────────────────────────────┘
```

实现详情（`src/server/engine.rs`）：
- 构建时通过 `op_source_kind()` 分类每个算子为 `"user"` / `"item"` / `"cross"`
- 预计算阶段：用 1 个物品 + 用户特征执行完整 DAG，缓存 user 算子的输出列
- 批量阶段：用户特征广播到 N 行，跳过 user 算子，直接注入预计算值

**特征来源传播规则**：
- `User` / `Context` 源的直接特征 → `"user"`
- `Item` / `ItemStats` 源 → `"item"`
- 算子输入同时包含 user 和 item → `"cross"`
- 仅 user 输入 → `"user"`；仅 item 输入 → `"item"`

---

## 6. Pipeline 最佳实践

### 6.1 多级级联解析

处理复杂嵌套数据的典型链路：JSON 提取 → 切分 → 映射。

```yaml
# 原始字段: stock_json = '[{"code":"600519,17"}, {"code":"000001,33"}]'
# 目标: 提取纯股票代码并映射为 Embedding ID

- name: step1_json
  op_type: JsonExtractList
  inputs: [stock_json]
  outputs: [codes_raw]
  params: { key: "code", pad_len: 5, pad_val: "" }

- name: step2_split
  op_type: ListStringParser
  inputs: [codes_raw]
  outputs: [pure_codes]
  params: { sep: ",", key_index: 0 }

- name: step3_map
  op_type: DictMapper
  inputs: [pure_codes]
  outputs: [stock_ids]
  params: { mapping: { "600519": 1, "000001": 2 }, default_idx: 0 }
  embed: { vocab_size: 1000, embed_dim: 16 }
```

### 6.2 动态特征计算与分桶

```yaml
# 原始特征: click_cnt, expo_cnt
# 目标: 计算平滑 CTR 并分桶

- name: calc_ctr
  op_type: ExpressionOp
  inputs: [click_cnt, expo_cnt]
  outputs: [ctr]
  params: { script: "v0 / (v1 + 1.0)" }

- name: bucket_ctr
  op_type: Bucketing
  inputs: [ctr]
  outputs: [ctr_bucket]
  params: { boundaries: [0.01, 0.05, 0.1, 0.2] }
  embed: { vocab_size: 5, embed_dim: 8 }
```

### 6.3 序列特征上的哈希交叉

```yaml
# 对变长行为序列和候选物品进行哈希交叉
# StringConcatHash 的 process_batch 自动处理序列中的每个元素

- name: seq_pad
  op_type: SequenceOp
  inputs: [click_seq]
  outputs: [padded_seq]
  params: { max_len: 20, pad_val: 0 }

- name: seq_cross
  op_type: StringConcatHash
  inputs: [padded_seq, item_id]
  outputs: [cross_ids]
  params:
    vocab_size: 10000
    oov_reserve: 1000
    mode: train
  embed: { vocab_size: 10000, embed_dim: 16 }
```

### 6.4 用户-物品交互信号

```yaml
# 标签重叠 + 标签交叉组合

- name: tag_overlap
  op_type: ListOverlap
  inputs: [user_tags, item_tags]
  outputs: [match_flag]

- name: tag_cross
  op_type: CrossFeature
  inputs: [user_tags, item_tags]
  outputs: [cross_tags]
  params: { cross_type: cartesian }
```

---

## 7. 算子速查表

| 算子 | 输入类型 | 输出类型 | 核心参数 | 典型场景 |
|---|---|---|---|---|
| Bucketing | Int/Float | Int | boundaries | 连续值离散化 |
| DictMapper | Any/List | Int/IntList | mapping, default_idx | 类别→索引 |
| StringParser | Str | StrList | sep1, sep2, key_index | 拼接字符串解析 |
| JsonExtractList | Str | StrList | key, pad_len | JSON 数组提取 |
| ListStringParser | StrList | StrList | sep, key_index | 列表元素二次拆解 |
| ExpressionOp | Numeric | Float | script | 数学变换 |
| CrossFeature | StrList×2 / IntList×2 | StrList / Float | cross_type | 特征交叉 |
| ListOverlap | StrList×2 | Int | — | 列表交集检测 |
| SequenceOp | IntList | IntList | max_len, pad_val | 序列定长对齐 |
| StringConcatHash | Str×2 | Int | vocab_size, mode | 在线哈希交叉 |
| PluginOp | Any | Any | path, op_name | 实验性外部算子 |

---

## 8. 扩展新算子

1. **Rust**：在 `src/feats/ops/` 下新建文件，实现 `CustomOp` trait（`name()` + `process()` + 可选 `process_batch()`）
2. **注册**：在 `src/feats/ops/mod.rs` 中添加 `pub mod` + `pub use`，在 `src/feats/dag.rs` 的 `create_op()` 中添加 match 分支
3. **Python**：在 `python/src/train/ops/` 下新建文件，实现 `process()` + `process_batch()` 方法
4. **注册 Python**：在 `ops/__init__.py` 中导出，在 `dag.py` 的 `_create_node()` 中添加分支
5. **测试**：Rust 侧 `#[cfg(test)]` 模块，Python 侧 `tests/test_ops.py`
6. **文档**：更新本文档的算子列表和速查表
