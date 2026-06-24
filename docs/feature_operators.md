# 特征预处理系统文档

本文档完整描述 scale-rec 特征预处理系统的架构、配置格式、17 个基础算子及其执行模式，面向算法工程师和系统开发者。

如果你想先理解这些算子在整个推荐系统里的位置，建议先读 [推荐排序系统教程 - 03. 特征工程契约](tutorial/03_feature_contract.md)。教程讲的是“为什么要这么配”，本文档讲的是“每个算子具体怎么做”。

---

## 1. 架构概览

特征预处理管线由三层组成：

```
FeatureConfig (YAML)
  ├── data_sources[] — 在线取数来源目录
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
data_sources:
  - name: user_profile_hbase
    kind: hbase
    description: user profile and behavior features
sources:
  - name: user_id
    source: User          # 来源分组: User | Item | Context
    data_source: user_profile_hbase # 在线取数来源，引用 data_sources[].name
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
      pooling: mean       # 可选：first | flatten | mean | sum | max (默认 first)
      seq_len: 3          # 可选：序列长度（默认由列表特征的 schema 自动继承）
      truncation: tail    # 可选：截断方向 head (头部截断，默认) | tail (尾部截断)
```

### 2.2 DataSourceDef — 在线取数来源

`data_sources` 是 feature config 的顶层目录，用来描述在线请求聚合层应该从哪些系统准备原始字段。它不改变 DAG 执行逻辑；Rust/Python 仍只消费已经传入请求或训练样本的字段。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 是 | 全局唯一来源名，被 `sources[].data_source` 引用 |
| `kind` | string | 是 | 来源类型，例如 `hbase`、`elasticsearch`、`flink`、`milvus`、`request` |
| `description` | string | 否 | 人类可读说明 |
| `params` | object | 否 | 连接别名、表名、索引名、字段映射等扩展参数 |

### 2.3 SourceDef — 原始输入源

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 是 | 全局唯一特征名 |
| `source` | string | 是 | 来源分组，用于 broadcast 模式优化 |
| `data_source` | string | 否 | 在线取数来源名，必须引用顶层 `data_sources[].name` |
| `dtype` | enum/object | 是 | `int` / `float` / `string` / `enum` / `list` |
| `default_val` | string | 是 | 默认值（字符串形式，按 dtype 解析） |
| `embed` | object | 否 | 直接嵌入配置（结构同下方 EmbedConfig） |

**List 类型**：`dtype` 支持嵌套列表声明。
```yaml
dtype: { list: { item_dtype: string, max_len: 10 } }
```

**Enum 类型**：`dtype` 支持枚举类型声明，可包含合法值列表及 OOV 映射规则。
```yaml
dtype:
  enum:
    values: [unknown, books, fashion]
    default: unknown
    oov: unknown
```

**Source 分组的语义**（详见第 5 节 broadcast 模式）：

| source 值 | 分类 | 含义 |
|---|---|---|
| `User` | user | 用户画像特征，一次请求内不变 |
| `Context` | user | 请求上下文特征，一次请求内不变 |
| `Item` | item | 候选物品特征，每个候选不同 |

### 2.4 OperatorDef — 算子节点

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 是 | 全局唯一算子名 |
| `op_type` | string | 是 | 算子类型标识 |
| `inputs` | string[] | 是 | 输入特征名列表 |
| `outputs` | string[] | 是 | 输出特征名列表 |
| `params` | object | 否 | 算子参数（各算子自行解析） |
| `embed` | object | 否 | 输出嵌入配置（见下方 2.5 EmbedConfig） |

### 2.5 EmbedConfig — 嵌入配置

特征编排 DAG 允许在算子输出的最终索引特征上声明 `embed`，从而将离散的特征索引转换为稠密特征向量。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `vocab_size` | int | — | 词表空间大小（词表索引范围为 `[0, vocab_size)`） |
| `embed_dim` | int | — | 嵌入向量维度 |
| `pooling` | string | `first` | 池化策略：`first` / `flatten` / `mean` / `sum` / `max`（详见下文） |
| `seq_len` | int | null | 固定截断/填充的序列长度。当输出为 `list` 且 `pooling != first` 时，默认从 `dtype.max_len` 自动继承。 |
| `truncation` | string | `head` | 列表超长时的截断方向：`head`（保留左侧/头部）或 `tail`（保留右侧/尾部，常用于最新行为序列） |

---

### 2.5 嵌入与池化深度解析

#### 2.5.1 单一定义源原则 (Single Source of Truth)
*   **统一声明在算子层**：所有 Embedding 应当统一配置在**算子节点（`OperatorDef`）的 `embed` 字段**中。输入源（`SourceDef`）中的 `embed` 声明已被弃用。
*   这可确保所有原始特征先经过算子处理（如字典映射 `DictMapper`、数值分桶 `Bucketing` 或哈希 `FeatureHash`）转化为致密整数索引后，再送入统一的 Embedding 空间，避免了冗余的 Embedding 层初始化。

#### 2.5.2 特征对齐与 Pooling 机制
对于列表类型的特征（如 `IntList`），为将其传入下游全连接层（Dense MLP），需将变长或多值的 Embedding 转化为固定大小的张量。支持五种池化模式：

1.  **`first`（首元素池化）**
    *   **行为**：仅取列表第一个元素的 Embedding 向量。
    *   **输出维度**：`[embed_dim]`
    *   **适用场景**：多值特征的单值提取，或列表退化为标量时的默认兜底。
2.  **`mean` / `sum` / `max`（均值/求和/最大值池化）**
    *   **行为**：对列表中所有元素的 Embedding 向量进行按元素（Element-wise）的均值、求和或最大值约简操作。
    *   **输出维度**：`[embed_dim]`
    *   **适用场景**：无序的标签集合（如 `user_tags`）、多值类别特征。
3.  **`flatten`（顺序打平）**
    *   **行为**：将列表中各个元素的 Embedding 向量按照先后顺序横向拼接。
    *   **输出维度**：`[seq_len * embed_dim]`
    *   **适用场景**：有序的行为序列特征（如 `historical_click_items`），常接 Attention 机制或直接送入 MLP。

#### 2.5.3 截断与填充 (Alignment & Truncation)
当特征输出为 `IntList` 且使用了 `pooling = flatten` 时，序列必须对齐为固定的 `seq_len`：
*   **长度推导**：若在 `embed` 中未显式指定 `seq_len`，DAG 在加载时将自动寻找该特征 schema 中的 `max_len`（通过上游算子或源的类型定义继承）。
*   **截断策略 (`truncation`)**：
    *   `head`：截断序列右侧，保留左侧头部元素（即第 `[0, seq_len)` 个元素）。
    *   `tail`：截断序列左侧，保留右侧尾部元素（即最新的 `seq_len` 个元素）。在推荐系统中，**行为序列特征通常应该配置为 `tail`**，以便在序列超长时保留最近的交互，捕捉最实时的兴趣。
*   **零向量 Padding**：当列表元素不足 `seq_len` 时，右侧（若为 `head` 截断）或左侧（若为 `tail` 截断）会填充索引值 `0`。下游的 PyTorch/Candle Embedding 层会将索引 `0` 固定映射为零向量（Zero Embedding），确保 Padding 不会引入干扰信号。

#### 2.5.4 典型 YAML 声明示例

##### 示例 1：用户无序兴趣标签（均值池化，哈希到 500 大小的词表）
```yaml
- name: user_tags_hash
  op_type: FeatureHash
  inputs: [user_raw_tags]  # 原始列表数据如 ["sports", "gaming"]
  outputs: [user_tag_indices]
  params:
    vocab_size: 500
  embed:
    vocab_size: 500
    embed_dim: 16
    pooling: mean         # 池化成 [16] 维的单向量
```

##### 示例 2：物品历史点击序列（打平，保留最近的 10 个行为，自动继承 `seq_len`）
```yaml
# 假设 upstream_item_seq 已经在类型 schema 中指定了 max_len 为 10
- name: item_history_align
  op_type: SequenceOp
  inputs: [upstream_item_seq]
  outputs: [aligned_item_ids]
  params:
    max_len: 10
    pad_val: 0
  embed:
    vocab_size: 10000
    embed_dim: 32
    pooling: flatten       # 打平为 [10 * 32] = 300 维的拼接向量
    truncation: tail      # 截断左侧，保留最近的 10 个交互行为
```

---

## 3. 类型系统

### 3.1 Fv — 强类型特征值

```rust
pub enum Fv {
    Int(i32),              // type_name: "int"
    Float(f32),            // type_name: "float"
    Str(String),           // type_name: "str"
    IntList(Vec<i32>),     // type_name: "list[int]"
    StrList(Vec<String>),  // type_name: "list[str]"
}
```

所有算子的输入输出均为 `Fv`。`IntList` 用于序列特征（ID 序列），`StrList` 用于标签/字符串列表特征。

### 3.2 类型转换规则

DAG 在初始化阶段根据 SourceDef.dtype 生成默认值及类型映射：
- `Int` → `Fv::Int(parsed)`
- `Float` → `Fv::Float(parsed)`
- `String` / `Enum` → `Fv::Str(string)`
  - *注意*：`Enum` 类型在在线 Rust 引擎中会对输入值进行严格校验与 OOV（Out Of Vocabulary）映射（映射至 oov 字段指定的 token），离线 Python 侧以 raw value 直接向下传递，因此建议将 OOV 归一化收拢于下游 `DictMapper` 等算子以保持完全一致性。
- `List { dtype: inner, max_len: N }` → 生成对应大小为 N 的列表类型特征：
  - `inner == int` → `Fv::IntList(vec![int; N])`
  - `inner == float` → `Fv::FloatList(vec![float; N])`
  - `inner in {string, enum}` → `Fv::StrList(vec![string; N])`

---

## 4. 全部 17 个算子

### 4.1 Bucketing — 数值分桶

将连续数值离散化为整数桶索引。使用二分查找在有序边界上确定输入值所属区间。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `boundaries` | float[] | — | 有序桶边界，N 个边界产生 N+1 个桶 |

**输入**：`Int` 或 `Float`
**输出**：`Int`（桶索引 0..N）

**处理流程**：

1. 取第一个输入值，转为浮点数
2. 在 `boundaries` 数组上二分查找，找到第一个不小于该值的边界索引
3. 索引即为桶号：桶 0 = `(-∞, boundaries[0])`，桶 i = `[boundaries[i-1], boundaries[i])`，桶 N = `[boundaries[N-1], +∞)`

**示例**：

```yaml
- name: hour_bucket
  op_type: Bucketing
  inputs: [ctx_hour]
  outputs: [hour_bucket]
  params:
    boundaries: [6, 12, 18, 22]
  embed: { vocab_size: 5, embed_dim: 4 }
```

```
boundaries = [6, 12, 18, 22]  →  5 个桶

输入 5   → 5 < 6         → 桶 0  (凌晨)
输入 9   → 9 ∈ [6,12)    → 桶 1  (上午)
输入 15  → 15 ∈ [12,18)  → 桶 2  (下午)
输入 20  → 20 ∈ [18,22)  → 桶 3  (晚间)
输入 23  → 23 ≥ 22       → 桶 4  (深夜)
```

---

### 4.2 DictMapper — 字典映射

将字符串或数值 key 映射为整数索引。支持单值输入和列表输入，是类别特征标准化的核心算子。

**索引约定**：mapping 值从 1 起始，`default_idx=0` 保留为「未命中/缺失」。下游 Embedding 可将 index 0 固定映射为零向量，从而区分 padding 占位符与真实特征。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `mapping` | object | — | key → index 映射表（值从 1 开始） |
| `default_idx` | int | 0 | 未命中时的默认索引（保留为 padding token） |

**输入**：`Int` / `Float` / `Str` / `StrList` / `IntList`
**输出**：`Int`（单值输入）或 `IntList`（列表输入）

**处理流程**：

1. 判断输入类型：单值还是列表
2. 单值：将值转为字符串，在 `mapping` 中查找，命中返回对应索引，未命中返回 `default_idx`
3. 列表：对列表中每个元素执行步骤 2，保持列表长度不变
4. 非 string/int/float 类型的值直接返回 `default_idx`

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
```

```
mapping = {phone:1, pad:2, pc:3}, default_idx=0

单值模式：
  输入 "phone"   → 命中 "phone" → 输出 1
  输入 "tablet"  → 未命中      → 输出 0

列表模式：
  输入 ["phone","tablet","pc"]
  逐元素映射               → 输出 [1, 0, 3]
```

---

### 4.3 StringParser — 两级字符串分词

解析 `K1#V1|K2#V2|...` 格式的拼接字符串，提取指定字段后补齐到固定长度。适用于用户标签（`tag#weight|...`）、证券持仓（`code#market#weight|...`）等结构化字符串。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `sep1` | string | `"#"` | 一级分隔符（分隔键值对） |
| `sep2` | string | `"\|"` | 二级分隔符（分隔 Key 和 Value） |
| `key_index` | int | 0 | 提取切分后的第几个字段：0=Key, 1=第一个 Value |
| `pad_len` | int | 0 | 结果列表固定长度，不足则填充 |
| `pad_val` | string | `"unknown"` | 填充值 |

**输入**：`Str`
**输出**：`StrList`（长度 = `pad_len`）

**处理流程**：

1. 将输入转为字符串，若为空串直接返回全填充列表
2. 用 `sep1` 切分 → 得到若干段（如 `["sports#1", "music#2", "gaming#3"]`）
3. 每段用 `sep2` 切分 → 取 `key_index` 位置的字段
4. 若结果数不足 `pad_len`，用 `pad_val` 填充至目标长度；超出则截断

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
    pad_len: 5
    pad_val: "none"
```

```
输入 "sports#0.9|music#0.8|gaming#0.7"

处理步骤：
  sep1("|")  → ["sports#0.9", "music#0.8", "gaming#0.7"]
  sep2("#")  → [[sports,0.9], [music,0.8], [gaming,0.7]]
  key_index=0 → ["sports", "music", "gaming"]
  pad_len=5  → ["sports", "music", "gaming", "none", "none"]
```

---

### 4.4 JsonExtractList — JSON 数组提取

解析 JSON 字符串中的数组，提取指定字段或直接取出元素值。支持对象数组（`[{key:val}, ...]`）和纯值数组（`["a","b"]`）两种格式。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `key` | string | null | 对象数组的提取字段名，null 表示数组元素为纯值 |
| `pad_len` | int | 0 | 结果列表固定长度 |
| `pad_val` | string | `""` | 填充值 |

**输入**：`Str`（JSON 格式字符串）
**输出**：`StrList`（长度 = `pad_len`）

**处理流程**：

1. 将输入解析为 JSON，取顶层数组
2. 若 `key` 非空：遍历数组中每个对象，提取 `key` 字段的值
3. 若 `key` 为空：遍历数组，将每个元素转为字符串
4. 若结果数不足 `pad_len`，用 `pad_val` 填充；超出则截断
5. 解析失败或空字符串 → 全填充值

**示例一：对象数组提取字段**：

```yaml
- name: extract_tags
  op_type: JsonExtractList
  inputs: [json_tags]
  outputs: [tag_list]
  params: { key: "tag", pad_len: 3, pad_val: "none" }
```

```
输入 '[{"score":0.99,"tag":"科技"},{"score":0.5,"tag":"数码"}]'

处理步骤：
  JSON 解析 → [{score:0.99,tag:"科技"}, {score:0.5,tag:"数码"}]
  key="tag" → ["科技", "数码"]
  pad_len=3 → ["科技", "数码", "none"]
```

**示例二：纯值数组**：

```yaml
- name: extract_codes
  op_type: JsonExtractList
  inputs: [stock_list]
  outputs: [codes]
  params: { pad_len: 5, pad_val: "" }
```

```
输入 '["600519,17","000001,33"]'

处理步骤：
  JSON 解析 → ["600519,17", "000001,33"]
  key=null  → ["600519,17", "000001,33"]
  pad_len=5 → ["600519,17", "000001,33", "", "", ""]
```

---

### 4.5 Split — 字符串直接切分

将单个字符串按分隔符切分为字符串列表，支持定长截断和填充。适用于 `"key1|key2|key3"` 这类简单分隔格式。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `sep` | string | `"\|"` | 分隔符 |
| `max_len` | int | `0` | 最大长度，0 表示不限制。超出截断，不足则填充 `pad_val` |
| `pad_val` | string | `""` | 填充值 |

**输入**：`Str`
**输出**：`StrList`（若 max_len > 0 则固定长度，否则变长）

**处理流程**：

1. 将输入转为字符串，若为空串直接返回空列表（或全填充列表）
2. 用 `sep` 切分字符串
3. 若 `max_len > 0`：截断至 `max_len`，不足用 `pad_val` 补齐

**与 StringParser 的区别**：Split 只有一级分隔符，不做字段提取，适合 `"a|b|c"` 这种扁平列表格式。

**示例**：

```yaml
- name: tag_split
  op_type: Split
  inputs: [interest_keywords]
  outputs: [interest_list]
  params:
    sep: "|"
    max_len: 5
    pad_val: "none"
```

```
输入 "新能源|半导体|医药|消费"

处理步骤：
  sep("|")  → ["新能源", "半导体", "医药", "消费"]
  max_len=5 → ["新能源", "半导体", "医药", "消费", "none"]
```

---

### 4.6 FlatSplit — 列表打平分割

将字符串列表中每个元素按分隔符切分后打平为单层列表。用于从序列化的向量序列中提取全部语义 ID。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `sep` | string | `","` | 元素内分隔符 |
| `max_len` | int | `0` | 最大长度，0 表示不限制。超出截断，不足则填充 `pad_val` |
| `pad_val` | string | `""` | 填充值 |

**输入**：`StrList`（每个元素是含分隔符的字符串）
**输出**：`StrList`

**处理流程**：

1. 遍历列表中每个字符串元素
2. 每个元素用 `sep` 切分，将切分结果追加到累加列表
3. 若 `max_len > 0`：截断或填充至目标长度

**与 ListStringParser 的区别**：ListStringParser 从每个元素只提取一个字段（`key_index`），FlatSplit 收集所有切分后的部分并打平。

**示例**：

```yaml
- name: semantic_ids_flat
  op_type: FlatSplit
  inputs: [item_vectors]
  outputs: [all_semantic_ids]
  params:
    sep: ","
    max_len: 40
    pad_val: ""
```

```
输入 ["a_93,b_129,c_140,d_53", "a_51,b_245,c_205,d_157"]

处理步骤：
  "a_93,b_129,c_140,d_53".split(",")  → ["a_93","b_129","c_140","d_53"]
  "a_51,b_245,c_205,d_157".split(",") → ["a_51","b_245","c_205","d_157"]
  打平 → ["a_93","b_129","c_140","d_53","a_51","b_245","c_205","d_157"]
```

典型应用场景：RQ-VAE 语义 ID 序列解析。

```yaml
# historical_click_items: "a_xx,b_xx,c_xx,d_xx#ts|..."
- name: parse_hist
  op_type: StringParser
  inputs: [historical_click_items]
  outputs: [hist_vectors]
  params:
    sep1: "|"
    sep2: "#"
    key_index: 0
    pad_len: 10
    pad_val: ""

- name: flatten_ids
  op_type: FlatSplit
  inputs: [hist_vectors]
  outputs: [all_ids]
  params:
    sep: ","
    max_len: 40       # 10 个向量 × 4 个 ID
    pad_val: ""

- name: map_ids
  op_type: DictMapper
  inputs: [all_ids]
  outputs: [all_ids_mapped]
  params:
    mapping: { a_00: 1, a_01: 2, ..., d_49: 200 }
    default_idx: 0
  embed: { vocab_size: 201, embed_dim: 4 }
```

---

### 4.7 ListStringParser — 列表二次切分

对 `StrList` 中每个元素进行分隔符切分，提取指定索引位置的值。适用于从 `"code,market"` 格式的列表中提取纯代码。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `sep` | string | `","` | 元素内分隔符 |
| `key_index` | int | 0 | 提取切分后的第几个字段 |

**输入**：`StrList`
**输出**：`StrList`

**处理流程**：

1. 遍历列表中每个元素，转为字符串
2. 用 `sep` 切分该元素
3. 取切分结果中 `key_index` 位置的字段
4. 若该位置不存在（数组越界），跳过该元素

**示例**：

```yaml
- name: extract_code
  op_type: ListStringParser
  inputs: [code_with_market]
  outputs: [pure_codes]
  params: { sep: ",", key_index: 0 }
```

```
输入 ["600519,17", "000001,33"]

处理步骤：
  "600519,17".split(",") → ["600519","17"], key_index=0 → "600519"
  "000001,33".split(",") → ["000001","33"], key_index=0 → "000001"

输出 ["600519", "000001"]
```

---

### 4.8 ExpressionOp — 脚本表达式

使用 Rhai 脚本执行数学计算。变量 `v0, v1, ...` 对应输入列表的索引位置。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `script` | string | — | Rhai 表达式，支持 `log`, `abs`, `max`, `min`, `sqrt` |

**输入**：任意 `Fv`（在脚本中自动转为数值）
**输出**：`Float`

**处理流程**：

1. 将每个输入值转为 f64 浮点数
2. 注入变量 `v0, v1, ...` 对应 inputs[0], inputs[1], ...
3. 注入数学函数：`log`, `abs`, `max`, `min`, `sqrt`
4. 执行 Rhai 脚本，返回计算结果

**示例**：

```yaml
# CTR 平滑
- name: calc_smooth_ctr
  op_type: ExpressionOp
  inputs: [click_cnt, expo_cnt]
  outputs: [smooth_ctr]
  params: { script: "v0 / (v1 + 1.0)" }

# 对数变换
- name: log_price
  op_type: ExpressionOp
  inputs: [item_price]
  outputs: [log_price]
  params: { script: "log(v0 + 0.01)" }
```

```
输入 [3, 10], script="v0 / (v1 + 1.0)"
  v0=3, v1=10 → 3 / (10 + 1.0) → 0.272727...

输入 [9999], script="log(v0 + 0.01)"
  v0=9999 → log(9999.01) → 9.21034...
```

---

### 4.9 Log1p — 对数平滑变换

计算单个数值输入的 `ln(1 + x)`，用于曝光、点击、成交额等非负计数或金额特征的平滑压缩。

**参数**：无

**输入**：`Int` 或 `Float`
**输出**：`Float`

**处理流程**：

1. 取第一个输入值，转为浮点数
2. 若输入值 `<= -1`，返回错误，避免 Rust/Python 在 `-inf`/异常行为上不一致
3. 返回 `ln(1 + x)`

**示例**：

```yaml
- name: expo_log
  op_type: Log1p
  inputs: [expo_cnt]
  outputs: [expo_log]
  params: {}
```

```
输入 5999 → ln(6000) → 8.699515...
输入 0    → ln(1)    → 0
```

---

### 4.10 CrossFeature — 特征交叉

对两个列表特征进行组合操作，支持笛卡尔积和内积。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `cross_type` | string | `"cartesian"` | `"cartesian"`（笛卡尔积）或 `"inner_product"`（内积） |

**输入**：`StrList` × 2（cartesian）或 `IntList` × 2（inner_product）
**输出**：`StrList`（cartesian）或 `Float`（inner_product）

**处理流程（笛卡尔积）**：

1. 将两个列表元素转为字符串
2. 双层遍历：`list1[i] + "_" + list2[j]` → 结果列表

**处理流程（内积）**：

1. 将两个列表元素转为 f32
2. 按较短列表的长度，逐元素相乘后求和

**示例**：

```yaml
# 笛卡尔积
- name: tag_cross
  op_type: CrossFeature
  inputs: [user_tags, item_tags]
  outputs: [cross_tags]
  params: { cross_type: cartesian }

# 内积
- name: vector_dot
  op_type: CrossFeature
  inputs: [user_vec, item_vec]
  outputs: [dot_score]
  params: { cross_type: inner_product }
```

```
笛卡尔积：
  输入 ["a","b"] × ["x","y"]
  → ["a_x", "a_y", "b_x", "b_y"]

内积：
  输入 [1.0, 2.0, 3.0] × [4.0, 5.0, 6.0]
  → 1×4 + 2×5 + 3×6 = 32.0
```

---

### 4.11 ListOverlap — 列表交集检测

判断两个字符串列表是否有交集，用于用户-物品交互信号提取。

**参数**：无

**输入**：两个 `StrList`
**输出**：`Int`（1=有交集，0=无交集）

**处理流程**：

1. 将第一个列表转为 HashSet
2. 遍历第二个列表，检查是否有元素存在于 HashSet 中
3. 命中任一元素返回 1，遍历完未命中返回 0

**示例**：

```yaml
- name: tag_overlap
  op_type: ListOverlap
  inputs: [user_tags, item_tags]
  outputs: [overlap_flag]
  embed: { vocab_size: 2, embed_dim: 4 }
```

```
输入 ["sports","music","gaming"] 和 ["music","travel","food"]
  HashSet: {sports, music, gaming}
  遍历第二个列表: "music" 命中 → 输出 1

输入 ["sports","music"] 和 ["travel","food"]
  HashSet: {sports, music}
  遍历第二个列表: 无命中 → 输出 0
```

---

### 4.12 SequenceOp — 序列截断填充

将整数序列裁剪或填充至固定长度，确保下游模型接收统一长度的序列输入。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `max_len` | int | 10 | 目标序列长度 |
| `pad_val` | int | 0 | 填充值（需在 Embedding 词表内，通常保留 0 为 padding token） |

**输入**：`IntList`
**输出**：`IntList`（长度 = `max_len`）

**处理流程**：

1. 若输入长度 > `max_len`：截取前 `max_len` 个元素
2. 若输入长度 < `max_len`：尾部用 `pad_val` 补齐至 `max_len`
3. 若恰好等于 `max_len`：原样返回

**示例**：

```yaml
- name: pad_history
  op_type: SequenceOp
  inputs: [click_seq]
  outputs: [padded_seq]
  params: { max_len: 5, pad_val: 0 }
```

```
max_len=5, pad_val=0

输入 [3, 7, 15]           → 补齐 → [3, 7, 15, 0, 0]
输入 [1, 2, 3, 4, 5]     → 不变 → [1, 2, 3, 4, 5]
输入 [1, 2, 3, 4, 5, 6]  → 截断 → [1, 2, 3, 4, 5]
```

---

### 4.13 StringConcat — 字符串拼接

将多路输入拼接到单个字符串，类型不限。作为特征交叉前的桥接算子，与 `FeatureHash` 配合完成特征哈希。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `separator` | string | `"_"` | 拼接分隔符 |

**输入**：任意数量和类型（内部转为字符串）
**输出**：`Str`

**处理流程**：

1. 将每个输入值转为字符串
2. 用 `separator` 连接所有字符串

**示例**：

```yaml
- name: user_item_concat
  op_type: StringConcat
  inputs: [user_id, item_category]
  outputs: [user_item_str]
  params:
    separator: "_"
```

```
输入 [42, "electronics"]
  42.toString()  → "42"
  join("_")      → "42_electronics"
```

---

### 4.14 FeatureHash — 特征哈希

无状态 DJB2 多种子哈希。将输入拼接后用 k 个独立种子分别哈希，输出一个或一组索引。此外，原生支持对列表输入进行逐元素哈希。Python 与 Rust 实现逐位一致。

**参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `vocab_size` | int | 1000 | 哈希空间大小 [0, vocab_size) |
| `num_hashes` | int | 1 | 独立哈希函数数量（>1 降低碰撞率） |
| `separator` | string | `"\|"` | 输入拼接分隔符 |
| `namespace` | string | `""` | 命名空间前缀，用于区分同一字段的不同 hash 空间 |
| `salt` | string | `""` | 盐值前缀，进一步离散化 hash 结果 |
| `version` | string | `""` | 版本前缀，用于 hash 空间版本迁移 |

**输入**：任意数量和类型
**输出**：`Int`（标量输入且 `num_hashes=1`）或 `IntList`（标量输入且 `num_hashes>1`，或列表输入）

**处理流程**：

*   **标量哈希模式**（输入均为标量）：
    1. 将全部输入值转为字符串，用 `separator` 拼接为单 key。
    2. 对每个种子 s ∈ [0, num_hashes)：计算 `(djb2_seeded(key, s) % vocab_size)`。
    3. `num_hashes=1` 返回单个 `Int`，否则返回 `IntList`。
*   **列表哈希模式**（输入中包含 `StrList` / `IntList` / `FloatList`）：
    1. 自动切换为**逐元素哈希**，固定使用 `seed=0` 分别对列表中每个元素计算 `(djb2_seeded(elem, seed=0) % vocab_size)`。
    2. 忽略 `num_hashes` 参数（始终产生单路哈希列表），输出类型总是 `IntList`。
    3. *注：批量执行时，同一批次内不允许混合标量行与列表行。*

**DJB2 算法**：
`h = 5381; for byte: h = h * 33 + byte`，32 位回绕后取 `0x7FFFFFFF` 低 31 位。

**示例**：

#### 示例 A：标量哈希

```yaml
- name: cross_hash
  op_type: FeatureHash
  inputs: [user_item_str]
  outputs: [hash_idx]
  params:
    vocab_size: 500
    num_hashes: 1
  embed: { vocab_size: 500, embed_dim: 8 }
```

```
单哈希 (num_hashes=1)：
  输入 "42_electronics"
  → djb2_seeded("42_electronics", seed=0) = 1442432207
  → 1442432207 % 500 = 207

多哈希 (num_hashes=4)：
  输入 "invest"
  → [djb2_seeded(..., seed=0)%500, ..., djb2_seeded(..., seed=3)%500]
  → [312, 89, 457, 23]
```

#### 示例 B：列表逐元素哈希

```yaml
- name: tag_list_hash
  op_type: FeatureHash
  inputs: [item_tags]  # 假设 item_tags 类型为 StrList
  outputs: [tag_indices]
  params:
    vocab_size: 1000
    num_hashes: 1
  embed: { vocab_size: 1000, embed_dim: 16 }
```

```
列表逐元素哈希 (使用 seed=0)：
  输入 ["sports", "gaming", "music"]
  → djb2_seeded("sports", seed=0) = 1834126079 → 1834126079 % 1000 = 79
  → djb2_seeded("gaming", seed=0) = 1346630663 → 1346630663 % 1000 = 663
  → djb2_seeded("music", seed=0) = 1480303541 → 1480303541 % 1000 = 541
  → 输出 [79, 663, 541]
```

#### 4.13.1 特殊情况与下游 Embedding 行为警告

使用 `FeatureHash` 时，请务必关注以下两种特殊情况下的下游 Embedding 行为：

##### 情况 A：标量特征多哈希时（`num_hashes > 1`），必须显式指定 `pooling`
当输入为标量特征且 `num_hashes = 4` 时，输出为长度为 4 的 `IntList`（如 `[312, 89, 457, 23]`），这在下游 Embedding 层查表后会得到 `[batch_size, 4, embed_dim]` 的三维张量。
*   **⚠️ 避坑警告**：如果此时 `embed` 中**未显式指定 `pooling`**，默认会采用 `pooling: first` 策略。这将导致下游仅读取第一个哈希索引（`seed=0` 对应的嵌入值），而**其余 3 个哈希值会被无声无息地丢弃**，导致计算冗余且无法起到降低哈希碰撞的效果。
*   **最佳实践**：多重哈希场景下，必须显式在 `embed` 中指定以下池化方式之一：
    *   `pooling: mean` 或 `sum`：将 4 个独立哈希值对应的 Embedding 向量取均值或求和，以获得抗碰撞的稳定表达（输出 Shape `[batch_size, embed_dim]`）。
    *   `pooling: flatten`：将 4 个独立哈希向量横向打平拼接（输出 Shape `[batch_size, 4 * embed_dim]`），完整保留 4 个投影空间的信息。

##### 情况 B：列表特征进行多哈希时，会发生退化
*   如果输入已经是列表类型（如 `StrList` 等标签序列），即使在参数中配置了 `num_hashes > 1`，`FeatureHash` 在执行时也会**强制忽略 `num_hashes` 限制，退化为列表逐元素单哈希模式**（固定只使用 `seed=0` 哈希每个元素）。
*   **设计原因**：避免如果对列表叠加多重哈希，会在下游产生 `[batch_size, list_len, num_hashes, embed_dim]` 的 4 维冗余结构，下游的全连接层及交叉层（如 FM）无法直接消费。

---

### 4.15 融合预处理算子

融合预处理算子把“先解析，再逐元素哈希”的链路合并成一个节点，用来减少 DAG 深度、降低运行时中间值分配，并简化配置。

#### 4.14.1 ParsedFeatureHash — 解析后哈希

`ParsedFeatureHash` 面向“字符串解析 + hash”的高频链路，支持以下模式：

| 模式 | 输入 | 行为 |
|---|---|---|
| `json` | `Str` | 解析 JSON 数组，按 `key` 提取字段后逐元素 hash |
| `structured` | `Str` | 先按 `sep1` 切块，再按 `sep2` 取 `key_index` 字段后逐元素 hash |
| `structured_flat_split` | `Str` | 适用于 `K1#V1|K2#V2` 这类结构化字符串：先按 `sep1` / `sep2` 取字段，再按 `sep` 继续打平后 hash |
| `split` | `Str` | 直接按 `sep` 切分后逐元素 hash |
| `list_split` | `StrList` | 对列表中每个元素按 `sep` 切分后 hash，输出长度与输入列表对齐 |
| `flat_split` | `StrList` | 对列表中每个元素切分后打平，再 hash |

**适用场景**：
- 标签序列、兴趣词序列、结构化字符串字段只需要进入 embedding，不再参与后续 `ListOverlap` 或其它解析算子
- 希望把配置里的 `StringParser + Split + FeatureHash` 合并成一个节点

**注意**：
- 如果下游还需要原始列表用于 `ListOverlap`、交集统计或可读调试，请保留拆分算子，不要强行全链路融合
- `structured_flat_split` 适合“先取字段，再打平子串”的场景，不适合纯列表类型输入

#### 4.14.2 ConcatHash — 拼接后哈希

`ConcatHash` 把多路标量输入先按分隔符拼接，再直接做 `FeatureHash`。它等价于 `StringConcat + FeatureHash`，但少了一层中间字符串节点。

**适用场景**：
- `user_id + item_id`、`source + type` 这类二元或多元交叉特征
- 只需要哈希索引，不需要保留拼接后的中间字符串

**优点**：
- DAG 更短
- 中间字符串不落地
- 配置更紧凑

---

### 4.16 PluginOp — 外部插件

通过 `cdylib` 动态加载外部算子，用于实验性特征快速验证。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `path` | string | — | 动态库路径 |
| `op_name` | string | `"custom_plugin"` | 算子标识 |

**输入**：任意
**输出**：任意

**约束**：外部库必须导出 `process_custom` 符号，签名为 `fn(&[&(dyn Any)]) -> Result<Box<dyn Any>, String>`。不建议在生产环境使用。

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
- `Item` 源 → `"item"`
- 算子输入同时包含 user 和 item → `"cross"`
- 仅 user 输入 → `"user"`；仅 item 输入 → `"item"`

---

## 6. Pipeline 最佳实践

### 6.1 多级级联解析

复杂嵌套数据的典型处理链路：JSON 提取 → 二次切分 → 字典映射。

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
  params:
    mapping: { "600519": 1, "000001": 2 }
    default_idx: 0
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

### 6.3 语义 ID 序列完整解析

RQ-VAE 语义 ID 序列的完整处理链路：两级解析 → 打平 → 映射。

```yaml
# 原始字段: historical_click_items
# 格式: "a_93,b_129,c_140,d_53#1773893763|a_51,b_245,c_205,d_157#1773843030|..."
# 目标: 提取全部语义 ID 并映射为 Embedding

- name: parse_hist
  op_type: StringParser
  inputs: [historical_click_items]
  outputs: [hist_vectors]
  params:
    sep1: "|"
    sep2: "#"
    key_index: 0
    pad_len: 10
    pad_val: ""

- name: flatten_ids
  op_type: FlatSplit
  inputs: [hist_vectors]
  outputs: [all_semantic_ids]
  params:
    sep: ","
    max_len: 40
    pad_val: ""

- name: map_ids
  op_type: DictMapper
  inputs: [all_semantic_ids]
  outputs: [mapped_ids]
  params:
    mapping: { a_00: 1, a_01: 2, ..., d_49: 200 }
    default_idx: 0
  embed: { vocab_size: 201, embed_dim: 4 }
```

### 6.4 特征哈希交叉

将高基数 ID 特征拼接后哈希到固定词表。

```yaml
- name: concat
  op_type: StringConcat
  inputs: [user_id, item_category]
  outputs: [cross_str]
  params: { separator: "_" }

- name: hash_cross
  op_type: FeatureHash
  inputs: [cross_str]
  outputs: [cross_idx]
  params:
    vocab_size: 500
    num_hashes: 1
  embed: { vocab_size: 500, embed_dim: 8 }
```

### 6.5 用户-物品交互信号

```yaml
- name: tag_overlap
  op_type: ListOverlap
  inputs: [user_tags, item_tags]
  outputs: [match_flag]
  embed: { vocab_size: 2, embed_dim: 4 }

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
| StringParser | Str | StrList | sep1, sep2, key_index, pad_len | 结构化字符串解析 |
| JsonExtractList | Str | StrList | key, pad_len | JSON 数组提取 |
| Split | Str | StrList | sep, max_len | 简单分隔字符串切分 |
| FlatSplit | StrList | StrList | sep, max_len | 序列化向量打平 |
| ListStringParser | StrList | StrList | sep, key_index | 列表元素字段提取 |
| ExpressionOp | Numeric | Float | script | 数学变换 |
| Log1p | Int/Float | Float | — | 对数平滑变换 |
| CrossFeature | StrList×2 / IntList×2 | StrList / Float | cross_type | 特征交叉 |
| ListOverlap | StrList×2 | Int | — | 列表交集检测 |
| SequenceOp | IntList | IntList | max_len, pad_val | 序列定长对齐 |
| StringConcat | Any×N | Str | separator | 多值字符串拼接 |
| FeatureHash | Any×N | Int/IntList | vocab_size, num_hashes | 特征哈希 |
| PluginOp | Any | Any | path, op_name | 实验性外部算子 |

---

## 8. 扩展新算子

现在使用 **registry 模式**，新增算子无需修改 DAG 核心代码：

1. **Rust**：
   - 在 `src/feats/ops/<name>.rs` 下新建文件，实现 `CustomOp` trait（`name()` + `process()` + 可选 `process_batch()`）
   - 实现 `pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String>` 工厂函数
   - 在 `src/feats/ops/registry.rs` 的 `OP_REGISTRY` 中插入一行

2. **Python**：
   - 在 `python/src/train/ops/<name>.py` 下新建文件，实现 `process()` + `process_batch()` 方法
   - 添加 `@register_op("<OpType>")` 装饰器 + `@classmethod from_config(params) -> Self`

3. **验证**：
   - Rust：`cargo test`（`all_17_ops_are_registered` 测试自动验证注册完整性）
   - Python：`PYTHONPATH=python/src uv run --directory python pytest tests/ -v`
   - 双端一致性：`PYTHONPATH=python/src uv run --project python python -m scale_rec_demo.verify_all`

4. **文档**：更新本文档的算子列表和速查表

> **注意**：不再需要修改 `dag.rs` 或 `dag.py` 的 `create_op` 方法。
