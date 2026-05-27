# 设计改进建议

本文只记录当前代码里仍然能确认的缺陷、待优化点和风险点。已落地的内容不再重复记录。排序按影响优先级和修复收益综合考虑。

## 已修复

本轮已经落地的修复如下：

- `flatten` 序列特征运行时长度现在强制使用配置里的 `seq_len`
- `FeatureHash.process_batch()` 不再靠前几行猜 batch 形态，混合 scalar/list 直接报错
- item 文件读取现在会识别 `NULL` / `\\N` 等缺失标记
- `int` 默认值和训练侧原始值解析改成严格模式，不再静默截断小数
- Rust 批量 DAG 对缺失中间列直接报错，不再回退到 `Int(0)`

## P1：会影响线上稳定性和演进成本的问题

### 1. 模型加载仍然绑定单一全局 feature config

`src/server/registry.rs` 仍然只持有一个 `feature_config_path`，所有模型都用这份配置构建 DAG 和 tokenizer。manifest 虽然带了 per-model 的 feature config 元数据，但运行时并没有真正按模型选择。

这会带来两个问题：
- 混合 schema 的模型无法共存
- manifest 中的 feature config 信息在运行时只起到“对齐校验”作用，不能参与真正的模型恢复

如果系统明确只支持单一共享 schema，这没问题；如果要支持按模型独立演进特征配置，就需要把 feature config 从“全局常量”改成“模型版本的一部分”。

建议：
- 明确部署模式是“全局共享 schema”还是“模型级 schema”
- 如果要支持后者，registry 要从 manifest 反解 feature config 路径
- `/models`、`/health` 返回的模型信息要带 schema 版本和状态

### 2. 请求和数据预处理的错误仍然偏“字符串化”

`src/server/routes.rs` 的 `map_predict_error()` 还是靠字符串包含关系来猜错误类型。这个做法短期能用，但长期会让错误分类变得脆弱，尤其是当底层错误信息变长或改词时。

建议：
- 训练和推理路径统一 typed error
- `BAD_REQUEST` / `FEATURE_ERROR` / `MODEL_ERROR` / `REGISTRY_ERROR` / `INTERNAL_ERROR` 分层明确
- 错误结构带上 `request_id`、`model_id` 和可选 `details`

## P2：模型工程和数据工程的性能热点

### 1. Python 读表路径还比较偏 row-wise

`python/src/train/app/data.py` 里大量使用 `iterrows()`、逐行 dict 拼装和逐列缺失填充。对小样本 demo 没问题，但在真实推荐训练数据上会非常吃 CPU。

主要热点：
- `build_item_index()` 的逐行 `iterrows()`
- `stream_file_batches()` 的 chunk 后再转 dict 再清洗
- discover 训练已经不再保留单独的 `stream_join()` 分支，但单文件流式读取仍然是 row-wise

建议：
- 优先用列式或向量化方式做 join / 缺失填充
- 大数据场景把数据准备迁到 Polars 或更强的批处理路径
- 明确 demo 路径和生产路径的性能预期，不要共用同一条慢路径

### 2. Rust `FeatureHash` 的缓存是全局锁热点

`src/feats/ops/feature_hash.rs:13-189` 里用了 `RwLock<HashMap<String, Fv>>` 做缓存。这个实现正确性没问题，但在高并发下会形成锁热点，而且缓存没有任何上限控制。

如果特征基数高、请求量大，缓存命中率未必足以抵掉锁竞争和内存增长。

建议：
- 先量化命中率和锁竞争成本
- 低收益场景考虑关闭缓存
- 真要保留缓存，考虑 bounded cache 或分片 cache

### 3. 推理路径的可观测性还可以再细

当前虽然已经拆了 `parse_us / dag_us / tensor_us / forward_us / response_us`，但对排障来说还不够细，尤其是当 batch 里混有 list 特征、broadcast 路径、skip-op 路径时。

建议：
- 把 batch size、item count、缺失值命中率、序列截断次数一起记录
- 对 broadcast 和 pointwise 分开统计
- 把 `feature quality` 指标和线上请求统计一起落到 manifest 或日志里

## P3：工程收敛建议

### 1. 把“类型信息”真正下沉到执行层

现在 schema 层已经能推导很多信息，但执行层还有不少 `Any`、字符串判断和默认值兜底。短期能跑，长期会让 Python/Rust 两端越来越难维护。

建议：
- 让 `FeatureSchema` 成为训练和推理的唯一输入契约
- 所有默认值、pooling、seq_len、label role 都从 schema 读取
- 新算子、新模型、新任务塔都先补 schema 再补运行时

### 2. 训练数据、模型产物和配置文件的边界继续收紧

训练侧已经有 `run_dir / checkpoints / published_weights / manifest` 这套结构，但代码里仍然会出现“临时路径”“复制配置”“发布权重”三种语义混在一起的情况。

建议：
- 将 run 级产物和发布级产物分开命名
- manifest 只写稳定语义，不写临时文件
- 文档里明确哪些文件是可删的，哪些是源文件

## 建议执行顺序

1. 修 `flatten` 的固定长度语义，让 Python/Rust 运行时都严格尊重 `seq_len`。
2. 修 `FeatureHash` 的 batch 形态识别，不要再靠前几个样本猜列类型。
3. 统一 item 文件和 user 文件的缺失值语义。
4. 把 `int` 默认值和配置解析改成严格模式，禁止静默截断。
5. 让 Rust 批量 DAG 对缺失中间列直接报错，不要回退到 `Int(0)`。
6. 决定 registry 是否支持模型级 feature config，再收敛 manifest 语义。
