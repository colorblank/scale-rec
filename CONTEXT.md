# Domain Glossary

- **Output contract**: `output_contract.version: 1` configuration that defines typed tower and
  relation nodes, training objectives, evaluation metrics, and public serving outputs.
- **Backbone representation**: Named tensor produced by a model structure and consumed by
  `OutputHead`. Shared-backbone models expose `shared`; MMoE exposes one representation per unique
  `graph.towers[].input`.
- **Internal node**: Tower or relation result available through `ModelExecution.nodes` for
  objectives and metrics.
- **Public output**: Stable serving field selected by `output_contract.outputs` and returned from
  ordinary `forward()`.
- **Legacy task config**: Compatibility path based on `tasks`, `task_config`, `label_col_map`, and
  `metrics`. It remains supported but new example configurations use the native output contract.
