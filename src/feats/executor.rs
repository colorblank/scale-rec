//! DAG 执行器：ExecutionPlan + DagExecutor，统一 plan-based 执行路径。
use super::ops::{CustomOp, Fv};
use crate::feats::config::SourceDef;
use std::collections::{HashMap, HashSet};

/// 预编译执行步骤：算子索引 + 输入/输出列索引。
pub struct ExecStep {
    /// `ExecutionPlan.ops` 中的算子索引。
    pub op_idx: usize,
    /// 输入列在运行时列数组中的索引。
    pub input_cols: Vec<usize>,
    /// 输出列在运行时列数组中的索引。
    pub output_cols: Vec<usize>,
}

/// 预编译执行计划：运算符 + 整数索引列，运行时零 HashMap 查找。
pub struct ExecutionPlan {
    /// 按拓扑顺序排列的执行步骤。
    pub steps: Vec<ExecStep>,
    /// 已构造好的算子实例。
    pub ops: Vec<Box<dyn CustomOp>>,
    source_cols: Vec<usize>,
    source_names: Vec<String>,
    col_names: Vec<Option<String>>,
    source_defaults: Vec<Fv>,
    col_count: usize,
    embed_ids: Vec<usize>,
}

impl ExecutionPlan {
    /// 创建预编译执行计划。
    pub fn new(
        steps: Vec<ExecStep>,
        ops: Vec<Box<dyn CustomOp>>,
        source_cols: Vec<usize>,
        source_names: Vec<String>,
        col_names: Vec<Option<String>>,
        source_defaults: Vec<Fv>,
        col_count: usize,
        embed_ids: Vec<usize>,
    ) -> Self {
        Self {
            steps,
            ops,
            source_cols,
            source_names,
            col_names,
            source_defaults,
            col_count,
            embed_ids,
        }
    }

    /// 按列式输入执行预编译计划，并返回完整列上下文。
    pub fn execute_plan(
        &self,
        columns: &HashMap<String, Vec<Fv>>,
        skip_op_idx: &HashSet<usize>,
        precomputed: &HashMap<usize, Fv>,
    ) -> Result<Vec<Vec<Fv>>, String> {
        let n_rows = columns.values().next().map(|v| v.len()).unwrap_or(0);
        if n_rows == 0 {
            return Ok(Vec::new());
        }

        let mut context: Vec<Vec<Fv>> = vec![Vec::with_capacity(n_rows); self.col_count];

        for i in 0..self.source_cols.len() {
            let name = &self.source_names[i];
            let cid = self.source_cols[i];
            let source_default = self.source_defaults.get(i).cloned().unwrap_or(Fv::Int(0));
            if let Some(col) = columns.get(name) {
                if col.len() == n_rows {
                    context[cid] = col.clone();
                } else {
                    let mut fixed = vec![source_default; n_rows];
                    for (row, val) in col.iter().take(n_rows).enumerate() {
                        fixed[row] = val.clone();
                    }
                    context[cid] = fixed;
                }
            } else {
                context[cid] = vec![source_default; n_rows];
            }
        }

        for (&col_id, val) in precomputed.iter() {
            if col_id < context.len() {
                context[col_id] = vec![val.clone(); n_rows];
            }
        }

        for step in &self.steps {
            if skip_op_idx.contains(&step.op_idx) {
                continue;
            }
            let op = &self.ops[step.op_idx];
            let input_slices: Vec<&[Fv]> = step
                .input_cols
                .iter()
                .map(|&cid| {
                    context.get(cid).map(|c| c.as_slice()).ok_or_else(|| {
                        let name = self
                            .col_names
                            .get(cid)
                            .and_then(|n| n.as_deref())
                            .unwrap_or("<unknown>");
                        format!("missing required column '{}' (id={})", name, cid)
                    })
                })
                .collect::<Result<_, String>>()?;
            let result_vec = op
                .process_batch(&input_slices, n_rows)
                .map_err(|e| format!("step {}: {}", step.op_idx, e))?;
            for &cid in &step.output_cols {
                context[cid] = result_vec.clone();
            }
        }
        Ok(context)
    }

    /// 返回可嵌入特征对应的列 id 列表。
    pub fn embed_ids(&self) -> &[usize] {
        &self.embed_ids
    }

    /// 返回 source 名称到列 id 的映射。
    pub fn source_col_map(&self) -> HashMap<String, usize> {
        self.source_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect()
    }
}

/// DAG 执行器：包装 ExecutionPlan，提供统一的执行接口。
pub struct DagExecutor {
    plan: ExecutionPlan,
    sources: HashMap<String, SourceDef>,
    execution_order: Vec<String>,
    data_sources: Vec<crate::feats::config::DataSourceDef>,
}

impl DagExecutor {
    /// 创建 DAG 执行器。
    pub fn new(
        plan: ExecutionPlan,
        sources: HashMap<String, SourceDef>,
        execution_order: Vec<String>,
        data_sources: Vec<crate::feats::config::DataSourceDef>,
    ) -> Self {
        Self {
            plan,
            sources,
            execution_order,
            data_sources,
        }
    }

    /// 委托内部 `ExecutionPlan` 执行列式输入。
    pub fn execute_plan(
        &self,
        columns: &HashMap<String, Vec<Fv>>,
        skip_op_idx: &HashSet<usize>,
        precomputed: &HashMap<usize, Fv>,
    ) -> Result<Vec<Vec<Fv>>, String> {
        self.plan.execute_plan(columns, skip_op_idx, precomputed)
    }

    /// 返回内部预编译执行计划。
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// 返回 DAG 的 source 定义。
    pub fn source_defs(&self) -> &HashMap<String, SourceDef> {
        &self.sources
    }

    /// 返回算子拓扑执行顺序。
    pub fn execution_order(&self) -> &[String] {
        &self.execution_order
    }

    /// 返回 DAG 绑定的数据源定义。
    pub fn data_sources(&self) -> &[crate::feats::config::DataSourceDef] {
        &self.data_sources
    }
}
