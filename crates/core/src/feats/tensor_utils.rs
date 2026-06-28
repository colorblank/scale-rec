//! 特征列 → 数值向量的转换（共享训练/推理的 pooling/padding 逻辑）。

use crate::feats::config::{FeatureSpec, PoolingStrategy, TruncationSide};
use crate::feats::ops::Fv;

/// 特征列转换后的形状。Scalar = [n], Sequence = [n, seq_len]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureColumn {
    /// 标量特征（FIRST pooling），形状 [n]。
    Scalar(Vec<i32>),
    /// 序列特征（MEAN/SUM/MAX/FLATTEN），形状 [n, seq_len]。
    Sequence(Vec<Vec<i32>>),
}

/// 将 Fv 列转换为数值向量，pooling/padding/truncation 逻辑与推理侧一致。
pub fn feature_column_to_vec(
    spec: &FeatureSpec,
    col: &[Fv],
    n: usize,
) -> Result<FeatureColumn, String> {
    let use_sequence = spec.pooling != PoolingStrategy::First
        && col.iter().any(|v| matches!(v, Fv::IntList(_)));

    if use_sequence {
        let seq_len = spec
            .seq_len
            .ok_or_else(|| {
                format!(
                    "feature '{}' sequence pooling requires seq_len > 0",
                    spec.name
                )
            })?
            .max(1);
        let mut result = Vec::with_capacity(n);
        for val in col.iter().take(n) {
            match val {
                Fv::IntList(values) => {
                    let start_offset =
                        if spec.truncation == TruncationSide::Tail && values.len() > seq_len {
                            values.len() - seq_len
                        } else {
                            0
                        };
                    let mut row = Vec::with_capacity(seq_len);
                    for idx in 0..seq_len {
                        row.push(values.get(start_offset + idx).copied().unwrap_or(0).max(0));
                    }
                    result.push(row);
                }
                Fv::Int(i) => {
                    let mut row = vec![(*i).max(0)];
                    row.extend(std::iter::repeat(0).take(seq_len - 1));
                    result.push(row);
                }
                _ => {
                    result.push(vec![0; seq_len]);
                }
            }
        }
        Ok(FeatureColumn::Sequence(result))
    } else {
        let indices: Vec<i32> = col
            .iter()
            .take(n)
            .map(|val| match val {
                Fv::Int(i) => (*i).max(0),
                Fv::IntList(values) => values.first().copied().unwrap_or(0).max(0),
                _ => 0,
            })
            .collect();
        Ok(FeatureColumn::Scalar(indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq_spec(seq_len: usize, truncation: TruncationSide) -> FeatureSpec {
        FeatureSpec {
            name: "test".into(),
            vocab_size: 100,
            embed_dim: 8,
            pooling: PoolingStrategy::Mean,
            seq_len: Some(seq_len),
            truncation,
        }
    }

    fn first_spec() -> FeatureSpec {
        FeatureSpec {
            name: "test".into(),
            vocab_size: 100,
            embed_dim: 8,
            pooling: PoolingStrategy::First,
            seq_len: None,
            truncation: TruncationSide::Head,
        }
    }

    #[test]
    fn scalar_first_pooling() {
        let col = vec![Fv::Int(42), Fv::Int(7), Fv::Int(0)];
        let result = feature_column_to_vec(&first_spec(), &col, 3).unwrap();
        match result {
            FeatureColumn::Scalar(v) => assert_eq!(v, vec![42, 7, 0]),
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn scalar_takes_first_from_list() {
        let col = vec![Fv::IntList(vec![1, 2, 3]), Fv::Int(99)];
        let result = feature_column_to_vec(&first_spec(), &col, 2).unwrap();
        match result {
            FeatureColumn::Scalar(v) => assert_eq!(v, vec![1, 99]),
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn seq_head_truncation() {
        let col = vec![Fv::IntList(vec![1, 2, 3, 4, 5])];
        let result = feature_column_to_vec(&seq_spec(3, TruncationSide::Head), &col, 1).unwrap();
        match result {
            FeatureColumn::Sequence(v) => assert_eq!(v, vec![vec![1, 2, 3]]),
            _ => panic!("expected Sequence"),
        }
    }

    #[test]
    fn seq_tail_truncation() {
        let col = vec![Fv::IntList(vec![1, 2, 3, 4, 5])];
        let result = feature_column_to_vec(&seq_spec(3, TruncationSide::Tail), &col, 1).unwrap();
        match result {
            FeatureColumn::Sequence(v) => assert_eq!(v, vec![vec![3, 4, 5]]),
            _ => panic!("expected Sequence"),
        }
    }

    #[test]
    fn seq_padding() {
        let col = vec![Fv::IntList(vec![1, 2])];
        let result = feature_column_to_vec(&seq_spec(5, TruncationSide::Head), &col, 1).unwrap();
        match result {
            FeatureColumn::Sequence(v) => assert_eq!(v, vec![vec![1, 2, 0, 0, 0]]),
            _ => panic!("expected Sequence"),
        }
    }

    #[test]
    fn non_list_column_uses_scalar_tensor_shape() {
        let col = vec![Fv::Int(7)];
        let result = feature_column_to_vec(&seq_spec(3, TruncationSide::Head), &col, 1).unwrap();
        match result {
            FeatureColumn::Scalar(v) => assert_eq!(v, vec![7]),
            _ => panic!("expected Scalar"),
        }
    }

    #[test]
    fn sequence_pooling_requires_seq_len_for_list_columns() {
        let mut spec = seq_spec(3, TruncationSide::Head);
        spec.seq_len = None;
        let col = vec![Fv::IntList(vec![1, 2, 3])];
        let err = feature_column_to_vec(&spec, &col, 1).unwrap_err();
        assert!(err.contains("requires seq_len"));
    }

    #[test]
    fn empty_column_returns_empty_result() {
        let col: Vec<Fv> = vec![];
        let result = feature_column_to_vec(&first_spec(), &col, 0).unwrap();
        match result {
            FeatureColumn::Scalar(v) => assert!(v.is_empty()),
            _ => panic!("expected Scalar"),
        }
    }
}
