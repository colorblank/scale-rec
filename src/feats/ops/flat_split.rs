//! 列表打平分割算子：StrList 输入 → 每个元素按分隔符切分后打平为单层列表。

use crate::feats::ops::{CustomOp, Fv};

/// 将字符串列表中每个元素按分隔符切分，打平为定长字符串列表。
pub struct FlatSplit {
    sep: String,
    max_len: usize,
    pad_val: String,
}

impl FlatSplit {
    pub fn new(sep: String, max_len: usize, pad_val: String) -> Self {
        Self {
            sep,
            max_len,
            pad_val,
        }
    }

    fn normalize(&self, mut parts: Vec<String>) -> Vec<String> {
        if self.max_len == 0 {
            return parts;
        }
        parts.truncate(self.max_len);
        while parts.len() < self.max_len {
            parts.push(self.pad_val.clone());
        }
        parts
    }
}

impl CustomOp for FlatSplit {
    fn name(&self) -> &str {
        "FlatSplit"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let str_list = match &inputs[0] {
            Fv::StrList(list) => list,
            _ => return Ok(Fv::StrList(self.normalize(Vec::new()))),
        };
        let mut all: Vec<String> = Vec::new();
        for s in str_list {
            if !s.is_empty() {
                for part in s.split(&self.sep) {
                    all.push(part.to_string());
                }
            }
        }
        Ok(Fv::StrList(self.normalize(all)))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        if n_rows == 0 {
            return Ok(vec![]);
        }
        let col = inputs.first().map(|c| *c).unwrap_or(&[]);
        let mut results: Vec<Fv> = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let str_list = if row < col.len() {
                match &col[row] {
                    Fv::StrList(list) => list.clone(),
                    _ => Vec::new(),
                }
            } else {
                Vec::new()
            };
            let mut all: Vec<String> = Vec::new();
            for s in &str_list {
                if !s.is_empty() {
                    for part in s.split(&self.sep) {
                        all.push(part.to_string());
                    }
                }
            }
            results.push(Fv::StrList(self.normalize(all)));
        }
        Ok(results)
    }
}

// ── 测试 ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let op = FlatSplit::new(",".into(), 0, "".into());
        let result = op
            .process(&[Fv::StrList(vec![
                "a_93,b_129,c_140,d_53".into(),
                "a_51,b_245,c_205,d_157".into(),
            ])])
            .unwrap();
        assert_eq!(
            result,
            Fv::StrList(vec![
                "a_93".into(),
                "b_129".into(),
                "c_140".into(),
                "d_53".into(),
                "a_51".into(),
                "b_245".into(),
                "c_205".into(),
                "d_157".into(),
            ])
        );
    }

    #[test]
    fn test_truncate() {
        let op = FlatSplit::new(",".into(), 4, "".into());
        let result = op.process(&[Fv::StrList(vec!["a,b,c".into()])]).unwrap();
        assert_eq!(
            result,
            Fv::StrList(vec!["a".into(), "b".into(), "c".into(), "".into()])
        );
    }

    #[test]
    fn test_pad() {
        let op = FlatSplit::new("|".into(), 4, "none".into());
        let result = op.process(&[Fv::StrList(vec!["x|y".into()])]).unwrap();
        assert_eq!(
            result,
            Fv::StrList(vec!["x".into(), "y".into(), "none".into(), "none".into()])
        );
    }

    #[test]
    fn test_empty_input() {
        let op = FlatSplit::new(",".into(), 0, "".into());
        let result = op.process(&[Fv::StrList(vec![])]).unwrap();
        assert_eq!(result, Fv::StrList(Vec::<String>::new()));
    }

    #[test]
    fn test_batch() {
        let op = FlatSplit::new(",".into(), 0, "".into());
        let col = vec![
            Fv::StrList(vec!["a,b".into()]),
            Fv::StrList(vec!["c,d".into(), "e".into()]),
        ];
        let results = op.process_batch(&[&col], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], Fv::StrList(vec!["a".into(), "b".into()]));
        assert_eq!(
            results[1],
            Fv::StrList(vec!["c".into(), "d".into(), "e".into()])
        );
    }
}
