//! 列表重叠检测算子：判断两个列表是否存在共同元素。
use super::{CustomOp, Fv};

/// 判断两个字符串列表是否存在共同元素。
pub struct ListOverlap;

impl ListOverlap {
    /// 创建列表重叠检测算子。
    pub fn new() -> Self {
        Self
    }
}

impl CustomOp for ListOverlap {
    fn name(&self) -> &str {
        "ListOverlap"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        Ok(Fv::Int(if has_overlap(&inputs[0], &inputs[1]) {
            1
        } else {
            0
        }))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let ca = inputs[0];
        let cb = inputs[1];
        let mut results = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let v = if has_overlap(&ca[i], &cb[i]) { 1 } else { 0 };
            results.push(Fv::Int(v));
        }
        Ok(results)
    }
}

/// 从 YAML params 创建 ListOverlap 算子。
pub fn create(_params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    Ok(Box::new(ListOverlap::new()))
}

fn has_overlap(a: &Fv, b: &Fv) -> bool {
    let (Fv::StrList(a), Fv::StrList(b)) = (a, b) else {
        return false;
    };
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    small
        .iter()
        .filter(|item| !item.is_empty())
        .any(|item| large.iter().any(|other| other == item))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_overlap_ignores_empty_values() {
        let op = ListOverlap::new();
        let result = op
            .process(&[
                Fv::StrList(vec!["".into(), "a".into()]),
                Fv::StrList(vec!["".into(), "b".into()]),
            ])
            .unwrap();
        assert_eq!(result, Fv::Int(0));
    }

    #[test]
    fn list_overlap_detects_shared_value() {
        let op = ListOverlap::new();
        let result = op
            .process(&[
                Fv::StrList(vec!["a".into(), "b".into()]),
                Fv::StrList(vec!["c".into(), "b".into()]),
            ])
            .unwrap();
        assert_eq!(result, Fv::Int(1));
    }

    #[test]
    fn list_overlap_one_side_empty() {
        let op = ListOverlap::new();
        assert_eq!(
            op.process(&[Fv::StrList(vec![]), Fv::StrList(vec!["a".into()]),])
                .unwrap(),
            Fv::Int(0)
        );
        assert_eq!(
            op.process(&[Fv::StrList(vec!["a".into()]), Fv::StrList(vec![]),])
                .unwrap(),
            Fv::Int(0)
        );
    }

    #[test]
    fn list_overlap_both_empty() {
        let op = ListOverlap::new();
        assert_eq!(
            op.process(&[Fv::StrList(vec![]), Fv::StrList(vec![]),])
                .unwrap(),
            Fv::Int(0)
        );
    }

    #[test]
    fn list_overlap_duplicate_items() {
        let op = ListOverlap::new();
        assert_eq!(
            op.process(&[
                Fv::StrList(vec!["a".into(), "a".into()]),
                Fv::StrList(vec!["a".into()]),
            ])
            .unwrap(),
            Fv::Int(1)
        );
    }

    #[test]
    fn list_overlap_batch_matches_single_row() {
        let op = ListOverlap::new();
        let left = vec![
            Fv::StrList(vec!["a".into(), "b".into()]),
            Fv::StrList(vec!["x".into()]),
        ];
        let right = vec![
            Fv::StrList(vec!["c".into(), "b".into()]),
            Fv::StrList(vec!["y".into()]),
        ];
        let result = op.process_batch(&[&left, &right], 2).unwrap();
        assert_eq!(result, vec![Fv::Int(1), Fv::Int(0)]);
    }
}
