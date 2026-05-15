//! 列表重叠检测算子：判断两个列表是否存在共同元素。
use std::collections::HashSet;
use super::{CustomOp, Fv};

pub struct ListOverlap;

impl ListOverlap { pub fn new() -> Self { Self } }

impl CustomOp for ListOverlap {
    fn name(&self) -> &str { "ListOverlap" }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let a = strs(&inputs[0]); let b = strs(&inputs[1]);
        if a.is_empty() || b.is_empty() { return Ok(Fv::Int(0)); }
        Ok(Fv::Int(if a.intersection(&b).next().is_some() { 1 } else { 0 }))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let ca = inputs[0]; let cb = inputs[1];
        let mut results = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let a = strs(&ca[i]); let b = strs(&cb[i]);
            let v = if a.is_empty() || b.is_empty() { 0 }
                    else if a.intersection(&b).next().is_some() { 1 } else { 0 };
            results.push(Fv::Int(v));
        }
        Ok(results)
    }
}

fn strs(v: &Fv) -> HashSet<&str> {
    match v { Fv::StrList(l) => l.iter().filter(|s| !s.is_empty()).map(|s| s.as_str()).collect(), _ => HashSet::new() }
}
