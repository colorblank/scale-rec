//! 字符串列表切分提取算子：对 StrList 中每个字符串进行 split 并提取指定索引内容。
use super::{CustomOp, Fv};

/// 对 StrList 中每个字符串按分隔符切分并提取指定索引。
pub struct ListStringParser {
    sep: String,
    key_index: usize,
}

impl ListStringParser {
    /// 创建列表字符串解析算子。
    pub fn new(sep: String, key_index: usize) -> Self {
        Self { sep, key_index }
    }
}

impl CustomOp for ListStringParser {
    fn name(&self) -> &str {
        "ListStringParser"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let list = match &inputs[0] {
            Fv::StrList(l) => l,
            _ => return Err("ListStringParser requires StrList input".into()),
        };
        let mut result = Vec::with_capacity(list.len());
        for item in list {
            let parts: Vec<&str> = item.split(&self.sep).collect();
            if self.key_index < parts.len() {
                result.push(parts[self.key_index].to_string());
            } else {
                result.push("".to_string()); // Or preserve pad_val if needed, but simple is better.
            }
        }
        Ok(Fv::StrList(result))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let col = inputs[0];
        let mut results = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            match &col[i] {
                Fv::StrList(list) => {
                    let mut result = Vec::with_capacity(list.len());
                    for item in list {
                        let parts: Vec<&str> = item.split(&self.sep).collect();
                        if self.key_index < parts.len() {
                            result.push(parts[self.key_index].to_string());
                        } else {
                            result.push("".to_string());
                        }
                    }
                    results.push(Fv::StrList(result));
                }
                _ => return Err("ListStringParser requires StrList input".into()),
            }
        }
        Ok(results)
    }
}

pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let sep = params.get("sep").and_then(|v| v.as_str()).unwrap_or(",").to_string();
    let key_index = params.get("key_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    Ok(Box::new(ListStringParser::new(sep, key_index)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_string_parser() {
        let op = ListStringParser::new(",".into(), 0);
        let input = Fv::StrList(vec!["603538,17".into(), "000001,33".into()]);
        let res = op.process(&[input]).unwrap();
        assert_eq!(res, Fv::StrList(vec!["603538".into(), "000001".into()]));

        let op2 = ListStringParser::new(",".into(), 1);
        let input2 = Fv::StrList(vec!["603538,17".into(), "000001,33".into()]);
        let res2 = op2.process(&[input2]).unwrap();
        assert_eq!(res2, Fv::StrList(vec!["17".into(), "33".into()]));
    }
}
