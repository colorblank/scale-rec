//! 字符串解析算子：结构化字符串分词与填充。
use std::any::Any;

/// 字符串解析算子。
///
/// 两级分隔符解析结构化字符串：`sep1` 分割条目，`sep2` 分割键值对，提取 `key_index` 位置的字段，填充至 `pad_len`。
pub struct StringParser {
    sep1: String,
    sep2: String,
    key_index: usize,
    pad_len: usize,
    pad_val: String,
}

impl StringParser {
    /// 构造解析算子。
    pub fn new(
        sep1: String,
        sep2: String,
        key_index: usize,
        pad_len: usize,
        pad_val: String,
    ) -> Self {
        Self {
            sep1,
            sep2,
            key_index,
            pad_len,
            pad_val,
        }
    }
}

impl super::CustomOp for StringParser {
    /// 执行解析: `String` → `Vec<String>`(长度 = pad_len)。
    fn name(&self) -> &str {
        "StringParser"
    }
    fn process(
        &self,
        inputs: &[&(dyn Any + Send + Sync)],
    ) -> Result<Box<dyn Any + Send + Sync>, String> {
        let s = inputs[0]
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .unwrap_or("");
        let mut result: Vec<String> = s
            .split(&self.sep1)
            .filter_map(|item| {
                let parts: Vec<&str> = item.split(&self.sep2).collect();
                if self.key_index < parts.len() {
                    Some(parts[self.key_index].to_string())
                } else {
                    None
                }
            })
            .collect();
        while result.len() < self.pad_len {
            result.push(self.pad_val.clone());
        }
        result.truncate(self.pad_len);
        Ok(Box::new(result))
    }
}
