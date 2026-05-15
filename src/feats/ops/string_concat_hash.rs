//! 字符串拼接哈希算子：两个字符串拼接 → hash 映射到固定词表。
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::RwLock;
use super::{CustomOp, Fv};

fn djb2(s: &str) -> u32 {
    let mut h: u32 = 5381;
    for b in s.bytes() { h = h.wrapping_mul(33).wrapping_add(b as u32); }
    h & 0x7FFFFFFF
}

pub struct StringConcatHash {
    #[allow(dead_code)] vocab_size: usize,
    oov_reserve: usize, main_size: usize,
    hash_map_path: String, mode: String, separator: String,
    mapping: RwLock<HashMap<String, i32>>,
    next_idx: RwLock<i32>,
}

impl StringConcatHash {
    pub fn new(vocab_size: usize, oov_reserve: usize, hash_map_path: String, mode: String, separator: String) -> Self {
        let main_size = vocab_size - oov_reserve;
        let mut mapping = HashMap::new(); let mut next_idx = 0i32;
        if mode == "inference" && !hash_map_path.is_empty() {
            if let Ok(contents) = fs::read_to_string(&hash_map_path) {
                if let Ok(raw) = serde_yaml::from_str::<serde_yaml::Value>(&contents) {
                    if let Some(map) = raw.get("mapping").and_then(|v| v.as_mapping()) {
                        for (k, v) in map {
                            if let (Some(key), Some(val)) = (k.as_str(), v.as_i64()) {
                                mapping.insert(key.to_string(), val as i32);
                            }
                        }
                    }
                }
            }
            next_idx = mapping.len() as i32;
        }
        Self { vocab_size, oov_reserve, main_size, hash_map_path, mode, separator,
               mapping: RwLock::new(mapping), next_idx: RwLock::new(next_idx) }
    }

    fn save_mapping(&self) {
        if self.hash_map_path.is_empty() { return; }
        if let Some(parent) = Path::new(&self.hash_map_path).parent() { let _ = fs::create_dir_all(parent); }
        let mapping = self.mapping.read().unwrap();
        let mut m = serde_yaml::Mapping::new(); let mut inner = serde_yaml::Mapping::new();
        for (k, v) in mapping.iter() { inner.insert(serde_yaml::Value::String(k.clone()), serde_yaml::Value::Number((*v).into())); }
        m.insert(serde_yaml::Value::String("mapping".into()), serde_yaml::Value::Mapping(inner));
        let _ = fs::write(&self.hash_map_path, serde_yaml::to_string(&m).unwrap_or_default());
    }
}

impl CustomOp for StringConcatHash {
    fn name(&self) -> &str { "StringConcatHash" }
    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let s1 = match &inputs[0] { Fv::Str(s) => s.as_str(), Fv::Int(i) => { let s = i.to_string(); return self.process_str(&s, &inputs[1]); }, _ => "" };
        self.process_str(s1, &inputs[1])
    }
}

impl StringConcatHash {
    fn process_str(&self, s1: &str, v2: &Fv) -> Result<Fv, String> {
        let s2 = match v2 { Fv::Str(s) => s.as_str(), Fv::Int(i) => &i.to_string(), _ => "" };
        let key = format!("{}{}{}", s1, self.separator, s2);
        if self.mode == "inference" {
            let mapping = self.mapping.read().unwrap();
            if let Some(&idx) = mapping.get(&key) { return Ok(Fv::Int(idx)); }
            return Ok(Fv::Int((djb2(&key) as usize % self.oov_reserve + self.main_size) as i32));
        }
        { let mapping = self.mapping.read().unwrap(); if let Some(&idx) = mapping.get(&key) { return Ok(Fv::Int(idx)); } }
        let mut mapping = self.mapping.write().unwrap();
        let mut next = self.next_idx.write().unwrap();
        let idx = if (*next as usize) < self.main_size { let i = *next; *next += 1; i }
                  else { (djb2(&key) as usize % self.oov_reserve + self.main_size) as i32 };
        mapping.insert(key, idx);
        drop(mapping); drop(next); self.save_mapping();
        Ok(Fv::Int(idx))
    }
}
