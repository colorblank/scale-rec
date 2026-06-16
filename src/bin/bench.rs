//! 压测工具：支持 open-loop 固定到达率和 legacy 闭环压测。
use clap::Parser;
use csv::StringRecord;
use rand::Rng;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::{self, MissedTickBehavior};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use scale_rec::feats::config::{Role, SourceKind};

#[derive(Parser, Clone)]
struct Args {
    #[arg(long, default_value = "http://localhost:8080")]
    target: String,
    #[arg(long, default_value = "lr")]
    model: String,
    #[arg(long, default_value = "pointwise")]
    mode: String,
    #[arg(long, default_value = "100")]
    concurrency: usize,
    #[arg(long, default_value = "64")]
    batch_size: usize,
    #[arg(long, default_value = "10")]
    duration_secs: u64,
    #[arg(long, default_value = "0")]
    target_qps: usize,
    #[arg(long)]
    input_file: Option<PathBuf>,
    #[arg(long)]
    feature_config: Option<PathBuf>,
    #[arg(long)]
    no_header: bool,
    #[arg(long, default_value = "\t")]
    separator: char,
}

#[derive(Clone)]
enum InputData {
    Pointwise(Arc<Vec<serde_json::Value>>),
    Broadcast(Arc<Vec<BroadcastSample>>),
}

#[derive(Clone)]
struct BroadcastSample {
    user: serde_json::Value,
    item: serde_json::Value,
}

#[derive(Clone)]
struct RequestContext {
    target: String,
    model: String,
    mode: String,
    batch_size: usize,
    input_data: Option<InputData>,
    next_row: Arc<AtomicU64>,
}

impl RequestContext {
    fn url(&self) -> String {
        if self.mode == "broadcast" {
            format!("{}/predict/broadcast", self.target)
        } else {
            format!("{}/predict", self.target)
        }
    }

    fn body(&self) -> serde_json::Value {
        let mut rng = rand::thread_rng();
        if let Some(input_data) = &self.input_data {
            return match input_data {
                InputData::Pointwise(rows) => {
                    let idx = (self.next_row.fetch_add(1, Ordering::Relaxed) as usize) % rows.len();
                    let features: Vec<serde_json::Value> = (0..self.batch_size)
                        .map(|offset| rows[(idx + offset) % rows.len()].clone())
                        .collect();
                    serde_json::json!({"model": self.model, "features": features})
                }
                InputData::Broadcast(rows) => {
                    let idx = (self.next_row.fetch_add(1, Ordering::Relaxed) as usize) % rows.len();
                    let user = rows[idx].user.clone();
                    let items: Vec<serde_json::Value> = (0..self.batch_size)
                        .map(|offset| rows[(idx + offset) % rows.len()].item.clone())
                        .collect();
                    serde_json::json!({"model": self.model, "user": user, "items": items})
                }
            };
        }
        if self.mode == "broadcast" {
            let user = random_user(&mut rng);
            let items: Vec<serde_json::Value> = (0..self.batch_size)
                .map(|_| random_item(&mut rng))
                .collect();
            serde_json::json!({"model": self.model, "user": user, "items": items})
        } else {
            let features: Vec<serde_json::Value> =
                (0..self.batch_size).map(|_| random_row(&mut rng)).collect();
            serde_json::json!({"model": self.model, "features": features})
        }
    }
}

fn csv_record_to_json(headers: &StringRecord, record: &StringRecord) -> serde_json::Value {
    let mut row = serde_json::Map::new();
    for (key, value) in headers.iter().zip(record.iter()) {
        row.insert(key.to_string(), csv_field_to_json(value));
    }
    serde_json::Value::Object(row)
}

#[derive(Debug, Deserialize)]
struct FlowConfigForBench {
    sources: Vec<SourceForBench>,
}

#[derive(Debug, Deserialize)]
struct SourceForBench {
    name: String,
    source: SourceKind,
    #[serde(default)]
    role: Role,
    dtype: String,
    default_val: Option<String>,
}

fn load_input_data(args: &Args) -> Option<InputData> {
    let path = args.input_file.as_ref()?;
    if args.mode == "broadcast" {
        let feature_config_path = args
            .feature_config
            .as_ref()
            .expect("--feature-config is required with --input-file in broadcast mode");
        let feature_yaml =
            std::fs::read_to_string(feature_config_path).expect("failed to read feature config");
        let flow: FlowConfigForBench =
            serde_yaml::from_str(&feature_yaml).expect("failed to parse feature config");
        let feature_sources: Vec<SourceForBench> = flow
            .sources
            .into_iter()
            .filter(|source| source.role == Role::Feature)
            .collect();
        let rows = load_broadcast_samples(path, &feature_sources, args.separator, args.no_header);
        if rows.is_empty() {
            warn!("input file contains no rows");
            return None;
        }
        return Some(InputData::Broadcast(Arc::new(rows)));
    }

    let rows = load_pointwise_rows(path);
    if rows.is_empty() {
        warn!("input file contains no rows");
        return None;
    }
    Some(InputData::Pointwise(Arc::new(rows)))
}

fn load_pointwise_rows(path: &PathBuf) -> Vec<serde_json::Value> {
    let mut reader = csv::Reader::from_path(path).expect("failed to open input file");
    let headers = reader
        .headers()
        .expect("failed to read csv headers")
        .clone();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.expect("failed to read csv row");
        rows.push(csv_record_to_json(&headers, &record));
    }
    rows
}

fn load_broadcast_samples(
    path: &PathBuf,
    sources: &[SourceForBench],
    separator: char,
    no_header: bool,
) -> Vec<BroadcastSample> {
    let delimiter = separator as u32;
    if delimiter > u8::MAX as u32 {
        panic!("--separator must be a single-byte character");
    }
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(!no_header)
        .delimiter(delimiter as u8)
        .from_path(path)
        .expect("failed to open input file");
    let mut rows = Vec::new();
    for (row_idx, record) in reader.records().enumerate() {
        let record = record.expect("failed to read input row");
        if record.len() < sources.len() {
            panic!(
                "row {} has {} fields, expected at least {}",
                row_idx + 1,
                record.len(),
                sources.len()
            );
        }

        let mut user = serde_json::Map::new();
        let mut item = serde_json::Map::new();
        for (source, value) in sources.iter().zip(record.iter()) {
            let target = if source.source == SourceKind::Item {
                &mut item
            } else {
                &mut user
            };
            target.insert(source.name.clone(), source_field_to_json(value, source));
        }
        rows.push(BroadcastSample {
            user: serde_json::Value::Object(user),
            item: serde_json::Value::Object(item),
        });
    }
    rows
}

fn source_field_to_json(s: &str, source: &SourceForBench) -> serde_json::Value {
    let trimmed = s.trim();
    let value = if trimmed.is_empty() {
        source.default_val.as_deref().unwrap_or("")
    } else {
        trimmed
    };
    match source.dtype.as_str() {
        "int" => value
            .parse::<i64>()
            .map(serde_json::Value::from)
            .unwrap_or_else(|_| serde_json::json!(0)),
        "float" => value
            .parse::<f64>()
            .map(serde_json::Value::from)
            .unwrap_or_else(|_| serde_json::json!(0.0)),
        "bool" => match value {
            "true" | "TRUE" | "1" => serde_json::json!(true),
            "false" | "FALSE" | "0" => serde_json::json!(false),
            _ => serde_json::json!(false),
        },
        _ => serde_json::json!(value),
    }
}

fn csv_field_to_json(s: &str) -> serde_json::Value {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return serde_json::Value::Null;
    }
    if let Ok(v) = trimmed.parse::<i64>() {
        return serde_json::json!(v);
    }
    if let Ok(v) = trimmed.parse::<f64>() {
        return serde_json::json!(v);
    }
    match trimmed {
        "true" | "TRUE" => serde_json::json!(true),
        "false" | "FALSE" => serde_json::json!(false),
        _ => serde_json::json!(trimmed),
    }
}

fn random_user(rng: &mut impl Rng) -> serde_json::Value {
    let tags = [
        "sports", "music", "gaming", "reading", "travel", "food", "fashion", "tech", "fitness",
        "art", "movie", "pet", "car", "photo", "diy",
    ];
    let cat_vals = ["val_0", "val_1", "val_2", "val_3", "val_4"];
    let mut row = serde_json::Map::new();
    row.insert("user_id".into(), serde_json::json!(rng.gen_range(0..1000)));
    for i in 0..15 {
        row.insert(
            format!("user_stat_{}", i),
            serde_json::json!((rng.gen_range(0.0f64..1.0) * 1000.0).round() / 1000.0),
        );
    }
    for i in 0..15 {
        row.insert(
            format!("user_cat_{}", i),
            serde_json::json!(cat_vals[rng.gen_range(0..5)]),
        );
    }
    for i in 0..5 {
        let n = rng.gen_range(3..=8);
        let s: Vec<String> = (0..n)
            .map(|_| format!("{}#{}", tags[rng.gen_range(0..15)], rng.gen_range(0..5)))
            .collect();
        row.insert(format!("user_tags_{}", i), serde_json::json!(s.join("|")));
    }
    let hn = rng.gen_range(10..=20);
    let hist: Vec<String> = (0..hn)
        .map(|_| {
            format!(
                "{}:{}",
                rng.gen_range(100..=900) / 100 * 100,
                rng.gen_range(1..=5)
            )
        })
        .collect();
    row.insert("user_history".into(), serde_json::json!(hist.join(",")));
    row.insert("ctx_hour".into(), serde_json::json!(rng.gen_range(0..24)));
    let devices = ["phone", "pad", "pc"];
    let platforms = ["ios", "android", "web"];
    let networks = ["wifi", "4g", "5g"];
    let pages = ["home", "detail", "search", "cart"];
    row.insert(
        "ctx_device".into(),
        serde_json::json!(devices[rng.gen_range(0..3)]),
    );
    row.insert(
        "ctx_platform".into(),
        serde_json::json!(platforms[rng.gen_range(0..3)]),
    );
    row.insert(
        "ctx_network".into(),
        serde_json::json!(networks[rng.gen_range(0..3)]),
    );
    row.insert(
        "ctx_page".into(),
        serde_json::json!(pages[rng.gen_range(0..4)]),
    );
    serde_json::Value::Object(row)
}

fn random_item(rng: &mut impl Rng) -> serde_json::Value {
    let tags = [
        "sports", "music", "gaming", "reading", "travel", "food", "fashion", "tech", "fitness",
        "art", "movie", "pet", "car", "photo", "diy",
    ];
    let cat_vals = ["val_0", "val_1", "val_2", "val_3", "val_4"];
    let mut row = serde_json::Map::new();
    row.insert("item_id".into(), serde_json::json!(rng.gen_range(0..2000)));
    for i in 0..15 {
        row.insert(
            format!("item_stat_{}", i),
            serde_json::json!((rng.gen_range(0.0f64..1.0) * 1000.0).round() / 1000.0),
        );
    }
    for i in 0..15 {
        row.insert(
            format!("item_cat_{}", i),
            serde_json::json!(cat_vals[rng.gen_range(0..5)]),
        );
    }
    for i in 0..5 {
        let n = rng.gen_range(3..=8);
        let s: Vec<String> = (0..n)
            .map(|_| format!("{}#1", tags[rng.gen_range(0..15)]))
            .collect();
        row.insert(format!("item_tags_{}", i), serde_json::json!(s.join("|")));
    }
    for name in [
        "item_ctr_7d",
        "item_cvr_7d",
        "item_click_24h",
        "item_order_30d",
        "item_expo_7d",
    ] {
        row.insert(
            name.into(),
            serde_json::json!((rng.gen_range(0.0f64..0.5) * 10000.0).round() / 10000.0),
        );
    }
    serde_json::Value::Object(row)
}

fn random_row(rng: &mut impl Rng) -> serde_json::Value {
    let mut row = random_user(rng).as_object().unwrap().clone();
    let item = random_item(rng);
    for (k, v) in item.as_object().unwrap() {
        row.insert(k.clone(), v.clone());
    }
    serde_json::Value::Object(row)
}

async fn issue_request(
    client: reqwest::Client,
    ctx: RequestContext,
    latencies: Arc<Mutex<Vec<f64>>>,
    total: Arc<AtomicU64>,
    errors: Arc<AtomicU64>,
) {
    let body = ctx.body();
    let start = Instant::now();
    let result = client.post(ctx.url()).json(&body).send().await;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    if result.map_or(false, |r| r.status().is_success()) {
        latencies.lock().unwrap().push(ms);
        total.fetch_add(1, Ordering::Relaxed);
    } else {
        errors.fetch_add(1, Ordering::Relaxed);
    }
}

async fn run_open_loop(args: Args) {
    let input_data = load_input_data(&args);
    let ctx = RequestContext {
        target: args.target.clone(),
        model: args.model.clone(),
        mode: args.mode.clone(),
        batch_size: args.batch_size,
        input_data,
        next_row: Arc::new(AtomicU64::new(0)),
    };
    let client = reqwest::Client::new();
    let latencies = Arc::new(Mutex::new(Vec::<f64>::new()));
    let total_reqs = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::new();

    let total_requests = args.duration_secs.saturating_mul(args.target_qps as u64) as usize;
    let period = Duration::from_secs_f64(1.0 / args.target_qps as f64);
    let start = Instant::now();
    let mut interval = time::interval_at(time::Instant::now(), period);
    interval.set_missed_tick_behavior(MissedTickBehavior::Burst);

    for _ in 0..total_requests {
        interval.tick().await;
        let client = client.clone();
        let ctx = ctx.clone();
        let lats = latencies.clone();
        let total = total_reqs.clone();
        let errs = errors.clone();
        handles.push(tokio::spawn(async move {
            issue_request(client, ctx, lats, total, errs).await;
        }));
    }

    for handle in handles {
        let _ = handle.await;
    }

    let elapsed = start.elapsed().as_secs_f64();
    report(
        args,
        total_reqs,
        errors,
        latencies,
        elapsed,
        Some(total_requests),
    );
}

fn run_closed_loop(args: Args) {
    let client = reqwest::blocking::Client::new();
    let latencies = Arc::new(Mutex::new(Vec::<f64>::new()));
    let total_reqs = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    let running = Arc::new(AtomicU64::new(1));
    let rate_interval_ms = if args.target_qps > 0 {
        (args.concurrency as f64 * 1000.0 / args.target_qps as f64) as u64
    } else {
        0
    };

    let mut handles = vec![];
    for _ in 0..args.concurrency {
        let client = client.clone();
        let lats = latencies.clone();
        let total = total_reqs.clone();
        let errs = errors.clone();
        let run = running.clone();
        let target = args.target.clone();
        let model = args.model.clone();
        let mode = args.mode.clone();
        let batch = args.batch_size;

        handles.push(std::thread::spawn(move || {
            let mut rng = rand::thread_rng();
            while run.load(Ordering::Relaxed) != 0 {
                let t0 = Instant::now();
                let url = if mode == "broadcast" {
                    format!("{}/predict/broadcast", target)
                } else {
                    format!("{}/predict", target)
                };
                let body = if mode == "broadcast" {
                    let user = random_user(&mut rng);
                    let items: Vec<serde_json::Value> =
                        (0..batch).map(|_| random_item(&mut rng)).collect();
                    serde_json::json!({"model": model, "user": user, "items": items})
                } else {
                    let features: Vec<serde_json::Value> =
                        (0..batch).map(|_| random_row(&mut rng)).collect();
                    serde_json::json!({"model": model, "features": features})
                };
                let start = Instant::now();
                let result = client.post(&url).json(&body).send();
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                if result.map_or(false, |r| r.status().is_success()) {
                    lats.lock().unwrap().push(ms);
                    total.fetch_add(1, Ordering::Relaxed);
                } else {
                    errs.fetch_add(1, Ordering::Relaxed);
                }
                if rate_interval_ms > 0 {
                    let elapsed = t0.elapsed().as_millis() as u64;
                    if elapsed < rate_interval_ms {
                        std::thread::sleep(Duration::from_millis(rate_interval_ms - elapsed));
                    }
                }
            }
        }));
    }

    std::thread::sleep(Duration::from_secs(args.duration_secs));
    running.store(0, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }

    let elapsed = args.duration_secs as f64;
    report(args, total_reqs, errors, latencies, elapsed, None);
}

fn report(
    args: Args,
    total_reqs: Arc<AtomicU64>,
    errors: Arc<AtomicU64>,
    latencies: Arc<Mutex<Vec<f64>>>,
    elapsed_secs: f64,
    scheduled: Option<usize>,
) {
    let mut lats = latencies.lock().unwrap().clone();
    if lats.is_empty() {
        warn!("no successful requests");
        return;
    }
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = lats.len();
    let sum: f64 = lats.iter().sum();
    let success = total_reqs.load(Ordering::Relaxed);
    let errors = errors.load(Ordering::Relaxed);
    let rps = success as f64 / elapsed_secs.max(1e-9);

    if let Some(s) = scheduled {
        info!(scheduled = s, "benchmark schedule");
    }
    if args.target_qps > 0 {
        info!(target_qps = args.target_qps, "benchmark target");
    }
    info!(
        batch_size = args.batch_size,
        model = %args.model,
        mode = %args.mode,
        "benchmark config"
    );
    info!(
        success = success,
        errors = errors,
        rps = rps,
        "benchmark throughput"
    );
    info!(
        p50_ms = p(&lats, 50.0),
        p95_ms = p(&lats, 95.0),
        p99_ms = p(&lats, 99.0),
        p999_ms = p(&lats, 99.9),
        mean_ms = sum / n as f64,
        min_ms = lats[0],
        max_ms = lats[n - 1],
        "benchmark latency"
    );
}

fn p(samples: &[f64], pct: f64) -> f64 {
    let i = ((pct / 100.0) * (samples.len() - 1) as f64).round() as usize;
    samples[i.min(samples.len() - 1)]
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .try_init();

    let args = Args::parse();
    info!(
        target = %args.target,
        model = %args.model,
        mode = %args.mode,
        concurrency = args.concurrency,
        batch_size = args.batch_size,
        duration_secs = args.duration_secs,
        "benchmark start"
    );

    if args.target_qps > 0 {
        run_open_loop(args).await;
    } else {
        run_closed_loop(args);
    }
}
