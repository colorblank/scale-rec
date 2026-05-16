//! 压测工具：对推理服务进行并发压测，输出 P50/P95/P99 延迟和吞吐量。
use clap::Parser;
use rand::Rng;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Parser)]
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
    target_qps: usize,  // rate-limited mode: 0 = unlimited
}

fn random_user(rng: &mut impl Rng) -> serde_json::Value {
    let tags = ["sports","music","gaming","reading","travel","food","fashion","tech","fitness","art",
                "movie","pet","car","photo","diy"];
    let cat_vals = ["val_0","val_1","val_2","val_3","val_4"];
    let mut row = serde_json::Map::new();
    row.insert("user_id".into(), serde_json::json!(rng.gen_range(0..1000)));
    for i in 0..15 { row.insert(format!("user_stat_{}",i), serde_json::json!((rng.gen_range(0.0f64..1.0)*1000.0).round()/1000.0)); }
    for i in 0..15 { row.insert(format!("user_cat_{}",i), serde_json::json!(cat_vals[rng.gen_range(0..5)])); }
    for i in 0..5 {
        let n = rng.gen_range(3..=8);
        let s: Vec<String> = (0..n).map(|_| format!("{}#{}", tags[rng.gen_range(0..15)], rng.gen_range(0..5))).collect();
        row.insert(format!("user_tags_{}",i), serde_json::json!(s.join("|")));
    }
    let hn = rng.gen_range(10..=20);
    let hist: Vec<String> = (0..hn).map(|_| format!("{}:{}", rng.gen_range(100..=900)/100*100, rng.gen_range(1..=5))).collect();
    row.insert("user_history".into(), serde_json::json!(hist.join(",")));
    serde_json::Value::Object(row)
}

fn random_item(rng: &mut impl Rng) -> serde_json::Value {
    let tags = ["sports","music","gaming","reading","travel","food","fashion","tech","fitness","art",
                "movie","pet","car","photo","diy"];
    let cat_vals = ["val_0","val_1","val_2","val_3","val_4"];
    let mut row = serde_json::Map::new();
    row.insert("item_id".into(), serde_json::json!(rng.gen_range(0..2000)));
    for i in 0..15 { row.insert(format!("item_stat_{}",i), serde_json::json!((rng.gen_range(0.0f64..1.0)*1000.0).round()/1000.0)); }
    for i in 0..15 { row.insert(format!("item_cat_{}",i), serde_json::json!(cat_vals[rng.gen_range(0..5)])); }
    for i in 0..5 {
        let n = rng.gen_range(3..=8);
        let s: Vec<String> = (0..n).map(|_| format!("{}#1", tags[rng.gen_range(0..15)])).collect();
        row.insert(format!("item_tags_{}",i), serde_json::json!(s.join("|")));
    }
    serde_json::Value::Object(row)
}

fn random_row(rng: &mut impl Rng) -> serde_json::Value {
    let mut row = random_user(rng).as_object().unwrap().clone();
    let item = random_item(rng);
    for (k, v) in item.as_object().unwrap() { row.insert(k.clone(), v.clone()); }
    serde_json::Value::Object(row)
}

fn main() {
    let args = Args::parse();
    println!("Benchmark: target={} model={} mode={} concur={} batch={} dur={}s",
        args.target, args.model, args.mode, args.concurrency, args.batch_size, args.duration_secs);

    let client = reqwest::blocking::Client::new();
    let latencies = Arc::new(Mutex::new(Vec::<f64>::new()));
    let total_reqs = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    let running = Arc::new(AtomicBool::new(true));

    // Rate limiting: per-worker interval in ms
    let rate_interval_ms = if args.target_qps > 0 {
        (args.concurrency as f64 * 1000.0 / args.target_qps as f64) as u64
    } else { 0 };

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
            while run.load(Ordering::Relaxed) {
                let t0 = Instant::now();
                let url = if mode == "broadcast" {
                    format!("{}/predict/broadcast", target)
                } else {
                    format!("{}/predict", target)
                };
                let body = if mode == "broadcast" {
                    let user = random_user(&mut rng);
                    let items: Vec<serde_json::Value> = (0..batch).map(|_| random_item(&mut rng)).collect();
                    serde_json::json!({"model": model, "user": user, "items": items})
                } else {
                    let features: Vec<serde_json::Value> = (0..batch).map(|_| random_row(&mut rng)).collect();
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
                // Rate limiting
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
    running.store(false, Ordering::Relaxed);
    for h in handles { let _ = h.join(); }

    let mut lats = latencies.lock().unwrap().clone();
    if lats.is_empty() { println!("No successful requests."); return; }
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = lats.len();
    let total = total_reqs.load(Ordering::Relaxed);
    let sum: f64 = lats.iter().sum();

    let rps = total as f64 / args.duration_secs as f64;
    println!("\n  ═══════════════════════════════════════");
    if args.target_qps > 0 {
        println!("  Target QPS:  {}", args.target_qps);
    }
    println!("  Batch:       {}  Concur: {}  Model: {}  Mode: {}",
             args.batch_size, args.concurrency, args.model, args.mode);
    println!("  ───────────────────────────────────────");
    println!("  Total:       {}  Errors: {}  RPS: {:.0}",
             total, errors.load(Ordering::Relaxed), rps);
    println!("  P50:         {:.1} ms", p(&lats, 50.0));
    println!("  P95:         {:.1} ms", p(&lats, 95.0));
    println!("  P99:         {:.1} ms", p(&lats, 99.0));
    println!("  P99.9:       {:.1} ms", p(&lats, 99.9));
    println!("  Mean:        {:.1} ms", sum / n as f64);
    println!("  Min/Max:     {:.1}/{:.1} ms", lats[0], lats[n-1]);
    println!("  ═══════════════════════════════════════");
}

fn p(s: &[f64], pct: f64) -> f64 {
    let i = ((pct / 100.0) * (s.len() - 1) as f64).round() as usize;
    s[i.min(s.len() - 1)]
}
