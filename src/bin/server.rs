//! HTTP 推理服务：加载特征配置和模型权重，提供 REST API。
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use scale_rec::server::registry::ModelRegistry;
use scale_rec::server::routes;
use tower_http::cors::CorsLayer;
use tracing::info;
use tracing_subscriber::EnvFilter;

struct ServerArgs {
    model_dir: PathBuf,
    feature_config_path: PathBuf,
    port: u16,
    worker_threads: Option<usize>,
    blocking_threads: Option<usize>,
}

fn main() {
    let args = parse_args();

    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    runtime.enable_all();
    if let Some(worker_threads) = args.worker_threads {
        runtime.worker_threads(worker_threads);
    }
    if let Some(blocking_threads) = args.blocking_threads {
        runtime.max_blocking_threads(blocking_threads);
    }

    runtime
        .build()
        .expect("failed to build tokio runtime")
        .block_on(run(args));
}

fn parse_args() -> ServerArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir: Option<PathBuf> = None;
    let mut feature_config_path: Option<PathBuf> = None;
    let mut port: u16 = 8080;
    let mut worker_threads: Option<usize> = None;
    let mut blocking_threads: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                model_dir = Some(PathBuf::from(&args[i]));
            }
            "--feature-config" => {
                i += 1;
                feature_config_path = Some(PathBuf::from(&args[i]));
            }
            "--port" => {
                i += 1;
                port = args[i].parse().unwrap_or(8080);
            }
            "--worker-threads" => {
                i += 1;
                worker_threads = args[i].parse().ok();
            }
            "--blocking-threads" => {
                i += 1;
                blocking_threads = args[i].parse().ok();
            }
            _ => {
                eprintln!("unknown arg: {}", args[i]);
            }
        }
        i += 1;
    }

    ServerArgs {
        model_dir: model_dir.expect("--model-dir is required"),
        feature_config_path: feature_config_path.expect("--feature-config is required"),
        port,
        worker_threads,
        blocking_threads,
    }
}

async fn run(args: ServerArgs) {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("feature config: {}", args.feature_config_path.display());
    info!("model dir: {}", args.model_dir.display());
    info!("port: {}", args.port);
    info!("worker threads: {:?}", args.worker_threads);
    info!("blocking threads: {:?}", args.blocking_threads);

    let registry = Arc::new(
        ModelRegistry::new(&args.feature_config_path, &args.model_dir)
            .expect("Failed to create model registry"),
    );

    // Auto-load all .safetensors files in model dir
    if let Ok(entries) = std::fs::read_dir(&args.model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "safetensors") {
                let name = path.file_stem().unwrap().to_string_lossy();
                let model_name = name.to_string();
                match registry.load_model(&model_name) {
                    Ok(info) => info!("loaded: {:?}", info),
                    Err(e) => info!("skip {}: {}", model_name, e),
                }
            }
        }
    }

    info!("models: {:?}", registry.list());

    let app: Router = routes::router(registry).layer(CorsLayer::permissive());

    let addr = format!("0.0.0.0:{}", args.port);
    info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
