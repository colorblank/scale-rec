//! HTTP 推理服务：加载特征配置和模型权重，提供 REST API。
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use scale_rec::server::registry::ModelRegistry;
use scale_rec::server::routes;
use tower_http::cors::CorsLayer;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = PathBuf::from("python/demo/temp");
    let mut feature_config_path = PathBuf::from("python/demo/feature_config_demo.yaml");
    let mut port: u16 = 8080;
    let mut watch = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => { i += 1; model_dir = PathBuf::from(&args[i]); }
            "--feature-config" => { i += 1; feature_config_path = PathBuf::from(&args[i]); }
            "--port" => { i += 1; port = args[i].parse().unwrap_or(8080); }
            "--watch" => { watch = true; }
            _ => { eprintln!("unknown arg: {}", args[i]); }
        }
        i += 1;
    }

    info!("feature config: {}", feature_config_path.display());
    info!("model dir: {}", model_dir.display());
    info!("port: {}", port);
    info!("watch: {}", watch);

    let registry = Arc::new(
        ModelRegistry::new(&feature_config_path, &model_dir)
            .expect("Failed to create model registry"),
    );

    // Auto-load all .safetensors files in model dir
    if let Ok(entries) = std::fs::read_dir(&model_dir) {
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

    let addr = format!("0.0.0.0:{}", port);
    info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
