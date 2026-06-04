//! HTTP 推理服务：加载特征配置和模型权重，提供 REST API。
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use scale_rec::server::manifest::find_manifests;
use scale_rec::server::registry::ModelRegistry;
use scale_rec::server::routes;
use tower_http::cors::CorsLayer;
use tracing::info;
use tracing_subscriber::EnvFilter;

struct ServerArgs {
    model_dir: PathBuf,
    model_paths: Vec<PathBuf>,
    feature_config_path: Option<PathBuf>,
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
    let mut model_paths: Vec<PathBuf> = Vec::new();
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
            "--model-path" | "--model-manifest" => {
                i += 1;
                model_paths.push(PathBuf::from(&args[i]));
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
    let resolved_model_dir = model_dir
        .or_else(|| {
            model_paths
                .first()
                .and_then(|path| path.parent().map(Path::to_path_buf))
        })
        .unwrap_or_else(|| {
            eprintln!("--model-dir or --model-path is required");
            std::process::exit(2);
        });

    ServerArgs {
        model_dir: resolved_model_dir,
        model_paths,
        feature_config_path,
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

    if let Some(feature_config_path) = &args.feature_config_path {
        info!("feature config fallback: {}", feature_config_path.display());
    } else {
        info!("feature config fallback: none (manifest-driven loading only)");
    }
    info!("model dir: {}", args.model_dir.display());
    if !args.model_paths.is_empty() {
        info!(
            "explicit model paths: {:?}",
            args.model_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
        );
    }
    info!("port: {}", args.port);
    info!("worker threads: {:?}", args.worker_threads);
    info!("blocking threads: {:?}", args.blocking_threads);

    let registry = Arc::new(match &args.feature_config_path {
        Some(feature_config_path) => ModelRegistry::new(feature_config_path, &args.model_dir)
            .expect("Failed to create model registry"),
        None => {
            ModelRegistry::from_model_dir(&args.model_dir).expect("Failed to create model registry")
        }
    });

    if !args.model_paths.is_empty() {
        for model_path in &args.model_paths {
            load_model_path(&registry, model_path);
        }
    } else {
        let manifests = find_manifests(&args.model_dir);
        if !manifests.is_empty() {
            for manifest_path in manifests {
                match registry.load_manifest(&manifest_path) {
                    Ok(info) => info!("loaded manifest {}: {:?}", manifest_path.display(), info),
                    Err(e) => info!("skip manifest {}: {}", manifest_path.display(), e),
                }
            }
        } else if args.feature_config_path.is_some() {
            // Backward-compatible demo mode: load loose .safetensors files and infer configs by name.
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
        } else {
            info!(
                "no manifests found under {}; no models loaded",
                args.model_dir.display()
            );
        }
    }

    info!("models: {:?}", registry.list());

    let app: Router = routes::router(registry).layer(CorsLayer::permissive());

    let addr = format!("0.0.0.0:{}", args.port);
    info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn load_model_path(registry: &ModelRegistry, model_path: &Path) {
    if model_path.is_dir() {
        let manifests = find_manifests(model_path);
        if manifests.is_empty() {
            info!("skip {}: no serving manifests found", model_path.display());
            return;
        }
        for manifest_path in manifests {
            match registry.load_manifest(&manifest_path) {
                Ok(info) => info!("loaded manifest {}: {:?}", manifest_path.display(), info),
                Err(e) => info!("skip manifest {}: {}", manifest_path.display(), e),
            }
        }
        return;
    }

    let extension = model_path.extension().and_then(|ext| ext.to_str());
    match extension {
        Some("safetensors") => match registry.load_safetensors(model_path) {
            Ok(info) => info!("loaded safetensors {}: {:?}", model_path.display(), info),
            Err(e) => info!("skip safetensors {}: {}", model_path.display(), e),
        },
        Some("yaml") | Some("yml") => match registry.load_manifest(model_path) {
            Ok(info) => info!("loaded manifest {}: {:?}", model_path.display(), info),
            Err(e) => info!("skip manifest {}: {}", model_path.display(), e),
        },
        _ => info!(
            "skip {}: expected a serving manifest, .safetensors file, or directory",
            model_path.display()
        ),
    }
}
