//! HTTP 推理服务：加载特征配置和模型权重，提供 REST API。
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use axum::{
    extract::DefaultBodyLimit,
    http::{header, HeaderValue, Method},
    Router,
};
use scale_rec::server::manifest::find_manifests;
use scale_rec::server::registry::ModelRegistry;
use scale_rec::server::routes;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

struct ServerArgs {
    model_dir: PathBuf,
    model_paths: Vec<PathBuf>,
    feature_config_path: Option<PathBuf>,
    port: u16,
    worker_threads: Option<usize>,
    blocking_threads: Option<usize>,
    allowed_origins: Vec<String>,
    max_body_bytes: usize,
}

const DEFAULT_ALLOWED_ORIGINS: &[&str] = &[
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
];
const DEFAULT_MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

fn main() -> Result<()> {
    let args = parse_args()?;

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
        .context("failed to build tokio runtime")?
        .block_on(run(args))
}

fn parse_args() -> Result<ServerArgs> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir: Option<PathBuf> = None;
    let mut model_paths: Vec<PathBuf> = Vec::new();
    let mut feature_config_path: Option<PathBuf> = None;
    let mut port: u16 = 8080;
    let mut worker_threads: Option<usize> = None;
    let mut blocking_threads: Option<usize> = None;
    let mut allowed_origins: Vec<String> = std::env::var("SCALE_REC_ALLOWED_ORIGINS")
        .ok()
        .map(|raw| split_csv(&raw))
        .unwrap_or_default();
    let mut max_body_bytes: usize = std::env::var("SCALE_REC_MAX_BODY_BYTES")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(DEFAULT_MAX_BODY_BYTES);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                if i >= args.len() {
                    bail!("--model-dir requires a value");
                }
                model_dir = Some(PathBuf::from(&args[i]));
            }
            "--model-path" | "--model-manifest" => {
                i += 1;
                if i >= args.len() {
                    bail!("{} requires a value", args[i - 1]);
                }
                model_paths.push(PathBuf::from(&args[i]));
            }
            "--feature-config" => {
                i += 1;
                if i >= args.len() {
                    bail!("--feature-config requires a value");
                }
                feature_config_path = Some(PathBuf::from(&args[i]));
            }
            "--port" => {
                i += 1;
                if i >= args.len() {
                    bail!("--port requires a value");
                }
                port = args[i].parse().unwrap_or(8080);
            }
            "--worker-threads" => {
                i += 1;
                if i >= args.len() {
                    bail!("--worker-threads requires a value");
                }
                worker_threads = args[i].parse().ok();
            }
            "--blocking-threads" => {
                i += 1;
                if i >= args.len() {
                    bail!("--blocking-threads requires a value");
                }
                blocking_threads = args[i].parse().ok();
            }
            "--allowed-origin" => {
                i += 1;
                if i >= args.len() {
                    bail!("--allowed-origin requires a value");
                }
                allowed_origins.push(args[i].clone());
            }
            "--max-body-bytes" => {
                i += 1;
                if i >= args.len() {
                    bail!("--max-body-bytes requires a value");
                }
                max_body_bytes = args[i]
                    .parse()
                    .with_context(|| format!("invalid --max-body-bytes '{}'", args[i]))?;
            }
            _ => {
                bail!("unknown arg: {}", args[i]);
            }
        }
        i += 1;
    }
    if allowed_origins.is_empty() {
        allowed_origins = DEFAULT_ALLOWED_ORIGINS
            .iter()
            .map(|origin| origin.to_string())
            .collect();
    }
    if max_body_bytes == 0 {
        bail!("--max-body-bytes must be greater than zero");
    }
    let resolved_model_dir = model_dir
        .or_else(|| {
            model_paths
                .first()
                .and_then(|path| path.parent().map(Path::to_path_buf))
        })
        .ok_or_else(|| anyhow!("--model-dir or --model-path is required"))?;

    Ok(ServerArgs {
        model_dir: resolved_model_dir,
        model_paths,
        feature_config_path,
        port,
        worker_threads,
        blocking_threads,
        allowed_origins,
        max_body_bytes,
    })
}

async fn run(args: ServerArgs) -> Result<()> {
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
    info!("allowed origins: {:?}", args.allowed_origins);
    info!("max body bytes: {}", args.max_body_bytes);

    let registry = Arc::new(match &args.feature_config_path {
        Some(feature_config_path) => ModelRegistry::new(feature_config_path, &args.model_dir)
            .map_err(|e| anyhow!("failed to create model registry: {}", e))?,
        None => ModelRegistry::from_model_dir(&args.model_dir)
            .map_err(|e| anyhow!("failed to create model registry: {}", e))?,
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
                        let Some(name) = path.file_stem() else {
                            warn!("skip {}: invalid file stem", path.display());
                            continue;
                        };
                        let name = name.to_string_lossy();
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

    let app: Router = routes::router(registry)
        .layer(DefaultBodyLimit::max(args.max_body_bytes))
        .layer(build_cors_layer(&args.allowed_origins)?);

    let addr = format!("0.0.0.0:{}", args.port);
    info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind {}", addr))?;
    axum::serve(listener, app)
        .await
        .context("server exited with error")?;
    Ok(())
}

fn split_csv(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

fn build_cors_layer(origins: &[String]) -> Result<CorsLayer> {
    let allowed_origins: Vec<HeaderValue> = origins
        .iter()
        .map(|origin| {
            origin
                .parse()
                .with_context(|| format!("invalid allowed origin '{}'", origin))
        })
        .collect::<Result<_>>()?;
    Ok(CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([header::CONTENT_TYPE]))
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
