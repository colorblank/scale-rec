//! Axum 路由：/health, /models, /predict, /predict/broadcast。
use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::engine::{InferenceError, InferenceErrorKind};
use super::registry::{
    FeatureContract, ModelAliasInfo, ModelRegistry, ModelServingInfo, ResolvedModel, RoutingPolicy,
};
use super::tracing::RequestTimer;

pub type AppState = Arc<ModelRegistry>;

/// 单行特征
pub type FeatureRow = HashMap<String, Value>;

#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    pub model: String,
    pub version: Option<String>,
    pub alias: Option<String>,
    pub fallback_version: Option<String>,
    pub routing_key: Option<String>,
    pub features: Vec<FeatureRow>,
}

#[derive(Debug, Deserialize)]
pub struct BroadcastRequest {
    pub model: String,
    pub version: Option<String>,
    pub alias: Option<String>,
    pub fallback_version: Option<String>,
    pub routing_key: Option<String>,
    pub user: FeatureRow,
    pub items: Vec<FeatureRow>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictResponse {
    pub model: String,
    pub version: String,
    pub predictions: Vec<HashMap<String, f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiError {
    pub code: String,
    pub message: String,
    pub request_id: Option<String>,
    pub model_id: Option<String>,
    pub details: Option<Value>,
}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status = match self.code.as_str() {
            "BAD_REQUEST" => StatusCode::BAD_REQUEST,
            "REGISTRY_ERROR" => StatusCode::NOT_FOUND,
            "FEATURE_ERROR" => StatusCode::UNPROCESSABLE_ENTITY,
            "MODEL_ERROR" | "INTERNAL_ERROR" => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        tracing::error!(
            code = %self.code,
            message = %self.message,
            model_id = ?self.model_id,
            "API error response returned"
        );
        (status, Json(self)).into_response()
    }
}

fn map_predict_error(err: InferenceError, model_id: String) -> ApiError {
    let code = match err.kind() {
        InferenceErrorKind::BadRequest => "BAD_REQUEST",
        InferenceErrorKind::Feature => "FEATURE_ERROR",
        InferenceErrorKind::Model => "MODEL_ERROR",
        InferenceErrorKind::Internal => "INTERNAL_ERROR",
    };
    ApiError {
        code: code.to_string(),
        message: err.message().to_string(),
        request_id: None,
        model_id: Some(model_id),
        details: None,
    }
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub models: Vec<ModelServingInfo>,
}

#[derive(Debug, Deserialize)]
pub struct AliasUpdateRequest {
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AliasListResponse {
    pub model: String,
    pub aliases: Vec<ModelAliasInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RoutingUpdateRequest {
    #[serde(flatten)]
    pub policy: RoutingPolicy,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/models", get(list_models))
        .route("/models/{model}", get(get_model))
        .route("/models/{model}/features", get(get_model_features))
        .route("/models/{model}/aliases", get(get_model_aliases))
        .route("/models/{model}/aliases/{alias}", post(set_model_alias))
        .route(
            "/models/{model}/aliases/{alias}",
            delete(delete_model_alias),
        )
        .route("/models/{model}/routing", get(get_model_routing))
        .route("/models/{model}/routing", post(set_model_routing))
        .route(
            "/models/{model}/versions/{version}/features",
            get(get_model_version_features),
        )
        .route("/predict", post(predict))
        .route("/predict/broadcast", post(predict_broadcast))
        .with_state(state)
}

async fn health(State(reg): State<AppState>) -> Json<Value> {
    Json(serde_json::json!({
        "status": "ok",
        "models": reg.list_info()
    }))
}

async fn list_models(State(reg): State<AppState>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        models: reg.list_info(),
    })
}

async fn get_model(
    State(reg): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<ModelServingInfo>, ApiError> {
    reg.model_info(&model).map(Json).ok_or_else(|| ApiError {
        code: "REGISTRY_ERROR".to_string(),
        message: format!("model '{}' not found", model),
        request_id: None,
        model_id: Some(model),
        details: None,
    })
}

async fn get_model_features(
    State(reg): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<FeatureContract>, ApiError> {
    reg.feature_contract(&model, None)
        .map(Json)
        .ok_or_else(|| ApiError {
            code: "REGISTRY_ERROR".to_string(),
            message: format!("model '{}' not found", model),
            request_id: None,
            model_id: Some(model),
            details: None,
        })
}

async fn get_model_aliases(
    State(reg): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<AliasListResponse>, ApiError> {
    let model_id = model.clone();
    let aliases = reg.aliases(&model).ok_or_else(|| ApiError {
        code: "REGISTRY_ERROR".to_string(),
        message: format!("model '{}' not found", model),
        request_id: None,
        model_id: Some(model.clone()),
        details: None,
    })?;
    Ok(Json(AliasListResponse {
        model: model_id,
        aliases,
    }))
}

async fn set_model_alias(
    State(reg): State<AppState>,
    Path((model, alias)): Path<(String, String)>,
    Json(req): Json<AliasUpdateRequest>,
) -> Result<Json<AliasListResponse>, ApiError> {
    reg.set_alias(&model, &alias, &req.version)
        .map_err(|e| ApiError {
            code: "REGISTRY_ERROR".to_string(),
            message: e,
            request_id: None,
            model_id: Some(model.clone()),
            details: Some(serde_json::json!({ "alias": alias, "version": req.version })),
        })?;
    let aliases = reg.aliases(&model).ok_or_else(|| ApiError {
        code: "REGISTRY_ERROR".to_string(),
        message: format!("model '{}' not found", model),
        request_id: None,
        model_id: Some(model.clone()),
        details: None,
    })?;
    Ok(Json(AliasListResponse { model, aliases }))
}

async fn delete_model_alias(
    State(reg): State<AppState>,
    Path((model, alias)): Path<(String, String)>,
) -> Result<Json<AliasListResponse>, ApiError> {
    reg.delete_alias(&model, &alias).map_err(|e| ApiError {
        code: "REGISTRY_ERROR".to_string(),
        message: e,
        request_id: None,
        model_id: Some(model.clone()),
        details: Some(serde_json::json!({ "alias": alias })),
    })?;
    let aliases = reg.aliases(&model).ok_or_else(|| ApiError {
        code: "REGISTRY_ERROR".to_string(),
        message: format!("model '{}' not found", model),
        request_id: None,
        model_id: Some(model.clone()),
        details: None,
    })?;
    Ok(Json(AliasListResponse { model, aliases }))
}

async fn get_model_routing(
    State(reg): State<AppState>,
    Path(model): Path<String>,
) -> Result<Json<Option<RoutingPolicy>>, ApiError> {
    reg.routing_policy(&model)
        .map(Json)
        .ok_or_else(|| ApiError {
            code: "REGISTRY_ERROR".to_string(),
            message: format!("model '{}' not found", model),
            request_id: None,
            model_id: Some(model),
            details: None,
        })
}

async fn set_model_routing(
    State(reg): State<AppState>,
    Path(model): Path<String>,
    Json(req): Json<RoutingUpdateRequest>,
) -> Result<Json<Option<RoutingPolicy>>, ApiError> {
    reg.set_routing_policy(&model, Some(req.policy))
        .map_err(|e| ApiError {
            code: "REGISTRY_ERROR".to_string(),
            message: e,
            request_id: None,
            model_id: Some(model.clone()),
            details: None,
        })?;
    reg.routing_policy(&model)
        .map(Json)
        .ok_or_else(|| ApiError {
            code: "REGISTRY_ERROR".to_string(),
            message: format!("model '{}' not found", model),
            request_id: None,
            model_id: Some(model),
            details: None,
        })
}

async fn get_model_version_features(
    State(reg): State<AppState>,
    Path((model, version)): Path<(String, String)>,
) -> Result<Json<FeatureContract>, ApiError> {
    reg.feature_contract(&model, Some(&version))
        .map(Json)
        .ok_or_else(|| ApiError {
            code: "REGISTRY_ERROR".to_string(),
            message: format!("model '{}' version '{}' not found", model, version),
            request_id: None,
            model_id: Some(model),
            details: Some(serde_json::json!({ "requested_version": version })),
        })
}

fn resolve_model(
    reg: &ModelRegistry,
    model: &str,
    version: Option<&str>,
    alias: Option<&str>,
    fallback_version: Option<&str>,
    routing_key: Option<&str>,
) -> Result<ResolvedModel, ApiError> {
    reg.resolve_request(model, version, alias, fallback_version, routing_key)
        .ok_or_else(|| ApiError {
            code: "REGISTRY_ERROR".to_string(),
            message: match (version, fallback_version) {
                (Some(v), Some(fb)) => {
                    format!(
                        "model '{}' version '{}' not found and fallback version '{}' is unavailable",
                        model, v, fb
                    )
                }
                (Some(v), None) => format!("model '{}' version '{}' not found", model, v),
                (None, Some(fb)) if alias.is_some() => format!(
                    "model '{}' alias '{}' not found and fallback version '{}' is unavailable",
                    model,
                    alias.unwrap_or_default(),
                    fb
                ),
                (None, None) if alias.is_some() => {
                    format!("model '{}' alias '{}' not found", model, alias.unwrap_or_default())
                }
                (None, _) => format!("model '{}' not found", model),
            },
            request_id: None,
            model_id: Some(model.to_string()),
            details: Some(serde_json::json!({
                "requested_version": version,
                "alias": alias,
                "fallback_version": fallback_version,
            })),
        })
}

async fn predict(
    State(reg): State<AppState>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, ApiError> {
    let mut timer = RequestTimer::new();
    let model = req.model;
    let requested_version = req.version;
    let alias = req.alias;
    let fallback_version = req.fallback_version;
    let routing_key = req.routing_key;
    let features = req.features;
    let batch_size = features.len();

    let resolved = resolve_model(
        &reg,
        &model,
        requested_version.as_deref(),
        alias.as_deref(),
        fallback_version.as_deref(),
        routing_key.as_deref(),
    )?;

    let model_for_error = model.clone();
    let engine = resolved.engine.clone();
    let (result, metrics) = tokio::task::spawn_blocking(move || engine.predict(&features))
        .await
        .map_err(|e| ApiError {
            code: "INTERNAL_ERROR".to_string(),
            message: format!("inference worker join error: {}", e),
            request_id: None,
            model_id: Some(model_for_error.clone()),
            details: None,
        })?
        .map_err(|e| map_predict_error(e, model_for_error))?;

    timer.record(&metrics);
    timer.finish("/predict", &model, batch_size);

    Ok(Json(PredictResponse {
        model,
        version: resolved.version,
        predictions: result,
    }))
}

async fn predict_broadcast(
    State(reg): State<AppState>,
    Json(req): Json<BroadcastRequest>,
) -> Result<Json<PredictResponse>, ApiError> {
    let mut timer = RequestTimer::new();
    let model = req.model;
    let requested_version = req.version;
    let alias = req.alias;
    let fallback_version = req.fallback_version;
    let routing_key = req.routing_key;
    let user = req.user;
    let items = req.items;
    let batch_size = items.len();

    let resolved = resolve_model(
        &reg,
        &model,
        requested_version.as_deref(),
        alias.as_deref(),
        fallback_version.as_deref(),
        routing_key.as_deref(),
    )?;

    let model_for_error = model.clone();
    let engine = resolved.engine.clone();
    let (result, metrics) =
        tokio::task::spawn_blocking(move || engine.predict_broadcast(&user, &items))
            .await
            .map_err(|e| ApiError {
                code: "INTERNAL_ERROR".to_string(),
                message: format!("inference worker join error: {}", e),
                request_id: None,
                model_id: Some(model_for_error.clone()),
                details: None,
            })?
            .map_err(|e| map_predict_error(e, model_for_error))?;

    timer.record(&metrics);
    timer.finish("/predict/broadcast", &model, batch_size);

    Ok(Json(PredictResponse {
        model,
        version: resolved.version,
        predictions: result,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use candle_core::{safetensors, DType as CandleDType, Device, Tensor};
    use serde::de::DeserializeOwned;
    use serde_json::json;
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower::ServiceExt;

    fn sha256_bytes(input: &[u8]) -> [u8; 32] {
        const H0: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];

        let bit_len = (input.len() as u64) * 8;
        let mut msg = input.to_vec();
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_be_bytes());

        let mut h = H0;
        for chunk in msg.chunks_exact(64) {
            let mut w = [0u32; 64];
            for (i, word) in w.iter_mut().take(16).enumerate() {
                let j = i * 4;
                *word = u32::from_be_bytes([chunk[j], chunk[j + 1], chunk[j + 2], chunk[j + 3]]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }

            let mut a = h[0];
            let mut b = h[1];
            let mut c = h[2];
            let mut d = h[3];
            let mut e = h[4];
            let mut f = h[5];
            let mut g = h[6];
            let mut hh = h[7];

            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let temp1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);

                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
            h[5] = h[5].wrapping_add(f);
            h[6] = h[6].wrapping_add(g);
            h[7] = h[7].wrapping_add(hh);
        }

        let mut out = [0u8; 32];
        for (i, word) in h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn sha256_file(path: &Path) -> String {
        let bytes = fs::read(path).unwrap();
        let digest = sha256_bytes(&bytes);
        let mut out = String::with_capacity(digest.len() * 2);
        for b in digest {
            out.push_str(&format!("{:02x}", b));
        }
        out
    }

    fn write_lr_manifest_artifact(root: &Path, model_id: &str, version: &str) -> PathBuf {
        let version_dir = root.join(model_id).join(version);
        fs::create_dir_all(&version_dir).unwrap();
        let feature_config = version_dir.join("feature.yaml");
        let model_config = version_dir.join("model.yaml");
        let weights = version_dir.join("model.safetensors");
        let manifest_path = version_dir.join("model_manifest.yaml");

        let feature_yaml = "version: '1.0.0'\nsources:\n  - name: user_id\n    dtype: int\n    default_val: '0'\noperators:\n  - name: user_id_hash\n    op_type: FeatureHash\n    inputs: [user_id]\n    outputs: [user_id_idx]\n    params: {vocab_size: 100, num_hashes: 1}\n    embed: {vocab_size: 100, embed_dim: 8}\n";
        fs::write(&feature_config, feature_yaml).unwrap();
        fs::write(&model_config, "type: lr\n").unwrap();

        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "embeddings.emb_user_id_idx.weight".to_string(),
            Tensor::zeros((100, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors.insert(
            "mlp.output.weight".to_string(),
            Tensor::zeros((1, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors.insert(
            "mlp.output.bias".to_string(),
            Tensor::zeros((1,), CandleDType::F32, &device).unwrap(),
        );
        safetensors::save(&tensors, &weights).unwrap();

        let manifest = format!(
            "schema_version: 1\nmodel_id: {}\nmodel_version: {}\nmodel_type: lr\ncode_commit:\nweights_file: model.safetensors\nweights_sha256: {}\nfeature_config_file: feature.yaml\nfeature_config_sha256: {}\nmodel_config_file: model.yaml\nmodel_config_sha256: {}\n",
            model_id,
            version,
            sha256_file(&weights),
            sha256_file(&feature_config),
            sha256_file(&model_config),
        );
        fs::write(&manifest_path, manifest).unwrap();
        manifest_path
    }

    fn app_with_ranker(root: &Path) -> Router {
        let registry = ModelRegistry::from_model_dir(root).unwrap();
        let v1 = write_lr_manifest_artifact(root, "ranker", "20260604_010000");
        let v2 = write_lr_manifest_artifact(root, "ranker", "20260604_020000");
        registry.load_manifest(&v1).unwrap();
        registry.load_manifest(&v2).unwrap();
        router(Arc::new(registry))
    }

    async fn read_json<T: DeserializeOwned>(resp: axum::response::Response) -> T {
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn alias_routing_and_predict_endpoints_work() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-routes-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let app = app_with_ranker(&root);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/models/ranker/aliases")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let aliases: AliasListResponse = read_json(resp).await;
        assert_eq!(aliases.model, "ranker");
        assert!(aliases.aliases.is_empty());

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/models/ranker/aliases/prod")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"version":"20260604_020000"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let aliases: AliasListResponse = read_json(resp).await;
        assert_eq!(aliases.aliases.len(), 1);
        assert_eq!(aliases.aliases[0].alias, "prod");
        assert_eq!(aliases.aliases[0].version, "20260604_020000");

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/models/ranker/routing")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"type":"weighted","key_field":"user_id","salt":"salt","versions":[{"version":"20260604_010000","weight":0},{"version":"20260604_020000","weight":100}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let routing: Option<RoutingPolicy> = read_json(resp).await;
        assert!(matches!(routing, Some(RoutingPolicy::Weighted { .. })));

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/predict")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "ranker",
                            "routing_key": "user-42",
                            "features": [
                                { "user_id": 42 }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let prediction: PredictResponse = read_json(resp).await;
        assert_eq!(prediction.model, "ranker");
        assert_eq!(prediction.version, "20260604_020000");
        assert_eq!(prediction.predictions.len(), 1);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/predict/broadcast")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "ranker",
                            "alias": "prod",
                            "user": { "user_id": 42 },
                            "items": [
                                { "user_id": 42 }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let broadcast: PredictResponse = read_json(resp).await;
        assert_eq!(broadcast.model, "ranker");
        assert_eq!(broadcast.version, "20260604_020000");
        assert_eq!(broadcast.predictions.len(), 1);

        fs::remove_dir_all(root).unwrap();
    }

    async fn read_error(resp: axum::response::Response) -> ApiError {
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn unknown_model_returns_registry_error() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-routes-missing-model-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let app = app_with_ranker(&root);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/models/missing/aliases")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let err: ApiError = read_error(resp).await;
        assert_eq!(err.code, "REGISTRY_ERROR");
        assert!(err.message.contains("model 'missing' not found"));

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn alias_to_missing_version_returns_registry_error() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-routes-missing-alias-version-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let app = app_with_ranker(&root);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/models/ranker/aliases/prod")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"version":"missing"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let err: ApiError = read_error(resp).await;
        assert_eq!(err.code, "REGISTRY_ERROR");
        assert!(err.message.contains("version 'missing' not found"));

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn routing_policy_with_unknown_version_is_rejected() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-routes-bad-routing-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let app = app_with_ranker(&root);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/models/ranker/routing")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"type":"fixed","version":"missing"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let err: ApiError = read_error(resp).await;
        assert_eq!(err.code, "REGISTRY_ERROR");
        assert!(err
            .message
            .contains("routing policy references unknown version 'missing'"));

        fs::remove_dir_all(root).unwrap();
    }
}
