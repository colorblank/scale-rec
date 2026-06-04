//! Axum 路由：/health, /models, /predict, /predict/broadcast。
use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::engine::{InferenceError, InferenceErrorKind};
use super::registry::{ModelRegistry, ModelServingInfo, ResolvedModel};
use super::tracing::RequestTimer;

pub type AppState = Arc<ModelRegistry>;

/// 单行特征
pub type FeatureRow = HashMap<String, Value>;

#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    pub model: String,
    pub version: Option<String>,
    pub fallback_version: Option<String>,
    pub features: Vec<FeatureRow>,
}

#[derive(Debug, Deserialize)]
pub struct BroadcastRequest {
    pub model: String,
    pub version: Option<String>,
    pub fallback_version: Option<String>,
    pub user: FeatureRow,
    pub items: Vec<FeatureRow>,
}

#[derive(Debug, Serialize)]
pub struct PredictResponse {
    pub model: String,
    pub version: String,
    pub predictions: Vec<HashMap<String, f32>>,
}

#[derive(Debug, Serialize)]
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

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/models", get(list_models))
        .route("/models/{model}", get(get_model))
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

fn resolve_model(
    reg: &ModelRegistry,
    model: &str,
    version: Option<&str>,
    fallback_version: Option<&str>,
) -> Result<ResolvedModel, ApiError> {
    reg.resolve(model, version, fallback_version)
        .ok_or_else(|| ApiError {
            code: "REGISTRY_ERROR".to_string(),
            message: match (version, fallback_version) {
                (Some(v), Some(fb)) => format!(
                    "model '{}' version '{}' not found and fallback version '{}' is unavailable",
                    model, v, fb
                ),
                (Some(v), None) => format!("model '{}' version '{}' not found", model, v),
                (None, _) => format!("model '{}' not found", model),
            },
            request_id: None,
            model_id: Some(model.to_string()),
            details: Some(serde_json::json!({
                "requested_version": version,
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
    let fallback_version = req.fallback_version;
    let features = req.features;
    let batch_size = features.len();

    let resolved = resolve_model(
        &reg,
        &model,
        requested_version.as_deref(),
        fallback_version.as_deref(),
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
    let fallback_version = req.fallback_version;
    let user = req.user;
    let items = req.items;
    let batch_size = items.len();

    let resolved = resolve_model(
        &reg,
        &model,
        requested_version.as_deref(),
        fallback_version.as_deref(),
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
