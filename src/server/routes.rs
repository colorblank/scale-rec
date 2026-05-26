//! Axum 路由：/health, /models, /predict, /predict/broadcast。
use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::engine::InferenceEngine;
use super::registry::ModelRegistry;
use super::tracing::RequestTimer;

pub type AppState = Arc<ModelRegistry>;

/// 单行特征
pub type FeatureRow = HashMap<String, Value>;

#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    pub model: String,
    pub features: Vec<FeatureRow>,
}

#[derive(Debug, Deserialize)]
pub struct BroadcastRequest {
    pub model: String,
    pub user: FeatureRow,
    pub items: Vec<FeatureRow>,
}

#[derive(Debug, Serialize)]
pub struct PredictResponse {
    pub model: String,
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

fn map_predict_error(err: String, model_id: String) -> ApiError {
    if err.contains("column '") || err.contains("field '") || err.contains("value") {
        ApiError {
            code: "BAD_REQUEST".to_string(),
            message: err,
            request_id: None,
            model_id: Some(model_id),
            details: None,
        }
    } else if err.contains("Model:") {
        ApiError {
            code: "MODEL_ERROR".to_string(),
            message: err,
            request_id: None,
            model_id: Some(model_id),
            details: None,
        }
    } else {
        ApiError {
            code: "FEATURE_ERROR".to_string(),
            message: err,
            request_id: None,
            model_id: Some(model_id),
            details: None,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub models: Vec<String>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/models", get(list_models))
        .route("/predict", post(predict))
        .route("/predict/broadcast", post(predict_broadcast))
        .with_state(state)
}

async fn health(State(reg): State<AppState>) -> Json<Value> {
    Json(serde_json::json!({
        "status": "ok",
        "models": reg.list()
    }))
}

async fn list_models(State(reg): State<AppState>) -> Json<ModelListResponse> {
    Json(ModelListResponse { models: reg.list() })
}

async fn predict(
    State(reg): State<AppState>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, ApiError> {
    let mut timer = RequestTimer::new();

    let engine: Arc<InferenceEngine> = reg.get(&req.model).ok_or_else(|| ApiError {
        code: "REGISTRY_ERROR".to_string(),
        message: format!("model '{}' not found", req.model),
        request_id: None,
        model_id: Some(req.model.clone()),
        details: None,
    })?;

    let (result, metrics) = engine
        .predict(&req.features)
        .map_err(|e| map_predict_error(e, req.model.clone()))?;

    timer.record(&metrics);
    timer.finish("/predict", &req.model, req.features.len());

    Ok(Json(PredictResponse {
        model: req.model,
        predictions: result,
    }))
}

async fn predict_broadcast(
    State(reg): State<AppState>,
    Json(req): Json<BroadcastRequest>,
) -> Result<Json<PredictResponse>, ApiError> {
    let mut timer = RequestTimer::new();

    let engine: Arc<InferenceEngine> = reg.get(&req.model).ok_or_else(|| ApiError {
        code: "REGISTRY_ERROR".to_string(),
        message: format!("model '{}' not found", req.model),
        request_id: None,
        model_id: Some(req.model.clone()),
        details: None,
    })?;

    let (result, metrics) = engine
        .predict_broadcast(&req.user, &req.items)
        .map_err(|e| map_predict_error(e, req.model.clone()))?;

    timer.record(&metrics);
    timer.finish("/predict/broadcast", &req.model, req.items.len());

    Ok(Json(PredictResponse {
        model: req.model,
        predictions: result,
    }))
}
