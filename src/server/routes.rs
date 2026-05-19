//! Axum 路由：/health, /models, /predict, /predict/broadcast。
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

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
pub struct ErrorResponse {
    pub error: String,
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
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut timer = RequestTimer::new();

    let engine: Arc<InferenceEngine> = reg.get(&req.model).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("model '{}' not found", req.model),
            }),
        )
    })?;

    let dag_start = Instant::now();
    let result = engine.predict(&req.features).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e }),
        )
    })?;
    timer.record_dag(dag_start.elapsed().as_micros() as u64);
    timer.finish(&req.model, req.features.len());

    Ok(Json(PredictResponse {
        model: req.model,
        predictions: result,
    }))
}

async fn predict_broadcast(
    State(reg): State<AppState>,
    Json(req): Json<BroadcastRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut timer = RequestTimer::new();

    let engine: Arc<InferenceEngine> = reg.get(&req.model).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("model '{}' not found", req.model),
            }),
        )
    })?;

    let dag_start = Instant::now();
    let result = engine
        .predict_broadcast(&req.user, &req.items)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })?;
    timer.record_dag(dag_start.elapsed().as_micros() as u64);
    timer.finish(&req.model, req.items.len());

    Ok(Json(PredictResponse {
        model: req.model,
        predictions: result,
    }))
}
