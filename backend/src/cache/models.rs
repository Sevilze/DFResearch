use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use shared::ProcessingMode;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    // Cognito fields
    pub cognito_sub: String,
    pub cognito_access_token: Option<String>,
    pub cognito_refresh_token: Option<String>,
    // User fields
    pub email: String,
    pub name: String,
    pub picture_url: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageCacheEntry {
    pub id: Uuid,
    pub user_id: Uuid,
    pub image_hash: String,
    pub file_name: String,
    pub file_size: i64,
    pub mime_type: String,
    pub s3_key: String,
    pub s3_bucket: String,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCacheEntry {
    pub id: Uuid,
    pub user_id: Uuid,
    pub image_hash: String,
    pub processing_mode: String,
    pub predictions: serde_json::Value,
    pub is_ai: bool,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: i32,
}

impl User {
    pub fn update_last_login(&mut self) {
        self.last_login = Some(chrono::Utc::now());
        self.updated_at = chrono::Utc::now();
    }
}

impl ImageCacheEntry {
    pub fn new(
        user_id: Uuid,
        image_hash: String,
        file_name: String,
        file_size: i64,
        mime_type: String,
        s3_key: String,
        s3_bucket: String,
    ) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Uuid::new_v4(),
            user_id,
            image_hash,
            file_name,
            file_size,
            mime_type,
            s3_key,
            s3_bucket,
            created_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }
}

impl InferenceCacheEntry {
    pub fn from_inference_result(
        user_id: Uuid,
        image_hash: String,
        processing_mode: &ProcessingMode,
        predictions: &[f32],
        is_ai: bool,
        confidence: f32,
    ) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: Uuid::new_v4(),
            user_id,
            image_hash,
            processing_mode: format!("{:?}", processing_mode),
            predictions: serde_json::json!(predictions),
            is_ai,
            confidence,
            created_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }

    pub fn to_inference_response(&self) -> shared::InferenceResponse {
        let predictions: Vec<f32> =
            serde_json::from_value(self.predictions.clone()).unwrap_or_default();

        shared::InferenceResponse {
            predictions,
            class_labels: vec!["AI Generated".into(), "Human Created".into()],
            is_ai: self.is_ai,
            confidence: self.confidence,
            image_hash: Some(self.image_hash.clone()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheHistoryEntry {
    pub image_hash: String,
    pub file_name: String,
    pub file_size: i64,
    pub mime_type: String,
    pub s3_key: String,
    pub s3_bucket: String,
    pub image_created_at: DateTime<Utc>,
    pub inference_results: Vec<InferenceResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub processing_mode: String,
    pub predictions: serde_json::Value,
    pub is_ai: bool,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
}

impl CacheHistoryEntry {
    pub fn new(image_entry: ImageCacheEntry, inference_entries: Vec<InferenceCacheEntry>) -> Self {
        let inference_results = inference_entries
            .into_iter()
            .map(|entry| InferenceResult {
                processing_mode: entry.processing_mode,
                predictions: entry.predictions,
                is_ai: entry.is_ai,
                confidence: entry.confidence,
                created_at: entry.created_at,
            })
            .collect();

        Self {
            image_hash: image_entry.image_hash,
            file_name: image_entry.file_name,
            file_size: image_entry.file_size,
            mime_type: image_entry.mime_type,
            s3_key: image_entry.s3_key,
            s3_bucket: image_entry.s3_bucket,
            image_created_at: image_entry.created_at,
            inference_results,
        }
    }
}
