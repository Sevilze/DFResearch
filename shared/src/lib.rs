use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct InferenceRequest {
    pub image_data: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct InferenceResponse {
    pub predictions: Vec<f32>,
    pub class_labels: Vec<String>,
    pub is_ai: bool,
    pub confidence: f32,
}