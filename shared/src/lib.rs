use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct InferenceRequest{
    pub image_data: String,
}

#[derive(Serialize, Deserialize)]
pub struct InferenceResponse{
    pub predictions: Vec<f32>,
    pub class_labels: Vec<String>
}