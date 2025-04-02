use tch::{Device, Tensor, nn::ModuleT};
use tch::CModule;
use core::num;
use std::fs;
use std::sync::Arc;
use std::sync::Mutex;
use image::io::Reader as ImageReader;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyBytes};
use pyo3::exceptions::PyException;
use shared::InferenceResponse;
use crate::pyprocess::augmentations::preprocess;

#[allow(dead_code)]
#[derive(Debug)]
pub enum InferenceError {
    PreprocessingError(String),
    ModelError(tch::TchError),
    PythonError(String),
}

impl From<PyErr> for InferenceError {
    fn from(err: PyErr) -> Self {
        InferenceError::PythonError(err.to_string())
    }
}

impl From<pyo3::PyDowncastError<'_>> for InferenceError {
    fn from(err: pyo3::PyDowncastError<'_>) -> Self {
        InferenceError::PythonError(err.to_string())
    }
}

#[derive(Clone)]
pub struct Model {
    model: Arc<Mutex<CModule>>,
}

impl Model {
    pub fn new(model_path: &str) -> Self {
        let device = Device::cuda_if_available();
        let model = Arc::new(Mutex::new(tch::CModule::load_on_device(model_path, device).unwrap()));
        Self { model }
    }

    pub fn inference(&self, image: &[u8]) -> Result<Vec<f32>, InferenceError> {
        let tensor = preprocess(image)?;
        let output = self.model.lock().unwrap().forward_t(&tensor, false);
        let output = output.softmax(-1, tch::Kind::Float);
        let output_flat = output.to_kind(tch::Kind::Float).view([-1]);
        let num_elements = output_flat.size()[0] as usize;
        let mut output_vec = vec![0.0f32; num_elements];
        output_flat.copy_data(&mut output_vec, num_elements);
        Ok(output_vec)
    }

    pub fn calculate_result(&self, predictions: &[f32]) -> (bool, f32) {
        if predictions.is_empty() {
            return (false, 0.0);
        }

        let confidence = predictions[0] * 100.0;
        let is_ai = confidence > 50.0;
        (is_ai, confidence)
    }
}
