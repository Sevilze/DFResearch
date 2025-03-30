use tch::{Device, Tensor, nn::ModuleT};
use tch::CModule;
use std::fs;
use std::sync::Arc;
use std::sync::Mutex;
use image::io::Reader as ImageReader;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyBytes};
use pyo3::exceptions::PyException;

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
        let tensor = self.preprocess(image)?;
        let output = self.model.lock().unwrap().forward_t(&tensor, false);
        let output = output.softmax(-1, tch::Kind::Float);
        
        let output_vec = output
            .to_kind(tch::Kind::Float)
            .view([-1])
            .try_into()
            .map_err(|e| InferenceError::ModelError(e))?;
        
        Ok(output_vec)
    }

    fn preprocess(&self, image: &[u8]) -> Result<Tensor, InferenceError> {
        Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let path: &pyo3::types::PyList = sys.getattr("path")?.downcast()?;
            let abs_path = fs::canonicalize("../pyproject")
                .map_err(|e| InferenceError::PythonError(e.to_string()))?;
            path.insert(0, abs_path.to_str().unwrap())?;
    
            let dataloader = PyModule::import(py, "dfresearch.dataloader")?;
            let channel_aug_class = dataloader.getattr("ChannelAugmentation")?;
            let channel_aug_instance = channel_aug_class.call0()?;
            let py_image = PyBytes::new(py, image);
    
            let nested: Vec<f32> = channel_aug_instance
                .call_method1("process_image", (py_image,))?
                .extract()?;
            
            let num_channels = self.check_channels(&nested)?;
            let tensor = Tensor::f_from_slice(&nested)
                .map_err(|e| InferenceError::ModelError(e))?
                .view([1, num_channels as i64, 224, 224])
                .to_kind(tch::Kind::Float);
    
            Ok(tensor)
        })
    }

    fn check_channels(&self, tensor_data: &[f32]) -> Result<usize, InferenceError> {
        let expected_spatial: usize = 224 * 224;
        let total_elements = tensor_data.len();
    
        if total_elements % expected_spatial != 0 {
            return Err(InferenceError::PreprocessingError(format!(
                "Data size {} not divisible by {} (224x224). Channels: {}",
                total_elements,
                expected_spatial,
                total_elements as f32 / expected_spatial as f32
            )));
        }
    
        Ok(total_elements / expected_spatial)
    }
}
