use tch::{Device, Tensor, nn::ModuleT};
use tch::CModule;
use std::sync::Arc;
use std::sync::Mutex;

#[allow(dead_code)]
#[derive(Debug)]
pub enum InferenceError {
    PreprocessingError(String),
    ModelError(tch::TchError),
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
        let output_vec = output.
            to_kind(tch::Kind::Float).
            view([-1]).
            try_into().
            map_err(|e| InferenceError::ModelError(e))?;
        Ok(output_vec)
    }

    fn preprocess(&self, _image: &[u8]) -> Result<Tensor, InferenceError> {
        todo!()
    }
}