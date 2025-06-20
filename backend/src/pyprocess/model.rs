use super::augmentations::preprocess;
use log::error;
use shared::ProcessingMode;
use std::fmt;
use std::sync::{Arc, Mutex};
use tch::CModule;
use tch::{nn::ModuleT, Device};

#[derive(Debug)]
pub enum InferenceError {
    PreprocessingError(String),
    ModelError(tch::TchError),
    ModeNotSupported(String),
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferenceError::PreprocessingError(msg) => write!(f, "Preprocessing error: {}", msg),
            InferenceError::ModelError(err) => write!(f, "Model error: {}", err),
            InferenceError::ModeNotSupported(msg) => write!(f, "Mode not supported: {}", msg),
        }
    }
}

impl std::error::Error for InferenceError {}

#[derive(Clone)]
pub struct Model {
    intermediate_model: Option<Arc<Mutex<CModule>>>,
    resnet_model: Option<Arc<Mutex<CModule>>>,
    densenet_model: Option<Arc<Mutex<CModule>>>,
    regnet_model: Option<Arc<Mutex<CModule>>>,
    mode: ProcessingMode,
}

impl Model {
    pub fn new() -> Self {
        Self {
            intermediate_model: None,
            resnet_model: None,
            densenet_model: None,
            regnet_model: None,
            mode: ProcessingMode::IntermediateFusionEnsemble,
        }
    }

    pub fn load_intermediate(model_path: &str) -> Result<Self, InferenceError> {
        let device = Device::cuda_if_available();
        let intermediate_model = match CModule::load_on_device(model_path, device) {
            Ok(model) => Arc::new(Mutex::new(model)),
            Err(e) => {
                error!("Failed to load intermediate model: {:?}", e);
                return Err(InferenceError::ModelError(e));
            }
        };

        Ok(Self {
            intermediate_model: Some(intermediate_model),
            resnet_model: None,
            densenet_model: None,
            regnet_model: None,
            mode: ProcessingMode::IntermediateFusionEnsemble,
        })
    }

    pub fn load_late(
        resnet_path: &str,
        densenet_path: &str,
        regnet_path: &str,
    ) -> Result<Self, InferenceError> {
        let device = Device::cuda_if_available();

        let resnet_model = match CModule::load_on_device(resnet_path, device) {
            Ok(model) => Some(Arc::new(Mutex::new(model))),
            Err(e) => {
                error!("Failed to load ResNet model: {:?}", e);
                return Err(InferenceError::ModelError(e));
            }
        };

        let densenet_model = match CModule::load_on_device(densenet_path, device) {
            Ok(model) => Some(Arc::new(Mutex::new(model))),
            Err(e) => {
                error!("Failed to load DenseNet model: {:?}", e);
                return Err(InferenceError::ModelError(e));
            }
        };

        let regnet_model = match CModule::load_on_device(regnet_path, device) {
            Ok(model) => Some(Arc::new(Mutex::new(model))),
            Err(e) => {
                error!("Failed to load RegNet model: {:?}", e);
                return Err(InferenceError::ModelError(e));
            }
        };

        Ok(Self {
            intermediate_model: None,
            resnet_model,
            densenet_model,
            regnet_model,
            mode: ProcessingMode::LateFusionEnsemble,
        })
    }

    pub fn persistent_load(&mut self, mode: &ProcessingMode) -> Result<(), InferenceError> {
        match mode {
            ProcessingMode::IntermediateFusionEnsemble => {
                if self.intermediate_model.is_some() {
                    return Ok(());
                }

                let intermediate_model_path = Self::get_model_path(
                    "IntermediateFusionEnsemble/best_model/IntermediateFusionEnsemble_scripted.pt",
                )?;
                let model = Self::load_intermediate(&intermediate_model_path)?;

                self.intermediate_model = model.intermediate_model;
                self.mode = ProcessingMode::IntermediateFusionEnsemble;
                Ok(())
            }

            ProcessingMode::LateFusionEnsemble => {
                if self.resnet_model.is_some()
                    && self.densenet_model.is_some()
                    && self.regnet_model.is_some()
                {
                    return Ok(());
                }

                let resnet_path = Self::get_model_path(
                    "ResnetClassifier/best_model/ResnetClassifier_scripted.pt",
                )?;
                let densenet_path = Self::get_model_path(
                    "DensenetClassifier/best_model/DensenetClassifier_scripted.pt",
                )?;
                let regnet_path = Self::get_model_path(
                    "RegnetClassifier/best_model/RegnetClassifier_scripted.pt",
                )?;
                let model = Self::load_late(&resnet_path, &densenet_path, &regnet_path)?;

                self.resnet_model = model.resnet_model;
                self.densenet_model = model.densenet_model;
                self.regnet_model = model.regnet_model;
                self.intermediate_model = model.intermediate_model;
                self.mode = ProcessingMode::LateFusionEnsemble;
                Ok(())
            }
        }
    }

    fn get_model_path(relative_path: &str) -> Result<String, InferenceError> {
        let production_path = format!("/usr/src/app/pyproject/models/{}", relative_path);
        if std::path::Path::new(&production_path).exists() {
            return Ok(production_path);
        }

        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let dev_path = format!("{}/../pyproject/models/{}", manifest_dir, relative_path);
            if std::path::Path::new(&dev_path).exists() {
                return Ok(dev_path);
            }
        }

        Err(InferenceError::ModeNotSupported(format!(
            "Failed to locate model file: {}. Tried production path: {}",
            relative_path, production_path
        )))
    }

    pub fn inference(
        &mut self,
        image: &[u8],
        mode: &ProcessingMode,
    ) -> Result<Vec<f32>, InferenceError> {
        match mode {
            ProcessingMode::IntermediateFusionEnsemble => {
                let intermediate_model = self.intermediate_model.as_ref();
                let tensor = preprocess(image)?;
                let output = intermediate_model
                    .as_ref()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .forward_t(&tensor, false);
                let output = output.softmax(-1, tch::Kind::Float);
                let output_flat = output.to_kind(tch::Kind::Float).view([-1]);
                let num_elements = output_flat.size()[0] as usize;
                let mut output_vec = vec![0.0f32; num_elements];
                output_flat.copy_data(&mut output_vec, num_elements);
                Ok(output_vec)
            }

            ProcessingMode::LateFusionEnsemble => {
                let tensor = preprocess(image)?;
                let resnet_output = self
                    .resnet_model
                    .as_ref()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .forward_t(&tensor, false);
                let densenet_output = self
                    .densenet_model
                    .as_ref()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .forward_t(&tensor, false);
                let regnet_output = self
                    .regnet_model
                    .as_ref()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .forward_t(&tensor, false);

                let resnet_probs = resnet_output.softmax(-1, tch::Kind::Float);
                let densenet_probs = densenet_output.softmax(-1, tch::Kind::Float);
                let regnet_probs = regnet_output.softmax(-1, tch::Kind::Float);
                let combined = (&resnet_probs + &densenet_probs + &regnet_probs) / 3.0;

                let output_flat = combined.to_kind(tch::Kind::Float).view([-1]);
                let num_elements = output_flat.size()[0] as usize;
                let mut output_vec = vec![0.0f32; num_elements];
                output_flat.copy_data(&mut output_vec, num_elements);

                Ok(output_vec)
            }
        }
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
