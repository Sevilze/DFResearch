use tch::Tensor;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyBytes};
use crate::pyprocess::model::InferenceError;
use std::fs;

pub fn preprocess(image: &[u8]) -> Result<Tensor, InferenceError> {
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let path: &pyo3::types::PyList = sys.getattr("path")?.downcast()?;

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let pyproject_path = format!("{}/../pyproject", manifest_dir);
        let abs_path = fs::canonicalize(pyproject_path)
            .map_err(|e| InferenceError::PythonError(e.to_string()))?;
        path.insert(0, abs_path.to_str().unwrap())?;

        let dataloader = PyModule::import(py, "dfresearch.dataloader")?;
        let channel_aug_class = dataloader.getattr("ChannelAugmentation")?;
        let channel_aug_instance = channel_aug_class.call0()?;
        let py_image = PyBytes::new(py, image);

        let nested: Vec<f32> = channel_aug_instance
            .call_method1("process_image", (py_image,))?
            .extract()?;

        let num_channels = check_channels(&nested)?;
        let tensor = Tensor::f_from_slice(&nested)
            .map_err(|e| InferenceError::ModelError(e))?
            .view([1, num_channels as i64, 224, 224])
            .to_kind(tch::Kind::Float);

        Ok(tensor)
    })
}

pub fn check_channels(tensor_data: &[f32]) -> Result<usize, InferenceError> {
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
