use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let path: &PyList = sys.getattr("path")?.downcast()?;
        path.insert(0, "../../pyproject/dfresearch")?;

        let dataloader = PyModule::import(py, "dataloader")?;
        let channel_aug_class = dataloader.getattr("ChannelAugmentation")?;
        let channel_aug_instance = channel_aug_class.call0()?;
        
        let dummy_img = py.None(); 
        let result = channel_aug_instance.call1((dummy_img,))?;
        println!("Python function output: {:?}", result); // Log the result
        
        println!("Result: {:?}", result);
        Ok(())
    })
}
