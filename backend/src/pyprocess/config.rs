use serde::{Deserialize, Serialize};
use serde_yaml;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct AugmentationConfig {
    pub version: f32,
    pub augmentations: Augmentations,
    pub defaults: Defaults,
    pub image: ImageConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Augmentations {
    pub dwt: AugmentationSpec,
    pub dct: AugmentationSpec,
    pub sobel: AugmentationSpec,
    pub grayscale: AugmentationSpec,
    pub hog: AugmentationSpec,
    pub fft: AugmentationSpec,
    pub lbp: AugmentationSpec,
    pub ltp: AugmentationSpec,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AugmentationSpec {
    pub enabled: bool,
    pub params: HashMap<String, serde_yaml::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Defaults {
    pub normalization: NormalizationConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NormalizationConfig {
    pub epsilon: f32,
    pub min_max: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageConfig {
    pub size: Vec<u32>,
    pub channels: u32,
    pub preprocessing: PreprocessingConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub resize_method: String,
}

impl AugmentationConfig {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let manifest_dir =
            std::env::var("CARGO_MANIFEST_DIR").map_err(|_| "Failed to get manifest directory")?;
        let config_path = format!("{}/../config/augmentations.yaml", manifest_dir);
        let config_str = std::fs::read_to_string(config_path)?;
        let config: AugmentationConfig = serde_yaml::from_str(&config_str)?;
        Ok(config)
    }

    pub fn to_channel_augmentation(&self) -> super::augmentations::ChannelAugmentation {
        super::augmentations::ChannelAugmentation {
            add_dwt: self.augmentations.dwt.enabled,
            add_dct: self.augmentations.dct.enabled,
            add_sobel: self.augmentations.sobel.enabled,
            add_gray: self.augmentations.grayscale.enabled,
            add_hog: self.augmentations.hog.enabled,
            add_fft: self.augmentations.fft.enabled,
            add_lbp: self.augmentations.lbp.enabled,
            add_ltp: self.augmentations.ltp.enabled,
            hog_orientations: self
                .augmentations
                .hog
                .params
                .get("orientations")
                .and_then(|v| v.as_u64())
                .unwrap_or(9) as usize,
            hog_pixels_per_cell: {
                let pixels = self
                    .augmentations
                    .hog
                    .params
                    .get("pixels_per_cell")
                    .and_then(|v| v.as_sequence())
                    .map(|seq| {
                        (
                            seq[0].as_u64().unwrap_or(8) as usize,
                            seq[1].as_u64().unwrap_or(8) as usize,
                        )
                    })
                    .unwrap_or((8, 8));
                pixels
            },
            lbp_n_points: self
                .augmentations
                .lbp
                .params
                .get("n_points")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as usize,
            lbp_radius: self
                .augmentations
                .lbp
                .params
                .get("radius")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize,
            ltp_threshold: self
                .augmentations
                .ltp
                .params
                .get("threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32,
        }
    }
}
