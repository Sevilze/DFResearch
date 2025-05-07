import yaml
from dataclasses import dataclass
from typing import Dict, Any, List, Callable, TypeVar
import os

T = TypeVar('T')

@dataclass
class AugmentationSpec:
    enabled: bool
    params: Dict[str, Any]

@dataclass
class NormalizationConfig:
    epsilon: float
    min_max: bool

@dataclass
class PreprocessingConfig:
    resize_method: str
    center_crop: bool

@dataclass
class ImageConfig:
    size: List[int]
    channels: int
    preprocessing: PreprocessingConfig

@dataclass
class AugmentationConfig:
    version: float
    augmentations: Dict[str, AugmentationSpec]
    defaults: Dict[str, Any]
    image: ImageConfig

    @classmethod
    def load(cls) -> 'AugmentationConfig':
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'augmentations.yaml')
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            version=data['version'],
            augmentations={
                name: AugmentationSpec(spec['enabled'], spec['params'])
                for name, spec in data['augmentations'].items()
            },
            defaults=data['defaults'],
            image=ImageConfig(
                size=data['image']['size'],
                channels=data['image']['channels'],
                preprocessing=PreprocessingConfig(**data['image']['preprocessing'])
            )
        )

    def to_channel_augmentation(self, factory: Callable[..., T]) -> T:
        return factory(
            add_dwt=self.augmentations['dwt'].enabled,
            add_dct=self.augmentations['dct'].enabled,
            add_sobel=self.augmentations['sobel'].enabled,
            add_gray=self.augmentations['grayscale'].enabled,
            add_hog=self.augmentations['hog'].enabled,
            add_fft=self.augmentations['fft'].enabled,
            add_lbp=self.augmentations['lbp'].enabled,
            add_ltp=self.augmentations['ltp'].enabled,
            dwt_wavelet=self.augmentations['dwt'].params.get('python_wavelet', 'bior2.2'),
            hog_orientations=self.augmentations['hog'].params.get('orientations', 9),
            hog_pixels_per_cell=tuple(self.augmentations['hog'].params.get('pixels_per_cell', (8, 8))),
            lbp_n_points=self.augmentations['lbp'].params.get('n_points', 8),
            lbp_radius=self.augmentations['lbp'].params.get('radius', 1),
            ltp_threshold=self.augmentations['ltp'].params.get('threshold', 0.1),
        )
