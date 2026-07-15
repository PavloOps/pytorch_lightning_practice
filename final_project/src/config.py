from dataclasses import asdict, dataclass, field
from pathlib import Path

from torch.cuda import is_available

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GeneralConfig:
    seed: int = 2026
    project_name: str = "Машинное обучение с помощью ClearML и Pytorch Lighting"
    experiment_name: str = "Final Project"
    tags: tuple[str, ...] = ("CV", "Food101", "ConvNeXT")


@dataclass
class DataConfig:
    data_dir: str = str(PROJECT_ROOT / "data" / "raw")
    external_dir: str = str(PROJECT_ROOT / "data" / "external")
    samples_dir: str = str(PROJECT_ROOT / "data" / "samples")
    figures_dir: str = str(PROJECT_ROOT / "reports" / "figures")
    download: bool = True
    archive_url: str = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    archive_name: str = "food-101.tar.gz"
    archive_hash: str = "85eeb15f3717b99a5da872d97d918f87"
    archive_hash_algorithm: str = "md5"
    val_size: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    image_size: int = 224
    num_smoke_samples: int = 8


@dataclass
class TransformConfig:
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    random_resized_crop_scale: tuple[float, float] = (0.75, 1.0)
    random_horizontal_flip_p: float = 0.5
    random_rotation_degrees: int = 10
    random_affine_translate: tuple[float, float] = (0.05, 0.05)
    random_affine_scale: tuple[float, float] = (0.95, 1.05)
    color_jitter_brightness: float = 0.15
    color_jitter_contrast: float = 0.15
    color_jitter_saturation: float = 0.15
    random_erasing_p: float = 0.15
    random_erasing_scale: tuple[float, float] = (0.02, 0.12)
    random_erasing_ratio: tuple[float, float] = (0.3, 3.3)
    eval_resize_offset: int = 32
    antialias: bool = True

    @property
    def color_jitter_params(self):
        return {
            "brightness": self.color_jitter_brightness,
            "contrast": self.color_jitter_contrast,
            "saturation": self.color_jitter_saturation,
        }

    @property
    def normalize_params(self):
        return {
            "mean": self.normalize_mean,
            "std": self.normalize_std,
        }

    @property
    def random_affine_params(self):
        return {
            "degrees": 0,
            "translate": self.random_affine_translate,
            "scale": self.random_affine_scale,
        }

    @property
    def random_erasing_params(self):
        return {
            "p": self.random_erasing_p,
            "scale": self.random_erasing_scale,
            "ratio": self.random_erasing_ratio,
        }


@dataclass
class ModelConfig:
    model_name: str = "convnext_tiny"
    weights: str = "DEFAULT"
    weights_cache_dir: str | None = None
    num_classes: int = 101
    freeze_backbone: bool = True
    unfreeze_backbone_epoch: int | None = 2


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    backbone_lr: float = 3e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    monitor_metric: str = "val/macro_f1"
    monitor_mode: str = "max"
    early_stopping_patience: int = 3
    save_top_k: int = 3
    debug_samples_epoch: int = 1
    num_debug_samples: int = 4
    debug_top_k: int = 5
    gradcam_alpha: float = 0.45


@dataclass
class TrainerConfig:
    max_epochs: int = 10
    accelerator: str = "gpu" if is_available() else "cpu"
    devices: int = 1
    precision: str = "32-true"
    log_every_n_steps: int = 10
    check_val_every_n_epoch: int = 1
    progress_bar_refresh_rate: int = 10


@dataclass
class CFG:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


if __name__ == "__main__":
    print(asdict(CFG()))
