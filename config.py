from dataclasses import asdict, dataclass, field

from torch.cuda import is_available


@dataclass
class GeneralConfig:
    seed: int = 2025
    num_workers: int = 4


@dataclass
class TrainerConfig:
    max_epochs: int = 20
    accelerator: str = "gpu" if is_available() else "cpu"
    devices: int = 1
    log_every_n_steps: int = 10
    check_val_every_n_epoch: int = 2


@dataclass
class TrainingProcessConfig:
    val_size: float = 0.2
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 20
    dropout: float = 0.3
    weight_decay: float = 0.05


@dataclass
class AugmentationConfig:
    normalize_mean: float = 159.0
    normalize_std: float = 40.0
    random_horizontal_flip_p: float = 0.1
    random_rotation_degrees: tuple = (-180, 180)
    random_rotation_p: float = 0.2


@dataclass
class DataConfig:
    data_dir: str = "dataset"
    saved_models_dir: str = "saved_models"
    train_url: str = (
        "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_train.csv.zip"
    )
    test_url: str = (
        "https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_test.csv.zip"
    )
    train_hash: str = "4c2897f19fab2b0ae2a7e4fa82e969043315d9f3a1a9cc0948b576bf1189a7e5"
    test_hash: str = "0e9d67bae23e67f40728e0b63bf15ad4bd5175947b8a9fac5dd9f17ce133c47b"
    train_name: str = "sign_mnist_train.csv"
    test_name: str = "sign_mnist_test.csv"


@dataclass
class ModelConfig:
    n_classes: int = 25
    image_size: int = 28
    stride: int = 1
    dilation: int = 1
    kernel_size_block1: int = 3
    kernel_size_block2: int = 3
    padding_block1: int = 1
    padding_block2: int = 1


@dataclass
class CFG:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    training: TrainingProcessConfig = field(default_factory=TrainingProcessConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


if __name__ == "__main__":
    print(asdict(CFG()))
