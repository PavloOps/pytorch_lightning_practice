from dataclasses import asdict, dataclass, field


@dataclass
class GeneralConfig:
    seed: int = 2025
    num_workers: int = 4


@dataclass
class TrainerConfig:
    max_epochs: int = 10
    accelerator: str = "gpu"
    devices: int = 1
    log_every_n_steps: int = 10
    check_val_every_n_epoch: int = 2


@dataclass
class TrainingProcessConfig:
    batch_size: int = 64
    lr: float = 2e-4
    noise_dim: int = 100
    betas: tuple[float, float] = (0.5, 0.999)
    debug_samples_epoch: int = 1


@dataclass
class DataConfig:
    data_dir: str = "../data"
    saved_models_dir: str = "saved_models"
    file_name: str = "dataset.pt"

@dataclass
class CFG:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    training: TrainingProcessConfig = field(default_factory=TrainingProcessConfig)
    data: DataConfig = field(default_factory=DataConfig)


if __name__ == "__main__":
    print(asdict(CFG()))
