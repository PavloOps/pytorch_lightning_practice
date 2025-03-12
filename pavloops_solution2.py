import contextlib
import io
import logging
from dataclasses import asdict, dataclass, field

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as transforms
from lightning import Trainer, seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torchview import draw_graph

from src.convolutional_network import MyConvNet
from src.network_trainer import create_trainer, pick_best_model
from src.sign_data_module import SignLanguageLightning

torch.set_float32_matmul_precision("medium")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class GeneralConfig:
    seed: int = 2025
    num_workers: int = 4


@dataclass
class TrainerConfig:
    max_epochs: int = 20
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu"
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
    training: TrainingProcessConfig = field(default_factory=TrainingProcessConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def show_image(image):
    image = image.squeeze()
    image = image * 40.0 + 159.0
    image_numpy = image.detach().numpy()
    plt.imshow(image_numpy, interpolation="bicubic")
    plt.savefig("test_picture")


def visualize_network(model_to_visualize, picture_file_name):
    model_graph = draw_graph(
        model_to_visualize, input_size=(1, 1, 28, 28), expand_nested=True
    )
    model_graph.visual_graph.format = "png"
    model_graph.visual_graph.render(picture_file_name, format="png", cleanup=True)
    logger.info(model_to_visualize)


def simple_visualize_metrics(metrics_file_path):
    df = pd.read_csv(metrics_file_path)

    if df.empty:
        raise ValueError("Check for metrics dataframe existence.")

    metrics_to_plot = ["fbeta", "fdr", "roc_auc", "loss"]
    plt.figure(figsize=(14, 10))

    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 2, i)

        train_col = f"train/{metric}_epoch"
        valid_col = f"valid/{metric}"

        epochs_train = df[["epoch", train_col]].dropna()
        epochs_valid = df[["epoch", valid_col]].dropna()

        plt.plot(
            epochs_train["epoch"],
            epochs_train[train_col],
            marker="o",
            label=f"Train {metric}",
        )
        plt.plot(
            epochs_valid["epoch"],
            epochs_valid[valid_col],
            marker="s",
            label=f"Valid {metric}",
        )

        plt.title(f"{metric.upper()} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("pics/training_plot.png", dpi=300)
    plt.close()


def run_experiment(config, need_dev_run=True):
    seed_everything(config.general.seed)

    dataset = SignLanguageLightning(
        config=config,
        normalize=transforms.Compose(
            [
                transforms.Normalize(
                    config.augmentation.normalize_mean,
                    config.augmentation.normalize_std,
                )
            ]
        ),
        normalize_and_augmentation=transforms.Compose(
            [
                transforms.Normalize(
                    config.augmentation.normalize_mean,
                    config.augmentation.normalize_std,
                ),
                transforms.RandomHorizontalFlip(
                    p=config.augmentation.random_horizontal_flip_p
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomRotation(
                            degrees=config.augmentation.random_rotation_degrees
                        )
                    ],
                    p=config.augmentation.random_rotation_p,
                ),
            ]
        ),
    )
    model = MyConvNet(config)
    trainer_params = TrainerConfig()

    if need_dev_run:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                trainer = Trainer(fast_dev_run=True)
                trainer.fit(model, datamodule=dataset)
            logger.info("Тестовый прогон успешно пройден")
        except (MisconfigurationException, RuntimeError, ValueError) as e:
            logger.error(f"Тестовый прогон завершился с ошибкой: {e}")
            raise
        except Exception as e:
            logger.error(f"Произошла непредвиденная ошибка: {e}")
            raise

    trainer = create_trainer(
        dir_path=config.data.saved_models_dir, params=asdict(trainer_params)
    )
    trainer.fit(model, datamodule=dataset)
    dataset.teardown()


def make_one_picture_inference(config, dir_path, wanted_index):
    model_path = pick_best_model(dir_path)
    logger.info(f"Used model: {model_path}")
    restored_model = MyConvNet.load_from_checkpoint(model_path, config=config)
    restored_model = restored_model.cpu()
    restored_model.eval()

    dataset = SignLanguageLightning(config=config)
    dataset.setup("test")
    inference_loader = dataset.test_dataloader()

    testiter = iter(inference_loader)
    img, label = next(testiter)
    pred = restored_model(img)
    logger.info(
        f"Fact: {label[wanted_index]}, Prediction: {(torch.argmax(pred[wanted_index], dim=0))}"
    )
    show_image(img[wanted_index])


@click.command()
@click.option(
    "--fast_dev_run", "-f", is_flag=True, help="Firstly, run in fast development mode."
)
def cli(fast_dev_run):
    cfg = CFG()
    logger.info("Convolutional Neural Network Architecture is:")
    visualize_network(MyConvNet(cfg), "pavloops_myconvnet_graph")
    logger.info(f"Augmentation params are: {cfg.augmentation}")
    run_experiment(cfg, need_dev_run=fast_dev_run)
    make_one_picture_inference(config=cfg, dir_path="saved_models", wanted_index=12)
    simple_visualize_metrics("lightning_logs/MyConvNet/version_0/metrics.csv")
    logger.info("Process finished successfully.")


if __name__ == "__main__":
    cli()
