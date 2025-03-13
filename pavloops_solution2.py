import contextlib
import io
import logging
from dataclasses import asdict

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as transforms
from lightning import Trainer, seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torchview import draw_graph

from config import CFG
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
        dir_path=config.data.saved_models_dir, params=asdict(config.trainer)
    )
    trainer.fit(model, datamodule=dataset)
    dataset.teardown()


def make_one_picture_inference(config, dir_path, wanted_index):
    seed_everything(config.general.seed)
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
