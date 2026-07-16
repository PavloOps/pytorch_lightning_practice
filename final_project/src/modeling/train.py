import logging
import os
import sys
import warnings
from getpass import getpass
from pathlib import Path

import click
import torch
from dotenv import load_dotenv
from lightning import seed_everything
from torchview import draw_graph

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
sys.path.append(str(SRC_DIR))

from config import CFG  # noqa: E402
from convolutional_network import Food101ConvNeXt  # noqa: E402
from dataset import Food101DataModule  # noqa: E402
from modeling.error_analysis import Food101ErrorAnalyzer  # noqa: E402
from modeling.trainer import create_trainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


warnings.filterwarnings(
    "ignore",
    message=r".*LeafSpec.*deprecated.*",
    category=FutureWarning,
)


def setup_clearml_environment():
    os.environ["CLEARML_WEB_HOST"] = "https://app.clear.ml"
    os.environ["CLEARML_API_HOST"] = "https://api.clear.ml"
    os.environ["CLEARML_FILES_HOST"] = "https://files.clear.ml"

    if not os.getenv("CLEARML_API_ACCESS_KEY"):
        os.environ["CLEARML_API_ACCESS_KEY"] = getpass(prompt="Enter ClearML access key: ")

    if not os.getenv("CLEARML_API_SECRET_KEY"):
        os.environ["CLEARML_API_SECRET_KEY"] = getpass(prompt="Enter ClearML secret key: ")


class Food101TrainingPipeline:
    def __init__(
        self,
        config: CFG,
        weights_path: str,
        onnx_path: str | None = None,
        fast_dev_run: bool = False,
    ):
        self.config = config
        self.weights_path = self.resolve_project_path(weights_path)
        self.onnx_path = self.resolve_project_path(onnx_path) if onnx_path is not None else None
        self.fast_dev_run = fast_dev_run

    @staticmethod
    def resolve_project_path(path):
        path = Path(path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def visualize_network(self):
        model = Food101ConvNeXt(config=self.config)
        figures_dir = Path(self.config.data.figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        graph_path = figures_dir / "convnext_tiny_food101_graph"

        model_graph = draw_graph(
            model,
            input_size=(1, 3, self.config.data.image_size, self.config.data.image_size),
            expand_nested=True,
            depth=3,
        )
        model_graph.visual_graph.format = "png"
        rendered_path = model_graph.visual_graph.render(
            str(graph_path),
            format="png",
            cleanup=True,
        )
        logger.info("Network graph is saved to %s.", Path(rendered_path))

    def run_training(self):
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)

        datamodule = Food101DataModule(config=self.config)
        model = Food101ConvNeXt(config=self.config)
        trainer = create_trainer(
            config=self.config,
            checkpoint_dir=self.weights_path.parent,
            fast_dev_run=self.fast_dev_run,
        )

        if self.fast_dev_run:
            logger.info("Start Food-101 fast development run.")
        else:
            logger.info("Start Food-101 training.")

        trainer.fit(model=model, datamodule=datamodule)
        best_model = model if self.fast_dev_run else self.load_best_model(model, trainer)

        if not self.fast_dev_run:
            logger.info("Start Food-101 validation error analysis.")
            Food101ErrorAnalyzer(config=self.config, datamodule=datamodule).run(best_model)

        if self.fast_dev_run:
            logger.info("Start Food-101 test step for fast development run.")
        else:
            logger.info("Start Food-101 testing.")

        test_metrics = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=None if self.fast_dev_run else "best",
        )
        logger.info("Test metrics: %s", test_metrics)

        trainer.save_checkpoint(self.weights_path)
        logger.info("Final checkpoint is saved to %s.", self.weights_path)

        if self.onnx_path is not None:
            self.export_model_to_onnx(best_model)

    def load_best_model(self, model, trainer):
        best_model_path = trainer.checkpoint_callback.best_model_path

        if not best_model_path:
            return model

        logger.info("Load best model checkpoint for error analysis: %s.", best_model_path)
        best_model = Food101ConvNeXt.load_from_checkpoint(best_model_path, config=self.config)
        best_model.to(model.device)
        return best_model

    def export_model_to_onnx(self, model):
        if self.onnx_path is None:
            raise ValueError("ONNX path is not defined.")

        onnx_path = self.onnx_path
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        model = model.cpu().eval()
        example_input = torch.randn(
            1,
            3,
            self.config.data.image_size,
            self.config.data.image_size,
        )

        torch.onnx.export(
            model,
            (example_input,),
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        logger.info("ONNX model is saved to %s.", onnx_path)


def update_config_from_cli(config, data_dir, lr, hard_cases_dir):
    if data_dir is not None:
        config.data.data_dir = data_dir
    if lr is not None:
        config.training.lr = lr
    if hard_cases_dir is not None:
        hard_cases_path = Path(hard_cases_dir)
        if not hard_cases_path.is_absolute():
            hard_cases_path = PROJECT_ROOT / hard_cases_path
        config.data.hard_cases_dir = str(hard_cases_path)
        config.data.hard_cases_manifest_path = str(hard_cases_path.parent / "hard_cases_manifest.csv")
    return config


@click.command()
@click.option(
    "--data_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Path to Food-101 raw data directory.",
)
@click.option(
    "--lr",
    type=float,
    default=None,
    help="Learning rate for classifier head.",
)
@click.option(
    "--weights_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Path where the final Lightning checkpoint will be saved.",
)
@click.option(
    "--onnx_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional path where the trained model will be exported in ONNX format.",
)
@click.option(
    "--fast_dev_run",
    is_flag=True,
    help="Run a single train/val/test batch for a smoke check.",
)
@click.option(
    "--visualize_network",
    is_flag=True,
    help="Save model graph PNG locally without ClearML connection.",
)
@click.option(
    "--hard_cases_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Directory where hard validation samples will be saved for later debug runs.",
)
def cli(data_dir, lr, weights_path, onnx_path, fast_dev_run, visualize_network, hard_cases_dir):
    load_dotenv(PROJECT_ROOT / ".env")

    config = update_config_from_cli(CFG(), data_dir, lr, hard_cases_dir)
    seed_everything(config.general.seed)

    pipeline = Food101TrainingPipeline(
        config=config,
        weights_path=weights_path,
        onnx_path=onnx_path,
        fast_dev_run=fast_dev_run,
    )

    if visualize_network:
        pipeline.visualize_network()
        return

    setup_clearml_environment()
    pipeline.run_training()


if __name__ == "__main__":
    cli()
