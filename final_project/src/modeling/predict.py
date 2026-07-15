import csv
import logging
import sys
from pathlib import Path

import click
import torch
from torchvision.datasets.folder import default_loader

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
sys.path.append(str(SRC_DIR))

from config import CFG  # noqa: E402
from convolutional_network import Food101ConvNeXt  # noqa: E402
from dataset import Food101DataModule  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Food101PredictionPipeline:
    def __init__(self, config: CFG, weights_path: str, data_dir: str, predictions_path: str):
        self.config = config
        self.weights_path = self.resolve_project_path(weights_path)
        self.data_path = self.resolve_project_path(data_dir)
        self.predictions_path = self.resolve_project_path(predictions_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def resolve_project_path(path):
        path = Path(path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def run(self):
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {self.weights_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Image path not found: {self.data_path}")

        datamodule = Food101DataModule(config=self.config)
        datamodule.prepare_data()
        datamodule.setup(stage="test")

        model = Food101ConvNeXt.load_from_checkpoint(
            checkpoint_path=str(self.weights_path),
            config=self.config,
        )
        model = model.to(self.device).eval()

        image_paths = self.collect_image_paths()
        predictions = self.predict_images(
            model=model,
            image_paths=image_paths,
            transform=datamodule.eval_transform,
            classes=datamodule.classes,
        )
        self.save_predictions(predictions)
        logger.info("Predictions are saved to %s.", self.predictions_path)

    def collect_image_paths(self):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if self.data_path.is_file():
            return [self.data_path]

        image_paths = sorted(path for path in self.data_path.rglob("*") if path.suffix.lower() in extensions)
        if not image_paths:
            raise ValueError(f"No image files found in {self.data_path}")

        return image_paths

    @torch.no_grad()
    def predict_images(self, model, image_paths, transform, classes):
        predictions = []

        for image_path in image_paths:
            image = default_loader(image_path)
            image = transform(image).unsqueeze(0).to(self.device)

            logits = model(image)
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            predicted_class = classes[prediction.item()]

            predictions.append(
                {
                    "image_path": str(image_path),
                    "prediction": predicted_class,
                    "confidence": confidence.item(),
                }
            )

            logger.info(
                "Prediction: %s | confidence=%.4f | image=%s",
                predicted_class,
                confidence.item(),
                image_path,
            )

        return predictions

    def save_predictions(self, predictions):
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.predictions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image_path", "prediction", "confidence"],
            )
            writer.writeheader()
            writer.writerows(predictions)


@click.command()
@click.option(
    "--weights_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to trained Lightning checkpoint.",
)
@click.option(
    "--data_dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to image file or directory with images.",
)
@click.option(
    "--predictions_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Path where prediction CSV will be saved.",
)
def cli(weights_path, data_dir, predictions_path):
    config = CFG()
    pipeline = Food101PredictionPipeline(
        config=config,
        weights_path=weights_path,
        data_dir=data_dir,
        predictions_path=predictions_path,
    )
    pipeline.run()


if __name__ == "__main__":
    cli()
