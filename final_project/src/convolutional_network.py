import logging
from dataclasses import asdict

import torch
import torchmetrics as tm
from config import CFG
from lightning import LightningModule
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

torch.set_float32_matmul_precision("medium")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Food101ConvNeXt(LightningModule):
    def __init__(self, config: CFG):
        super().__init__()
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.setup_weights_cache()

        self.num_classes = config.model.num_classes
        self.lr = config.training.lr
        self.backbone_lr = config.training.backbone_lr
        self.weight_decay = config.training.weight_decay

        self.model = self.build_model()
        self.backbone = self.model.features
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)

        self.train_accuracy = tm.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = tm.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = tm.Accuracy(task="multiclass", num_classes=self.num_classes)

        self.train_macro_f1 = tm.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_macro_f1 = tm.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_macro_f1 = tm.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

        self.val_top5_accuracy = tm.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=5)
        self.test_top5_accuracy = tm.Accuracy(task="multiclass", num_classes=self.num_classes, top_k=5)

    def build_model(self):
        weights = self.get_weights()
        model = convnext_tiny(weights=weights)

        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, self.num_classes)

        return model

    def get_weights(self):
        if self.config.model.weights is None:
            return None
        if self.config.model.weights == "DEFAULT":
            return ConvNeXt_Tiny_Weights.DEFAULT
        return ConvNeXt_Tiny_Weights[self.config.model.weights]

    def forward(self, x):
        return self.model(x)

    def basic_step(self, batch, step: str):
        images = batch["image"]
        targets = batch["label"]
        logits = self(images)
        loss = self.criterion(logits, targets)

        metrics = {f"{step}/loss": loss}

        if step == "train":
            self.train_accuracy.update(logits, targets)
            self.train_macro_f1.update(logits, targets)
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log(
                "train/accuracy",
                self.train_accuracy,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "train/macro_f1",
                self.train_macro_f1,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )

        elif step == "val":
            self.val_accuracy.update(logits, targets)
            self.val_macro_f1.update(logits, targets)
            self.val_top5_accuracy.update(logits, targets)
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(
                "val/accuracy",
                self.val_accuracy,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "val/macro_f1",
                self.val_macro_f1,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "val/top5_accuracy",
                self.val_top5_accuracy,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        elif step == "test":
            self.test_accuracy.update(logits, targets)
            self.test_macro_f1.update(logits, targets)
            self.test_top5_accuracy.update(logits, targets)
            self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(
                "test/accuracy",
                self.test_accuracy,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "test/macro_f1",
                self.test_macro_f1,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "test/top5_accuracy",
                self.test_top5_accuracy,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.basic_step(batch, "train")
        return metrics["train/loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self.basic_step(batch, "val")
        return metrics["val/loss"]

    def test_step(self, batch, batch_idx):
        metrics = self.basic_step(batch, "test")
        return metrics["test/loss"]

    def predict_step(self, batch, batch_idx):
        logits = self(batch["image"])
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

        return {
            "image_path": batch.get("image_path"),
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def configure_optimizers(self):
        head_parameters = list(self.model.classifier.parameters())

        if self.config.model.freeze_backbone:
            optimizer_parameters = [
                {
                    "params": head_parameters,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                }
            ]
        else:
            optimizer_parameters = [
                {
                    "params": self.model.features.parameters(),
                    "lr": self.backbone_lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": head_parameters,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
            ]

        optimizer = torch.optim.AdamW(optimizer_parameters)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.trainer.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def setup_weights_cache(self):
        if self.config.model.weights_cache_dir is not None:
            torch.hub.set_dir(self.config.model.weights_cache_dir)


if __name__ == "__main__":
    cfg = CFG()
    network = Food101ConvNeXt(config=cfg)
    curr_batch = {
        "image": torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size),
        "label": torch.tensor([0, 1]),
    }
    output = network(curr_batch["image"])
    loss = network.criterion(output, curr_batch["label"])

    logger.info("Check is done!")
    logger.info("Input size: %s", curr_batch["image"].shape)
    logger.info("Output size: %s", output.shape)
    logger.info("Loss: %.4f", loss.item())
