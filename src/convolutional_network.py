import torch
import torchmetrics as tm
from lightning import LightningModule
from torch import nn

from src.custom_metric import FalseDiscoveryRate


def calc_out_size(img_size, kernel_size, stride=1, padding=1, dilation=1):
    out_size = (
        (img_size + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride
    ) + 1
    return int(out_size)


class MyConvNet(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.lr = config.training.lr
        self.train_fbeta = tm.FBetaScore(
            task="multiclass", num_classes=self.config.model.n_classes
        )
        self.valid_fbeta = tm.FBetaScore(
            task="multiclass", num_classes=self.config.model.n_classes
        )
        self.train_roc_auc = tm.AUROC(
            task="multiclass", num_classes=self.config.model.n_classes
        )
        self.valid_roc_auc = tm.AUROC(
            task="multiclass", num_classes=self.config.model.n_classes
        )
        self.train_fdr = FalseDiscoveryRate(
            task="multiclass", num_classes=self.config.model.n_classes
        )
        self.valid_fdr = FalseDiscoveryRate(
            task="multiclass", num_classes=self.config.model.n_classes
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=self.config.model.kernel_size_block1,
                padding=self.config.model.padding_block1,
                stride=self.config.model.stride,
                dilation=self.config.model.dilation,
            ),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=self.config.model.kernel_size_block2,
                padding=self.config.model.padding_block2,
                stride=self.config.model.stride,
                dilation=self.config.model.dilation,
            ),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )

        block1_out_size = (
            calc_out_size(
                config.model.image_size,
                config.model.kernel_size_block1,
                config.model.stride,
                config.model.padding_block1,
            )
            // 2
        )  # AvgPool2d(2)

        block2_out_size = (
            calc_out_size(
                block1_out_size,
                config.model.kernel_size_block2,
                config.model.stride,
                config.model.padding_block2,
            )
            // 2
        )  # AvgPool2d(2)

        self.lin1 = nn.Linear(
            in_features=16 * block2_out_size * block2_out_size, out_features=100
        )
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=self.config.training.dropout)
        self.lin2 = nn.Linear(100, self.config.model.n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view((x.shape[0], -1))
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)
        return x

    def basic_step(self, batch, step: str):
        current_data, target = batch
        prediction = self(current_data)

        metrics = {f"{step}/loss": self.criterion(prediction, target)}

        if step == "train":
            metrics[f"{step}/fbeta"] = self.train_fbeta(prediction, target)
            metrics[f"{step}/roc_auc"] = self.train_roc_auc(prediction, target)
            metrics[f"{step}/fdr"] = self.train_fdr(prediction, target)
        elif step == "valid":
            metrics[f"{step}/fbeta"] = self.valid_fbeta(prediction, target)
            metrics[f"{step}/roc_auc"] = self.valid_roc_auc(prediction, target)
            metrics[f"{step}/fdr"] = self.valid_fdr(prediction, target)

        self.log_dict(metrics, prog_bar=True, on_epoch=True, sync_dist=True)
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.basic_step(batch, "train")
        return metrics["train/loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self.basic_step(batch, "valid")
        return metrics["valid/loss"]

    def test_step(self, batch, batch_idx):
        metrics = self.basic_step(batch, batch_idx, "test")
        return metrics["test/loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.training.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
