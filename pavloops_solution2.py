import contextlib
import gc
import hashlib
import io
import logging
import os
import subprocess
from dataclasses import dataclass, field

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from dotenv import load_dotenv
from lightning import (LightningDataModule, LightningModule, Trainer,
                       seed_everything)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from sklearn.model_selection import train_test_split
from torch import nn
from torchview import draw_graph

torch.set_float32_matmul_precision('medium')
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%b-%d %H:%M:%S")
logger = logging.getLogger(__name__)


@dataclass
class GeneralConfig:
    seed: int = 2025
    device: str = "gpu" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4


@dataclass
class TrainingConfig:
    val_size: float = 0.2
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 20
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
    data_dir: str = './raw_data'
    train_url: str = 'https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_train.csv.zip'
    test_url: str = 'https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/sign_mnist_test.csv.zip'
    train_hash: str = '4c2897f19fab2b0ae2a7e4fa82e969043315d9f3a1a9cc0948b576bf1189a7e5'
    test_hash: str = '0e9d67bae23e67f40728e0b63bf15ad4bd5175947b8a9fac5dd9f17ce133c47b'
    train_name: str = 'sign_mnist_train.csv'
    test_name: str = 'sign_mnist_test.csv'


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
    weights_file_name: str = 'model_weights.ckpt'


@dataclass
class CFG:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def calc_out_size(img_size, kernel_size, stride=1, padding=1, dilation=1):
    out_size = ((img_size + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride) + 1
    return int(out_size)


class MyConvNet(LightningModule):
    def __init__(self, config: CFG):
        super().__init__()
        self.config = config
        self.lr = config.training.lr

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=self.config.model.kernel_size_block1,
                padding=self.config.model.padding_block1,
                stride=self.config.model.stride,
                dilation=self.config.model.dilation
            ),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=self.config.model.kernel_size_block2,
                padding=self.config.model.padding_block2,
                stride=self.config.model.stride,
                dilation=self.config.model.dilation
            ),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU()
        )

        block1_out_size = calc_out_size(
            config.model.image_size,
            config.model.kernel_size_block1,
            config.model.stride,
            config.model.padding_block1) // 2    # AvgPool2d(2)

        block2_out_size = calc_out_size(
            block1_out_size,
            config.model.kernel_size_block2,
            config.model.stride,
            config.model.padding_block2) // 2    # AvgPool2d(2)

        self.lin1 = nn.Linear(in_features=16 * block2_out_size * block2_out_size, out_features=100)
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
        loss = self.criterion(prediction, target)
        loss_dict = {f"{step}/loss": loss}
        self.log_dict(loss_dict, prog_bar=True)
        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self.basic_step(batch, "train")
        return loss_dict["train/loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.basic_step(batch, "valid")
        return loss_dict["valid/loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.training.epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


class SignLanguageDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        item_label = self.df.iloc[index, 0]

        current_image = self.df.iloc[index, 1:].values.reshape(28, 28)
        current_image = torch.Tensor(current_image).unsqueeze(0)

        if self.transform is not None:
            current_image = self.transform(current_image)

        return current_image, item_label


class SignLanguageLightning(LightningDataModule):
    def __init__(self, config: CFG, normalize=None, normalize_and_augmentation=None):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.normalize = normalize
        self.normalize_and_augmentation = normalize_and_augmentation

    @staticmethod
    def _calculate_sha256(file_path):
        assert os.path.exists(file_path), f"File not found: {file_path}"

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _file_is_available(self, file_name, ideal_file_hash):
        return os.path.exists(os.path.join(self.config.data.data_dir, file_name)) \
            and self._calculate_sha256(os.path.join(self.config.data.data_dir, file_name)) == ideal_file_hash

    def _download_file(self, file_path, file_url):
        logger.info("Start files' downloading:")
        subprocess.run(["wget", "-O", file_path, file_url], check=True)
        subprocess.run(["unzip", file_path, "-d", self.config.data.data_dir], check=True)
        subprocess.run(["rm", file_path], check=True)

    def prepare_data(self):
        os.makedirs(self.config.data.data_dir, exist_ok=True)

        if not self._file_is_available(self.config.data.train_name, self.config.data.train_hash):
            self._download_file(
                file_path=os.path.join(self.config.data.data_dir, 'train.csv.zip'),
                file_url=self.config.data.train_url
            )
        else:
            logger.info('Train file already downloaded.')

        if not self._file_is_available(self.config.data.test_name, self.config.data.test_hash):
            self._download_file(
                file_path=os.path.join(self.config.data.data_dir, 'test.csv.zip'),
                file_url=self.config.data.test_url
            )
        else:
            logger.info('Test file already downloaded.')

    def _load_dataset_to_ram(self, df, mode):
        if mode == 'train':
            return SignLanguageDataset(df, transform=self.normalize_and_augmentation)
        else:
            return SignLanguageDataset(df, transform=self.normalize)

    def setup(self, stage):
        if stage in ('fit', 'train', None):
            raw_data = pd.read_csv(os.path.join(self.config.data.data_dir, self.config.data.train_name))

            train_data, val_data = train_test_split(
                raw_data,
                test_size=self.config.training.val_size,
                random_state=self.config.general.seed
            )

            self.train_dataset = self._load_dataset_to_ram(train_data, mode='train')
            self.val_dataset = self._load_dataset_to_ram(val_data, mode='val')
            logger.info("Train and validation are loaded to RAM.")
            del raw_data, train_data, val_data
            gc.collect()

        if stage in ('test', None) and self.test_dataset is None:
            raw_data = pd.read_csv(os.path.join(self.config.data.data_dir, self.config.data.test_name))
            self.test_dataset = self._load_dataset_to_ram(raw_data, mode='test')
            del raw_data
            gc.collect()
            logger.info("Test is loaded to RAM.")

    def _make_dataloader(self, needed_dataset, need_shuffle):
        return data.DataLoader(
            needed_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.general.num_workers,
            pin_memory=True,
            shuffle=need_shuffle
        )

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, need_shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset, need_shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataset, need_shuffle=False)

    def teardown(self, stage=None):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        gc.collect()


def show_image(image):
    image = image.squeeze()
    image = image * 40. + 159.
    image_numpy = image.detach().numpy()
    plt.imshow(image_numpy, interpolation='bicubic')
    plt.savefig('test_picture')


def visualize_network(model_to_visualize, picture_file_name):
    model_graph = draw_graph(model_to_visualize, input_size=(1, 1, 28, 28), expand_nested=True)
    model_graph.visual_graph.format = "png"
    model_graph.visual_graph.render(picture_file_name, format="png", cleanup=True)
    logger.info(model_to_visualize)


def run_experiment(config, need_dev_run=False):
    seed_everything(config.general.seed)

    dataset = SignLanguageLightning(
        config=config,
        normalize=transforms.Compose([
            transforms.Normalize(config.augmentation.normalize_mean, config.augmentation.normalize_std)]),
        normalize_and_augmentation=transforms.Compose([
            transforms.Normalize(config.augmentation.normalize_mean, config.augmentation.normalize_std),
            transforms.RandomHorizontalFlip(p=config.augmentation.random_horizontal_flip_p),
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=config.augmentation.random_rotation_degrees)],
                p=config.augmentation.random_rotation_p
            ),
        ])
    )
    model = MyConvNet(config)

    if need_dev_run:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                trainer = Trainer(fast_dev_run=True)
                trainer.fit(model, datamodule=dataset)
            logger.info("Тестовый прогон успешно пройден")
        except (MisconfigurationException, RuntimeError, ValueError) as e:
            logger.error(f"Тестовый прогон завершился с ошибкой: {e}")
            raise
        except Exception as e:
            logger.error(f"Произошла непредвиденная ошибка: {e}")
            raise

    trainer = Trainer(max_epochs=config.training.epochs, log_every_n_steps=1)
    trainer.fit(model, datamodule=dataset)
    trainer.save_checkpoint(config.model.weights_file_name)
    dataset.teardown()


def make_one_picture_inference(config, wanted_index):
    seed_everything(config.general.seed)

    restored_model = MyConvNet.load_from_checkpoint(config.model.weights_file_name, config=config)
    restored_model = restored_model.cpu()
    restored_model.eval()

    dataset = SignLanguageLightning(config=config)
    dataset.setup('test')
    inference_loader = dataset.test_dataloader()

    testiter = iter(inference_loader)
    img, label = next(testiter)
    pred = restored_model(img)
    logger.info(f'Fact: {label[wanted_index]}, Prediction: {(torch.argmax(pred[wanted_index], dim=0))}')
    show_image(img[wanted_index])


@click.command()
@click.option('--fast_dev_run', '-f', is_flag=True, help='Firstly, run in fast development mode.')
def cli(fast_dev_run):
    cfg = CFG()
    logger.info('Convolutional Neural Network Architecture is:')
    visualize_network(MyConvNet(cfg), "pavloops_myconvnet_graph")
    logger.info(f"Augmentation params are: {cfg.augmentation}")
    run_experiment(cfg, need_dev_run=fast_dev_run)
    make_one_picture_inference(config=cfg, wanted_index=12)
    logger.info("Process finished successfully.")


if __name__ == '__main__':
    cli()
