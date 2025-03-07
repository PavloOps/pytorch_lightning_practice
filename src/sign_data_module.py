import gc
import hashlib
import logging
import os
import subprocess

import pandas as pd
import torch
import torch.utils.data as data
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%b-%d %H:%M:%S")
logger = logging.getLogger(__name__)


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
    def __init__(self, config, normalize=None, normalize_and_augmentation=None):
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
