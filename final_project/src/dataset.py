import gc
import hashlib
import json
import logging
import ssl
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import torch
from config import CFG
from lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import Food101
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2
from torchvision.utils import save_image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Food101Dataset(Dataset):
    def __init__(self, image_files, labels, classes, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = default_loader(self.image_files[index])
        label = self.labels[index]
        image_path = str(self.image_files[index])

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "image_path": image_path,
            "class_name": self.classes[label],
        }


class Food101DataModule(LightningDataModule):
    def __init__(self, config: CFG, train_transform=None, eval_transform=None):
        super().__init__()
        self.config = config
        self.data_config = config.data
        self.data_dir = Path(self.data_config.data_dir)
        self.external_dir = Path(self.data_config.external_dir)
        self.archive_path = self.external_dir / self.data_config.archive_name
        transform_config = config.transform

        self.train_transform = (
            train_transform
            if train_transform is not None
            else v2.Compose(
                [
                    v2.RandomResizedCrop(
                        size=(self.data_config.image_size, self.data_config.image_size),
                        scale=transform_config.random_resized_crop_scale,
                        antialias=transform_config.antialias,
                    ),
                    v2.RandomHorizontalFlip(p=transform_config.random_horizontal_flip_p),
                    v2.RandomRotation(degrees=transform_config.random_rotation_degrees),
                    v2.RandomAffine(**transform_config.random_affine_params),
                    v2.ColorJitter(**transform_config.color_jitter_params),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(**transform_config.normalize_params),
                    v2.RandomErasing(**transform_config.random_erasing_params),
                ]
            )
        )
        self.eval_transform = (
            eval_transform
            if eval_transform is not None
            else v2.Compose(
                [
                    v2.Resize(
                        size=(self.data_config.image_size + transform_config.eval_resize_offset),
                        antialias=transform_config.antialias,
                    ),
                    v2.CenterCrop(size=(self.data_config.image_size, self.data_config.image_size)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(**transform_config.normalize_params),
                ]
            )
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.classes = None
        self.class_to_idx = None

    @staticmethod
    def calculate_file_hash(file_path, algorithm):
        hash_func = hashlib.new(algorithm)

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def file_is_available(self, file_path):
        return (
            file_path.exists()
            and self.calculate_file_hash(file_path, self.data_config.archive_hash_algorithm)
            == self.data_config.archive_hash
        )

    def remove_broken_archive(self):
        if self.archive_path.exists() and not self.file_is_available(self.archive_path):
            logger.warning(
                "Food-101 archive exists but hash is invalid. Remove broken file: %s",
                self.archive_path,
            )
            self.archive_path.unlink()

    def food101_is_available(self):
        dataset_dir = self.data_dir / "food-101"
        return (
            (dataset_dir / "images").is_dir()
            and (dataset_dir / "meta" / "train.json").is_file()
            and (dataset_dir / "meta" / "test.json").is_file()
        )

    def save_response_to_archive(self, response):
        total_size = int(response.headers.get("Content-Length", 0))

        with open(self.archive_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading Food-101",
            ) as progress_bar:
                while chunk := response.read(8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))

    def download_archive(self):
        logger.info("Start Food-101 archive downloading.")

        try:
            with urllib.request.urlopen(self.data_config.archive_url) as response:
                self.save_response_to_archive(response)
        except (urllib.error.URLError, ssl.SSLError) as error:
            logger.warning(
                "Regular download failed because of SSL/network error: %s. "
                "Retrying with unverified SSL context; archive hash will be checked.",
                error,
            )
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(self.data_config.archive_url, context=context) as response:
                self.save_response_to_archive(response)

    def extract_archive(self):
        logger.info("Extract Food-101 archive to %s.", self.data_dir)

        with tarfile.open(self.archive_path, "r:gz") as archive:
            archive.extractall(path=self.data_dir)

    def get_food101_dataset_part(self, dataset_part):
        metadata_path = self.data_dir / "food-101" / "meta" / f"{dataset_part}.json"
        images_dir = self.data_dir / "food-101" / "images"

        with open(metadata_path) as f:
            metadata = json.loads(f.read())

        classes = sorted(metadata.keys())
        class_to_idx = dict(zip(classes, range(len(classes))))
        image_files = []
        labels = []

        for class_label, image_rel_paths in metadata.items():
            image_files.extend(
                images_dir.joinpath(*f"{image_rel_path}.jpg".split("/")) for image_rel_path in image_rel_paths
            )
            labels.extend([class_to_idx[class_label]] * len(image_rel_paths))

        return image_files, labels, classes, class_to_idx

    def prepare_data(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)

        if self.food101_is_available():
            logger.info("Food-101 dataset already exists in %s.", self.data_dir)
            return

        if not self.data_config.download:
            raise FileNotFoundError(f"Food-101 dataset not found in {self.data_dir}, " "and data.download is False.")

        self.remove_broken_archive()

        if not self.file_is_available(self.archive_path):
            self.download_archive()

        if not self.file_is_available(self.archive_path):
            raise RuntimeError("Food-101 archive hash mismatch after download: " f"{self.archive_path}")

        self.extract_archive()

        Food101(root=str(self.data_dir), split="train", download=False)
        Food101(root=str(self.data_dir), split="test", download=False)
        logger.info("Food-101 dataset is available in %s.", self.data_dir)

    def setup(self, stage=None):
        if stage in ("fit", None):
            image_files, labels, self.classes, self.class_to_idx = self.get_food101_dataset_part("train")

            val_len = int(len(image_files) * self.data_config.val_size)
            train_len = len(image_files) - val_len
            split_indices = torch.randperm(
                len(image_files),
                generator=torch.Generator().manual_seed(self.config.general.seed),
            )

            train_indices = split_indices[:train_len].tolist()
            val_indices = split_indices[train_len:].tolist()

            train_dataset = Food101Dataset(
                image_files=image_files,
                labels=labels,
                classes=self.classes,
                transform=self.train_transform,
            )
            val_dataset = Food101Dataset(
                image_files=image_files,
                labels=labels,
                classes=self.classes,
                transform=self.eval_transform,
            )

            self.train_dataset = Subset(train_dataset, train_indices)
            self.val_dataset = Subset(val_dataset, val_indices)
            logger.info(
                "Food-101 train/val split is ready: train=%s, val=%s.",
                train_len,
                val_len,
            )

        if stage in ("test", "predict", None):
            image_files, labels, classes, class_to_idx = self.get_food101_dataset_part("test")

            self.test_dataset = Food101Dataset(
                image_files=image_files,
                labels=labels,
                classes=classes,
                transform=self.eval_transform,
            )

            if self.classes is None:
                self.classes = classes
                self.class_to_idx = class_to_idx

            logger.info("Food-101 test split is ready: test=%s.", len(self.test_dataset))

    def make_dataloader(self, dataset, shuffle: bool):
        persistent_workers = self.data_config.persistent_workers and self.data_config.num_workers > 0
        pin_memory = self.data_config.pin_memory and torch.cuda.is_available()

        return DataLoader(
            dataset,
            batch_size=self.data_config.batch_size,
            shuffle=shuffle,
            num_workers=self.data_config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(self):
        return self.make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.make_dataloader(self.test_dataset, shuffle=False)

    def teardown(self, stage=None):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        gc.collect()


if __name__ == "__main__":
    cfg = CFG()
    seed_everything(cfg.general.seed)
    dm = Food101DataModule(config=cfg)
    dm.prepare_data()

    dm.setup(stage="fit")
    loader = dm.train_dataloader()

    batch = next(iter(loader))
    images_ = batch["image"]
    labels_ = batch["label"]

    logger.info("Check is done!")
    logger.info("Image size: %s", images_.shape)
    logger.info("Batch size: %s", labels_.shape)
    logger.info("Image type: %s", images_.dtype)
    logger.info("Labels sample: %s", labels_[:10])

    samples_dir = Path(cfg.data.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    mean = torch.tensor(cfg.transform.normalize_mean).view(3, 1, 1)
    std = torch.tensor(cfg.transform.normalize_std).view(3, 1, 1)
    sample_count = min(cfg.data.num_smoke_samples, images_.shape[0])

    for sample_idx in range(sample_count):
        sample_image = images_[sample_idx].cpu() * std + mean
        sample_image = sample_image.clamp(0, 1)
        sample_label = labels_[sample_idx].item()
        sample_class = batch["class_name"][sample_idx].replace("/", "_")
        sample_path = samples_dir / f"{sample_idx:02d}_{sample_label}_{sample_class}.png"
        save_image(sample_image, sample_path)

    logger.info("Saved %s sample images to %s.", sample_count, samples_dir)
