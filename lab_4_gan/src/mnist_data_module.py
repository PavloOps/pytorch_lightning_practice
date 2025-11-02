import gc
import logging
import os

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from config import CFG
from lightning import LightningDataModule

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MNISTDataset(data.Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        image, label = self.mnist_dataset[index]
        image = image.unsqueeze(0) if image.ndim == 2 else image  # (1, 28, 28)

        return image, label


class MNISTLightning(LightningDataModule):
    def __init__(self, config, transform=None):
        super().__init__()
        self.config = config
        self.dataset_file_path = os.path.join(
            self.config.data.data_dir, self.config.data.file_name
        )
        self.dataset = None
        self.transform = (
            transform
            if transform is not None
            else transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
        )

    def prepare_data(self):
        if not os.path.exists(self.dataset_file_path):
            logger.info("Processed dataset not found — downloading raw MNIST...")
            mnist = datasets.MNIST(
                root=self.config.data.data_dir, train=True, download=True
            )
            images = torch.stack(
                [transforms.ToTensor()(mnist[i][0]) for i in range(len(mnist))]
            )
            labels = torch.tensor([mnist[i][1] for i in range(len(mnist))])

            torch.save(
                {"images": images, "labels": labels}, str(self.dataset_file_path)
            )
            logger.info(f"Processed MNIST saved to {self.dataset_file_path}")
        else:
            logger.info("Processed dataset already exists — skipping download.")

    def setup(self, stage):
        if stage in ("fit", None):
            loaded_data = torch.load(str(self.dataset_file_path))
            images, labels = loaded_data["images"], loaded_data["labels"]
            self.dataset = data.TensorDataset(images, labels)
            logger.info("Dataset is loaded to RAM.")

    def train_dataloader(self):
        return data.DataLoader(
            self.dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True,
        )

    def teardown(self, stage=None):
        self.dataset = None
        gc.collect()


if __name__ == "__main__":
    cfg = CFG()
    dm = MNISTLightning(config=cfg)
    dm.prepare_data()

    dm.setup(stage="fit")
    loader = dm.train_dataloader()

    batch = next(iter(loader))
    images_, labels_ = batch

    print("✅ Check is done!")
    print(f"Image size: {images_.shape}")
    print(f"Batch size: {labels_.shape}")
    print(f"Image type: {images_.dtype}")
    print(labels_[:10])
