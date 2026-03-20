import gc
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from final_project.src.config import CFG
from final_project.src.data_module.two_tower_dataset import (
    NumericStats,
    TwoTowerDataset,
)
from lightning import LightningDataModule
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TwoTowerDataModule(LightningDataModule):
    def __init__(self, config: CFG) -> None:
        super().__init__()
        self.config = config

        self.train_dataset: Optional[TwoTowerDataset] = None
        self.val_dataset: Optional[TwoTowerDataset] = None
        self.test_dataset: Optional[TwoTowerDataset] = None

        self.category_maps: Optional[dict[str, dict[str, int]]] = None
        self.numeric_stats: Optional[dict[str, NumericStats]] = None

    def _csv_path(self, file_name: Optional[str]) -> Optional[Path]:
        data_config = self.config.data
        if not file_name:
            return None
        return Path(data_config.data_dir) / file_name

    def _dataloader_kwargs(self) -> dict[str, object]:
        data_config = self.config.data
        return {
            "batch_size": data_config.batch_size,
            "num_workers": data_config.num_workers,
            "pin_memory": data_config.pin_memory,
            "persistent_workers": (
                data_config.persistent_workers if data_config.num_workers > 0 else False
            ),
        }

    def prepare_data(self) -> None:
        train_csv_path = self._csv_path(self.config.data.train_file_name)
        val_csv_path = self._csv_path(self.config.data.val_file_name)
        test_csv_path = self._csv_path(self.config.data.test_file_name)

        if train_csv_path is None or not train_csv_path.exists():
            raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")
        if val_csv_path and not val_csv_path.exists():
            raise FileNotFoundError(f"Val CSV not found: {val_csv_path}")
        if test_csv_path and not test_csv_path.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

        logger.info("Data files are available.")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            if self.train_dataset is None:
                self._setup_fit_datasets()

        if stage in ("test", None):
            if self.test_dataset is None:
                self._setup_test_dataset()

    def _setup_fit_datasets(self) -> None:
        train_csv_path = self._csv_path(self.config.data.train_file_name)
        val_csv_path = self._csv_path(self.config.data.val_file_name)

        if train_csv_path is None:
            raise FileNotFoundError("Train CSV path is not configured.")
        if val_csv_path is None:
            raise FileNotFoundError("Val CSV path is not configured.")

        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)

        self.train_dataset = TwoTowerDataset(
            frame=train_df,
            fit=True,
            **{
                "user_cat_cols": list(self.config.data.user_cat_cols),
                "item_cat_cols": list(self.config.data.item_cat_cols),
                "user_num_cols": list(self.config.data.user_num_cols),
                "item_num_cols": list(self.config.data.item_num_cols),
                "label_col": self.config.data.label_col,
                "add_month_feature": self.config.data.add_month_feature,
            },
        )
        self.category_maps = self.train_dataset.category_maps
        self.numeric_stats = self.train_dataset.numeric_stats

        self.val_dataset = TwoTowerDataset(
            frame=val_df,
            user_cat_cols=self.train_dataset.user_cat_cols,
            item_cat_cols=self.train_dataset.item_cat_cols,
            user_num_cols=self.train_dataset.user_num_cols,
            item_num_cols=self.train_dataset.item_num_cols,
            fit=False,
            category_maps=self.category_maps,
            numeric_stats=self.numeric_stats,
            label_col=self.config.data.label_col,
            add_month_feature=self.config.data.add_month_feature,
        )

        logger.info(
            "Fit datasets are ready: train=%d, val=%d",
            len(self.train_dataset),
            len(self.val_dataset),
        )

    def _setup_test_dataset(self) -> None:
        test_csv_path = self._csv_path(self.config.data.test_file_name)

        if self.train_dataset is None:
            self._setup_fit_datasets()
        if test_csv_path is None:
            raise FileNotFoundError("Test CSV path is not configured.")

        test_df = pd.read_csv(test_csv_path)

        self.test_dataset = TwoTowerDataset(
            frame=test_df,
            user_cat_cols=self.train_dataset.user_cat_cols,
            item_cat_cols=self.train_dataset.item_cat_cols,
            user_num_cols=self.train_dataset.user_num_cols,
            item_num_cols=self.train_dataset.item_num_cols,
            fit=False,
            category_maps=self.category_maps,
            numeric_stats=self.numeric_stats,
            label_col=self.config.data.label_col,
            add_month_feature=self.config.data.add_month_feature,
        )

        logger.info("Test dataset is ready: test=%d", len(self.test_dataset))

    @property
    def cardinalities(self) -> dict[str, int]:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before reading cardinalities.")
        return self.train_dataset.cardinalities

    @property
    def user_cat_feature_names(self) -> list[str]:
        if self.train_dataset is None:
            return []
        return self.train_dataset.user_cat_cols

    @property
    def item_cat_feature_names(self) -> list[str]:
        if self.train_dataset is None:
            return []
        return self.train_dataset.item_cat_cols

    @property
    def user_num_feature_names(self) -> list[str]:
        if self.train_dataset is None:
            return []
        return self.train_dataset.user_num_cols

    @property
    def item_num_feature_names(self) -> list[str]:
        if self.train_dataset is None:
            return []
        return self.train_dataset.item_num_cols

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not initialized. Call setup('fit').")
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=self.config.data.drop_last,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Val dataset is not initialized. Call setup('fit').")
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            drop_last=False,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not initialized. Call setup('test').")
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            drop_last=False,
            **self._dataloader_kwargs(),
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.category_maps = None
        self.numeric_stats = None
        gc.collect()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    cfg = CFG()
    cfg.data.batch_size = 16
    cfg.data.num_workers = 0
    cfg.data.data_dir = str(project_root / "data")
    cfg.data.train_file_name = "train_example.csv"

    dm = TwoTowerDataModule(config=cfg)
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    train_batch = next(iter(dm.train_dataloader()))
    print("Train batch keys:", sorted(train_batch.keys()))
    for k, v in train_batch.items():
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    print("Cardinalities:", dm.cardinalities)
