import gc
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from final_project.data.prepare_two_tower_dataset import (
    build_train_dataset,
    build_validation_dataset,
    normalize_raw_frame,
)
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
        path = Path(file_name)
        if path.is_absolute():
            return path
        return Path(data_config.data_dir) / path

    def _prepared_csv_path(self, csv_path: Path) -> Path:
        return csv_path.with_name(f"{csv_path.stem}_prepared{csv_path.suffix}")

    def _required_prepared_columns(self) -> set[str]:
        data_config = self.config.data
        return {
            data_config.label_col,
            *data_config.user_cat_cols,
            *data_config.item_cat_cols,
            *data_config.user_num_cols,
            *data_config.item_num_cols,
        }

    def _is_prepared_frame(self, frame: pd.DataFrame) -> bool:
        return self._required_prepared_columns().issubset(frame.columns)

    def _resolve_fit_input_paths(self) -> tuple[Path, Path]:
        train_csv_path = self._csv_path(self.config.data.train_file_name)
        val_csv_path = self._csv_path(self.config.data.val_file_name)
        if train_csv_path is None:
            raise FileNotFoundError("Train CSV path is not configured.")
        if val_csv_path is None:
            raise FileNotFoundError("Val CSV path is not configured.")
        return train_csv_path, val_csv_path

    def _resolve_fit_dataset_paths(self) -> tuple[Path, Path]:
        train_input_path, val_input_path = self._resolve_fit_input_paths()
        prepared_train_path = self._prepared_csv_path(train_input_path)
        prepared_val_path = self._prepared_csv_path(val_input_path)

        train_candidate_path = (
            train_input_path
            if train_input_path == prepared_train_path or prepared_train_path.exists()
            else train_input_path
        )
        val_candidate_path = (
            val_input_path
            if val_input_path == prepared_val_path or prepared_val_path.exists()
            else val_input_path
        )

        if prepared_train_path.exists():
            train_candidate_path = prepared_train_path
        if prepared_val_path.exists():
            val_candidate_path = prepared_val_path
        return train_candidate_path, val_candidate_path

    def _prepare_fit_data(self, train_csv_path: Path, val_csv_path: Path) -> tuple[Path, Path]:
        logger.info("Reading fit inputs: train=%s val=%s", train_csv_path, val_csv_path)
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        logger.info(
            "Loaded fit inputs: train_shape=%s val_shape=%s",
            train_df.shape,
            val_df.shape,
        )

        train_is_prepared = self._is_prepared_frame(train_df)
        val_is_prepared = self._is_prepared_frame(val_df)
        logger.info(
            "Input format detection: train_prepared=%s val_prepared=%s",
            train_is_prepared,
            val_is_prepared,
        )

        if train_is_prepared and val_is_prepared:
            logger.info("Prepared train/val datasets already provided.")
            return train_csv_path, val_csv_path

        if train_is_prepared != val_is_prepared:
            raise ValueError(
                "Train/val inputs must both be prepared datasets or both be raw datasets."
            )

        prepared_train_path = self._prepared_csv_path(train_csv_path)
        prepared_val_path = self._prepared_csv_path(val_csv_path)
        if prepared_train_path.exists() and prepared_val_path.exists():
            logger.info(
                "Using cached prepared datasets: train=%s val=%s",
                prepared_train_path,
                prepared_val_path,
            )
            return prepared_train_path, prepared_val_path

        logger.info("Detected raw train/val CSVs. Preparing supervised datasets on disk.")
        train_raw = normalize_raw_frame(train_df)
        val_raw = normalize_raw_frame(val_df)
        logger.info(
            "Normalized raw inputs: train_shape=%s val_shape=%s",
            train_raw.shape,
            val_raw.shape,
        )
        logger.info("Building prepared train dataset...")
        prepared_train_df = build_train_dataset(
            raw_df=train_raw,
            seed=self.config.general.seed,
        )
        logger.info("Prepared train dataset built: shape=%s", prepared_train_df.shape)
        logger.info("Building prepared validation dataset...")
        prepared_val_df = build_validation_dataset(
            history_df=train_raw,
            target_df=val_raw,
            seed=self.config.general.seed,
        )
        logger.info("Prepared validation dataset built: shape=%s", prepared_val_df.shape)
        prepared_train_df.to_csv(prepared_train_path, index=False)
        prepared_val_df.to_csv(prepared_val_path, index=False)
        logger.info(
            "Saved prepared datasets: train=%s val=%s",
            prepared_train_path,
            prepared_val_path,
        )
        return prepared_train_path, prepared_val_path

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
        train_csv_path, val_csv_path = self._resolve_fit_input_paths()
        test_csv_path = self._csv_path(self.config.data.test_file_name)
        logger.info(
            "prepare_data started: train=%s val=%s test=%s",
            train_csv_path,
            val_csv_path,
            test_csv_path,
        )

        if train_csv_path is None or not train_csv_path.exists():
            raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")
        if val_csv_path and not val_csv_path.exists():
            raise FileNotFoundError(f"Val CSV not found: {val_csv_path}")
        if test_csv_path and not test_csv_path.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

        self._prepare_fit_data(train_csv_path=train_csv_path, val_csv_path=val_csv_path)
        logger.info("Data files are available.")

    def setup(self, stage: Optional[str] = None) -> None:
        logger.info("setup called with stage=%s", stage)
        if stage in ("fit", None):
            if self.train_dataset is None:
                self._setup_fit_datasets()

        if stage in ("test", None):
            if self.test_dataset is None:
                self._setup_test_dataset()

    def _setup_fit_datasets(self) -> None:
        train_csv_path, val_csv_path = self._resolve_fit_dataset_paths()
        logger.info(
            "Setting up fit datasets from disk: train=%s val=%s",
            train_csv_path,
            val_csv_path,
        )

        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        logger.info(
            "Prepared data loaded into memory: train_shape=%s val_shape=%s",
            train_df.shape,
            val_df.shape,
        )

        logger.info("Building train dataset tensors...")
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
        logger.info("Train dataset tensors ready. Building validation dataset tensors...")

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

        logger.info("Setting up test dataset from disk: test=%s", test_csv_path)
        test_df = pd.read_csv(test_csv_path)
        logger.info("Prepared test data loaded into memory: test_shape=%s", test_df.shape)
        if not self._is_prepared_frame(test_df):
            raise ValueError(
                "Test CSV must already be prepared. Raw test preprocessing is not implemented."
            )

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

    @property
    def model_feature_config(self) -> dict[str, object]:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before reading model feature config.")
        return {
            "cardinalities": self.cardinalities,
            "user_cat_feature_names": self.user_cat_feature_names,
            "item_cat_feature_names": self.item_cat_feature_names,
            "user_num_dim": len(self.user_num_feature_names),
            "item_num_dim": len(self.item_num_feature_names),
        }

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
    cfg.data.train_file_name = "/home/pavloops/PycharmProjects/pytorch_lightning_practice/final_project/data/train_dataset.csv"
    cfg.data.val_file_name = "/home/pavloops/PycharmProjects/pytorch_lightning_practice/final_project/data/valid_dataset.csv"
    cfg.data.test_file_name = ""

    dm = TwoTowerDataModule(config=cfg)
    dm.prepare_data()
    dm.setup("fit")
    # dm.setup("test")

    val_batch = next(iter(dm.val_dataloader()))
    print("Train batch keys:", sorted(val_batch.keys()))
    for k, v in val_batch.items():
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    print("Cardinalities:", dm.cardinalities)
