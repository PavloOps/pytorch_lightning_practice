import gc
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from final_project.data.prepare_two_tower_dataset import (
    RAW_COLUMNS,
    OUTPUT_COLUMNS,
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

    @staticmethod
    def _path_looks_prepared(path: Path) -> bool:
        return path.stem.endswith("_prepared")

    @staticmethod
    def _prepared_parquet_path(input_path: Path) -> Path:
        return input_path.with_name(f"{input_path.stem}_prepared.parquet")

    @staticmethod
    def _prepared_csv_path(input_path: Path) -> Path:
        return input_path.with_name(f"{input_path.stem}_prepared.csv")

    @staticmethod
    def _csv_columns(path: Path) -> set[str]:
        return set(pd.read_csv(path, nrows=0).columns.tolist())

    @staticmethod
    def _parquet_columns(path: Path) -> set[str]:
        try:
            import pyarrow.parquet as pq
        except Exception:
            return set(pd.read_parquet(path).head(0).columns.tolist())
        return set(pq.ParquetFile(path).schema.names)

    def _read_columns(self, path: Path) -> set[str]:
        if path.suffix.lower() == ".parquet":
            return self._parquet_columns(path)
        return self._csv_columns(path)

    @staticmethod
    def _cast_frame_dtypes(
        frame: pd.DataFrame,
        dtype_map: dict[str, str],
    ) -> pd.DataFrame:
        for col, dtype in dtype_map.items():
            if col in frame.columns:
                frame[col] = frame[col].astype(dtype, copy=False)
        return frame

    @staticmethod
    def _raw_dtype_map() -> dict[str, str]:
        return {
            "user_id": "int64",
            "year_month": "int32",
            "item_id": "int64",
            "cnt": "float32",
            "avg_price": "float32",
            "brand": "category",
            "category": "category",
            "country": "category",
            "inn": "category",
            "owner": "category",
            "city": "category",
            "region": "category",
        }

    @staticmethod
    def _prepared_dtype_map() -> dict[str, str]:
        return {
            "user_id": "int64",
            "year_month": "int32",
            "region": "category",
            "city": "category",
            "item_id": "int64",
            "label": "int8",
            "avg_price": "float32",
            "brand": "category",
            "category": "category",
            "country": "category",
            "inn": "category",
            "owner": "category",
            "user_total_cnt": "float32",
            "user_unique_items": "int32",
        }

    def _read_raw_frame(self, path: Path) -> pd.DataFrame:
        dtype_map = self._raw_dtype_map()
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path, columns=RAW_COLUMNS)
            return self._cast_frame_dtypes(frame=frame, dtype_map=dtype_map)
        return pd.read_csv(path, usecols=RAW_COLUMNS, dtype=dtype_map)

    def _prepared_read_columns(self) -> list[str]:
        required = set(OUTPUT_COLUMNS)
        required.update(self._required_prepared_columns())
        required.update({"year_month"})
        return sorted(required)

    def _read_prepared_frame(self, path: Path) -> pd.DataFrame:
        columns = self._prepared_read_columns()
        dtype_map = self._prepared_dtype_map()
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path, columns=columns)
            return self._cast_frame_dtypes(frame=frame, dtype_map=dtype_map)
        return pd.read_csv(path, usecols=columns, dtype=dtype_map)

    def _save_prepared_frame(self, path: Path, frame: pd.DataFrame) -> Path:
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception as exc:
            fallback_path = path.with_suffix(".csv")
            logger.warning(
                "Failed to save parquet (%s). Falling back to CSV: %s",
                exc,
                fallback_path,
            )
            frame.to_csv(fallback_path, index=False)
            return fallback_path

    def _csv_path(self, file_name: Optional[str]) -> Optional[Path]:
        data_config = self.config.data
        if not file_name:
            return None
        path = Path(file_name)
        if path.is_absolute():
            return path
        return Path(data_config.data_dir) / path

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
        if self._path_looks_prepared(train_input_path) and self._path_looks_prepared(
            val_input_path
        ):
            train_parquet = train_input_path.with_suffix(".parquet")
            val_parquet = val_input_path.with_suffix(".parquet")
            if train_parquet.exists() and val_parquet.exists():
                return train_parquet, val_parquet
            return train_input_path, val_input_path

        train_prepared_parquet = self._prepared_parquet_path(train_input_path)
        val_prepared_parquet = self._prepared_parquet_path(val_input_path)
        if train_prepared_parquet.exists() and val_prepared_parquet.exists():
            return train_prepared_parquet, val_prepared_parquet

        train_prepared_csv = self._prepared_csv_path(train_input_path)
        val_prepared_csv = self._prepared_csv_path(val_input_path)
        if train_prepared_csv.exists() and val_prepared_csv.exists():
            return train_prepared_csv, val_prepared_csv

        return train_input_path, val_input_path

    def _prepare_fit_data(self, train_csv_path: Path, val_csv_path: Path) -> tuple[Path, Path]:
        logger.info("Reading fit inputs: train=%s val=%s", train_csv_path, val_csv_path)
        train_header = self._read_columns(train_csv_path)
        val_header = self._read_columns(val_csv_path)

        train_is_prepared = self._required_prepared_columns().issubset(train_header)
        val_is_prepared = self._required_prepared_columns().issubset(val_header)
        logger.info(
            "Input format detection: train_prepared=%s val_prepared=%s",
            train_is_prepared,
            val_is_prepared,
        )

        if train_is_prepared and val_is_prepared:
            if train_csv_path.suffix.lower() == ".csv" and val_csv_path.suffix.lower() == ".csv":
                train_parquet_path = train_csv_path.with_suffix(".parquet")
                val_parquet_path = val_csv_path.with_suffix(".parquet")
                if train_parquet_path.exists() and val_parquet_path.exists():
                    logger.info(
                        "Using cached parquet copies of prepared datasets: train=%s val=%s",
                        train_parquet_path,
                        val_parquet_path,
                    )
                    return train_parquet_path, val_parquet_path

                logger.info("Converting prepared CSV datasets to parquet for faster I/O.")
                train_prepared_df = self._read_prepared_frame(train_csv_path)
                train_output_path = self._save_prepared_frame(
                    path=train_parquet_path,
                    frame=train_prepared_df,
                )
                del train_prepared_df
                gc.collect()

                val_prepared_df = self._read_prepared_frame(val_csv_path)
                val_output_path = self._save_prepared_frame(
                    path=val_parquet_path,
                    frame=val_prepared_df,
                )
                del val_prepared_df
                gc.collect()
                return train_output_path, val_output_path

            logger.info("Prepared train/val datasets already provided.")
            return train_csv_path, val_csv_path

        if train_is_prepared != val_is_prepared:
            raise ValueError(
                "Train/val inputs must both be prepared datasets or both be raw datasets."
            )

        prepared_train_parquet_path = self._prepared_parquet_path(train_csv_path)
        prepared_val_parquet_path = self._prepared_parquet_path(val_csv_path)
        if prepared_train_parquet_path.exists() and prepared_val_parquet_path.exists():
            logger.info(
                "Using cached prepared datasets: train=%s val=%s",
                prepared_train_parquet_path,
                prepared_val_parquet_path,
            )
            return prepared_train_parquet_path, prepared_val_parquet_path

        prepared_train_csv_path = self._prepared_csv_path(train_csv_path)
        prepared_val_csv_path = self._prepared_csv_path(val_csv_path)
        if prepared_train_csv_path.exists() and prepared_val_csv_path.exists():
            logger.info(
                "Using cached prepared datasets: train=%s val=%s",
                prepared_train_csv_path,
                prepared_val_csv_path,
            )
            return prepared_train_csv_path, prepared_val_csv_path

        logger.info("Detected raw train/val CSVs. Preparing supervised datasets on disk.")
        train_df = self._read_raw_frame(train_csv_path)
        val_df = self._read_raw_frame(val_csv_path)
        logger.info(
            "Loaded raw fit inputs: train_shape=%s val_shape=%s",
            train_df.shape,
            val_df.shape,
        )
        train_raw = normalize_raw_frame(train_df)
        val_raw = normalize_raw_frame(val_df)
        del train_df
        del val_df
        gc.collect()
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
        prepared_train_output_path = self._save_prepared_frame(
            path=prepared_train_parquet_path,
            frame=prepared_train_df,
        )
        prepared_val_output_path = self._save_prepared_frame(
            path=prepared_val_parquet_path,
            frame=prepared_val_df,
        )
        del train_raw
        del val_raw
        del prepared_train_df
        del prepared_val_df
        gc.collect()
        logger.info(
            "Saved prepared datasets: train=%s val=%s",
            prepared_train_output_path,
            prepared_val_output_path,
        )
        return prepared_train_output_path, prepared_val_output_path

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

        logger.info("Loading train dataframe into memory...")
        train_df = self._read_prepared_frame(train_csv_path)
        logger.info("Prepared train data loaded into memory: train_shape=%s", train_df.shape)

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
                "example_size": 30,
                "example_random_state": self.config.general.seed,
            },
        )
        del train_df
        gc.collect()
        self.category_maps = self.train_dataset.category_maps
        self.numeric_stats = self.train_dataset.numeric_stats
        logger.info("Train dataset tensors ready. Loading validation dataframe...")

        val_df = self._read_prepared_frame(val_csv_path)
        logger.info("Prepared validation data loaded into memory: val_shape=%s", val_df.shape)
        logger.info("Building validation dataset tensors...")

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
            example_size=30,
            example_random_state=self.config.general.seed,
        )
        del val_df
        gc.collect()

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
        test_df = self._read_prepared_frame(test_csv_path)
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
            example_size=30,
            example_random_state=self.config.general.seed,
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
            drop_last=self.config.data.drop_last,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not initialized. Call setup('test').")
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            drop_last=self.config.data.drop_last,
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
    cfg.data.train_file_name = "/final_project/data/archive/train_dataset.csv"
    cfg.data.val_file_name = "/final_project/data/archive/valid_dataset.csv"
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
