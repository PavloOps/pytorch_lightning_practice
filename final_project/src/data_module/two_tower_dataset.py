from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


MISSING_TOKEN = "__MISSING__"


@dataclass
class NumericStats:
    mean: float
    std: float


class TwoTowerDataset(Dataset):
    """
    Minimal Dataset for pointwise Two-Tower training on tabular interactions.

    Output format for each sample:
    {
        "user_cat": LongTensor [n_user_cat],
        "user_num": FloatTensor [n_user_num],
        "item_cat": LongTensor [n_item_cat],
        "item_num": FloatTensor [n_item_num],
        "label":    FloatTensor []   # if label_col is present
    }
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        user_cat_cols: Optional[List[str]] = None,
        item_cat_cols: Optional[List[str]] = None,
        user_num_cols: Optional[List[str]] = None,
        item_num_cols: Optional[List[str]] = None,
        label_col: str = "label",
        category_maps: Optional[Dict[str, Dict[str, int]]] = None,
        numeric_stats: Optional[Dict[str, NumericStats]] = None,
        fit: bool = False,
        add_month_feature: bool = True,
    ) -> None:
        self.user_cat_cols = user_cat_cols or ["user_id"]
        self.item_cat_cols = item_cat_cols or ["item_id", "brand", "category", "country", "inn", "owner"]
        self.user_num_cols = user_num_cols or ["user_total_cnt", "user_unique_items"]
        self.item_num_cols = item_num_cols or ["avg_price"]
        self.label_col = label_col
        self.add_month_feature = add_month_feature

        self.df = frame.copy()
        self._prepare_month_feature()

        self.cat_cols = self.user_cat_cols + self.item_cat_cols
        self.num_cols = self.user_num_cols + self.item_num_cols

        if fit:
            self.category_maps = self._fit_category_maps(self.df, self.cat_cols)
            self.numeric_stats = self._fit_numeric_stats(self.df, self.num_cols)
        else:
            if category_maps is None or numeric_stats is None:
                raise ValueError("For fit=False, provide both category_maps and numeric_stats.")
            self.category_maps = category_maps
            self.numeric_stats = numeric_stats

        self.user_cat_tensor = self._encode_cat_block(self.user_cat_cols)
        self.item_cat_tensor = self._encode_cat_block(self.item_cat_cols)
        self.user_num_tensor = self._encode_num_block(self.user_num_cols)
        self.item_num_tensor = self._encode_num_block(self.item_num_cols)

        self.label_tensor = None
        if self.label_col in self.df.columns:
            self.label_tensor = torch.tensor(self.df[self.label_col].astype("float32").values, dtype=torch.float32)

    @staticmethod
    def _fit_category_maps(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, Dict[str, int]]:
        maps: Dict[str, Dict[str, int]] = {}
        for col in cols:
            values = (
                df[col]
                .fillna(MISSING_TOKEN)
                .astype(str)
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            maps[col] = {v: i + 1 for i, v in enumerate(values)}  # 0 reserved for UNK
        return maps

    @staticmethod
    def _fit_numeric_stats(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, NumericStats]:
        stats: Dict[str, NumericStats] = {}
        for col in cols:
            series = pd.to_numeric(df[col], errors="coerce")
            mean = float(series.mean()) if series.notna().any() else 0.0
            std = float(series.std()) if series.notna().any() else 1.0
            if std == 0.0:
                std = 1.0
            stats[col] = NumericStats(mean=mean, std=std)
        return stats

    def _prepare_month_feature(self) -> None:
        if self.add_month_feature and "year_month" in self.df.columns and "month" not in self.df.columns:
            self.df["month"] = pd.to_numeric(self.df["year_month"], errors="coerce").fillna(0).astype(int) % 100
            if "month" not in self.user_num_cols:
                self.user_num_cols = [*self.user_num_cols, "month"]

    def _encode_cat_block(self, cols: List[str]) -> torch.Tensor:
        encoded_columns = []
        for col in cols:
            if col not in self.df.columns:
                raise KeyError(f"Missing categorical column: {col}")
            mapping = self.category_maps[col]
            encoded = (
                self.df[col]
                .fillna(MISSING_TOKEN)
                .astype(str)
                .map(mapping)
                .fillna(0)
                .astype("int64")
                .values
            )
            encoded_columns.append(torch.tensor(encoded, dtype=torch.long))

        if not encoded_columns:
            return torch.empty((len(self.df), 0), dtype=torch.long)
        return torch.stack(encoded_columns, dim=1)

    def _encode_num_block(self, cols: List[str]) -> torch.Tensor:
        encoded_columns = []
        for col in cols:
            if col not in self.df.columns:
                raise KeyError(f"Missing numeric column: {col}")
            stat = self.numeric_stats[col]
            values = pd.to_numeric(self.df[col], errors="coerce").fillna(stat.mean).astype("float32")
            normed = (values - stat.mean) / stat.std
            encoded_columns.append(torch.tensor(normed.values, dtype=torch.float32))

        if not encoded_columns:
            return torch.empty((len(self.df), 0), dtype=torch.float32)
        return torch.stack(encoded_columns, dim=1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "user_cat": self.user_cat_tensor[idx],
            "user_num": self.user_num_tensor[idx],
            "item_cat": self.item_cat_tensor[idx],
            "item_num": self.item_num_tensor[idx],
        }
        if self.label_tensor is not None:
            sample["label"] = self.label_tensor[idx]
        return sample

    @property
    def cardinalities(self) -> Dict[str, int]:
        return {col: len(mapping) + 1 for col, mapping in self.category_maps.items()}

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        fit: bool = True,
        category_maps: Optional[Dict[str, Dict[str, int]]] = None,
        numeric_stats: Optional[Dict[str, NumericStats]] = None,
        **kwargs,
    ) -> "TwoTowerDataset":
        df = pd.read_csv(csv_path)
        return cls(
            frame=df,
            fit=fit,
            category_maps=category_maps,
            numeric_stats=numeric_stats,
            **kwargs,
        )


if __name__ == "__main__":
    dataset = TwoTowerDataset.from_csv(str("/home/pavloops/PycharmProjects/pytorch_lightning_practice/final_project/data/train_example.csv"), fit=True)
    first = dataset[0]

    print(f"Dataset length: {len(dataset)}")
    print("First train sample:")
    for key, value in first.items():
        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}, value={value}")
