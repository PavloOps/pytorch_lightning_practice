from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_columns(df: pd.DataFrame, required: list[str], optional: list[str] | None = None) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in optional or []:
        if col is not None and col not in df.columns:
            raise ValueError(f"Optional column `{col}` was requested but not found in dataframe.")


def prepare_output_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def normalize_string_column(df: pd.DataFrame, column: str) -> None:
    df[column] = df[column].astype(str)


def print_stats(stats: dict[str, object]) -> None:
    for key, value in stats.items():
        print(f"{key}: {value}")
