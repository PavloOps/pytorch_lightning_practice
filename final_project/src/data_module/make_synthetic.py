from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd


CORE_COLUMNS = ["receipt_id", "bonus_card_id", "product_group_id", "count", "date_"]
OPTIONAL_COLUMNS = ["category", "is_vital", "pharmacy_id"]


def _make_id_map(values: pd.Series, prefix: str, rng: np.random.Generator) -> dict[str, str]:
    unique_values = values.astype(str).dropna().unique().tolist()
    shuffled = unique_values.copy()
    rng.shuffle(shuffled)
    return {
        original: f"{prefix}_{idx:07d}" for idx, original in enumerate(shuffled, start=1)
    }


def _apply_k_anonymity(df: pd.DataFrame, column: str, min_group_size: int) -> pd.DataFrame:
    if min_group_size <= 1:
        return df
    value_counts = df[column].value_counts()
    allowed_values = value_counts[value_counts >= min_group_size].index
    return df[df[column].isin(allowed_values)].reset_index(drop=True)


@click.command()
@click.option("--input-path", required=True, help="Path to real (private) CSV data.")
@click.option(
    "--output-path",
    default="data/sample_interactions.csv",
    show_default=True,
    help="Path to output synthetic CSV.",
)
@click.option(
    "--n-rows",
    default=100000,
    type=int,
    show_default=True,
    help="Number of rows in synthetic dataset.",
)
@click.option("--seed", default=2025, type=int, show_default=True, help="Random seed.")
@click.option(
    "--min-group-size",
    default=20,
    type=int,
    show_default=True,
    help="Minimum frequency for product_group_id after synthesis (k-anonymity).",
)
def cli(input_path: str, output_path: str, n_rows: int, seed: int, min_group_size: int):
    rng = np.random.default_rng(seed)

    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    real_df = pd.read_csv(input_file)

    if "count" not in real_df.columns and "cnt" in real_df.columns:
        real_df = real_df.rename(columns={"cnt": "count"})

    missing_core_cols = [col for col in CORE_COLUMNS if col not in real_df.columns]
    if missing_core_cols:
        raise ValueError(f"Missing required columns: {missing_core_cols}")

    used_columns = CORE_COLUMNS + [col for col in OPTIONAL_COLUMNS if col in real_df.columns]
    df = real_df[used_columns].copy()
    df = df.dropna(subset=CORE_COLUMNS).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after dropping NaN in required columns.")

    df["date_"] = pd.to_datetime(df["date_"], errors="coerce")
    df = df.dropna(subset=["date_"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows with parseable date_ values.")

    sampled = df.sample(n=max(1, n_rows), replace=True, random_state=seed).reset_index(drop=True)

    user_map = _make_id_map(df["bonus_card_id"], "user", rng)
    item_map = _make_id_map(df["product_group_id"], "product_group", rng)
    receipt_map = _make_id_map(df["receipt_id"], "receipt", rng)
    sampled["receipt_id"] = sampled["receipt_id"].astype(str).map(receipt_map)
    sampled["bonus_card_id"] = sampled["bonus_card_id"].astype(str).map(user_map)
    sampled["product_group_id"] = sampled["product_group_id"].astype(str).map(item_map)

    if "pharmacy_id" in sampled.columns:
        pharmacy_map = _make_id_map(df["pharmacy_id"], "pharmacy", rng)
        sampled["pharmacy_id"] = sampled["pharmacy_id"].astype(str).map(pharmacy_map)

    counts = pd.to_numeric(sampled["count"], errors="coerce").fillna(1).clip(lower=1)
    noisy_counts = np.maximum(
        1,
        np.round(counts.to_numpy() * rng.uniform(0.8, 1.2, size=len(counts))).astype(int),
    )
    sampled["count"] = noisy_counts

    min_date = df["date_"].min()
    max_date = df["date_"].max()
    day_noise = rng.integers(-7, 8, size=len(sampled))
    sampled["date_"] = (sampled["date_"] + pd.to_timedelta(day_noise, unit="D")).clip(
        lower=min_date, upper=max_date
    )
    sampled["date_"] = sampled["date_"].dt.strftime("%Y-%m-%d")

    sampled = _apply_k_anonymity(
        df=sampled, column="product_group_id", min_group_size=min_group_size
    )

    sampled.to_csv(output_file, index=False)

    print(f"Input rows: {len(real_df)}")
    print(f"Synthetic rows: {len(sampled)}")
    print(f"Columns: {list(sampled.columns)}")
    print(f"Saved synthetic data to: {output_file}")


if __name__ == "__main__":
    cli()
