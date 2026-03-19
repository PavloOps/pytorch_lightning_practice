from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from util import normalize_string_column, prepare_output_path, print_stats


@dataclass(frozen=True)
class NewUserSliceConfig:
    user_col: str = "user_id"
    min_interactions: int = 2
    new_user_share: float = 0.1
    seed: int = 42


def build_new_user_slice(
    df: pd.DataFrame,
    user_col: str = "user_id",
    min_interactions: int = 2,
    new_user_share: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a user-level holdout for 'new user' evaluation.

    Returns:
      - new_user_ids_df: dataframe with one column [user_col]
      - new_user_eval_df: all rows from original df for those users
    """
    cfg = NewUserSliceConfig(
        user_col=user_col,
        min_interactions=min_interactions,
        new_user_share=new_user_share,
        seed=seed,
    )
    work = _prepare_user_frame(df=df, user_col=cfg.user_col)
    eligible_users = _get_eligible_users(
        df=work,
        user_col=cfg.user_col,
        min_interactions=cfg.min_interactions,
    )
    selected_users = _sample_new_users(
        eligible_users=eligible_users,
        share=cfg.new_user_share,
        seed=cfg.seed,
    )
    return _build_outputs(df=work, selected_users=selected_users, user_col=cfg.user_col)


def _prepare_user_frame(df: pd.DataFrame, user_col: str) -> pd.DataFrame:
    if user_col not in df.columns:
        raise ValueError(f"Missing required column: {user_col}")

    work = df.copy()
    normalize_string_column(work, user_col)
    return work


def _get_eligible_users(df: pd.DataFrame, user_col: str, min_interactions: int) -> np.ndarray:
    if min_interactions <= 0:
        raise ValueError("`min_interactions` must be > 0")

    user_counts = df.groupby(user_col).size().rename("n_interactions")
    eligible_users = user_counts[user_counts >= min_interactions].index.to_numpy()
    if len(eligible_users) == 0:
        raise ValueError("No eligible users found for new-user split.")
    return eligible_users


def _sample_new_users(eligible_users: np.ndarray, share: float, seed: int) -> np.ndarray:
    if not (0.0 < share < 1.0):
        raise ValueError("`new_user_share` must be in (0, 1)")

    n_new_users = max(1, int(round(len(eligible_users) * share)))
    n_new_users = min(n_new_users, len(eligible_users))

    rng = np.random.default_rng(seed)
    return rng.choice(eligible_users, size=n_new_users, replace=False)


def _build_outputs(
    df: pd.DataFrame,
    selected_users: np.ndarray,
    user_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    new_user_ids_df = pd.DataFrame({user_col: selected_users}).sort_values(user_col).reset_index(drop=True)
    selected_set = set(selected_users.tolist())
    new_user_eval_df = df[df[user_col].isin(selected_set)].copy().reset_index(drop=True)
    return new_user_ids_df, new_user_eval_df


if __name__ == "__main__":
    input_path = Path("/home/pavloops/PycharmProjects/pytorch_lightning_practice/final_project/data/dataset.csv")
    out_ids = prepare_output_path("/home/pavloops/PycharmProjects/pytorch_lightning_practice/final_project/data/new_user_ids.csv")
    out_eval = prepare_output_path("/home/pavloops/PycharmProjects/pytorch_lightning_practice/final_project/data/new_user_eval_slice.csv")

    source_df = pd.read_csv(input_path)
    new_user_ids_df, new_user_eval_df = build_new_user_slice(
        df=source_df,
        user_col="user_id",
        min_interactions=2,
        new_user_share=0.1,
        seed=42,
    )

    new_user_ids_df.to_csv(out_ids, index=False)
    new_user_eval_df.to_csv(out_eval, index=False)

    total_users = source_df["user_id"].astype(str).nunique()
    holdout_users = new_user_ids_df["user_id"].astype(str).nunique()
    holdout_rows = len(new_user_eval_df)
    total_rows = len(source_df)

    print_stats(
        {
            "Total users": total_users,
            "New users (holdout)": f"{holdout_users} ({holdout_users / max(1, total_users):.2%})",
            "Total rows": total_rows,
            "Rows in new-user eval slice": f"{holdout_rows} ({holdout_rows / max(1, total_rows):.2%})",
            "Saved user ids": out_ids.resolve(),
            "Saved eval slice": out_eval.resolve(),
        }
    )
