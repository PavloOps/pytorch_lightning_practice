from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from util import ensure_columns, prepare_output_path, print_stats


@dataclass(frozen=True)
class DebugSetConfig:
    per_case: int = 50
    candidates_per_sample: int = 200
    seed: int = 42
    cold_user_quantile: float = 0.2
    tail_item_quantile: float = 0.2
    head_item_quantile: float = 0.8
    short_history_quantile: float = 0.2
    ambiguous_quantile: float = 0.8
    popular_share: float = 0.5
    same_category_share: float = 0.3
    random_share: float = 0.2


@dataclass(frozen=True)
class DebugColumnConfig:
    user: str = "user_id"
    item: str = "item_id"
    month: str = "year_month"
    count: str = "cnt"
    category: str | None = "category"


@dataclass(frozen=True)
class DebugThresholds:
    cold_user: float
    short_history_user: float
    ambiguous_user: float
    tail_item: float
    head_item: float


@dataclass(frozen=True)
class CandidatePools:
    month_to_items: dict[int, list[str]]
    month_to_popular: dict[int, list[str]]
    user_month_positives: dict[tuple[str, int], set[str]]
    item_to_category: dict[str, str]
    category_month_to_items: dict[tuple[int, str], list[str]]
    all_items: list[str]


def build_debug_dataset(
    df: pd.DataFrame,
    per_case: int = 50,
    candidates_per_sample: int = 200,
    seed: int = 42,
    user_col: str = "user_id",
    item_col: str = "item_id",
    month_col: str = "year_month",
    count_col: str = "cnt",
    category_col: str | None = "category",
) -> pd.DataFrame:
    """
    Build a stratified debug set for recommender training diagnostics.

    Output columns:
      - debug_id
      - stratum
      - user_id / month / target_item_id
      - candidates (list[item_id], first element is always target)
      - candidate_count
      - user_prior_interactions
      - user_prior_unique_items
      - user_prior_diversity
      - item_popularity_in_month

    Notes:
      - One row in output is one ranking task for debugging.
      - Stratification is done on positive interactions.
      - Negative candidates are sampled from same month pool.
    """
    cfg = DebugSetConfig(
        per_case=per_case,
        candidates_per_sample=candidates_per_sample,
        seed=seed,
    )
    columns = DebugColumnConfig(
        user=user_col,
        item=item_col,
        month=month_col,
        count=count_col,
        category=category_col,
    )
    _validate_config(cfg, columns.category, df.columns)

    positives = _prepare_positive_interactions(df=df, columns=columns)
    thresholds = _compute_thresholds(positives=positives, cfg=cfg)
    rng = np.random.default_rng(cfg.seed)

    sampled = _sample_debug_strata(
        positives=positives,
        thresholds=thresholds,
        per_case=cfg.per_case,
        rng=rng,
        columns=columns,
    )
    sampled = _attach_stratum_flags(sampled=sampled, thresholds=thresholds)

    pools = _build_candidate_pools(positives=positives, columns=columns)
    sampled = _attach_candidates(
        sampled=sampled,
        positives=positives,
        pools=pools,
        cfg=cfg,
        columns=columns,
        rng=rng,
    )
    return _select_output_columns(sampled=sampled, columns=columns)


def _prepare_positive_interactions(df: pd.DataFrame, columns: DebugColumnConfig) -> pd.DataFrame:
    ensure_columns(
        df=df,
        required=[columns.user, columns.item, columns.month, columns.count],
        optional=[columns.category] if columns.category else [],
    )

    selected_columns = [columns.user, columns.item, columns.month, columns.count]
    if columns.category:
        selected_columns.append(columns.category)

    work = df[selected_columns].copy()
    work[columns.user] = work[columns.user].astype(str)
    work[columns.item] = work[columns.item].astype(str)
    work[columns.month] = pd.to_numeric(work[columns.month], errors="coerce")
    work[columns.count] = pd.to_numeric(work[columns.count], errors="coerce").fillna(0.0)
    work = work.dropna(subset=[columns.month]).copy()
    work[columns.month] = work[columns.month].astype(int)

    if columns.category:
        work[columns.category] = work[columns.category].fillna("UNKNOWN").astype(str)

    positives = work.loc[work[columns.count] > 0].copy()
    if positives.empty:
        raise ValueError("No positive interactions found (`count_col` > 0).")

    positives = positives.sort_values([columns.user, columns.month, columns.item]).reset_index(drop=True)
    positives["_row_id"] = np.arange(len(positives))
    positives = _attach_user_history_features(positives=positives, columns=columns)
    positives = _attach_item_popularity(positives=positives, columns=columns)
    return positives


def _attach_user_history_features(positives: pd.DataFrame, columns: DebugColumnConfig) -> pd.DataFrame:
    enriched = positives.copy()
    grouped = enriched.groupby(columns.user, sort=False)
    enriched["user_prior_interactions"] = grouped.cumcount()
    enriched["user_prior_unique_items"] = grouped[columns.item].transform(_prior_unique_count)

    prior_unique = enriched["user_prior_unique_items"].to_numpy(dtype=float)
    prior_interactions = enriched["user_prior_interactions"].to_numpy(dtype=float)
    enriched["user_prior_diversity"] = prior_unique / np.maximum(prior_interactions, 1.0)
    return enriched


def _attach_item_popularity(positives: pd.DataFrame, columns: DebugColumnConfig) -> pd.DataFrame:
    item_month_popularity = (
        positives.groupby([columns.month, columns.item], as_index=False)[columns.count]
        .sum()
        .rename(columns={columns.count: "item_popularity_in_month"})
    )
    return positives.merge(item_month_popularity, on=[columns.month, columns.item], how="left")


def _compute_thresholds(positives: pd.DataFrame, cfg: DebugSetConfig) -> DebugThresholds:
    return DebugThresholds(
        cold_user=positives["user_prior_interactions"].quantile(cfg.cold_user_quantile),
        short_history_user=positives["user_prior_unique_items"].quantile(cfg.short_history_quantile),
        ambiguous_user=positives["user_prior_diversity"].quantile(cfg.ambiguous_quantile),
        tail_item=positives["item_popularity_in_month"].quantile(cfg.tail_item_quantile),
        head_item=positives["item_popularity_in_month"].quantile(cfg.head_item_quantile),
    )


def _sample_debug_strata(
    positives: pd.DataFrame,
    thresholds: DebugThresholds,
    per_case: int,
    rng: np.random.Generator,
    columns: DebugColumnConfig,
) -> pd.DataFrame:
    available = positives.copy()
    sampled_frames: list[pd.DataFrame] = []

    for name, mask in _iter_strata_masks(available=available, thresholds=thresholds):
        pool = available.loc[mask].copy()
        if pool.empty:
            continue

        n_take = min(per_case, len(pool))
        chosen_idx = rng.choice(pool.index.to_numpy(), size=n_take, replace=False)
        chosen = pool.loc[chosen_idx].copy()
        chosen["stratum"] = name
        sampled_frames.append(chosen)
        available = available.drop(index=chosen.index)

    if not sampled_frames:
        raise ValueError("Could not sample debug strata from provided dataframe.")

    sampled = pd.concat(sampled_frames, axis=0, ignore_index=True)
    return sampled.sort_values(["stratum", columns.user, columns.month, columns.item]).reset_index(drop=True)


def _iter_strata_masks(
    available: pd.DataFrame,
    thresholds: DebugThresholds,
) -> list[tuple[str, pd.Series]]:
    return [
        ("cold_user", available["user_prior_interactions"] <= thresholds.cold_user),
        ("tail_item", available["item_popularity_in_month"] <= thresholds.tail_item),
        ("head_item", available["item_popularity_in_month"] >= thresholds.head_item),
        ("short_history_user", available["user_prior_unique_items"] <= thresholds.short_history_user),
        ("ambiguous_user", available["user_prior_diversity"] >= thresholds.ambiguous_user),
        (
            "easy_control",
            (available["item_popularity_in_month"] >= thresholds.head_item)
            & (available["user_prior_interactions"] > thresholds.cold_user),
        ),
    ]


def _attach_stratum_flags(sampled: pd.DataFrame, thresholds: DebugThresholds) -> pd.DataFrame:
    enriched = sampled.copy()
    enriched["is_cold_user"] = enriched["user_prior_interactions"] <= thresholds.cold_user
    enriched["is_tail_item"] = enriched["item_popularity_in_month"] <= thresholds.tail_item
    enriched["is_head_item"] = enriched["item_popularity_in_month"] >= thresholds.head_item
    enriched["is_short_history_user"] = enriched["user_prior_unique_items"] <= thresholds.short_history_user
    enriched["is_ambiguous_user"] = enriched["user_prior_diversity"] >= thresholds.ambiguous_user
    enriched["is_easy_control"] = (
        (enriched["item_popularity_in_month"] >= thresholds.head_item)
        & (enriched["user_prior_interactions"] > thresholds.cold_user)
    )
    return enriched


def _build_candidate_pools(positives: pd.DataFrame, columns: DebugColumnConfig) -> CandidatePools:
    month_to_items = (
        positives.groupby(columns.month)[columns.item]
        .apply(lambda series: sorted(set(series.astype(str).tolist())))
        .to_dict()
    )
    month_item_popularity = (
        positives.groupby([columns.month, columns.item], as_index=False)[columns.count]
        .sum()
        .sort_values([columns.month, columns.count], ascending=[True, False])
    )
    month_to_popular = month_item_popularity.groupby(columns.month)[columns.item].apply(list).to_dict()
    user_month_positives = (
        positives.groupby([columns.user, columns.month])[columns.item]
        .apply(lambda series: set(series.astype(str).tolist()))
        .to_dict()
    )

    item_to_category: dict[str, str] = {}
    category_month_to_items: dict[tuple[int, str], list[str]] = {}
    if columns.category:
        item_to_category = (
            positives.groupby(columns.item)[columns.category]
            .agg(lambda series: series.mode().iloc[0] if not series.mode().empty else "UNKNOWN")
            .to_dict()
        )
        category_month_to_items = (
            positives.groupby([columns.month, columns.category])[columns.item]
            .apply(lambda series: sorted(set(series.astype(str).tolist())))
            .to_dict()
        )

    return CandidatePools(
        month_to_items=month_to_items,
        month_to_popular=month_to_popular,
        user_month_positives=user_month_positives,
        item_to_category=item_to_category,
        category_month_to_items=category_month_to_items,
        all_items=positives[columns.item].astype(str).drop_duplicates().tolist(),
    )


def _attach_candidates(
    sampled: pd.DataFrame,
    positives: pd.DataFrame,
    pools: CandidatePools,
    cfg: DebugSetConfig,
    columns: DebugColumnConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_popular, n_same_category, n_random = _compute_candidate_mix(cfg=cfg, has_category=columns.category is not None)

    candidates_column: list[list[str]] = []
    candidate_counts: list[int] = []
    for row in sampled.itertuples(index=False):
        candidates = _build_candidates_for_row(
            row=row,
            positives=positives,
            pools=pools,
            rng=rng,
            columns=columns,
            n_popular=n_popular,
            n_same_category=n_same_category,
            n_random=n_random,
            total_candidates=cfg.candidates_per_sample,
        )
        candidates_column.append(candidates)
        candidate_counts.append(len(candidates))

    enriched = sampled.copy()
    enriched["target_item_id"] = enriched[columns.item]
    enriched["candidates"] = candidates_column
    enriched["candidate_count"] = candidate_counts
    return enriched


def _compute_candidate_mix(cfg: DebugSetConfig, has_category: bool) -> tuple[int, int, int]:
    n_popular = int(round(cfg.candidates_per_sample * cfg.popular_share))
    n_same_category = int(round(cfg.candidates_per_sample * cfg.same_category_share)) if has_category else 0
    n_random = max(0, cfg.candidates_per_sample - n_popular - n_same_category)
    return n_popular, n_same_category, n_random


def _build_candidates_for_row(
    row: object,
    positives: pd.DataFrame,
    pools: CandidatePools,
    rng: np.random.Generator,
    columns: DebugColumnConfig,
    n_popular: int,
    n_same_category: int,
    n_random: int,
    total_candidates: int,
) -> list[str]:
    user_id = getattr(row, columns.user)
    month = int(getattr(row, columns.month))
    target_item = getattr(row, columns.item)

    month_items = set(pools.month_to_items.get(month, []))
    if not month_items:
        month_items = set(pools.all_items or positives[columns.item].astype(str).unique().tolist())

    excluded = set(pools.user_month_positives.get((user_id, month), set()))
    excluded.add(target_item)
    allowed = list(month_items - excluded)

    popular_pool = [item for item in pools.month_to_popular.get(month, []) if item not in excluded]
    same_category_pool = _get_same_category_pool(
        month=month,
        target_item=target_item,
        excluded=excluded,
        pools=pools,
        category_col=columns.category,
    )

    negatives: list[str] = []
    used: set[str] = set()
    _extend_unique(negatives, used, _sample_without_replacement(rng, popular_pool, n_popular))
    _extend_unique(negatives, used, _sample_without_replacement(rng, same_category_pool, n_same_category))

    random_pool = [item for item in allowed if item not in used]
    _extend_unique(negatives, used, _sample_without_replacement(rng, random_pool, n_random))

    if len(negatives) < total_candidates:
        backfill_pool = [item for item in allowed if item not in used]
        missing = total_candidates - len(negatives)
        _extend_unique(negatives, used, _sample_without_replacement(rng, backfill_pool, missing))

    return [target_item] + negatives


def _get_same_category_pool(
    month: int,
    target_item: str,
    excluded: set[str],
    pools: CandidatePools,
    category_col: str | None,
) -> list[str]:
    if category_col is None:
        return []

    category = pools.item_to_category.get(target_item, "UNKNOWN")
    return [
        item
        for item in pools.category_month_to_items.get((month, category), [])
        if item not in excluded
    ]


def _extend_unique(target: list[str], used: set[str], candidates: list[str]) -> None:
    for item in candidates:
        if item not in used:
            target.append(item)
            used.add(item)


def _select_output_columns(sampled: pd.DataFrame, columns: DebugColumnConfig) -> pd.DataFrame:
    out = sampled[
        [
            "stratum",
            columns.user,
            columns.month,
            "target_item_id",
            "candidates",
            "candidate_count",
            "user_prior_interactions",
            "user_prior_unique_items",
            "user_prior_diversity",
            "item_popularity_in_month",
            "is_cold_user",
            "is_tail_item",
            "is_head_item",
            "is_short_history_user",
            "is_ambiguous_user",
            "is_easy_control",
        ]
    ].copy()
    out.insert(0, "debug_id", [f"dbg_{index:05d}" for index in range(1, len(out) + 1)])
    return out.reset_index(drop=True)


def _prior_unique_count(series: pd.Series) -> pd.Series:
    seen: set[str] = set()
    result = np.zeros(len(series), dtype=np.int64)
    values = series.astype(str).tolist()
    for index, value in enumerate(values):
        result[index] = len(seen)
        seen.add(value)
    return pd.Series(result, index=series.index)


def _sample_without_replacement(rng: np.random.Generator, pool: list[str], n: int) -> list[str]:
    if n <= 0 or not pool:
        return []
    take = min(n, len(pool))
    indices = rng.choice(len(pool), size=take, replace=False)
    return [pool[int(index)] for index in indices]


def _validate_config(cfg: DebugSetConfig, category_col: str | None, columns: pd.Index) -> None:
    if cfg.per_case <= 0:
        raise ValueError("`per_case` must be > 0.")
    if cfg.candidates_per_sample <= 0:
        raise ValueError("`candidates_per_sample` must be > 0.")
    if not (0.0 <= cfg.popular_share <= 1.0):
        raise ValueError("`popular_share` must be in [0, 1].")
    if not (0.0 <= cfg.same_category_share <= 1.0):
        raise ValueError("`same_category_share` must be in [0, 1].")
    if not (0.0 <= cfg.random_share <= 1.0):
        raise ValueError("`random_share` must be in [0, 1].")
    if abs((cfg.popular_share + cfg.same_category_share + cfg.random_share) - 1.0) > 1e-6:
        raise ValueError("`popular_share + same_category_share + random_share` must equal 1.0.")
    if category_col is None or category_col not in columns:
        if cfg.same_category_share > 0:
            raise ValueError(
                "`same_category_share` > 0 requires existing category column. "
                "Set `category_col=None` and same_category_share=0."
            )


if __name__ == "__main__":
    input_path = Path("/home/pavloops/PycharmProjects/pytorch_lightning_practice/final_project/data/dataset.csv")
    output_path = Path("/home/pavloops/PycharmProjects/pytorch_lightning_practice/final_project/data/debug_dataset.csv")

    source_df = pd.read_csv(
        input_path,
        usecols=["user_id", "item_id", "year_month", "cnt", "category"],
    )
    debug_df = build_debug_dataset(
        df=source_df,
        per_case=100,
        candidates_per_sample=200,
        seed=42,
        user_col="user_id",
        item_col="item_id",
        month_col="year_month",
        count_col="cnt",
        category_col="category",
    )

    output_path = prepare_output_path(output_path)
    debug_df.to_csv(output_path, index=False)
    print_stats(
        {
            "Input rows": len(source_df),
            "Debug rows": len(debug_df),
            "Saved debug dataset to": output_path.resolve(),
        }
    )
    print("Strata counts:")
    print(debug_df["stratum"].value_counts().to_string())
