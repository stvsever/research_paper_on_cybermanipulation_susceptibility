from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def one_hot_profile_categoricals(df: pd.DataFrame, categorical_prefix: str = "profile_cat__") -> pd.DataFrame:
    categorical_cols = [c for c in df.columns if c.startswith("profile_cat_")]
    if not categorical_cols:
        return df
    encoded = pd.get_dummies(df[categorical_cols], prefix=categorical_cols, dtype=float)
    encoded.columns = [f"{categorical_prefix}{col}" for col in encoded.columns]
    df = df.drop(columns=categorical_cols)
    return pd.concat([df, encoded], axis=1)


def zscore_series(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def infer_analysis_mode(df: pd.DataFrame) -> str:
    if "attack_present" not in df.columns or len(df) == 0:
        return "mixed_condition"
    unique_values = sorted(pd.Series(df["attack_present"]).dropna().astype(int).unique().tolist())
    if unique_values == [1]:
        return "treated_only"
    if unique_values == [0]:
        return "control_only"
    return "mixed_condition"


def choose_primary_moderator_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns and df[preferred].nunique() > 1:
        return preferred

    preferred = [
        "profile_cont_age_years",
        "profile_cont_big_five_neuroticism_mean_pct",
        "profile_cont_big_five_openness_to_experience_mean_pct",
        "profile_cont_big_five_conscientiousness_mean_pct",
        "profile_cont_big_five_extraversion_mean_pct",
        "profile_cont_big_five_agreeableness_mean_pct",
    ]
    for col in preferred:
        if col in df.columns and df[col].nunique() > 1:
            return col

    candidates = [
        col
        for col in df.columns
        if col.startswith("profile_cont_") and df[col].nunique() > 1
    ]
    if not candidates:
        return "baseline_score"
    return sorted(candidates)[0]


def available_moderator_columns(
    df: pd.DataFrame,
    preferred_order: Optional[Iterable[str]] = None,
) -> List[str]:
    ordered = list(preferred_order or [])
    candidates: List[str] = []

    for column in ordered:
        if column in df.columns and df[column].nunique() > 1 and column not in candidates:
            candidates.append(column)

    for column in sorted(df.columns):
        is_continuous = column.startswith("profile_cont_")
        is_binary = column.startswith("profile_cat__")
        if (is_continuous or is_binary) and df[column].nunique() > 1 and column not in candidates:
            candidates.append(column)

    return candidates
