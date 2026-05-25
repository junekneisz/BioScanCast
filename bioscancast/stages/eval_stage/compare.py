from __future__ import annotations

from typing import Iterable

import pandas as pd


REQUIRED_RESULT_COLUMNS = {
    "question_id",
    "forecast_source",
    "brier_score",
    "log_score",
    "accuracy",
    "rps",
    "top_probability",
    "normalized_entropy",
    "true_probability",
}


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("results_df is missing required columns: " + ", ".join(missing))


def compare_sources(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scoring metrics by forecast source."""
    _require_columns(results_df, REQUIRED_RESULT_COLUMNS)

    summary = (
        results_df.groupby("forecast_source", dropna=False)
        .agg(
            n_questions=("question_id", "nunique"),
            mean_brier_score=("brier_score", "mean"),
            mean_log_score=("log_score", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_rps=("rps", "mean"),
            mean_top_probability=("top_probability", "mean"),
            mean_normalized_entropy=("normalized_entropy", "mean"),
            mean_true_probability=("true_probability", "mean"),
        )
        .reset_index()
        .sort_values("mean_brier_score", ascending=True)
    )
    return summary


def compare_sources_by_question_type(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scoring metrics by forecast source and question type."""
    _require_columns(results_df, REQUIRED_RESULT_COLUMNS)

    if "question_type" not in results_df.columns:
        raise ValueError("results_df must contain a 'question_type' column to compare by question type.")

    summary = (
        results_df.groupby(["forecast_source", "question_type"], dropna=False)
        .agg(
            n_questions=("question_id", "nunique"),
            mean_brier_score=("brier_score", "mean"),
            mean_log_score=("log_score", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_rps=("rps", "mean"),
            mean_top_probability=("top_probability", "mean"),
            mean_normalized_entropy=("normalized_entropy", "mean"),
            mean_true_probability=("true_probability", "mean"),
        )
        .reset_index()
        .sort_values(["forecast_source", "question_type"])
    )
    return summary
