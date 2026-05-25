from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioscancast.stages.eval_stage.calibration import calibration_table, plot_calibration_curve


def _ensure_parent_dir(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _title_metric(metric: str) -> str:
    return metric.replace("_", " ").title()


def plot_source_comparison(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    required = {"forecast_source", "mean_brier_score", "mean_log_score", "mean_accuracy", "mean_rps"}
    missing = [c for c in required if c not in summary_df.columns]
    if missing:
        raise ValueError("summary_df is missing required columns: " + ", ".join(missing))

    output_path = _ensure_parent_dir(output_path)
    plot_df = summary_df.set_index("forecast_source")[["mean_brier_score", "mean_log_score", "mean_accuracy", "mean_rps"]]
    ax = plot_df.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_xlabel("Forecast source")
    ax.set_ylabel("Metric value")
    ax.set_title("Forecast performance by source")
    ax.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_metric_boxplot(results_df: pd.DataFrame, metric: str, output_path: str | Path, ylabel: str | None = None) -> None:
    required = {"forecast_source", metric}
    missing = [c for c in required if c not in results_df.columns]
    if missing:
        raise ValueError("results_df is missing required columns: " + ", ".join(missing))

    output_path = _ensure_parent_dir(output_path)
    sources = list(dict.fromkeys(results_df["forecast_source"].tolist()))
    data = [results_df.loc[results_df["forecast_source"] == s, metric].dropna().tolist() for s in sources]

    plt.figure(figsize=(8.5, 5))
    plt.boxplot(
    data,
    labels=sources,
    showmeans=True,
    whis=(0, 100)
)
    for i, vals in enumerate(data, start=1):
        if vals:
            xvals = np.random.normal(i, 0.04, size=len(vals))
            plt.plot(xvals, vals, "o", alpha=0.75)
    plt.ylabel(ylabel or _title_metric(metric))
    plt.title(f"Distribution of {metric.replace('_', ' ')} by source")
    plt.figtext(0.5, -0.03, "Dots = questions, orange line = median, green triangle = mean", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()




def plot_accuracy_by_source(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    required = {"forecast_source", "mean_accuracy"}
    missing = [c for c in required if c not in summary_df.columns]
    if missing:
        raise ValueError("summary_df is missing required columns: " + ", ".join(missing))

    output_path = _ensure_parent_dir(output_path)
    plot_df = summary_df.set_index("forecast_source")[["mean_accuracy"]]
    ax = plot_df.plot(kind="bar", figsize=(7.5, 5), legend=False, rot=0)
    ax.set_xlabel("Forecast source")
    ax.set_ylabel("Accuracy")
    ax.set_title("Mean accuracy by source")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_metric_distribution(results_df: pd.DataFrame, metric: str, output_path: str | Path) -> None:
    if metric not in results_df.columns:
        raise ValueError(f"results_df must contain a '{metric}' column.")

    output_path = _ensure_parent_dir(output_path)
    series = results_df[metric].dropna().astype(float)
    plt.figure(figsize=(8, 5))
    bins = min(10, max(3, len(series)))
    plt.hist(series, bins=bins)
    plt.xlabel(_title_metric(metric))
    plt.ylabel("Count")
    plt.title(f"Distribution of question-level {_title_metric(metric).lower()}")
    if series.nunique() == 1:
        val = float(series.iloc[0])
        plt.xlim(val - 0.05, val + 0.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_question_level_scatter(comparison_df: pd.DataFrame, metric: str, source_a: str, source_b: str, output_path: str | Path) -> None:
    col_a = f"{metric}_{source_a}"
    col_b = f"{metric}_{source_b}"
    required = {"question_id", col_a, col_b}
    missing = [c for c in required if c not in comparison_df.columns]
    if missing:
        raise ValueError("comparison_df is missing required columns: " + ", ".join(missing))

    output_path = _ensure_parent_dir(output_path)
    x = comparison_df[col_a].astype(float)
    y = comparison_df[col_b].astype(float)

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(x, y)
    mn = float(min(x.min(), y.min()))
    mx = float(max(x.max(), y.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel(f"{source_a.title()} {metric.replace('_', ' ')}")
    plt.ylabel(f"{source_b.title()} {metric.replace('_', ' ')}")
    plt.title(f"Question-level {metric.replace('_', ' ')} comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_question_level_differences(comparison_df: pd.DataFrame, metric: str, source_a: str, source_b: str, output_path: str | Path) -> None:
    col_a = f"{metric}_{source_a}"
    col_b = f"{metric}_{source_b}"
    required = {"question_id", col_a, col_b}
    missing = [c for c in required if c not in comparison_df.columns]
    if missing:
        raise ValueError("comparison_df is missing required columns: " + ", ".join(missing))

    output_path = _ensure_parent_dir(output_path)
    diffs = comparison_df[col_a].astype(float) - comparison_df[col_b].astype(float)
    plt.figure(figsize=(8, 4.5))
    plt.axhline(0, linestyle="--")
    plt.scatter(range(len(diffs)), diffs)
    plt.xticks(range(len(diffs)), comparison_df["question_id"].tolist(), rotation=45)
    plt.ylabel(f"{source_a.title()} - {source_b.title()} {metric.replace('_', ' ')}")
    plt.title(f"Per-question difference in {metric.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_win_rate(comparison_df: pd.DataFrame, metric: str, source_a: str, source_b: str, output_path: str | Path, lower_is_better: bool = True) -> None:
    col_a = f"{metric}_{source_a}"
    col_b = f"{metric}_{source_b}"
    diffs = comparison_df[col_a].astype(float) - comparison_df[col_b].astype(float)

    if lower_is_better:
        a_wins = int((diffs < 0).sum())
        b_wins = int((diffs > 0).sum())
    else:
        a_wins = int((diffs > 0).sum())
        b_wins = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())

    output_path = _ensure_parent_dir(output_path)
    plt.figure(figsize=(6.5, 4.5))
    plt.bar([source_a.title(), source_b.title(), "Ties"], [a_wins, b_wins, ties])
    plt.ylabel("Questions")
    plt.title(f"Win rate by question for {metric.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_confidence_calibration(results_df: pd.DataFrame, output_path: str | Path, source: str | None = None) -> None:
    if not {"top_probability", "accuracy"}.issubset(results_df.columns):
        raise ValueError("results_df must contain 'top_probability' and 'accuracy'.")

    if source is not None and "forecast_source" in results_df.columns:
        results_df = results_df[results_df["forecast_source"] == source]

    table = calibration_table(results_df["top_probability"].tolist(), results_df["accuracy"].tolist(), bins=5)
    plot_calibration_curve(table, output_path)


def plot_reliability_overview(results_df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, group in results_df.groupby("forecast_source"):
        plot_confidence_calibration(group, output_dir / f"calibration_{source}.png")
