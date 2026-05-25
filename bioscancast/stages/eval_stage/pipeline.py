from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd

from bioscancast.stages.eval_stage.calibration import calibration_table
from bioscancast.stages.eval_stage.compare import (
    compare_sources,
    compare_sources_by_question_type,
)
from bioscancast.stages.eval_stage.loaders import load_forecasts, load_questions
from bioscancast.stages.eval_stage.scoring import (
    accuracy,
    log_score,
    multiclass_brier_score,
    normalized_entropy,
    ranked_probability_score,
    top_probability,
    true_probability,
)
from bioscancast.stages.eval_stage.statistics import (
    exact_mcnemar_test,
    paired_t_test,
    wilcoxon_signed_rank_test,
)
from bioscancast.stages.eval_stage.visualisation import (
    plot_accuracy_by_source,
    plot_confidence_calibration,
    plot_metric_boxplot,
    plot_metric_distribution,
    plot_question_level_differences,
    plot_question_level_scatter,
    plot_reliability_overview,
    plot_source_comparison,
    plot_win_rate,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _canonicalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    return text.strip()


def _prepare_questions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = {"question_id", "question_status", "resolved_option"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError("questions dataframe is missing required columns: " + ", ".join(missing))

    df["question_id"] = df["question_id"].astype(str).str.strip()
    df["question_status"] = df["question_status"].astype(str).str.lower().str.strip()
    df["resolved_option"] = df["resolved_option"].apply(_canonicalize_text)

    if "question_type" in df.columns:
        df["question_type"] = df["question_type"].astype(str).str.lower().str.strip()
    else:
        df["question_type"] = "unknown"

    if "topic" not in df.columns:
        df["topic"] = ""

    return df


def _infer_source_name(path: str | Path) -> str:
    stem = Path(path).stem.lower().strip()
    for suffix in ("_forecasts", "_forecast", "_mock", "_data"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or "forecast"


def _prepare_forecasts(df: pd.DataFrame, source_name: str | None = None) -> pd.DataFrame:
    df = df.copy()

    required_cols = {"question_id", "option", "probability"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError("forecasts dataframe is missing required columns: " + ", ".join(missing))

    df["question_id"] = df["question_id"].astype(str).str.strip()
    df["option"] = df["option"].apply(_canonicalize_text)
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    if df["probability"].isna().any():
        bad_rows = df[df["probability"].isna()]
        raise ValueError(
            "Some forecast probabilities could not be parsed as numeric values. "
            "Problematic rows: " + str(bad_rows.index.tolist())
        )

    if df["probability"].max() > 1.0:
        df["probability"] = df["probability"]  # keep as-is if already scaled

    if "forecast_source" not in df.columns:
        df["forecast_source"] = source_name or "forecast"
    else:
        df["forecast_source"] = df["forecast_source"].astype(str).str.strip()
        if source_name and (df["forecast_source"].nunique() == 1) and (
            df["forecast_source"].iloc[0] in {"", "nan", "none"}
        ):
            df["forecast_source"] = source_name

    if "forecast_version" in df.columns:
        df["forecast_version"] = df["forecast_version"].astype(str).str.strip()

    return df


def _get_resolved_option_for_group(group: pd.DataFrame, question_row: pd.Series) -> Tuple[str, str]:
    resolved_option = _canonicalize_text(question_row["resolved_option"])
    options = [_canonicalize_text(opt) for opt in group["option"].tolist()]

    if resolved_option in options:
        return resolved_option, ""

    lowered = resolved_option.lower()
    if lowered in {"", "tbd", "na", "n/a", "ambiguous"}:
        return "", f"unscorable_status:{lowered or 'empty'}"
    if lowered.startswith("resolved on "):
        return "", "placeholder_resolution_text"

    return "", "resolved_option_not_in_forecast_options"


def build_distribution(group: pd.DataFrame) -> Tuple[List[str], List[float]]:
    group = group.copy()
    if "option_order" in group.columns:
        group = group.sort_values("option_order")

    options = [_canonicalize_text(opt) for opt in group["option"].tolist()]
    probabilities = group["probability"].astype(float).tolist()
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("Forecast probabilities must sum to a positive value.")
    probabilities = [p / total for p in probabilities]
    return options, probabilities


def score_all_forecasts(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    skipped = []

    grouped = merged_df.groupby(["question_id", "forecast_source"], dropna=False)
    for (question_id, source), group in grouped:
        question_row = group.iloc[0]

        if str(question_row["question_status"]).lower() != "resolved":
            skipped.append(
                {
                    "question_id": question_id,
                    "forecast_source": source,
                    "skip_reason": f"question_status={question_row['question_status']}",
                }
            )
            continue

        options, probabilities = build_distribution(group)
        resolved_option, skip_reason = _get_resolved_option_for_group(group=group, question_row=question_row)
        if skip_reason:
            skipped.append({"question_id": question_id, "forecast_source": source, "skip_reason": skip_reason})
            continue

        if resolved_option not in options:
            skipped.append(
                {
                    "question_id": question_id,
                    "forecast_source": source,
                    "skip_reason": "resolved_option_missing_after_normalization",
                }
            )
            continue

        true_index = options.index(resolved_option)
        brier = multiclass_brier_score(probabilities, true_index)
        logscore = log_score(probabilities, true_index)
        acc = accuracy(probabilities, true_index)
        rps = ranked_probability_score(probabilities, true_index)
        top_prob = top_probability(probabilities)
        norm_entropy = normalized_entropy(probabilities)
        true_prob = true_probability(probabilities, true_index)

        results.append(
            {
                "question_id": question_id,
                "topic": question_row.get("topic", ""),
                "question_type": question_row.get("question_type", "unknown"),
                "forecast_source": source,
                "resolved_option": resolved_option,
                "brier_score": brier,
                "log_score": logscore,
                "accuracy": acc,
                "rps": rps,
                "top_probability": top_prob,
                "normalized_entropy": norm_entropy,
                "true_probability": true_prob,
            }
        )

    return pd.DataFrame(results), pd.DataFrame(skipped)


def _build_pairwise_comparison(
    results_df: pd.DataFrame,
    source_a: str,
    source_b: str,
) -> pd.DataFrame:
    metrics = [
        "brier_score",
        "log_score",
        "accuracy",
        "rps",
        "top_probability",
        "normalized_entropy",
        "true_probability",
    ]

    a = results_df[results_df["forecast_source"] == source_a].set_index("question_id")
    b = results_df[results_df["forecast_source"] == source_b].set_index("question_id")
    common = sorted(set(a.index) & set(b.index))

    rows = []
    for qid in common:
        row = {"question_id": qid}
        for metric in metrics:
            row[f"{metric}_{source_a}"] = float(a.loc[qid, metric])
            row[f"{metric}_{source_b}"] = float(b.loc[qid, metric])
        rows.append(row)

    return pd.DataFrame(rows)


def _choose_pairs(results_df: pd.DataFrame) -> List[Tuple[str, str]]:
    sources = list(dict.fromkeys(results_df["forecast_source"].tolist()))
    if len(sources) < 2:
        return []
    return list(combinations(sources, 2))


def _significance_table(comparison_df: pd.DataFrame, source_a: str, source_b: str) -> pd.DataFrame:
    rows = []
    for metric in ["brier_score", "log_score", "rps"]:
        x = comparison_df[f"{metric}_{source_a}"].astype(float).tolist()
        y = comparison_df[f"{metric}_{source_b}"].astype(float).tolist()
        rows.append({"metric": metric, **paired_t_test(x, y).__dict__})
        rows.append({"metric": metric, **wilcoxon_signed_rank_test(x, y).__dict__})

    accuracy_a = comparison_df[f"accuracy_{source_a}"].astype(int).tolist()
    accuracy_b = comparison_df[f"accuracy_{source_b}"].astype(int).tolist()
    rows.append({"metric": "accuracy", **exact_mcnemar_test(accuracy_a, accuracy_b).__dict__})
    return pd.DataFrame(rows)


def _confidence_calibration_overview(results_df: pd.DataFrame) -> None:
    for source, group in results_df.groupby("forecast_source"):
        table = calibration_table(group["top_probability"].tolist(), group["accuracy"].tolist(), bins=5)
        table.to_csv(OUTPUT_DIR / f"calibration_table_{source}.csv", index=False)
        plot_confidence_calibration(group, OUTPUT_DIR / f"calibration_{source}.png")


def run_evaluation(
    forecasts_path: str | Sequence[str] = "bioscancast_forecasts.csv",
    questions_path: str = "bioscancast_questions.csv",
) -> None:
    """End-to-end evaluation entry point."""
    _ensure_output_dir()

    if isinstance(forecasts_path, (str, Path)):
        forecast_paths = [str(forecasts_path)]
    else:
        forecast_paths = [str(p) for p in forecasts_path]

    questions = _prepare_questions(load_questions(questions_path))
    resolved_questions = questions[questions["question_status"] == "resolved"].copy()
    ambiguous_questions = questions[questions["question_status"] == "ambiguous"].copy()
    unresolved_questions = questions[questions["question_status"] == "unresolved"].copy()

    if not ambiguous_questions.empty:
        ambiguous_questions.to_csv(OUTPUT_DIR / "ambiguous_questions.csv", index=False)
    if not unresolved_questions.empty:
        unresolved_questions.to_csv(OUTPUT_DIR / "unresolved_questions.csv", index=False)

    forecast_frames = []
    for forecast_path in forecast_paths:
        source_name = _infer_source_name(forecast_path)
        forecast_df = _prepare_forecasts(load_forecasts(forecast_path), source_name=source_name)
        forecast_frames.append(forecast_df)

    forecasts = pd.concat(forecast_frames, ignore_index=True)
    merged = forecasts.merge(
        resolved_questions,
        on="question_id",
        how="inner",
        suffixes=("_forecast", "_question"),
    )

    results_df, skipped_df = score_all_forecasts(merged)

    results_path = OUTPUT_DIR / "question_level_metrics.csv"
    skipped_path = OUTPUT_DIR / "skipped_questions.csv"
    summary_path = OUTPUT_DIR / "summary_metrics.csv"
    by_type_path = OUTPUT_DIR / "summary_metrics_by_question_type.csv"

    results_df.to_csv(results_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    if results_df.empty:
        print("No questions could be scored. Check the resolved_option values.")
        return

    summary_df = compare_sources(results_df)
    summary_df.to_csv(summary_path, index=False)

    by_type_df = compare_sources_by_question_type(results_df)
    by_type_df.to_csv(by_type_path, index=False)

    plot_source_comparison(summary_df, OUTPUT_DIR / "source_comparison.png")
    plot_accuracy_by_source(summary_df, OUTPUT_DIR / "accuracy_by_source.png")

    plot_metric_distribution(results_df, "brier_score", OUTPUT_DIR / "brier_distribution.png")
    plot_metric_distribution(results_df, "log_score", OUTPUT_DIR / "log_score_distribution.png")
    plot_metric_distribution(results_df, "rps", OUTPUT_DIR / "rps_distribution.png")

    plot_metric_boxplot(
        results_df,
        "brier_score",
        OUTPUT_DIR / "brier_boxplot.png",
        ylabel="Brier score (lower is better)",
    )
    plot_metric_boxplot(
        results_df,
        "log_score",
        OUTPUT_DIR / "log_boxplot.png",
        ylabel="Log score (lower is better)",
    )
    plot_metric_boxplot(
        results_df,
        "rps",
        OUTPUT_DIR / "rps_boxplot.png",
        ylabel="RPS (lower is better)",
    )

    _confidence_calibration_overview(results_df)
    plot_reliability_overview(results_df, OUTPUT_DIR)

    pairs = _choose_pairs(results_df)

    all_stats = []
    if not pairs:
        print("\nNo pairwise comparisons could be built.")
    else:
        for source_a, source_b in pairs:
            comparison_df = _build_pairwise_comparison(results_df, source_a, source_b)

            if comparison_df.empty:
                continue

            pair_tag = f"{source_a}_vs_{source_b}"

            comparison_df.to_csv(
                OUTPUT_DIR / f"pairwise_comparison_{pair_tag}.csv",
                index=False,
            )

            stats_df = _significance_table(comparison_df, source_a, source_b)
            stats_df.insert(0, "source_a", source_a)
            stats_df.insert(1, "source_b", source_b)
            stats_df.to_csv(
                OUTPUT_DIR / f"significance_tests_{pair_tag}.csv",
                index=False,
            )

            all_stats.append(stats_df)

            for metric in ["brier_score", "log_score", "rps", "accuracy"]:
                plot_question_level_scatter(
                    comparison_df,
                    metric,
                    source_a,
                    source_b,
                    OUTPUT_DIR / f"scatter_{metric}_{pair_tag}.png",
                )
                plot_question_level_differences(
                    comparison_df,
                    metric,
                    source_a,
                    source_b,
                    OUTPUT_DIR / f"differences_{metric}_{pair_tag}.png",
                )
                plot_win_rate(
                    comparison_df,
                    metric,
                    source_a,
                    source_b,
                    OUTPUT_DIR / f"win_rate_{metric}_{pair_tag}.png",
                    lower_is_better=(metric != "accuracy"),
                )

        if all_stats:
            combined_stats_df = pd.concat(all_stats, ignore_index=True)
            combined_stats_df.to_csv(OUTPUT_DIR / "significance_tests_all_pairs.csv", index=False)

    print("\nEvaluation complete.")
    print(summary_df.to_string(index=False))

    if all_stats:
        print("\nPairwise statistical tests saved to:")
        print(OUTPUT_DIR / "significance_tests_all_pairs.csv")

    if not skipped_df.empty:
        print("\nSkipped questions:")
        print(skipped_df.to_string(index=False))


if __name__ == "__main__":
    run_evaluation()