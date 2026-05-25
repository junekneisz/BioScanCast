"""Legacy helpers kept for backwards compatibility with older notebooks/tests."""

from bioscancast.stages.eval_stage.scoring import (
    accuracy,
    binary_brier_score,
    binary_log_score,
    log_score,
    multiclass_brier_score,
    ranked_probability_score,
)


def evaluate_binary(predictions):
    results = []
    for p in predictions:
        prob = p["prob"]
        outcome = p["outcome"]
        results.append({
            "brier": binary_brier_score(prob, outcome),
            "log": binary_log_score(prob, outcome),
        })
    return results


def evaluate_range(predictions):
    results = []
    for p in predictions:
        probs = p["probs"]
        true_idx = p["true_bucket"]
        results.append({
            "range_brier": multiclass_brier_score(probs, true_idx),
            "range_rps": ranked_probability_score(probs, true_idx),
        })
    return results


def evaluate_open(predictions):
    results = []
    for p in predictions:
        results.append({"confidence": p.get("confidence", None)})
    return results
