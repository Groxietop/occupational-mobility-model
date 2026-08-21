"""Scoring the gravity model against the baselines it has to beat.

A model of occupational transitions can look excellent and be worthless, so
the baselines here are not a formality -- they are the argument.

  size_only          Predict flow proportional to destination employment and
                     nothing else. This is the base-rate trap made explicit.
                     Most moves land in big occupations, so this scores far
                     better than intuition suggests. Any model that does not
                     clearly beat it has discovered nothing.

  equal_similarity   The current `similarity.py` behaviour: average the eight
                     O*NET domains with equal weight. This is the incumbent,
                     and the specific thing phase 1 is trying to improve on.

  onet_related       O*NET's own expert-curated related-occupation list. A
                     human judgment of "could you move here", with no
                     reference to whether anyone did.

Metrics are computed on held-out *years*, never a random split of pairs --
splitting pairs at random leaks i->j into the training set while testing on
j->i.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pairs import similarity_columns

EPSILON = 1e-9


def poisson_deviance(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Poisson deviance. Proper scoring rule for counts; lower is better."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), EPSILON, None)
    # y * log(y / mu) is taken as 0 where y == 0, which is its limit.
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(actual > 0, actual * np.log(actual / predicted), 0.0)
    return float(2.0 * np.mean(term - (actual - predicted)))


def recall_at_k(panel: pd.DataFrame, predicted: np.ndarray, k: int = 10) -> float:
    """Per-origin recall@k against the true top-k destinations, averaged.

    The practical question a career tool answers is "given where I am, what
    are my likely next moves" -- so the metric that matters is whether the
    model's top few destinations for an origin are the ones people actually
    moved to.
    """
    frame = panel[["soc_from", "weighted_count"]].copy()
    frame["predicted"] = predicted

    scores = []
    for _, group in frame.groupby("soc_from"):
        observed = group[group["weighted_count"] > 0]
        if len(observed) == 0:
            continue
        top_n = min(k, len(observed))
        actual_top = set(observed.nlargest(top_n, "weighted_count").index)
        predicted_top = set(group.nlargest(top_n, "predicted").index)
        scores.append(len(actual_top & predicted_top) / top_n)

    return float(np.mean(scores)) if scores else float("nan")


def spearman_on_observed(panel: pd.DataFrame, predicted: np.ndarray) -> float:
    """Rank correlation restricted to pairs where a move was actually seen."""
    mask = panel["weighted_count"].to_numpy() > 0
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(panel["weighted_count"].to_numpy()[mask], predicted[mask])
    return float(rho)


def _rescale(panel: pd.DataFrame, raw_score: np.ndarray) -> np.ndarray:
    """Put a baseline's arbitrary score on the same total scale as the flows.

    The baselines produce affinity scores, not counts. Comparing them to the
    model on deviance requires them to predict the same total mass, so each
    origin's scores are normalised to that origin's observed outflow.
    """
    frame = pd.DataFrame(
        {
            "soc_from": panel["soc_from"].to_numpy(),
            "score": np.clip(raw_score, EPSILON, None),
            "actual": panel["weighted_count"].to_numpy(),
        }
    )
    totals = frame.groupby("soc_from")["score"].transform("sum")
    outflow = frame.groupby("soc_from")["actual"].transform("sum")
    return (frame["score"] / totals.replace(0, np.nan) * outflow).fillna(EPSILON).to_numpy()


def baseline_predictions(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """The three reference models, each scaled to per-origin observed outflow."""
    out: dict[str, np.ndarray] = {}

    out["size_only"] = _rescale(panel, panel["dest_employment"].to_numpy(dtype=float))

    sim_cols = similarity_columns(panel)
    equal = panel[sim_cols].mean(axis=1).to_numpy(dtype=float)
    # Cosine similarity runs [-1, 1]; shift to keep the affinity non-negative.
    equal_affinity = np.clip(equal + 1.0, EPSILON, None)
    out["equal_similarity"] = _rescale(
        panel, equal_affinity * panel["dest_employment"].to_numpy(dtype=float)
    )

    related = panel["onet_related"].to_numpy(dtype=float)
    # O*NET lists ~10 related occupations per job; give listed pairs the bulk
    # of the mass and leave a floor so unlisted pairs aren't predicted at zero.
    out["onet_related"] = _rescale(
        panel, (related * 10.0 + 0.1) * panel["dest_employment"].to_numpy(dtype=float)
    )

    return out


def score(panel: pd.DataFrame, predicted: np.ndarray, k: int = 10) -> dict[str, float]:
    return {
        "poisson_deviance": poisson_deviance(
            panel["weighted_count"].to_numpy(), predicted
        ),
        "spearman_observed": spearman_on_observed(panel, predicted),
        f"recall_at_{k}": recall_at_k(panel, predicted, k=k),
    }


def compare(
    panel: pd.DataFrame, model_predictions: np.ndarray, k: int = 10
) -> pd.DataFrame:
    """Scorecard: the fitted model against every baseline, on the same panel."""
    rows = []
    for name, values in baseline_predictions(panel).items():
        rows.append({"model": name, **score(panel, values, k=k)})
    rows.append({"model": "learned_gravity", **score(panel, model_predictions, k=k)})

    frame = pd.DataFrame(rows)
    order = ["size_only", "equal_similarity", "onet_related", "learned_gravity"]
    frame["__order"] = frame["model"].map({n: i for i, n in enumerate(order)})
    return frame.sort_values("__order").drop(columns="__order").reset_index(drop=True)
