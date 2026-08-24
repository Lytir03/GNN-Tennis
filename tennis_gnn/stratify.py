# Stratified comparison of two models over graph-structural subgroups.
#
# This exists to answer one question: is the graph buying anything? The
# tabular baseline already gets one-hop history aggregates for each
# player, so the only thing message passing can add is relational - how
# the two players connect through shared opponents.
#
# Two warnings are baked into how this reports, because subgroup analysis
# is about the easiest way there is to fool yourself:
#
# 1. Multiple comparisons. Every stratum is a hypothesis test. Slice a
#    null result finely enough and some cell will clear a 95% interval.
#    The number of comparisons gets printed with the results so it can't
#    be quietly forgotten.
#
# 2. Confounding. Structural descriptors correlate with how much data a
#    player has, and data richness independently changes which model
#    wins. So a marginal subgroup difference isn't evidence about
#    structure until sparsity is controlled - and controlling it here
#    flips the sign, which is exactly the trap this module is meant to
#    make visible.
#
# The trustworthy test isn't in this file at all - it's the one_hop /
# three_hop ablation in config.py, which intervenes on the receptive
# field instead of just correlating with it. Use these tables to describe
# where models differ, and the depth ablation to figure out why.

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_gnn.experiment_tracking import (  # noqa: E402
    load_prediction_artifact,
)
# One definition of the t multiplier for the whole project.  It lived here as
# `2.776 if count == 5 else 1.96`, which silently used a normal quantile at
# three seeds and made every three-seed interval less than half its true width.
from tennis_gnn.compare import t_critical  # noqa: E402


KEY = ["block_idx", "row_in_block"]


def log_loss_per_match(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_frame(
    root: Path,
    scope: str,
    seeds,
    candidate: str,
    baseline: str,
    structure: pd.DataFrame,
    *,
    phase: str = "test",
) -> pd.DataFrame:
    # Per-match paired losses for two models, joined to structural strata.
    frames = []
    for seed in seeds:
        directory = root / "results/frozen_predictions" / scope / f"seed_{seed}"
        merged = None
        for name in (candidate, baseline):
            frame, _ = load_prediction_artifact(directory, name)
            frame = frame[frame["phase"] == phase][
                KEY + ["y_true", "probability"]
            ].rename(columns={"probability": name})
            merged = (
                frame
                if merged is None
                else merged.merge(frame.drop(columns="y_true"), on=KEY)
            )
        merged["seed"] = seed
        y = merged["y_true"].to_numpy()
        merged["loss_candidate"] = log_loss_per_match(
            y, merged[candidate].to_numpy()
        )
        merged["loss_baseline"] = log_loss_per_match(
            y, merged[baseline].to_numpy()
        )
        merged["delta"] = merged["loss_candidate"] - merged["loss_baseline"]
        frames.append(merged.merge(structure, on=KEY, how="left"))
    combined = pd.concat(frames, ignore_index=True)
    if combined["head_to_head"].isna().any():
        raise ValueError(
            "Structural join left unmatched rows; the structure table and the "
            "artifacts describe different matches."
        )
    return combined


def stratum_table(data: pd.DataFrame, by: str) -> pd.DataFrame:
    # Seed-level paired deltas within each stratum.
    #
    # Averaging within (stratum, seed) first, then treating seeds as the
    # unit of replication, is what keeps the interval honest: matches
    # within a seed share a model fit and aren't independent.
    per_seed = (
        data.groupby([by, "seed"], observed=True)
        .agg(
            n=("delta", "size"),
            delta=("delta", "mean"),
            candidate=("loss_candidate", "mean"),
            baseline=("loss_baseline", "mean"),
        )
        .reset_index()
    )
    rows = []
    for stratum, group in per_seed.groupby(by, observed=True):
        deltas = group["delta"].to_numpy()
        count = len(deltas)
        if count < 2:
            continue
        half = t_critical(count) * deltas.std(ddof=1) / np.sqrt(count)
        mean = float(deltas.mean())
        rows.append(
            {
                "stratum": stratum,
                "n_matches": int(group["n"].mean()),
                "candidate_ll": float(group["candidate"].mean()),
                "baseline_ll": float(group["baseline"].mean()),
                "delta": mean,
                "ci_low": mean - half,
                "ci_high": mean + half,
                "wins": f"{int((deltas < 0).sum())}/{count}",
                "significant": "*" if (mean - half) * (mean + half) > 0 else "",
            }
        )
    return pd.DataFrame(rows)


def interaction_test(data: pd.DataFrame, by: str) -> pd.DataFrame:
    # Difference of paired deltas between two strata, paired by seed.
    #
    # This is the test a subgroup claim actually needs. "The model wins in
    # stratum A" isn't evidence the stratum matters - the model might win
    # everywhere. It's only about structure if the advantage in A is
    # bigger than the advantage outside A, which is this contrast.
    per_seed = (
        data.groupby([by, "seed"], observed=True)["delta"].mean().unstack(0)
    )
    if per_seed.shape[1] != 2:
        raise ValueError(
            f"interaction_test needs exactly 2 strata, got {per_seed.shape[1]}"
        )
    left, right = per_seed.columns
    contrast = (per_seed[right] - per_seed[left]).dropna().to_numpy()
    count = len(contrast)
    half = t_critical(count) * contrast.std(ddof=1) / np.sqrt(count)
    mean = float(contrast.mean())
    return pd.DataFrame(
        [
            {
                "contrast": f"({right}) minus ({left})",
                "n_seeds": count,
                "difference_of_deltas": mean,
                "ci_low": mean - half,
                "ci_high": mean + half,
                "significant": "*" if (mean - half) * (mean + half) > 0 else "",
            }
        ]
    )
