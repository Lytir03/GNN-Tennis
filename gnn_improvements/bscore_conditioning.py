# Can the GNN beat the B-score rating it is built on?
#
# The scope-matched literature comparison says it currently doesn't: on
# Slams+Masters this model scores Brier 0.1946 / LL 0.5703, and Spagnolo et
# al.'s B-score - weighted Bonacich centrality, no machine learning at all -
# scores 0.194 / 0.573 on the same tournament tiers.
#
# There is a mechanical reason it might not be able to. The direct-logit skip
# connection computes bscore_scale * (b_a - b_b), bscore_scale starts at 1.0,
# and the measured median |b_a - b_b| is 3.2e-03 - so bscore_scale has to
# reach about 15 before that term moves a logit by 1. Adam moves a scalar by
# at most about `lr` per step, so in this scope's 1344 steps at lr 1e-4 it can
# travel 0.13. The pathway the architecture was built around (the head's last
# layer is zero-initialised precisely so the model starts as pure scaled
# B-score) cannot actually do its job.
#
# The arms:
#   A1 scale15   raise the init to 15. If this alone recovers the gain, the
#                defect was purely scale.
#   A2 log       log-transform, floored. Tests whether the heavy tail matters
#                independently of the scale.
#   A3 logz      per-block standardised log: scale-free and tail-corrected,
#                and invariant to the ~10x drift in centrality magnitude
#                between scopes.
#   A4 logz+surf compose the winner with the surface-matched B-score, which
#                already won 5/5 seeds on its own.
#   B1 nodenorm  per-block node-feature normalisation. The existing
#                normalize_node_features flag standardises over whatever
#                shares the batch, so it uses cross-block statistics while
#                training and per-block statistics while predicting; this is
#                the corrected version. Node features mix height (~185) with
#                B-scores (~5e-4) and nothing normalises the encoder input.
#
# The control (bscore_general_control) already has 5 seeds on this scope and
# is not re-run.
#
# Run: python gnn_improvements/bscore_conditioning.py
#      python gnn_improvements/bscore_conditioning.py --arms logz --scope slams_masters_1990

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.config import BASE_MODEL  # noqa: E402
from tennis_gnn.run import TUNED_TRAINING, run_named  # noqa: E402

SEEDS = (42, 123, 456, 789, 2026)

ARMS = {
    "scale15": replace(BASE_MODEL, bscore_scale_init=15.0),
    "log": replace(BASE_MODEL, bscore_transform="log"),
    "logz": replace(BASE_MODEL, bscore_transform="logz"),
    "logz_surface": replace(
        BASE_MODEL, bscore_transform="logz", direct_bscore_surface=True
    ),
    "nodenorm": replace(BASE_MODEL, per_block_node_norm=True),
    # BASE_MODEL reads neither of these, so the reference config is blind to
    # per-player form while the GBDT baseline receives all twelve statistics.
    # Artifacts that switch them on score 0.594-0.596 against the control's
    # 0.6037, but every one of them was trained on a different recipe
    # (lr 3e-4, or 8 steps), so the comparison is confounded. These two arms
    # are the one-factor version on the tuned recipe.
    "history_decoder": replace(BASE_MODEL, history_to_decoder=True),
    "history_nodes": replace(BASE_MODEL, node_history_features=True),
    # The published artifacts that score best combine history with little or
    # no message passing, which is also what the closest prior work reports.
    "history_decoder_nomp": replace(
        BASE_MODEL, history_to_decoder=True, disable_message_passing=True
    ),
    # Elo beat B-score as a standalone baseline on both scopes, and blending
    # Elo into this model was the only thing all day that improved it. These
    # ask whether building the GNN on the better rating beats bolting it on
    # afterwards. "elo" keeps the input width identical, so it is a swap and
    # not a capacity change.
    "elo_rating": replace(BASE_MODEL, rating_source="elo"),
    "both_ratings": replace(BASE_MODEL, rating_source="both"),
    # The best combination available given everything measured today.
    "elo_history_decoder": replace(
        BASE_MODEL, rating_source="elo", history_to_decoder=True
    ),
    # Height supplies 99.41% of what reaches the node encoder and the four
    # B-scores supply 0.12%, so the encoder is a height detector and message
    # passing propagates height around the graph. These arms fix the input
    # rather than the architecture.
    "no_height": replace(BASE_MODEL, include_height=False),
    "scaled": replace(BASE_MODEL, node_feature_scaling="fixed"),
    "scaled_no_height": replace(
        BASE_MODEL, node_feature_scaling="fixed", include_height=False
    ),
    # history_nodes lost badly (0.6060 against 0.6024) while the same twelve
    # statistics routed past the encoder won. If the cause was drowning
    # rather than the features themselves, scaling should rescue it.
    "scaled_history_nodes": replace(
        BASE_MODEL, node_feature_scaling="fixed", node_history_features=True
    ),
    "scaled_no_height_history_nodes": replace(
        BASE_MODEL,
        node_feature_scaling="fixed",
        include_height=False,
        node_history_features=True,
    ),
}


def artifact_name(arm: str) -> str:
    return f"cond_{arm}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="slams_masters")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--arms", nargs="+", default=list(ARMS),
        help=f"any of {tuple(ARMS)}",
    )
    args = parser.parse_args()

    unknown = set(args.arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; choose from {tuple(ARMS)}")

    print(
        f"b-score conditioning: scope {args.scope}, arms {args.arms}, "
        f"seeds {args.seeds}\n",
        flush=True,
    )

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            start = time.time()
            row = run_named(
                "base",
                scope=args.scope,
                seed=seed,
                train_config=TUNED_TRAINING,
                verbose=False,
                artifact_name=artifact_name(arm),
                model_config=ARMS[arm],
            )
            row["arm"] = arm
            rows.append(row)
            print(
                f"{arm:14s} seed {seed:<5d} "
                f"val_ll={row['val_log_loss']:.4f} "
                f"test_ll={row['test_log_loss']:.4f} "
                f"[{time.time() - start:.0f}s]",
                flush=True,
            )

    print("\nmean by arm:")
    for arm in args.arms:
        vals = [r for r in rows if r["arm"] == arm]
        v = sum(r["val_log_loss"] for r in vals) / len(vals)
        t = sum(r["test_log_loss"] for r in vals) / len(vals)
        print(f"  {arm:14s} val_ll={v:.4f}  test_ll={t:.4f}  (n={len(vals)})")

    output = ROOT / "gnn_improvements" / "results"
    output.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    path = output / f"conditioning_{args.scope}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nWrote {path}")
    print(
        "\nNothing here is a result until it replicates on the extended scope "
        "(--scope slams_masters_1990); five arms were screened, and this "
        "project has a history of decisions resting on noise."
    )


if __name__ == "__main__":
    main()
