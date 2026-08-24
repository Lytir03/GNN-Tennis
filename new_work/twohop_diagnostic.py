# Was the 2-hop penalty in the tier-3 grid row a depth effect or a recipe artifact?
#
# The full-scope grid reported gain_2_vs_1 = +0.02368 for the richest
# feature tier - a large penalty for going from one hop to two. That
# number was produced under tier 3's recipe (lr 3e-4), which was
# selected on the smaller scope and never re-searched at full scope; the
# limitation is recorded in new_work/STATUS.md. Two hops touch far more
# of the graph per update, so a step size chosen for one hop is exactly
# the thing that would break first.
#
# The test: rerun the same 2-hop model, same seeds, same data, changing
# only the learning rate to 1e-4. If the penalty is depth, it survives.
# If it's the recipe, most of it disappears.
#
# Run: python new_work/twohop_diagnostic.py

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.compare import per_seed_metrics  # noqa: E402

SCOPE = "full"
SEEDS = (42, 123)
ARMS = {
    "1 hop, lr 3e-4 (grid)": "grid_3_history_decoder_1hop",
    "2 hop, lr 3e-4 (grid)": "grid_3_history_decoder_2hop",
    "2 hop, lr 1e-4 (diag)": "diag_2hop_lr1e4",
}


def main() -> None:
    rows = []
    for label, artifact in ARMS.items():
        for seed in SEEDS:
            frame = pd.read_csv(
                ROOT
                / "results"
                / "frozen_predictions"
                / SCOPE
                / f"seed_{seed}"
                / f"{artifact}.csv"
            )
            test = frame[frame["phase"] == "test"]
            p = test["probability"].to_numpy().clip(1e-7, 1 - 1e-7)
            y = test["y_true"].to_numpy()
            rows.append(
                {
                    "arm": label,
                    "seed": seed,
                    "n": len(test),
                    "log_loss": float(
                        -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                    ),
                    "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
                }
            )

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))

    print("\nPaired differences in test log loss (negative = first arm better):")
    wide = frame.pivot(index="seed", columns="arm", values="log_loss")
    for a, b in (
        ("2 hop, lr 1e-4 (diag)", "2 hop, lr 3e-4 (grid)"),
        ("2 hop, lr 1e-4 (diag)", "1 hop, lr 3e-4 (grid)"),
    ):
        diff = wide[a] - wide[b]
        print(
            f"  {a} - {b}: "
            + ", ".join(f"seed {s} {d:+.5f}" for s, d in diff.items())
            + f"  (mean {diff.mean():+.5f})"
        )

    output = ROOT / "new_work" / "results" / "twohop_diagnostic.csv"
    frame.to_csv(output, index=False)
    print(f"\nWrote {output}")
    print(
        "\nTwo seeds is a diagnostic, not an estimate.  It is enough to say the"
        "\ntier-3 2-hop row cannot be read as a depth result; it is not enough"
        "\nto quote a corrected depth effect."
    )


if __name__ == "__main__":
    main()
