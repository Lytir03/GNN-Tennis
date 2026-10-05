# Validation-only recipe search for an arbitrary ModelConfig.
#
# tennis_gnn/tune.py already does this, but its CLI resolves --model through
# one_factor_ablations(), so it can only search recipes for a *named*
# ablation. The configs in gnn_improvements/ are ad-hoc combinations that
# deliberately do not belong in one_factor_ablations() (adding them would
# break its guarantee that every entry differs from BASE_MODEL in exactly one
# field). So this reuses tune.py's spaces and run_experiment directly rather
# than duplicating a search.
#
# Selection is on validation only, never test - the rule the rest of the
# project follows.
#
# Run: python gnn_improvements/retune.py --arm scale15 --space focused

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.config import BASE_MODEL  # noqa: E402
from tennis_gnn.data import load_dataset  # noqa: E402
from tennis_gnn.snapshots import load_or_build  # noqa: E402
from tennis_gnn.train import metrics, run_experiment  # noqa: E402
from tennis_gnn.config import TrainConfig  # noqa: E402
from tennis_gnn.tune import focused_space, search_space  # noqa: E402
from gnn_improvements.bscore_conditioning import ARMS  # noqa: E402


def scale_space() -> list[TrainConfig]:
    # The two axes tune.py never searched properly.
    #
    # replay_buffer_size has never been varied at all: every configuration in
    # focused_space() and search_space() leaves it at the default 200, and
    # only replay_batch_size (None vs 32) moves. It is the axis most likely to
    # be wrong at scale, because the buffer holds 200 *blocks* - on
    # slams_masters_1990 that spans roughly a fifth of the history it covered
    # on the scope where the recipe was chosen.
    #
    # hidden_dim is only half-searched: search_space() tries (32, 64) but
    # pinned at lr 3e-4 / 8 steps, never at the lr 1e-4 / 4 steps recipe that
    # actually won and is used everywhere.
    #
    # Held at the winning recipe on every other axis so this is a clean
    # two-factor grid rather than another confound.
    return [
        TrainConfig(
            learning_rate=1e-4,
            steps_per_block=4,
            replay_batch_size=32,
            passes=1,
            calibrate=True,
            replay_buffer_size=buffer_size,
            hidden_dim=hidden,
        )
        for buffer_size in (100, 200, 400, 800)
        for hidden in (32, 64, 128)
    ]


def buffer_space() -> list[TrainConfig]:
    # scale_space() came back with its winner at 800, the edge of its range,
    # and a perfectly monotone improvement across it (0.5526, 0.5493, 0.5469,
    # 0.5452 at hidden_dim=32). A search that stops at its own boundary has
    # reported the range, not an optimum, so this continues past it.
    #
    # hidden_dim is pinned at 32: scale_space() settled that axis, with 64
    # slightly worse and 128 far worse at every buffer size, and dropping it
    # makes each run ~230s instead of ~800s.
    return [
        TrainConfig(
            learning_rate=1e-4,
            steps_per_block=4,
            replay_batch_size=32,
            passes=1,
            calibrate=True,
            replay_buffer_size=buffer_size,
            hidden_dim=32,
        )
        for buffer_size in (800, 1600, 3200, 6400)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="slams_masters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arm", default="control",
        help="'control' for BASE_MODEL, or any arm in bscore_conditioning.ARMS",
    )
    parser.add_argument(
        "--space", choices=("full", "focused", "scale", "buffer"),
        default="focused",
    )
    args = parser.parse_args()

    model_config = (
        BASE_MODEL if args.arm == "control" else ARMS[args.arm]
    )
    configs = {
        "focused": focused_space,
        "full": search_space,
        "scale": scale_space,
        "buffer": buffer_space,
    }[args.space]()

    dataset = load_dataset(ROOT, scope=args.scope)
    snapshots = load_or_build(
        ROOT, dataset, model_config.edge_preset, seed=args.seed, verbose=False
    )

    print(
        f"retune: arm={args.arm} scope={args.scope} seed={args.seed} "
        f"{len(configs)} configs, validation only\n",
        flush=True,
    )

    rows = []
    for index, train_config in enumerate(configs, start=1):
        start = time.time()
        result = run_experiment(
            snapshots,
            model_config,
            replace(train_config, seed=args.seed),
            verbose=False,
            eval_phases=("val",),
        )
        predictions = result["predictions"]
        validation = predictions[predictions["phase"] == "val"]
        scores = metrics(validation)
        rows.append(
            {
                **{f"val_{k}": v for k, v in scores.items()},
                "temperature": result["temperature"],
                "seconds": time.time() - start,
                **asdict(train_config),
            }
        )
        print(
            f"  [{index:2d}/{len(configs)}] lr={train_config.learning_rate:.0e} "
            f"steps={train_config.steps_per_block} "
            f"hidden={train_config.hidden_dim} "
            f"drop={train_config.dropout} "
            f"val_ll={scores['log_loss']:.4f} "
            f"[{time.time() - start:.0f}s]",
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values("val_log_loss").reset_index(drop=True)
    output = ROOT / "gnn_improvements" / "results"
    output.mkdir(parents=True, exist_ok=True)
    # The space belongs in the name: without it a second search silently
    # overwrites the first, and the boundary evidence that motivated it.
    path = (
        output
        / f"retune_{args.space}_{args.arm}_{args.scope}_seed{args.seed}.csv"
    )
    frame.to_csv(path, index=False)
    print(f"\nbest on validation:\n{frame.head(3).to_string(index=False)}")
    print(f"\nWrote {path}")
    print(
        "\nOne seed of a validation search picks a recipe; it does not measure "
        "one. Confirm the winner on the full seed set before quoting it."
    )


if __name__ == "__main__":
    main()
