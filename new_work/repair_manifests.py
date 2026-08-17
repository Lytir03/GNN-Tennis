"""Correct the split metadata on artifacts written before run.py knew about splits.

`tennis_gnn/run.py` recorded `train_end` / `validation_end` / `test_end` from the
module-level constants, which are the *original* split.  Once a second split
existed, every full-scope GNN artifact therefore claimed (2015, 2016, 2020) while
the data behind it actually used (2016, 2018, 2024).  The GBDT recorded the truth,
so `assert_compatible` refused to compare artifacts that are in fact comparable -
which is the guard working correctly on bad metadata.

The predictions themselves were never wrong: phases come from `load_dataset`,
which had the right split all along.  Only the recorded description was wrong.

This repairs the description, and refuses to touch anything whose
`evaluation_hash` does not still match its CSV - otherwise a genuine corruption
could be silently relabelled as a metadata fix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tennis_gnn.data import SCOPE_FILES  # noqa: E402
from tennis_gnn.experiment_tracking import evaluation_hash  # noqa: E402


def repair(scope: str, *, apply: bool) -> int:
    split = SCOPE_FILES[scope][2]
    expected = {
        "train_end": split.train_end,
        "validation_end": split.val_end,
        "test_end": int(split.rolling_end[:4]) - 1,
    }
    root = ROOT / "results" / "frozen_predictions" / scope
    if not root.is_dir():
        print(f"No artifacts for scope {scope!r}")
        return 0

    changed = refused = 0
    for manifest_path in sorted(root.glob("seed_*/*.manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        wrong = {
            field: (manifest.get(field), value)
            for field, value in expected.items()
            if manifest.get(field) != value
        }
        if not wrong:
            continue

        # Verify the artifact is intact before relabelling it.
        frame = pd.read_csv(manifest_path.with_suffix("").with_suffix(".csv"))
        if evaluation_hash(frame) != manifest["evaluation_hash"]:
            print(
                f"  REFUSED {manifest_path.parent.name}/"
                f"{manifest['experiment']}: evaluation_hash does not match its "
                "CSV, so this is not a metadata-only problem"
            )
            refused += 1
            continue

        label = f"{manifest_path.parent.name}/{manifest['experiment']}"
        print(f"  {'fixing' if apply else 'would fix'} {label}: " + ", ".join(
            f"{field} {old} -> {new}" for field, (old, new) in wrong.items()
        ))
        if apply:
            manifest.update(expected)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        changed += 1

    print(
        f"\n{changed} artifact(s) {'repaired' if apply else 'need repair'}"
        + (f", {refused} refused" if refused else "")
    )
    return 1 if refused else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="full")
    parser.add_argument(
        "--apply", action="store_true", help="without this, only reports"
    )
    args = parser.parse_args()
    return repair(args.scope, apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
