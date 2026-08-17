"""Graph-structural descriptors for each predicted match.

The point of these is to test *where* a graph model could possibly beat a
tabular one.  The GBDT baseline receives 72 features, and every history feature
among them is a **one-hop aggregate of a single player's own record** (result
balance, game and set margins, straight-sets rate, completion rate, on all
surfaces and on the match surface).  It has no head-to-head feature and no
common-opponent feature.

So the tabular model already has the one-hop view.  The only information a
message-passing model can add is relational: how *these two players* connect
through the graph - a direct previous meeting, or a shared opponent two hops
away.  If the GNN has any structural advantage at all, it has to show up in the
matches where that connection exists and is informative.  If it does not show up
there, the graph is not buying anything anywhere.

These descriptors are symmetric in (a, b) and depend only on the graph, never on
the orientation draw, so one pass over any seed's snapshots labels every seed.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_gnn.snapshots import BlockSnapshot  # noqa: E402


def block_structure(snapshot: BlockSnapshot) -> pd.DataFrame:
    """Per-match structural descriptors for one block.

    ``edge_index`` holds the matches played *before* this block, already
    bidirectional, so each historical meeting contributes two directed edges.
    """

    neighbours: dict[int, set[int]] = defaultdict(set)
    meetings: dict[tuple[int, int], int] = defaultdict(int)
    source, target = snapshot.edge_index.tolist()
    for u, v in zip(source, target):
        neighbours[u].add(v)
        meetings[(u, v)] += 1

    rows = []
    for position, (a, b) in enumerate(
        zip(snapshot.player_a.tolist(), snapshot.player_b.tolist())
    ):
        neighbours_a = neighbours.get(a, set())
        neighbours_b = neighbours.get(b, set())
        # Each prior meeting appears once in each direction; count one side.
        head_to_head = meetings.get((a, b), 0)
        shared = neighbours_a & neighbours_b
        # A player is not their own common opponent.
        shared = shared - {a, b}
        rows.append(
            {
                "block_idx": snapshot.block_idx,
                "row_in_block": position,
                "phase": snapshot.phase,
                "year": snapshot.year,
                "head_to_head": head_to_head,
                "common_opponents": len(shared),
                "degree_a": len(neighbours_a),
                "degree_b": len(neighbours_b),
                "degree_min": min(len(neighbours_a), len(neighbours_b)),
                "connected": bool(head_to_head or shared),
                "intransitivity_level": snapshot.intransitivity_level[position],
            }
        )
    return pd.DataFrame(rows)


def structure_table(snapshots) -> pd.DataFrame:
    """Structural descriptors for every match in every block."""

    return pd.concat(
        [block_structure(snapshot) for snapshot in snapshots],
        ignore_index=True,
    )


def add_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the pre-specified stratum labels used in the analysis.

    Cut points are chosen from the *distribution* of the descriptors, not from
    any model's performance, so that the strata cannot be tuned to produce a
    result.
    """

    frame = frame.copy()
    frame["h2h_stratum"] = pd.cut(
        frame["head_to_head"],
        bins=[-0.5, 0.5, 1.5, float("inf")],
        labels=["no prior meeting", "1 prior meeting", "2+ prior meetings"],
    )
    frame["common_stratum"] = pd.cut(
        frame["common_opponents"],
        bins=[-0.5, 0.5, 4.5, 14.5, float("inf")],
        labels=["0", "1-4", "5-14", "15+"],
    )
    # Cold start.  `degree_min` is the number of distinct prior opponents of
    # the *thinner-recorded* of the two players, so it measures how much of a
    # per-player history either model could possibly have.  The 0-5 cut is the
    # one the smaller-scope analysis used and is carried over unchanged, so
    # this is a pre-specified stratum here rather than a fresh search.
    frame["degree_stratum"] = pd.cut(
        frame["degree_min"],
        bins=[-0.5, 5.5, 20.5, float("inf")],
        labels=["0-5 (cold)", "6-20", "21+"],
    )
    frame["cold_start"] = frame["degree_min"] <= 5
    # The hypothesis stratum: no direct meeting, so head-to-head cannot help,
    # but the two players are joined through shared opponents.  This is where a
    # two-hop model has information the one-hop tabular features do not.
    frame["two_hop_only"] = (frame["head_to_head"] == 0) & (
        frame["common_opponents"] > 0
    )
    return frame
