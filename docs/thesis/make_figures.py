# Builds the data figures for the thesis chapters in this folder.
#
# Everything is read from files already on disk - the processed match data,
# the frozen prediction artifacts and the result CSVs.  Nothing is retrained.
# Each figure is written as a vector PDF (for the thesis) and a PNG preview.
#
# Run from the repository root:
#   ~/miniconda3/envs/tennis-gnn/bin/python docs/thesis/make_figures.py
#
# Sign convention used in every "gain" figure: positive means the graph (or the
# GNN) is better.  Log-loss gains are therefore baseline minus candidate.

from __future__ import annotations

import os
import sys
from pathlib import Path

# The pip PyTorch wheel and conda NumPy ship separate OpenMP runtimes on macOS;
# needed only for the strata figure, which loads cached graph snapshots.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FROZEN = ROOT / "results" / "frozen_predictions"
SEEDS = (42, 123, 456, 789, 2026)

# Validated categorical slots (dataviz reference palette, light mode), used in
# fixed order.  Slot 3 is below 3:1 contrast, so every series also carries a
# marker shape and a direct label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, NEUTRAL = "#e6e5e0", "#c9c8c2"
SCOPE_STYLE = {
    "slams_masters": ("Slams + Masters", BLUE, "o"),
    "full": ("Full tour", ORANGE, "s"),
    "slams_masters_1990": ("Slams + Masters, 1990", AQUA, "^"),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    }
)

WIDTH = 6.3  # A4 text width in inches


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


def t_critical(n: int) -> float:
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)


# ---------------------------------------------------------------- artifacts --

_cache: dict = {}


def test_frame(scope: str, name: str, seed: int) -> pd.DataFrame | None:
    key = (scope, name, seed)
    if key not in _cache:
        path = FROZEN / scope / f"seed_{seed}" / f"{name}.csv"
        if not path.is_file():
            _cache[key] = None
        else:
            frame = pd.read_csv(path)
            _cache[key] = frame[frame["phase"] == "test"].reset_index(drop=True)
    return _cache[key]


def metric(frame: pd.DataFrame, which: str) -> float:
    y = frame["y_true"].to_numpy(float)
    p = np.clip(frame["probability"].to_numpy(float), 1e-7, 1 - 1e-7)
    if which == "log_loss":
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    if which == "brier":
        return float(np.mean((p - y) ** 2))
    return float(np.mean((p >= 0.5) == y))


def per_seed(scope: str, name: str, which: str = "log_loss") -> dict[int, float]:
    out = {}
    for seed in SEEDS:
        frame = test_frame(scope, name, seed)
        if frame is not None:
            out[seed] = metric(frame, which)
    return out


def paired_gain(scope, candidate, baseline, which="log_loss"):
    # Gain of candidate over baseline, positive = candidate better.
    a, b = per_seed(scope, candidate, which), per_seed(scope, baseline, which)
    seeds = sorted(set(a) & set(b))
    sign = -1.0 if which in {"log_loss", "brier"} else 1.0
    d = np.array([sign * (a[s] - b[s]) for s in seeds])
    half = t_critical(len(d)) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return {"mean": d.mean(), "low": d.mean() - half, "high": d.mean() + half,
            "n": len(d), "seeds": d}


def zero_line(ax, orientation="h"):
    (ax.axhline if orientation == "h" else ax.axvline)(0, color=MUTED, lw=0.8, zorder=1)


# ------------------------------------------------------------------ fig 04 --

def fig04_split_timeline():
    scopes = [
        # label, (warmup start, rolling start, train end, val end, test end), counts
        ("Slams + Masters", (2006, 2011, 2015, 2016, 2020), (5391, 5367, 1075, 3770)),
        ("Full tour", (2006, 2011, 2016, 2018, 2024), (13710, 15809, 5325, 14340)),
        ("Slams + Masters,\n1990 extension", (1980, 1990, 2011, 2013, 2020), (7360, 23338, 2142, 6995)),
    ]
    colours = {"Warm-up": NEUTRAL, "Train": BLUE, "Validation": ORANGE, "Test": AQUA}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    for row, (label, (w0, r0, tr, va, te), counts) in enumerate(scopes):
        y = len(scopes) - 1 - row
        spans = [("Warm-up", w0, r0), ("Train", r0, tr + 1),
                 ("Validation", tr + 1, va + 1), ("Test", va + 1, te + 1)]
        for (phase, start, end), n in zip(spans, counts):
            ax.barh(y, end - start, left=start, height=0.62, color=colours[phase],
                    edgecolor="white", linewidth=1.5, zorder=2)
            if end - start >= 3:
                ax.text((start + end) / 2, y, f"{start}–{end - 1}\n{n:,}", ha="center",
                        va="center", fontsize=7, zorder=3,
                        color="white" if phase == "Train" else INK)
            else:
                span = f"{start}" if end - start == 1 else f"{start}–{end - 1}"
                ax.text((start + end) / 2, y + 0.36, f"{span}: {n:,}", ha="center",
                        va="bottom", fontsize=6.5, color=INK_2, zorder=3)
        # First rolling year: online state accumulates, no gradient updates.
        ax.barh(y, 1, left=r0, height=0.62, facecolor="none", edgecolor="white",
                hatch="////", linewidth=0, zorder=3)
        ax.text(te + 1.6, y, f"{sum(counts[1:]):,} scored", va="center",
                fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(scopes)), [s[0] for s in reversed(scopes)])
    ax.set_xlim(1979, 2030)
    ax.set_xlabel("Calendar year")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(color=c, label=k) for k, c in colours.items()]
    handles.append(Patch(facecolor=BLUE, edgecolor="white", hatch="////",
                         label="First training year (no gradient updates)"))
    ax.legend(handles=handles, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1.2, columnspacing=1.0, fontsize=7.5)
    save(fig, "fig04_split_timeline")


# ------------------------------------------------------------------ fig 05 --

def fig05_bscore_example():
    # Toy win graph: X has two wins against the strong group, Y three wins
    # against the weak group.  Scores come from the same networkx call as
    # new_work/build_bscore.py, all matches recent (weight 1).
    import networkx as nx

    matches = [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"),   # strong group
               ("X", "A"), ("X", "B"), ("C", "X"),
               ("Y", "D"), ("Y", "E"), ("Y", "F"), ("A", "Y"),
               ("D", "E"), ("E", "F"), ("F", "C")]                # weak group, one upset
    graph = nx.DiGraph()
    for winner, loser in matches:
        graph.add_edge(loser, winner, weight=1.0)
    score = nx.eigenvector_centrality(graph, weight="weight", max_iter=1000, tol=1e-9)
    wins = {n: sum(w == n for w, _ in matches) for n in graph}
    losses = {n: sum(l == n for _, l in matches) for n in graph}

    pos = {"A": (1.2, 2.2), "B": (0.3, 1.2), "C": (2.1, 1.2), "X": (-0.6, 2.2),
           "Y": (3.4, 2.2), "D": (2.8, 0.9), "E": (4.0, 0.9), "F": (3.4, -0.1)}
    colour = {"X": BLUE, "Y": ORANGE}
    fig, (ax_g, ax_b) = plt.subplots(1, 2, figsize=(WIDTH, 2.9),
                                     gridspec_kw={"width_ratios": [1.45, 1]})

    radius = {n: 0.16 + 0.24 * score[n] for n in graph}
    for loser, winner in graph.edges:
        (x0, y0), (x1, y1) = np.array(pos[loser]), np.array(pos[winner])
        d = np.array([x1 - x0, y1 - y0])
        u = d / np.linalg.norm(d)
        both = graph.has_edge(winner, loser)
        # Offset the two arcs of a return match sideways so they do not overlap.
        side = np.array([-u[1], u[0]]) * (0.07 if both else 0.0)
        start = np.array(pos[loser]) + u * (radius[loser] + 0.03) + side
        end = np.array(pos[winner]) - u * (radius[winner] + 0.03) + side
        ax_g.annotate("", xy=end, xytext=start, zorder=2,
                      arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9,
                                      mutation_scale=9, shrinkA=0, shrinkB=0))
    for n, (x, y) in pos.items():
        c = colour.get(n, NEUTRAL)
        ax_g.add_patch(plt.Circle((x, y), radius[n], facecolor=c, edgecolor="white", lw=1.2, zorder=3))
        ax_g.text(x, y, n, ha="center", va="center", fontsize=8.5, zorder=4, fontweight="bold",
                  color="white" if n in colour else INK)
    ax_g.text(1.2, 0.55, "strong group", ha="center", fontsize=7.5, color=INK_2, style="italic")
    ax_g.text(4.25, 0.1, "weak group", ha="center", fontsize=7.5, color=INK_2, style="italic")
    ax_g.set_xlim(-1.0, 4.6)
    ax_g.set_ylim(-0.4, 2.7)
    ax_g.set_aspect("equal")
    ax_g.axis("off")
    ax_g.set_title("(a) Win graph: arrow from loser to winner", loc="left")

    order = sorted(score, key=score.get)
    y = np.arange(len(order))
    ax_b.barh(y, [score[n] for n in order], height=0.62, zorder=2,
              color=[colour.get(n, NEUTRAL) for n in order])
    for yi, n in zip(y, order):
        ax_b.text(score[n] + 0.012, yi, f"{score[n]:.2f}   ({wins[n]}–{losses[n]})",
                  va="center", fontsize=7.2, color=INK)
    ax_b.set_yticks(y, order, fontsize=8)
    ax_b.set_xlim(0, 0.78)
    ax_b.set_xlabel("B-score")
    ax_b.grid(axis="y", visible=False)
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_title("(b) B-score (win–loss record)", loc="left")
    fig.tight_layout()
    fig.text(0.01, -0.04, "Y has the better record (3–1) but beat only weak players; X (2–1) beat A and B "
             "and scores higher.\nAll matches weighted 1; in the thesis each match is weighted "
             "by 1/(1 + age/365) and weights of repeated pairings are summed.",
             fontsize=7, color=INK_2)
    save(fig, "fig05_bscore_example")


# ------------------------------------------------------------------ fig 06 --

def fig06_snapshot_timeline():
    # Schematic of the rolling snapshot procedure (section 4.2): two consecutive
    # prediction steps, each with its timeline of round blocks and its graph.
    from matplotlib.patches import FancyBboxPatch

    n_blocks, window = 12, 6
    matches = [2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 3, 2]
    fig = plt.figure(figsize=(WIDTH, 3.3))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], hspace=0.55, wspace=0.05)

    def timeline(ax, current, title, now):
        lo = current - window
        ax.axvspan(lo - 0.5, current - 0.5, color=BLUE, alpha=0.08, lw=0, zorder=0)
        for b in range(n_blocks):
            if b < lo:
                face, edge, hatch = "white", NEUTRAL, "////"
            elif b < current:
                face, edge, hatch = BLUE, BLUE, None
            elif b == current:
                face, edge, hatch = ORANGE, ORANGE, None
            else:
                face, edge, hatch = "white", NEUTRAL, None
            ax.add_patch(FancyBboxPatch((b - 0.36, -0.36), 0.72, 0.72,
                                        boxstyle="round,pad=0,rounding_size=0.08",
                                        facecolor=face, edgecolor=edge, hatch=hatch, lw=1.0, zorder=2))
            dot = "white" if lo <= b <= current else NEUTRAL
            for k in range(matches[b]):
                ax.plot(b + (k - (matches[b] - 1) / 2) * 0.2, 0, "o", ms=2.6, color=dot, zorder=3)
        # Window bracket [t - W, t).
        ax.annotate("", xy=(lo - 0.5, 0.62), xytext=(current - 0.5, 0.62),
                    arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.9, shrinkA=0, shrinkB=0))
        ax.text((lo + current - 1) / 2, 0.72, f"graph-history window  $[{now}-W,\\ {now})$",
                ha="center", va="bottom", fontsize=7.3, color=BLUE)
        ax.plot([current - 0.5] * 2, [-0.5, 0.62], color=INK, lw=0.8, ls="--", zorder=1)
        ax.text(current - 0.5, -0.55, f"${now}$", ha="center", va="top", fontsize=8.5)
        ax.text(current + 0.12, -0.55, "predict", ha="left", va="top", fontsize=7, color=ORANGE)
        ax.set_xlim(-0.6, n_blocks - 0.4)
        ax.set_ylim(-0.95, 1.1)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=8.5)

    def snapshot(ax, removed, added, title):
        pos = {"a": (0.0, 1.0), "b": (1.0, 1.25), "c": (0.55, 0.3), "d": (1.6, 0.55),
               "e": (-0.35, 0.2), "f": (1.2, -0.3)}
        edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("a", "e")]
        edges += [("d", "f")] if added else []
        for u, v in edges:
            if (u, v) == removed:
                continue
            new = (u, v) == ("d", "f")
            ax.plot(*zip(pos[u], pos[v]), color=BLUE, lw=1.6 if new else 1.0, zorder=1)
        if removed and not added:
            u, v = removed
            ax.plot(*zip(pos[u], pos[v]), color=BLUE, lw=1.0, zorder=1)
        # The pair being predicted in this step.
        pair = ("d", "f") if not added else ("c", "f")
        ax.plot(*zip(pos[pair[0]], pos[pair[1]]), color=ORANGE, lw=1.3, ls=(0, (3, 2)), zorder=1)
        for n, (x, y) in pos.items():
            gone = added and n == "e"
            ax.plot(x, y, "o", ms=10, color="white" if gone else NEUTRAL,
                    mec=NEUTRAL if gone else "white", mew=1.0, zorder=2)
            ax.text(x, y, n, ha="center", va="center", fontsize=7, zorder=3,
                    color=MUTED if gone else INK)
        ax.set_xlim(-0.7, 2.0)
        ax.set_ylim(-0.65, 1.55)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=8.5)

    ax_t0, ax_g0 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_t1, ax_g1 = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    timeline(ax_t0, 8, "(a) Step $t$: build $G_t$ from the window, predict the current round", "t")
    snapshot(ax_g0, None, False, "$G_t$")
    timeline(ax_t1, 9, "(b) Step $t+1$: add the completed round, slide the window", "t+1")
    snapshot(ax_g1, ("a", "e"), True, "$G_{t+1}$")

    ax_t1.annotate("drops out", xy=(2, 0.38), xytext=(1.3, 0.85), ha="center", fontsize=7,
                   color=INK_2, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7,
                                                mutation_scale=7))
    ax_t1.annotate("added", xy=(8, -0.38), xytext=(7.1, -0.78), ha="center", fontsize=7,
                   color=INK_2, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7,
                                                mutation_scale=7))
    handles = [Patch(facecolor="white", edgecolor=NEUTRAL, hatch="////", label="Outside window"),
               Patch(color=BLUE, label="Edges of the snapshot"),
               Patch(color=ORANGE, label="Round being predicted"),
               Patch(facecolor="white", edgecolor=NEUTRAL, label="Not yet played")]
    fig.legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0.06, 0.95),
               handlelength=1.2, columnspacing=1.2, fontsize=7.5)
    fig.text(0.06, -0.02, "Boxes: tournament-round blocks in chronological order; dots: matches. "
             "Right: snapshot graphs. Dashed orange: pair being predicted,\nwhose players are "
             "nodes even without a match in the window (f in $G_t$). "
             "In $G_{t+1}$ the match a–e has aged out, so e leaves the graph (hollow).",
             fontsize=7, color=INK_2)
    save(fig, "fig06_snapshot_timeline")


# ------------------------------------------------------------------ fig 07 --

def fig07_matches_per_year():
    rounds = {"R128", "R64", "R32", "R16", "QF", "SF", "F"}
    m = pd.read_csv(ROOT / "data" / "processed" / "atp_matches_full.csv",
                    usecols=["tourney_date", "tourney_level", "round",
                             "winner_name", "loser_name"], low_memory=False)
    m = m[m["round"].isin(rounds) & m["winner_name"].notna() & m["loser_name"].notna()]
    m["year"] = pd.to_datetime(m["tourney_date"].astype(str)).dt.year
    table = m.groupby(["year", "tourney_level"]).size().unstack(fill_value=0)
    levels = [("G", "Grand Slam", BLUE), ("M", "Masters 1000", ORANGE),
              ("A", "ATP 500 / 250", AQUA)]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    bottom = np.zeros(len(table))
    for code, label, colour in levels:
        values = table.get(code, pd.Series(0, index=table.index)).to_numpy()
        ax.bar(table.index, values, bottom=bottom, width=0.8, color=colour,
               edgecolor="white", linewidth=1.0, label=label, zorder=2)
        bottom += values
    ax.set_ylabel("Main-draw matches")
    ax.set_xticks(table.index[::2])
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    if 2020 in table.index:
        ax.annotate("2020 season\ninterrupted", xy=(2020, bottom[list(table.index).index(2020)]),
                    xytext=(2020, bottom.max() * 0.93), ha="center", fontsize=7,
                    color=INK_2, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    save(fig, "fig07_matches_per_year")


# ------------------------------------------------------------------ fig 08 --

def fig08_recency_weight():
    days = np.linspace(0, 1500, 600)
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.3))
    ax.plot(days, 1 / (1 + days / 365), color=BLUE, zorder=3)
    for d, text in ((365, "365 days\ngraph window (final)"),
                    (1095, "1095 days\nhistory features;\ndefault graph window")):
        w = 1 / (1 + d / 365)
        ax.axvline(d, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.plot([d], [w], "o", color=BLUE, mec="white", mew=1.2, zorder=4)
        ax.annotate(f"w = {w:.2f}", (d, w), xytext=(6, 6), textcoords="offset points",
                    fontsize=7.5, color=INK)
        ax.text(d + 18, 0.97, text, fontsize=7, color=INK_2, va="top")
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Age of the match at prediction time (days)")
    ax.set_ylabel("Recency weight  $w = 1/(1+\\mathrm{age}/365)$")
    save(fig, "fig08_recency_weight")


# ------------------------------------------------------------------ fig 09 --

TIER_LABELS = ["0\nno B-score,\nno history", "1\nB-score\n+ static",
               "2\n+ history\non nodes", "3\n+ history\nat decoder"]


def substitution_pairs(scope: str) -> list[tuple[str, str]]:
    # (one hop, no messages) per tier, as the grid stores them.
    if scope == "slams_masters":
        return [("grid_0_no_bscore_1hop", "grid_0_no_bscore_nonehop"),
                ("gnn_one_hop", "grid_1_bscore_nonehop"),
                ("gnn_history_tuned_one_hop", "grid_2_history_nodes_nonehop"),
                ("gnn_decoder_one_hop", "grid_3_history_decoder_nonehop")]
    if scope == "full":
        return [("grid_0_no_bscore_1hop", "grid_0_no_bscore_nonehop"),
                ("grid_1_bscore_1hop", "grid_1_bscore_nonehop"),
                ("grid_2_history_nodes_1hop", "grid_2_history_nodes_nonehop"),
                # Stable-recipe re-run; the grid's own lr 3e-4 cell handicaps
                # the message-passing arm (see chapter 6).
                ("depth_3_history_decoder_1hop_lr1e4", "depth_3_history_decoder_nonehop_lr1e4")]
    return [(f"hopgrid_{t}_1hop", f"hopgrid_{t}_nonehop")
            for t in ("t0_nothing", "t1_bscore", "t2_history_nodes", "t3_history_decoder")]


def fig09_substitution_curve():
    results = {scope: [paired_gain(scope, c, b) for c, b in substitution_pairs(scope)]
               for scope in SCOPE_STYLE}
    fig, (ax_all, ax_zoom) = plt.subplots(
        1, 2, figsize=(WIDTH, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    offsets = {"slams_masters": -0.12, "full": 0.0, "slams_masters_1990": 0.12}
    for ax, tiers in ((ax_all, range(4)), (ax_zoom, range(1, 4))):
        zero_line(ax)
        for scope, (label, colour, marker) in SCOPE_STYLE.items():
            xs = np.array(list(tiers)) + offsets[scope]
            r = [results[scope][t] for t in tiers]
            mean = np.array([x["mean"] for x in r])
            err = np.array([[x["mean"] - x["low"], x["high"] - x["mean"]] for x in r]).T
            ax.plot(xs, mean, color=colour, lw=1.2, zorder=2)
            ax.errorbar(xs, mean, yerr=err, fmt="none", ecolor=colour, elinewidth=1.0,
                        capsize=2, zorder=3)
            for x, t, m in zip(xs, tiers, mean):
                hollow = scope == "full" and t == 3
                ax.plot([x], [m], marker=marker, color=colour, zorder=4,
                        mfc="white" if hollow else colour, mec=colour if hollow else "white",
                        mew=1.2)
        ax.set_xticks(list(tiers), [TIER_LABELS[t] for t in tiers], fontsize=7.5)
        ax.grid(axis="x", visible=False)
    ax_all.set_ylabel("Log-loss gain of one hop\nover no message passing")
    ax_all.set_title("(a) All feature tiers", loc="left")
    ax_zoom.set_title("(b) Tiers 1–3, enlarged", loc="left")
    handles = [Line2D([], [], color=c, marker=m, mec="white", label=l)
               for l, c, m in SCOPE_STYLE.values()]
    handles.append(Line2D([], [], ls="none", marker="s", mfc="white", mec=ORANGE,
                          label="lr 1e-4 re-run"))
    fig.legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0.06, 0.98))
    fig.text(0.01, -0.04, "Positive = the graph helps. Bars: 95% paired CI over seeds "
             "(5 seeds; 3 for full-tour tiers 0–2). Default 1095-day graph window.",
             fontsize=7, color=INK_2)
    fig.tight_layout()
    save(fig, "fig09_substitution_curve")


# ------------------------------------------------------------------ fig 10 --

def fig10_window_sweep():
    sweep = pd.read_csv(ROOT / "gnn_improvements" / "results" / "window_hop_all.csv")
    sweep = sweep[sweep["tier"] == "t3_history_decoder"]
    grid90 = pd.read_csv(ROOT / "gnn_improvements" / "results" / "hop_grid_slams_masters_1990.csv")

    def gains(frame, window_col, value_col, hops, windows):
        rows = []
        for w in windows:
            sub = frame[frame[window_col] == w]
            a = sub[sub["hops"].astype(str) == hops].set_index("seed")[value_col]
            b = sub[sub["hops"].astype(str) == "none"].set_index("seed")[value_col]
            d = (b - a).dropna().to_numpy()
            half = t_critical(len(d)) * d.std(ddof=1) / np.sqrt(len(d))
            rows.append((w, d.mean(), half, int((d > 0).sum()), len(d)))
        return rows

    sm = sweep[sweep["scope"] == "slams_masters"]
    a_rows = gains(sm, "window", "test", "2", [90, 180, 270, 365, 547, 730, 1095])
    s90 = sweep[sweep["scope"] == "slams_masters_1990"]
    b_rows = gains(s90, "window", "test", "1", [365, 547])
    g = grid90[grid90["tier"] == "t3_history_decoder"].assign(window=1095)
    b_rows += gains(g, "window", "test_log_loss", "1", [1095])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH, 2.6),
                                     gridspec_kw={"width_ratios": [2.1, 1]}, sharey=True)
    for ax, rows, colour, marker, title in (
        (ax_a, a_rows, BLUE, "o", "(a) Slams + Masters, two hops"),
        (ax_b, b_rows, AQUA, "^", "(b) 1990 extension, one hop"),
    ):
        zero_line(ax)
        w = np.array([r[0] for r in rows])
        m = np.array([r[1] for r in rows])
        h = np.array([r[2] for r in rows])
        ax.plot(w, m, color=colour, lw=1.2, zorder=2)
        ax.errorbar(w, m, yerr=h, fmt="none", ecolor=colour, elinewidth=1.0, capsize=2, zorder=3)
        ax.plot(w, m, marker, color=colour, mec="white", mew=1.2, zorder=4)
        for wi, mi, hi, wins, n in rows:
            ax.annotate(f"{wins}/{n}", (wi, mi + hi), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=6.5, color=INK_2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Graph-history window (days)")
    ax_a.set_xticks([90, 180, 270, 365, 547, 730, 1095])
    ax_a.tick_params(axis="x", labelrotation=0, labelsize=7.5)
    ax_b.set_xticks([365, 547, 1095])
    ax_b.set_xlim(250, 1210)
    ax_a.set_ylabel("Log-loss gain over\nno message passing")
    fig.text(0.01, -0.05, "Tier 3 features. Positive = message passing helps. Bars: 95% paired CI "
             "over 5 seeds; labels: seeds in which message passing wins. "
             "(b) at 1095 days is the hop-grid run.", fontsize=7, color=INK_2)
    fig.tight_layout()
    save(fig, "fig10_window_sweep")


# ------------------------------------------------------------------ fig 11 --

def fig11_strata():
    strata = pd.read_csv(ROOT / "new_work" / "results" / "strata_full.csv")
    rows = [("1_vs_none", "1_bscore", "Tier 1: B-score + static\n(3 seeds, lr 3e-4)", BLUE, "o"),
            ("1_vs_none_stable", "3_history_decoder",
             "Tier 3: + history at decoder\n(5 seeds, lr 1e-4)", ORANGE, "s")]
    cols = [("degree_stratum", ["0-5 (cold)", "6-20", "21+"],
             "(a) Prior opponents (less-recorded player)"),
            ("common_stratum", ["0", "1-4", "5-14", "15+"],
             "(b) Common opponents")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.0), sharey="row",
                             gridspec_kw={"width_ratios": [3, 4]})
    for r, (intervention, tier, label, colour, marker) in enumerate(rows):
        for c, (by, order, title) in enumerate(cols):
            ax = axes[r, c]
            zero_line(ax)
            sub = strata[(strata["intervention"] == intervention) & (strata["tier"] == tier)
                         & (strata["by"] == by)].set_index("stratum").loc[order]
            x = np.arange(len(order))
            gain = -sub["delta"].to_numpy()
            low, high = -sub["ci_high"].to_numpy(), -sub["ci_low"].to_numpy()
            ax.errorbar(x, gain, yerr=np.vstack([gain - low, high - gain]), fmt="none",
                        ecolor=colour, elinewidth=1.0, capsize=2.5, zorder=3)
            ax.plot(x, gain, marker, color=colour, mec="white", mew=1.2, ms=6.5, zorder=4)
            ax.set_xlim(-0.5, len(order) - 0.5)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(title, loc="left", fontsize=8.5)
                ax.set_xticks(x, [""] * len(order))
            else:
                n = sub["n_matches"].to_numpy()
                ax.set_xticks(x, [f"{o}\nn = {k:,}" for o, k in zip(order, n)], fontsize=7.5)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.supylabel("Log-loss gain of one hop over no message passing", fontsize=8.5,
                  color=INK_2, x=0.0)
    fig.tight_layout()
    fig.text(0.02, -0.03, "Full tour, test 2019–24, 1095-day window. Positive = the graph helps. "
             "Bars: 95% paired CI over seeds. Descriptive; no multiplicity correction.",
             fontsize=7, color=INK_2)
    save(fig, "fig11_strata_full")


# ------------------------------------------------------------------ fig 12 --

def fig12_hop_contrasts():
    rows = [
        ("One hop vs no messages\n1095-day window",
         paired_gain("full", "depth_3_history_decoder_1hop_lr1e4", "depth_3_history_decoder_nonehop_lr1e4")),
        ("One hop vs no messages\n365-day window",
         paired_gain("full", "win365_t3_history_decoder_1hop_sum", "win365_t3_history_decoder_nonehop")),
        ("Two hops vs one hop\n1095-day window",
         paired_gain("full", "depth_3_history_decoder_2hop_lr1e4", "depth_3_history_decoder_1hop_lr1e4")),
        ("Two hops vs one hop\n365-day window",
         paired_gain("full", "win365_t3_history_decoder_2hop", "win365_t3_history_decoder_1hop_sum")),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    zero_line(ax, "v")
    for i, (label, r) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([r["low"], r["high"]], [y, y], color=BLUE, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(r["seeds"], np.full(r["n"], y) + 0.18, "o", ms=3.5, color=NEUTRAL,
                mec="white", mew=0.5, zorder=2)
        ax.plot([r["mean"]], [y], "o", color=BLUE, mec="white", mew=1.2, ms=6.5, zorder=4)
        wins = int((r["seeds"] > 0).sum())
        ax.annotate(f"{r['mean']:+.5f}  [{r['low']:+.5f}, {r['high']:+.5f}]  {wins}/{r['n']}",
                    (r["high"], y), xytext=(8, -3), textcoords="offset points",
                    fontsize=7.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8)
    ax.set_xlim(-0.016, 0.0125)
    ax.set_xlabel("Log-loss gain of the deeper model (positive = deeper is better)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.01, -0.12, "Full tour, tier 3, lr 1e-4. Line: 95% paired CI over 5 seeds; "
             "grey dots: individual seeds; label: mean [CI] and seeds favouring the deeper model.",
             fontsize=7, color=INK_2)
    save(fig, "fig12_hop_contrasts_full")


# ------------------------------------------------------------------ fig 13 --

def fig13_gnn_vs_gbdt():
    arms = [("GNN, 1095-day window", "depth_3_history_decoder_1hop_lr1e4", BLUE, "o"),
            ("GNN, 365-day window (final)", "win365_t3_history_decoder_1hop_sum", ORANGE, "s")]
    metrics_ = [("log_loss", "(a) Log loss ($\\times 10^{-3}$)", 1e3), ("brier", "(b) Brier score ($\\times 10^{-3}$)", 1e3),
                ("accuracy", "(c) Accuracy (percentage points)", 1e2)]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2), sharey=True)
    for ax, (which, title, scale) in zip(axes, metrics_):
        zero_line(ax, "v")
        for k, (label, name, colour, marker) in enumerate(arms):
            r = paired_gain("full", name, "gbdt_tuned", which)
            y = len(arms) - 1 - k
            ax.plot([r["low"] * scale, r["high"] * scale], [y, y], color=colour, lw=2,
                    solid_capstyle="round", zorder=3)
            ax.plot(r["seeds"] * scale, np.full(r["n"], y) + 0.22, "o", ms=3.2, color=NEUTRAL,
                    mec="white", mew=0.5, zorder=2)
            ax.plot([r["mean"] * scale], [y], marker, color=colour, mec="white", mew=1.2,
                    ms=6.5, zorder=4)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlim(left=-0.5)
    axes[0].set_yticks([1, 0], [a[0] for a in arms], fontsize=8)
    fig.tight_layout()
    fig.text(0.5, -0.03, "Improvement of the GNN over the tuned GBDT (positive = GNN better)",
             ha="center", fontsize=8.5, color=INK_2)
    fig.text(0.01, -0.1, "Full tour, test 2019–24, tier 3, one hop; same matches and labels. "
             "Line: 95% paired CI over 5 seeds; grey dots: individual seeds.",
             fontsize=7, color=INK_2)
    save(fig, "fig13_gnn_vs_gbdt_full")


# ------------------------------------------------------------------ fig 14 --

def fig14_baselines():
    panels = [
        ("Slams + Masters (test 2017–20)", "slams_masters",
         [("Elo", "elo_baseline"), ("B-score", "bscore_only"), ("GBDT (tuned)", "gbdt_tuned"),
          ("GNN (final)", "win365_t3_history_decoder_2hop_sum_buf800")]),
        ("Full tour (test 2019–24)", "full",
         [("GBDT (tuned)", "gbdt_tuned"), ("GNN (final)", "win365_t3_history_decoder_1hop_sum")]),
        ("1990 extension (test 2014–20)", "slams_masters_1990",
         [("Elo", "elo_baseline"), ("B-score", "bscore_only"),
          ("GNN (final)", "win365_t3_history_decoder_1hop")]),
    ]
    order = ["B-score", "Elo", "GBDT (tuned)", "GNN (final)"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2), sharey=True)
    for ax, (title, scope, models) in zip(axes, panels):
        values = {label: np.mean(list(per_seed(scope, name).values())) for label, name in models}
        for label in order:
            y = len(order) - 1 - order.index(label)
            if label in values:
                v = values[label]
                colour = BLUE if label == "GNN (final)" else INK_2
                ax.plot([v], [y], "o", color=colour, mec="white", mew=1.2, ms=6.5, zorder=3)
                ax.annotate(f"{v:.4f}", (v, y), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=7, color=INK)
            else:
                ax.text(0.5, y, "not computed", transform=ax.get_yaxis_transform(),
                        ha="center", va="center", fontsize=7, color=MUTED)
        lo, hi = min(values.values()), max(values.values())
        pad = (hi - lo) * 0.25 + 0.002
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(-0.6, len(order) - 0.3)
    axes[0].set_yticks(range(len(order)), list(reversed(order)), fontsize=8)
    fig.tight_layout()
    fig.text(0.5, -0.04, "Test log loss (lower is better; axes differ per scope)",
             ha="center", fontsize=8.5, color=INK_2)
    save(fig, "fig14_baselines")


# ------------------------------------------------------------------ fig 15 --

def fig15_reliability():
    arms = [("GNN (final)", "win365_t3_history_decoder_1hop_sum", BLUE, "o"),
            ("GBDT (tuned)", "gbdt_tuned", ORANGE, "s")]
    bins = np.linspace(0, 1, 11)
    fig, (ax, ax_h) = plt.subplots(2, 1, figsize=(WIDTH * 0.55, 3.6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", zorder=1)
    for k, (label, name, colour, marker) in enumerate(arms):
        frames = [test_frame("full", name, s) for s in SEEDS]
        pooled = pd.concat([f for f in frames if f is not None])
        p = pooled["probability"].to_numpy()
        y = pooled["y_true"].to_numpy()
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        mean_p = np.array([p[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)])
        freq = np.array([y[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)])
        count = np.array([(idx == b).sum() for b in range(10)])
        ok = count >= 50
        ax.plot(mean_p[ok], freq[ok], color=colour, lw=1.2, zorder=2)
        ax.plot(mean_p[ok], freq[ok], marker, color=colour, mec="white", mew=1.2, zorder=3,
                label=label)
        centres = (bins[:-1] + bins[1:]) / 2 + (k - 0.5) * 0.035
        ax_h.bar(centres, count / len(SEEDS), width=0.035, color=colour, zorder=2)
    ax.set_ylabel("Observed win frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax_h.set_ylabel("Matches\nper seed", fontsize=8)
    ax_h.set_xlabel("Predicted probability that player A wins")
    ax_h.grid(axis="x", visible=False)
    fig.text(0.01, -0.03, "Full tour, test 2019–24, pooled over 5 seeds; 10 equal-width bins, "
             "bins with < 50 matches omitted.", fontsize=7, color=INK_2)
    save(fig, "fig15_reliability_full")


# ------------------------------------------------------------------ fig 16 --

def fig16_recipe_sensitivity():
    arms = ["none", "1", "2"]
    labels = {"none": "No message\npassing", "1": "One hop", "2": "Two hops"}
    recipes = [("lr 3e-4 (feature × hop grid)", "grid_3_history_decoder_{}hop", ORANGE, "s"),
               ("lr 1e-4 (depth re-run)", "depth_3_history_decoder_{}hop_lr1e4", BLUE, "o")]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.6))
    means = {}
    for k, (label, pattern, colour, marker) in enumerate(recipes):
        xs, ys = [], []
        for i, arm in enumerate(arms):
            vals = np.array(list(per_seed("full", pattern.format(arm)).values()))
            x = i + (k - 0.5) * 0.2
            ax.plot(np.full(len(vals), x), vals, "o", ms=3, color=NEUTRAL, mec="white",
                    mew=0.4, zorder=2)
            ax.plot([x], [vals.mean()], marker, color=colour, mec="white", mew=1.2, ms=6.5,
                    zorder=4, label=label if i == 0 else None)
            xs.append(x)
            ys.append(vals.mean())
            means[(k, arm)] = vals.mean()
        ax.plot(xs, ys, color=colour, lw=1.0, zorder=3)
    for i, arm in enumerate(arms):
        diff = means[(0, arm)] - means[(1, arm)]
        ax.annotate(f"Δ {diff:+.4f}", (i, max(means[(0, arm)], means[(1, arm)])),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(3), [labels[a] for a in arms])
    ax.set_ylabel("Test log loss")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.text(0.01, -0.07, "Full tour, tier 3, 1095-day window, 5 seeds (grey dots). "
             "Δ = lr 3e-4 minus lr 1e-4.", fontsize=7, color=INK_2)
    save(fig, "fig16_recipe_sensitivity_full")


# ======================================================= Slams + Masters ==
#
# Slams + Masters is the principal scope.  Its final model is tier 3, two hops,
# 365-day window, replay buffer 800 (win365_t3_history_decoder_2hop_sum_buf800).
# The depth and window arms exist only with buffer 200, so hop contrasts are
# made within that family (win{window}_t3_history_decoder_{k}hop).

SM = "slams_masters"
SM_FINAL = "win365_t3_history_decoder_2hop_sum_buf800"


def sm_arm(window: int, hops) -> str:
    return f"win{window}_t3_history_decoder_{hops}hop"


def forest(ax, rows, colour=BLUE, value_fmt="{:+.4f}", label_x=None):
    # rows: (label, paired_gain dict) top to bottom; None label -> group header.
    zero_line(ax, "v")
    y_labels = []
    for i, (label, r) in enumerate(rows):
        y = len(rows) - 1 - i
        y_labels.append((y, label))
        if r is None:
            continue
        ax.plot([r["low"], r["high"]], [y, y], color=colour, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(r["seeds"], np.full(r["n"], y) + 0.22, "o", ms=3.2, color=NEUTRAL,
                mec="white", mew=0.5, zorder=2)
        ax.plot([r["mean"]], [y], "o", color=colour, mec="white", mew=1.2, ms=6.5, zorder=4)
        wins = int((r["seeds"] > 0).sum())
        text = (f"{value_fmt.format(r['mean'])}  [{value_fmt.format(r['low'])}, "
                f"{value_fmt.format(r['high'])}]  {wins}/{r['n']}")
        ax.annotate(text, (label_x if label_x is not None else r["high"], y),
                    xytext=(8, -3), textcoords="offset points", fontsize=7.2, color=INK)
    ax.set_yticks([y for y, _ in y_labels], [l for _, l in y_labels], fontsize=8)
    for tick, (_, label) in zip(ax.get_yticklabels(), y_labels):
        if label.isupper() or label.startswith("vs "):
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


# ------------------------------------------------------------------ fig 17 --

def fig17_sm_depth_window():
    series = [(2, "Two hops", BLUE, "o", [90, 180, 270, 365, 547, 730, 1095]),
              (1, "One hop", ORANGE, "s", [270, 365, 547, 730]),
              (3, "Three hops", AQUA, "^", [365, 547])]
    offsets = {2: 0, 1: -14, 3: 14}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    zero_line(ax)
    for hops, label, colour, marker, windows in series:
        r = [paired_gain(SM, sm_arm(w, hops), sm_arm(w, "none")) for w in windows]
        x = np.array(windows) + offsets[hops]
        mean = np.array([v["mean"] for v in r])
        err = np.array([[v["mean"] - v["low"], v["high"] - v["mean"]] for v in r]).T
        ax.plot(x, mean, color=colour, lw=1.2 if hops != 2 else 1.8, zorder=2)
        ax.errorbar(x, mean, yerr=err, fmt="none", ecolor=colour, elinewidth=1.0, capsize=2, zorder=3)
        ax.plot(x, mean, marker, color=colour, mec="white", mew=1.2, ms=6 if hops == 2 else 5,
                zorder=4, label=label)
        if hops == 2:
            for xi, v in zip(x, r):
                ax.annotate(f"{int((v['seeds'] > 0).sum())}/{v['n']}", (xi, v["high"]),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=6.5, color=INK_2)
    ax.set_xticks([90, 180, 270, 365, 547, 730, 1095])
    ax.set_xlabel("Graph-history window (days)")
    ax.set_ylabel("Log-loss gain over\nno message passing")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.text(0.01, -0.09, "Slams + Masters, test 2017–20, tier 3, lr 1e-4, replay buffer 200. "
             "Positive = message passing helps.\nBars: 95% paired CI over 5 seeds; "
             "labels: seeds in which two hops win.", fontsize=7, color=INK_2)
    save(fig, "fig17_sm_depth_by_window")


# ------------------------------------------------------------------ fig 18 --

def fig18_sm_hop_contrasts():
    rows = [
        ("One hop vs no messages", paired_gain(SM, sm_arm(365, 1), sm_arm(365, "none"))),
        ("Two hops vs no messages", paired_gain(SM, sm_arm(365, 2), sm_arm(365, "none"))),
        ("Two hops vs one hop", paired_gain(SM, sm_arm(365, 2), sm_arm(365, 1))),
        ("Three hops vs two hops", paired_gain(SM, sm_arm(365, 3), sm_arm(365, 2))),
        ("Three hops vs two hops\n(547-day window)", paired_gain(SM, sm_arm(547, 3), sm_arm(547, 2))),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    forest(ax, rows)
    ax.set_xlim(-0.004, 0.0075)
    ax.set_xlabel("Log-loss gain of the deeper model (positive = deeper is better)")
    fig.text(0.01, -0.14, "Slams + Masters, test 2017–20, tier 3, 365-day window unless stated, "
             "lr 1e-4, buffer 200.\nLine: 95% paired CI over 5 seeds; grey dots: seeds; "
             "label: mean [CI] and seeds favouring the deeper model.", fontsize=7, color=INK_2)
    save(fig, "fig18_sm_hop_contrasts")


# ------------------------------------------------------------------ fig 19 --

def fig19_sm_substitution():
    tiers = ["t0_nothing", "t1_bscore", "t2_history_nodes", "t3_history_decoder"]
    depth = [(1, "One hop", ORANGE, "s", -0.08), (2, "Two hops", BLUE, "o", 0.08)]
    at365 = {1: "win365_t1_bscore_{}hop", 3: "win365_t3_history_decoder_{}hop"}
    fig, (ax_all, ax_zoom) = plt.subplots(
        1, 2, figsize=(WIDTH, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, shown in ((ax_all, range(4)), (ax_zoom, range(1, 4))):
        zero_line(ax)
        for hops, label, colour, marker, off in depth:
            r = [paired_gain(SM, f"hopgrid_{tiers[t]}_{hops}hop", f"hopgrid_{tiers[t]}_nonehop")
                 for t in shown]
            x = np.array(list(shown)) + off
            mean = np.array([v["mean"] for v in r])
            err = np.array([[v["mean"] - v["low"], v["high"] - v["mean"]] for v in r]).T
            ax.plot(x, mean, color=colour, lw=1.2, zorder=2)
            ax.errorbar(x, mean, yerr=err, fmt="none", ecolor=colour, elinewidth=1.0, capsize=2, zorder=3)
            ax.plot(x, mean, marker, color=colour, mec="white", mew=1.2, zorder=4,
                    label=f"{label}, 1095-day window")
            if ax is ax_zoom:
                for t in (1, 3):
                    v = paired_gain(SM, at365[t].format(hops), at365[t].format("none"))
                    xt = t + off + 0.16
                    ax.errorbar([xt], [v["mean"]], yerr=[[v["mean"] - v["low"]], [v["high"] - v["mean"]]],
                                fmt="none", ecolor=colour, elinewidth=1.0, capsize=2, zorder=3)
                    ax.plot([xt], [v["mean"]], marker, mfc="white", mec=colour, mew=1.3, zorder=4,
                            label=f"{label}, 365-day window" if t == 1 else None)
        ax.set_xticks(list(shown), [TIER_LABELS[t] for t in shown], fontsize=7.5)
        ax.grid(axis="x", visible=False)
    ax_all.set_ylabel("Log-loss gain over\nno message passing")
    ax_all.set_title("(a) All feature tiers", loc="left")
    ax_zoom.set_title("(b) Tiers 1–3, with the 365-day window", loc="left")
    handles, labels = ax_zoom.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower left", bbox_to_anchor=(0.04, 0.98), fontsize=7.5)
    fig.tight_layout()
    fig.text(0.01, -0.04, "Slams + Masters, test 2017–20, lr 1e-4, buffer 200, 5 seeds. "
             "Positive = the graph helps. Bars: 95% paired CI. The 365-day window was run for tiers 1 and 3.",
             fontsize=7, color=INK_2)
    save(fig, "fig19_sm_substitution_curve")


# ------------------------------------------------------------------ fig 20 --

def sm_strata_table(candidate: str, baseline: str, window: int = 365, scope: str = SM) -> pd.DataFrame:
    # Per-match log-loss gain joined to structural descriptors of the block
    # graph the models actually saw, one row per (seed, match).
    import torch

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tennis_gnn.structure import add_strata, structure_table

    frames = []
    for seed in SEEDS:
        snaps = torch.load(ROOT / ".cache" / "snapshots" / f"{scope}__full__w{window}__seed{seed}__v4.pt",
                           weights_only=False)
        struct = add_strata(structure_table([s for s in snaps if s.phase == "test"]))
        a, b = test_frame(scope, candidate, seed), test_frame(scope, baseline, seed)
        key = ["block_idx", "row_in_block"]

        def per_match(frame):
            p = np.clip(frame["probability"].to_numpy(float), 1e-7, 1 - 1e-7)
            y = frame["y_true"].to_numpy(float)
            return frame[key].assign(ll=-(y * np.log(p) + (1 - y) * np.log(1 - p)), y=y)

        merged = per_match(a).merge(per_match(b), on=key, suffixes=("_a", "_b"))
        assert (merged["y_a"] == merged["y_b"]).all(), "labels differ between artifacts"
        merged = merged.merge(struct, on=key, how="left")
        assert merged["degree_min"].notna().all(), "structure join incomplete"
        merged["gain"] = merged["ll_b"] - merged["ll_a"]
        merged["seed"] = seed
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def strata_summary(table: pd.DataFrame, by: str, order: list[str]) -> pd.DataFrame:
    rows = []
    for stratum in order:
        per_seed_gain = table[table[by].astype(str) == stratum].groupby("seed")["gain"].mean()
        d = per_seed_gain.to_numpy()
        half = t_critical(len(d)) * d.std(ddof=1) / np.sqrt(len(d))
        n = int((table[by].astype(str) == stratum).sum() / len(SEEDS))
        rows.append({"stratum": stratum, "mean": d.mean(), "low": d.mean() - half,
                     "high": d.mean() + half, "wins": int((d > 0).sum()), "n": n})
    return pd.DataFrame(rows)


def fig20_sm_strata():
    arms = [("Two hops vs no messages", sm_arm(365, 2), sm_arm(365, "none"), BLUE, "o"),
            ("One hop vs no messages", sm_arm(365, 1), sm_arm(365, "none"), ORANGE, "s")]
    cols = [("degree_stratum", ["0-5 (cold)", "6-20", "21+"], "(a) Prior opponents (less-recorded player)"),
            ("common_stratum", ["0", "1-4", "5-14", "15+"], "(b) Common opponents")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.0), sharey=True,
                             gridspec_kw={"width_ratios": [3, 4]})
    summary_rows = []
    for r, (label, cand, base, colour, marker) in enumerate(arms):
        table = sm_strata_table(cand, base)
        for c, (by, order, title) in enumerate(cols):
            ax = axes[r, c]
            zero_line(ax)
            summ = strata_summary(table, by, order)
            x = np.arange(len(order))
            ax.errorbar(x, summ["mean"], yerr=[summ["mean"] - summ["low"], summ["high"] - summ["mean"]],
                        fmt="none", ecolor=colour, elinewidth=1.0, capsize=2.5, zorder=3)
            ax.plot(x, summ["mean"], marker, color=colour, mec="white", mew=1.2, ms=6.5, zorder=4)
            for xi, row in zip(x, summ.itertuples()):
                ax.annotate(f"{row.wins}/5", (xi, row.high), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=6.5, color=INK_2)
            ax.set_xlim(-0.5, len(order) - 0.5)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(title, loc="left", fontsize=8.5)
                ax.set_xticks(x, [""] * len(order))
            else:
                ax.set_xticks(x, [f"{o}\nn = {k:,}" for o, k in zip(order, summ["n"])], fontsize=7.5)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            summary_rows += [dict(arm=label, by=by, **row) for row in summ.to_dict("records")]
    pd.DataFrame(summary_rows).to_csv(OUT / "fig20_sm_strata.csv", index=False)
    fig.supylabel("Log-loss gain over no message passing", fontsize=8.5, color=INK_2, x=0.0)
    fig.tight_layout()
    fig.text(0.02, -0.05, "Slams + Masters, test 2017–20, tier 3, 365-day window, buffer 200. "
             "Strata from the 365-day block graphs.\nPositive = the graph helps. Bars: 95% paired CI "
             "over 5 seeds; labels: seeds in which message passing wins. Descriptive, no multiplicity correction.",
             fontsize=7, color=INK_2)
    save(fig, "fig20_sm_strata")


# ------------------------------------------------------------------ fig 21 --

def fig21_sm_gnn_vs_gbdt():
    arms = [("Final GNN\n(2 hops, buffer 800)", SM_FINAL, BLUE, "o"),
            ("GNN, 2 hops\n(buffer 200)", sm_arm(365, 2), ORANGE, "s"),
            ("No message passing\n(buffer 200)", sm_arm(365, "none"), AQUA, "^")]
    metrics_ = [("log_loss", "(a) Log loss ($\\times 10^{-3}$)", 1e3),
                ("brier", "(b) Brier score ($\\times 10^{-3}$)", 1e3),
                ("accuracy", "(c) Accuracy (percentage points)", 1e2)]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6), sharey=True)
    for ax, (which, title, scale) in zip(axes, metrics_):
        zero_line(ax, "v")
        for k, (label, name, colour, marker) in enumerate(arms):
            r = paired_gain(SM, name, "gbdt_tuned", which)
            y = len(arms) - 1 - k
            ax.plot([r["low"] * scale, r["high"] * scale], [y, y], color=colour, lw=2,
                    solid_capstyle="round", zorder=3)
            ax.plot(r["seeds"] * scale, np.full(r["n"], y) + 0.24, "o", ms=3.2, color=NEUTRAL,
                    mec="white", mew=0.5, zorder=2)
            ax.plot([r["mean"] * scale], [y], marker, color=colour, mec="white", mew=1.2, ms=6.5, zorder=4)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(arms) - 0.3)
    axes[0].set_yticks(range(len(arms)), [a[0] for a in reversed(arms)], fontsize=8)
    fig.tight_layout()
    fig.text(0.5, -0.03, "Improvement over the tuned GBDT (positive = GNN better)",
             ha="center", fontsize=8.5, color=INK_2)
    fig.text(0.01, -0.1, "Slams + Masters, test 2017–20, tier 3, 365-day window; same matches and labels. "
             "Line: 95% paired CI over 5 seeds; grey dots: seeds.", fontsize=7, color=INK_2)
    save(fig, "fig21_sm_gnn_vs_gbdt")


# ------------------------------------------------------------------ fig 22 --

def fig22_sm_ablations():
    ref2 = sm_arm(365, 2)
    rows = [
        ("vs final model (2 hops, 365 d, buffer 800)", None),
        ("Elo instead of B-score as rating", paired_gain(SM, "win365_t3_elo_rating_2hop_buf800", SM_FINAL)),
        ("No B-score node features", paired_gain(SM, "win365_t3_no_node_bscore_2hop_buf800", SM_FINAL)),
        ("No B-score at all", paired_gain(SM, "win365_t3_no_bscore_at_all_2hop_buf800", SM_FINAL)),
        ("Replay buffer 200", paired_gain(SM, ref2, SM_FINAL)),
        ("vs 2 hops, 365 d, buffer 200", None),
        ("No message passing", paired_gain(SM, sm_arm(365, "none"), ref2)),
        ("One hop", paired_gain(SM, sm_arm(365, 1), ref2)),
        ("Three hops", paired_gain(SM, sm_arm(365, 3), ref2)),
        ("Mean aggregation", paired_gain(SM, "win365_t3_history_decoder_2hop_mean", ref2)),
        ("1095-day window", paired_gain(SM, "win1095_t3_history_decoder_2hop", ref2)),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
    forest(ax, rows, label_x=0.0045)
    ax.set_xlim(-0.0095, 0.0045)
    ax.set_xlabel("Log-loss gain of the variant (negative = variant is worse)")
    fig.text(0.01, -0.09, "Slams + Masters, test 2017–20, tier 3. Each row changes one factor from the "
             "reference above it.\nLine: 95% paired CI over 5 seeds; grey dots: seeds; label: mean [CI] "
             "and seeds favouring the variant.", fontsize=7, color=INK_2)
    save(fig, "fig22_sm_ablations")


# ------------------------------------------------------------------ fig 23 --

def fig23_sm_reliability():
    reliability(SM, [("Final GNN", SM_FINAL, BLUE, "o"), ("GBDT (tuned)", "gbdt_tuned", ORANGE, "s")],
                "Slams + Masters, test 2017–20, pooled over 5 seeds; 10 equal-width bins, "
                "bins with < 50 matches omitted.", "fig23_sm_reliability", min_count=50)


def reliability(scope, arms, note, name, min_count=50):
    bins = np.linspace(0, 1, 11)
    fig, (ax, ax_h) = plt.subplots(2, 1, figsize=(WIDTH * 0.55, 3.6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", zorder=1)
    for k, (label, art, colour, marker) in enumerate(arms):
        pooled = pd.concat([f for f in (test_frame(scope, art, s) for s in SEEDS) if f is not None])
        p, y = pooled["probability"].to_numpy(), pooled["y_true"].to_numpy()
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        count = np.array([(idx == b).sum() for b in range(10)])
        mean_p = np.array([p[idx == b].mean() if count[b] else np.nan for b in range(10)])
        freq = np.array([y[idx == b].mean() if count[b] else np.nan for b in range(10)])
        ok = count >= min_count
        ax.plot(mean_p[ok], freq[ok], color=colour, lw=1.2, zorder=2)
        ax.plot(mean_p[ok], freq[ok], marker, color=colour, mec="white", mew=1.2, zorder=3, label=label)
        ax_h.bar((bins[:-1] + bins[1:]) / 2 + (k - 0.5) * 0.035, count / len(SEEDS), width=0.035,
                 color=colour, zorder=2)
    ax.set_ylabel("Observed win frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax_h.set_ylabel("Matches\nper seed", fontsize=8)
    ax_h.set_xlabel("Predicted probability that player A wins")
    ax_h.grid(axis="x", visible=False)
    fig.text(0.01, -0.03, note, fontsize=7, color=INK_2)
    save(fig, name)


# ------------------------------------------------------------------ fig 24 --

def fig24_strata_both_scopes():
    # Connectivity strata for both scopes in one figure, all at the final
    # 365-day window with tier 3 features, so the two scopes are comparable.
    arms = [("Slams + Masters, two hops (final)", SM, sm_arm(365, 2), sm_arm(365, "none"),
             BLUE, "o", "full", -0.2),
            ("Slams + Masters, one hop", SM, sm_arm(365, 1), sm_arm(365, "none"),
             BLUE, "o", "hollow", 0.0),
            ("Full tour, one hop (final)", "full", "win365_t3_history_decoder_1hop_sum",
             "win365_t3_history_decoder_nonehop", ORANGE, "s", "full", 0.2)]
    cols = [("degree_stratum", ["0-5 (cold)", "6-20", "21+"], "(a) Prior opponents (less-recorded player)"),
            ("common_stratum", ["0", "1-4", "5-14", "15+"], "(b) Common opponents")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True,
                             gridspec_kw={"width_ratios": [3, 4]})
    counts = {}
    rows = []
    for label, scope, cand, base, colour, marker, fill, off in arms:
        table = sm_strata_table(cand, base, scope=scope)
        for ax, (by, order, title) in zip(axes, cols):
            summ = strata_summary(table, by, order)
            counts[(scope, by)] = summ["n"].tolist()
            x = np.arange(len(order)) + off
            ax.errorbar(x, summ["mean"], yerr=[summ["mean"] - summ["low"], summ["high"] - summ["mean"]],
                        fmt="none", ecolor=colour, elinewidth=1.0, capsize=2, zorder=3)
            ax.plot(x, summ["mean"], marker, ms=6, zorder=4, color=colour,
                    mfc="white" if fill == "hollow" else colour,
                    mec=colour if fill == "hollow" else "white", mew=1.2,
                    label=label if ax is axes[0] else None)
            rows += [dict(arm=label, by=by, **r) for r in summ.to_dict("records")]
    for ax, (by, order, title) in zip(axes, cols):
        zero_line(ax)
        x = np.arange(len(order))
        ax.set_xticks(x, [f"{o}\n{a:,} / {b:,}" for o, a, b in
                          zip(order, counts[(SM, by)], counts[("full", by)])], fontsize=7.3)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=8.5)
    axes[0].set_ylabel("Log-loss gain over\nno message passing")
    fig.legend(ncol=3, loc="lower left", bbox_to_anchor=(0.06, 0.97), fontsize=7.5)
    pd.DataFrame(rows).to_csv(OUT / "fig24_strata_both_scopes.csv", index=False)
    fig.tight_layout()
    fig.text(0.01, -0.07, "Tier 3, 365-day window, test periods 2017–20 (Slams + Masters) and 2019–24 "
             "(full tour). Tick labels: matches per seed, Slams + Masters / full tour.\n"
             "Positive = message passing helps. Bars: 95% paired CI over 5 seeds. "
             "Descriptive; no multiplicity correction.", fontsize=7, color=INK_2)
    save(fig, "fig24_strata_both_scopes")


if __name__ == "__main__":
    fig04_split_timeline()
    fig05_bscore_example()
    fig06_snapshot_timeline()
    fig07_matches_per_year()
    fig08_recency_weight()
    fig09_substitution_curve()
    fig10_window_sweep()
    fig11_strata()
    fig12_hop_contrasts()
    fig13_gnn_vs_gbdt()
    fig14_baselines()
    fig15_reliability()
    fig16_recipe_sensitivity()
    fig17_sm_depth_window()
    fig18_sm_hop_contrasts()
    fig19_sm_substitution()
    fig20_sm_strata()
    fig21_sm_gnn_vs_gbdt()
    fig22_sm_ablations()
    fig23_sm_reliability()
    fig24_strata_both_scopes()
