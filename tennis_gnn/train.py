# Training, calibration and evaluation for the tennis GNN.
#
# Protocol (unchanged from the original experiment):
# - the graph for a block is built only from matches played before it
# - gradient updates happen on training blocks only (2012-2015)
# - validation (2016) and test (2017-2020) get predicted by the frozen
#   end-of-training model, on a graph that keeps growing
#
# Because no parameter ever changes after the last training block, training
# and evaluation can be split into two passes without touching a single
# validation or test prediction. That's what makes it possible to do
# several optimisation passes over the training years.

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from tennis_gnn.config import ModelConfig, TrainConfig
from tennis_gnn.data import WARMUP_END
from tennis_gnn.model import (
    LEGACY_CONTEXT_DIM,
    LEGACY_NODE_DIM,
    TennisGNN,
)
from tennis_gnn.snapshots import BlockSnapshot


DEVICE = torch.device("cpu")


_SURFACE_BSCORE_COLUMN = {"Hard": 1, "Clay": 2, "Grass": 3}


def to_pyg(snapshot: BlockSnapshot) -> Data:
    # Wraps a snapshot as a PyG graph, keeping the raw B-score channels.
    #
    # raw_bscore_general/raw_bscore_surface are kept beside x because the
    # decoder consumes them directly as a skill prior; they have to survive
    # the node-feature ablations that zero or standardise x.
    #
    # The block is single-surface (see snapshots.py:build_graphs), so one
    # column selection per block gives every node in it the right B-score
    # for the surface actually being played on. Unrecognised surfaces fall
    # back to column 0 (general) rather than raising.
    surface_col = _SURFACE_BSCORE_COLUMN.get(snapshot.surface, 0)
    return Data(
        x=snapshot.x,
        edge_index=snapshot.edge_index,
        edge_attr=snapshot.edge_attr,
        raw_bscore_general=snapshot.x[:, 0].clone(),
        raw_bscore_surface=snapshot.x[:, surface_col].clone(),
    )


def _make_batch(items: Sequence[tuple], device: torch.device):
    graphs = [item[0] for item in items]
    batch = Batch.from_data_list(graphs).to(device)
    offsets = batch.ptr[:-1]
    player_a = torch.cat(
        [item[1] + offsets[i] for i, item in enumerate(items)]
    ).to(device)
    player_b = torch.cat(
        [item[2] + offsets[i] for i, item in enumerate(items)]
    ).to(device)
    context = torch.cat([item[3] for item in items]).to(device)
    y = torch.cat([item[4] for item in items]).to(device)
    return batch, player_a, player_b, context, y


def fit_temperature(
    logits: np.ndarray, y: np.ndarray, *, max_iter: int = 300
) -> float:
    # Fits a single temperature by minimising validation NLL.
    #
    # Dividing every logit by one positive scalar can't reorder
    # predictions, so accuracy and AUC are untouched - only confidence
    # gets rescaled. This is the cheapest honest way to fix a model that
    # ranks well but is overconfident, and it's fitted on validation only.
    if len(logits) == 0:
        return 1.0
    logit_tensor = torch.tensor(logits, dtype=torch.float)
    target = torch.tensor(y, dtype=torch.float)
    log_temperature = torch.zeros(1, requires_grad=True)
    # Without a line search, LBFGS takes fixed-size steps and overshoots: on a
    # model that is *under*-confident (the optimum is T < 1) it ran straight
    # past the minimum to the clamp at T=0.0183 every time, so no under-confident
    # model in this project was ever calibrated.  Strong-Wolfe fixes it.
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        # Clamped inside the objective as well as after: when the logits carry
        # almost no signal the optimal temperature runs away to infinity, and
        # an unbounded log-temperature lets LBFGS step into overflow and return
        # NaN.  A NaN here is silent and total - it turns every probability in
        # the run into NaN, including phases the fit never touched.
        scaled = logit_tensor / log_temperature.clamp(-4.0, 4.0).exp()
        loss = criterion(scaled, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.clamp(-4.0, 4.0).exp().item())
    if not np.isfinite(temperature) or temperature <= 0.0:
        # Calibration failing is recoverable; silently emitting NaN is not.
        # T=1 is the identity, so the run degrades to uncalibrated output.
        warnings.warn(
            "Temperature fit did not converge to a finite positive value; "
            "falling back to T=1 (uncalibrated).",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0

    # T=1 is inside the feasible set, so a correct fit can never be worse than
    # not calibrating at all.  The clamp breaks that guarantee: it creates a
    # flat region with zero gradient, and LBFGS can come to rest on the
    # boundary.  Observed doing exactly that - a model whose logits carried
    # little signal was assigned T=0.0183, the clamp, which multiplies every
    # logit by 55 and took validation log loss from 0.64 to 1.16.  Silently
    # returning a "calibration" that makes the model worse is the failure mode
    # this guards.
    with torch.no_grad():
        fitted_nll = float(criterion(logit_tensor / temperature, target))
        identity_nll = float(criterion(logit_tensor, target))
    if not np.isfinite(fitted_nll) or fitted_nll > identity_nll:
        warnings.warn(
            f"Temperature fit landed on T={temperature:.4g}, which scores "
            f"{fitted_nll:.4f} against {identity_nll:.4f} uncalibrated; "
            "falling back to T=1.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0
    return temperature


def run_experiment(
    snapshots: Sequence[BlockSnapshot],
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    verbose: bool = True,
    eval_phases: Iterable[str] | None = None,
) -> dict:
    # Fits the model and returns frozen per-match predictions.
    #
    # eval_phases restricts the frozen evaluation pass. A hyperparameter
    # search only needs validation, and skipping the other 80% of the
    # timeline roughly halves the cost of each configuration.
    init_seed = train_config.effective_init_seed
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    generator = np.random.default_rng(init_seed)

    graphs = [to_pyg(snapshot) for snapshot in snapshots]
    edge_dim = snapshots[0].edge_attr.shape[1]
    stored_context_dim = snapshots[0].context.shape[1]
    if model_config.rich_match_context and stored_context_dim <= LEGACY_CONTEXT_DIM:
        raise ValueError(
            "These cached snapshots predate the richer match context "
            f"(stored width {stored_context_dim}). Rebuild them with "
            "load_or_build(..., rebuild=True), or set "
            "ModelConfig.rich_match_context=False."
        )
    context_dim = (
        stored_context_dim
        if model_config.rich_match_context
        else LEGACY_CONTEXT_DIM
    )

    stored_node_dim = snapshots[0].x.shape[1]
    if model_config.node_history_features and stored_node_dim <= LEGACY_NODE_DIM:
        raise ValueError(
            "These cached snapshots predate the per-node history features "
            f"(stored width {stored_node_dim}). Rebuild them with "
            "load_or_build(..., rebuild=True), or set "
            "ModelConfig.node_history_features=False."
        )
    node_dim = (
        stored_node_dim
        if model_config.node_history_features
        else LEGACY_NODE_DIM
    )
    if model_config.history_to_decoder and stored_node_dim <= LEGACY_NODE_DIM:
        raise ValueError(
            "history_to_decoder needs snapshots carrying the history block "
            f"(stored width {stored_node_dim}); rebuild with "
            "load_or_build(..., rebuild=True)."
        )
    decoder_extra_dim = (
        stored_node_dim - LEGACY_NODE_DIM
        if model_config.history_to_decoder
        else 0
    )

    model = TennisGNN(
        model_config,
        node_in_dim=node_dim,
        decoder_extra_dim=decoder_extra_dim,
        edge_in_dim=edge_dim,
        match_context_dim=context_dim,
        hidden_dim=train_config.hidden_dim,
        dropout=train_config.dropout,
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    trainable = [
        index
        for index, snapshot in enumerate(snapshots)
        if snapshot.phase == "train" and snapshot.year > WARMUP_END
    ]

    # ---- Pass 1: optimisation over the training years only ----------------
    if not trainable:
        raise ValueError(
            "No training blocks found; check the phase assignment and that "
            "snapshots cover the training years."
        )

    replay: list[tuple] = []
    step_count = 0
    last_loss = float("nan")
    for pass_index in range(train_config.passes):
        for index in trainable:
            snapshot = snapshots[index]
            replay.append(
                (
                    graphs[index],
                    snapshot.player_a,
                    snapshot.player_b,
                    snapshot.context,
                    snapshot.y,
                )
            )
            if len(replay) > train_config.replay_buffer_size:
                replay = replay[-train_config.replay_buffer_size :]

            model.train()
            for _ in range(train_config.steps_per_block):
                if (
                    train_config.replay_batch_size is not None
                    and len(replay) > train_config.replay_batch_size
                ):
                    chosen = generator.choice(
                        len(replay),
                        size=train_config.replay_batch_size,
                        replace=False,
                    )
                    items = [replay[i] for i in chosen]
                else:
                    items = replay

                optimizer.zero_grad()
                batch, player_a, player_b, context, y = _make_batch(
                    items, DEVICE
                )
                loss = criterion(
                    model(batch, player_a, player_b, context), y
                )
                loss.backward()
                if train_config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.grad_clip
                    )
                optimizer.step()
                step_count += 1
                last_loss = float(loss.item())

        if verbose:
            print(
                f"  pass {pass_index + 1}/{train_config.passes} "
                f"done ({step_count} steps, last loss {last_loss:.4f})",
                flush=True,
            )

    # ---- Pass 2: frozen evaluation over the whole timeline ----------------
    model.eval()
    wanted = None if eval_phases is None else set(eval_phases)
    rows: list[dict] = []
    with torch.no_grad():
        for index, snapshot in enumerate(snapshots):
            if wanted is not None and snapshot.phase not in wanted:
                continue
            logits = model(
                graphs[index],
                snapshot.player_a,
                snapshot.player_b,
                snapshot.context,
            )
            logit_values = logits.cpu().numpy()
            for position in range(len(logit_values)):
                rows.append(
                    {
                        "phase": snapshot.phase,
                        "tourney_id": snapshot.tourney_id,
                        "round_order": snapshot.round_order,
                        "block_idx": snapshot.block_idx,
                        "row_in_block": position,
                        "year": snapshot.year,
                        "y_true": float(snapshot.y[position]),
                        "logit": float(logit_values[position]),
                        "intransitivity_level": snapshot.intransitivity_level[
                            position
                        ],
                    }
                )

    predictions = pd.DataFrame(rows)

    temperature = 1.0
    if train_config.calibrate:
        validation = predictions[predictions["phase"] == "val"]
        temperature = fit_temperature(
            validation["logit"].to_numpy(), validation["y_true"].to_numpy()
        )
    predictions["temperature"] = temperature
    predictions["probability"] = 1.0 / (
        1.0 + np.exp(-predictions["logit"].to_numpy() / temperature)
    )

    return {
        "predictions": predictions,
        "temperature": temperature,
        "steps": step_count,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }


def run_ensemble(
    snapshots: Sequence[BlockSnapshot],
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    members: int = 5,
    verbose: bool = True,
) -> dict:
    # Averages several independently initialised models on the same data.
    #
    # The data seed is held fixed so every member solves an identical
    # problem with identical labels; only the initialisation and replay
    # sampling differ. Averaging probabilities preserves the decoder's
    # antisymmetry, since a mean of complementary pairs is still
    # complementary.
    #
    # Note this is only a fair comparison against a baseline that got the
    # same treatment. Gradient-boosted trees fitted on identical data with
    # a different random_state are very nearly the same model, so they
    # gain almost nothing from ensembling - that's a real property of the
    # two model families, not a trick, but it needs to be stated when
    # reporting.
    frames = []
    temperature_values = []
    for member in range(members):
        config = replace(train_config, init_seed=1000 + member)
        result = run_experiment(
            snapshots, model_config, config, verbose=False
        )
        frames.append(result["predictions"])
        temperature_values.append(result["temperature"])
        if verbose:
            subset = result["predictions"]
            print(
                f"  member {member + 1}/{members}: "
                f"val_ll={metrics(subset[subset['phase'] == 'val'])['log_loss']:.4f}",
                flush=True,
            )

    key = ["phase", "tourney_id", "round_order", "block_idx", "row_in_block"]
    combined = frames[0][key + ["y_true", "year", "intransitivity_level"]].copy()
    stacked = np.stack(
        [
            frame.sort_values(key)["probability"].to_numpy()
            for frame in frames
        ]
    )
    combined = combined.sort_values(key).reset_index(drop=True)
    combined["probability"] = stacked.mean(axis=0)
    return {
        "predictions": combined,
        "members": members,
        "temperature": float(np.mean(temperature_values)),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype=float)
    probability = np.clip(
        frame["probability"].to_numpy(dtype=float), 1e-7, 1 - 1e-7
    )
    return {
        "n_matches": int(len(frame)),
        "accuracy": float(np.mean((probability >= 0.5) == y)),
        "log_loss": float(
            -np.mean(
                y * np.log(probability) + (1 - y) * np.log(1 - probability)
            )
        ),
        "brier": float(np.mean((probability - y) ** 2)),
    }


def metrics_by_phase(
    predictions: pd.DataFrame, phases: Iterable[str] = ("train", "val", "test")
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            phase: metrics(predictions[predictions["phase"] == phase])
            for phase in phases
        }
    ).T
