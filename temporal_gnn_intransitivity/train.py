"""Train and evaluate the temporal GNN without hyperparameter search."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import (
    BASE_CONTEXT_COLUMNS,
    EDGE_COLUMNS,
    INTRA_CONTEXT_COLUMNS,
    build_event_stream,
)
from model import TemporalTennisGNN


@dataclass
class Block:
    phase: str
    metadata: pd.DataFrame
    player_a: torch.Tensor
    player_b: torch.Tensor
    y: torch.Tensor
    context: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    edge: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_blocks(
    frame: pd.DataFrame,
    *,
    include_intransitivity: bool,
    device: torch.device,
) -> list[Block]:
    context_columns = list(BASE_CONTEXT_COLUMNS)
    if include_intransitivity:
        context_columns += list(INTRA_CONTEXT_COLUMNS)
    blocks = []
    for _, group in frame.groupby("block_idx", sort=False):
        winner = torch.tensor(
            group["winner_idx"].to_numpy(), dtype=torch.long, device=device
        )
        loser = torch.tensor(
            group["loser_idx"].to_numpy(), dtype=torch.long, device=device
        )
        base_edge = torch.tensor(
            group[list(EDGE_COLUMNS)].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        winner_extra = torch.tensor(
            group[["winner_recency"]].to_numpy(dtype=np.float32),
            device=device,
        )
        loser_extra = torch.tensor(
            group[["loser_recency"]].to_numpy(dtype=np.float32),
            device=device,
        )
        positive = torch.ones((len(group), 1), device=device)
        negative = -positive
        winner_edge = torch.cat(
            [base_edge, winner_extra, positive], dim=1
        )
        loser_base = base_edge.clone()
        loser_base[:, 0:2] *= -1.0
        loser_edge = torch.cat([loser_base, loser_extra, negative], dim=1)
        if include_intransitivity:
            intra = torch.tensor(
                group[list(INTRA_CONTEXT_COLUMNS)].to_numpy(dtype=np.float32),
                device=device,
            )
            winner_edge = torch.cat([winner_edge, intra], dim=1)
            loser_edge = torch.cat([loser_edge, intra], dim=1)

        blocks.append(
            Block(
                phase=str(group["phase"].iloc[0]),
                metadata=group[
                    [
                        "phase",
                        "tourney_date",
                        "tourney_id",
                        "round_order",
                        "block_idx",
                        "row_in_block",
                        "y_true",
                        "intransitivity_score",
                        "intransitivity_level",
                    ]
                ].reset_index(drop=True),
                player_a=torch.tensor(
                    group["player_a_idx"].to_numpy(),
                    dtype=torch.long,
                    device=device,
                ),
                player_b=torch.tensor(
                    group["player_b_idx"].to_numpy(),
                    dtype=torch.long,
                    device=device,
                ),
                y=torch.tensor(
                    group["y_true"].to_numpy(dtype=np.float32),
                    device=device,
                ),
                context=torch.tensor(
                    group[context_columns].to_numpy(dtype=np.float32),
                    device=device,
                ),
                source=torch.cat([loser, winner]),
                target=torch.cat([winner, loser]),
                edge=torch.cat([winner_edge, loser_edge]),
            )
        )
    return blocks


def replay_until_train(
    model: TemporalTennisGNN, blocks: list[Block]
) -> torch.Tensor:
    state = model.new_state()
    with torch.no_grad():
        for block in blocks:
            if block.phase != "warmup":
                break
            state = model.update_block(
                state, block.source, block.target, block.edge
            )
    return state.detach()


def train_epoch(
    model: TemporalTennisGNN,
    blocks: list[Block],
    optimizer: torch.optim.Optimizer,
    *,
    chunk_blocks: int,
) -> float:
    model.train()
    state = replay_until_train(model, blocks)
    chunk_losses = []
    total_loss = 0.0
    total_matches = 0
    blocks_in_chunk = 0

    def optimise_chunk() -> None:
        nonlocal state, chunk_losses, blocks_in_chunk
        if not chunk_losses:
            return
        optimizer.zero_grad()
        loss = torch.cat(chunk_losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        state = state.detach()
        chunk_losses = []
        blocks_in_chunk = 0

    for block in blocks:
        if block.phase == "warmup":
            continue
        if block.phase != "train":
            break
        logits = model.predict(
            state, block.player_a, block.player_b, block.context
        )
        per_match = nn.functional.binary_cross_entropy_with_logits(
            logits, block.y, reduction="none"
        )
        chunk_losses.append(per_match)
        total_loss += float(per_match.detach().sum())
        total_matches += len(block.y)
        state = model.update_block(
            state, block.source, block.target, block.edge
        )
        blocks_in_chunk += 1
        if blocks_in_chunk >= chunk_blocks:
            optimise_chunk()
    optimise_chunk()
    return total_loss / max(total_matches, 1)


@torch.no_grad()
def evaluate_stream(
    model: TemporalTennisGNN, blocks: list[Block]
) -> pd.DataFrame:
    model.eval()
    state = model.new_state().detach()
    rows = []
    for block in blocks:
        if block.phase in {"val", "test"}:
            logits = model.predict(
                state, block.player_a, block.player_b, block.context
            )
            output = block.metadata.copy()
            output["probability"] = torch.sigmoid(logits).cpu().numpy()
            rows.append(output)
        state = model.update_block(
            state, block.source, block.target, block.edge
        )
    return pd.concat(rows, ignore_index=True)


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
                y * np.log(probability)
                + (1 - y) * np.log(1 - probability)
            )
        ),
        "brier": float(np.mean((probability - y) ** 2)),
    }


def run(
    root: Path,
    *,
    scope: str,
    seed: int,
    include_intransitivity: bool,
    epochs: int,
    hidden_dim: int,
    chunk_blocks: int,
) -> None:
    set_seed(seed)
    device = torch.device("cpu")
    cache_directory = root / "temporal_gnn_intransitivity" / "results" / scope
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache = cache_directory / f"events_cyclic_share_seed_{seed}.pkl"
    if cache.is_file():
        payload = pd.read_pickle(cache)
        frame = payload["frame"]
        player_to_idx = payload["player_to_idx"]
        intra_metadata = payload["intra_metadata"]
    else:
        frame, player_to_idx, intra_metadata = build_event_stream(
            root, scope=scope, seed=seed
        )
        pd.to_pickle(
            {
                "frame": frame,
                "player_to_idx": player_to_idx,
                "intra_metadata": intra_metadata,
            },
            cache,
        )

    blocks = make_blocks(
        frame,
        include_intransitivity=include_intransitivity,
        device=device,
    )
    context_dim = len(BASE_CONTEXT_COLUMNS) + (
        len(INTRA_CONTEXT_COLUMNS) if include_intransitivity else 0
    )
    edge_dim = 10 + (
        len(INTRA_CONTEXT_COLUMNS) if include_intransitivity else 0
    )
    model = TemporalTennisGNN(
        len(player_to_idx),
        hidden_dim=hidden_dim,
        edge_dim=edge_dim,
        context_dim=context_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    losses = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model, blocks, optimizer, chunk_blocks=chunk_blocks
        )
        losses.append(loss)
        print(f"epoch={epoch} train_loss={loss:.6f}", flush=True)

    predictions = evaluate_stream(model, blocks)
    test = predictions[predictions["phase"] == "test"].copy()
    validation = predictions[predictions["phase"] == "val"].copy()
    experiment = (
        "temporal_gnn_intransitivity"
        if include_intransitivity
        else "temporal_gnn"
    )
    artifact_directory = (
        root / "results" / "frozen_predictions" / scope / f"seed_{seed}"
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from edge_feature_ablation.experiment_tracking import (
        ArtifactManifest,
        save_prediction_artifact,
    )

    save_prediction_artifact(
        test,
        artifact_directory,
        ArtifactManifest(
            experiment=experiment,
            model_family="TemporalTennisGNN",
            tournament_scope=scope,
            seed=seed,
            train_end=2015,
            validation_end=2016,
            test_end=2020,
            update_phases=("warmup", "train", "val", "test"),
            config={
                "include_intransitivity": include_intransitivity,
                "intransitivity_threshold_source": "train_2011_2015",
                "intransitivity_thresholds": intra_metadata,
                "epochs": epochs,
                "hidden_dim": hidden_dim,
                "chunk_blocks": chunk_blocks,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "bscore": False,
            },
        ),
        probability_column="probability",
    )
    by_level = {
        level: metrics(group)
        for level, group in test.groupby("intransitivity_level")
    }
    report = {
        "experiment": experiment,
        "scope": scope,
        "seed": seed,
        "train_loss_by_epoch": losses,
        "validation": metrics(validation),
        "test": metrics(test),
        "test_by_intransitivity": by_level,
        "intransitivity_thresholds": intra_metadata,
    }
    report_path = cache_directory / f"{experiment}_seed_{seed}.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("slams", "slams_masters"),
        default="slams_masters",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        choices=("base", "intransitivity"),
        default="base",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--chunk-blocks", type=int, default=32)
    args = parser.parse_args()
    run(
        Path(__file__).resolve().parents[1],
        scope=args.scope,
        seed=args.seed,
        include_intransitivity=args.variant == "intransitivity",
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        chunk_blocks=args.chunk_blocks,
    )


if __name__ == "__main__":
    main()
