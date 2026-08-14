"""Generate the ablation notebook from the existing training experiment.

Only the imports, edge construction and edge dimension are changed.  The
temporal split, seed, target creation, model, optimiser and rolling training
loop remain byte-for-byte identical to the source notebook.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "NN_test" / "GNN_intransitivity_changes.ipynb"
OUTPUT = HERE / "GNN_edge_feature_ablation.ipynb"


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)
    cell["execution_count"] = None
    cell["outputs"] = []


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]
for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

imports = source(cells[0])
imports += """
import os
from dataclasses import asdict, replace

try:
    from edge_feature_ablation.edge_features import (
        ABLATION_PRESETS,
        build_bidirectional_edges,
        edge_feature_names,
    )
    from edge_feature_ablation.experiment_config import MODEL_EXPERIMENTS
except ModuleNotFoundError:
    # When the kernel working directory is this notebook's directory.
    from edge_features import (
        ABLATION_PRESETS,
        build_bidirectional_edges,
        edge_feature_names,
    )
    from experiment_config import MODEL_EXPERIMENTS
"""
set_source(cells[0], imports)

seed_cell = source(cells[1])
seed_cell = seed_cell.replace(
    "SEED = 42",
    'SEED = int(os.environ.get("TENNIS_SEED", "42"))',
)
seed_cell += """

# Change only this environment variable between ablations.  All experimental
# constants below remain identical to the source experiment.
MODEL_EXPERIMENT = os.environ.get("TENNIS_MODEL_EXPERIMENT", "edge_full")
if MODEL_EXPERIMENT not in MODEL_EXPERIMENTS:
    raise ValueError(
        f"Unknown model experiment {MODEL_EXPERIMENT!r}; "
        f"choose one of {tuple(MODEL_EXPERIMENTS)}"
    )
MODEL_CONFIG = MODEL_EXPERIMENTS[MODEL_EXPERIMENT]

ABLATION_NAME = os.environ.get(
    "TENNIS_EDGE_ABLATION", MODEL_CONFIG.edge_preset
)
if ABLATION_NAME not in ABLATION_PRESETS:
    raise ValueError(
        f"Unknown ablation {ABLATION_NAME!r}; "
        f"choose one of {tuple(ABLATION_PRESETS)}"
    )
EDGE_CONFIG = ABLATION_PRESETS[ABLATION_NAME]
if MODEL_CONFIG.tournament_context_edges:
    EDGE_CONFIG = replace(EDGE_CONFIG, include_tournament_context=True)

# Manual switch for the tournament population used by both the GNN and the
# final logistic-regression baseline.
TOURNAMENT_SCOPE = os.environ.get(
    "TENNIS_TOURNAMENT_SCOPE", "slams_masters"
)  # "slams" or "slams_masters"
if TOURNAMENT_SCOPE not in {"slams", "slams_masters"}:
    raise ValueError(
        "TOURNAMENT_SCOPE must be 'slams' or 'slams_masters'"
    )

print("model experiment:", MODEL_EXPERIMENT)
print("model config:", asdict(MODEL_CONFIG))
print("edge ablation:", ABLATION_NAME)
print("edge features:", edge_feature_names(EDGE_CONFIG))
print("tournament scope:", TOURNAMENT_SCOPE)
"""
set_source(cells[1], seed_cell)

paths_cell = source(cells[2])
paths_cell = paths_cell.replace(
    'MATCHES_PATH = Path(PROCESSED_DIR / "atp_matches_slams_masters.csv")',
    """MATCHES_FILENAME = (
    "atp_matches_slams.csv"
    if TOURNAMENT_SCOPE == "slams"
    else "atp_matches_slams_masters.csv"
)
MATCHES_PATH = Path(PROCESSED_DIR / MATCHES_FILENAME)""",
)
set_source(cells[2], paths_cell)

set_source(
    cells[9],
    """## Score parsing and status handling live in edge_features.py.
# Keeping this cell makes the generated notebook structure align with the
# source experiment while ensuring every ablation uses the tested parser.
""",
)

graph_cell = source(cells[13])
graph_cell = graph_cell.replace(
    "    x = torch.tensor(node_features, dtype=torch.float)\n",
    """    x = torch.tensor(node_features, dtype=torch.float)
    raw_bscore_general = x[:, 0].clone()
    if not MODEL_CONFIG.node_bscore_features:
        # Keep the architecture fixed while removing general and
        # surface-specific B-score information from message passing.
        x[:, :4] = 0.0
    if MODEL_CONFIG.normalize_node_features and x.shape[0] > 1:
        # Continuous node features only; the hand flag remains binary.
        continuous = x[:, :5]
        feature_mean = continuous.mean(dim=0, keepdim=True)
        feature_std = continuous.std(dim=0, keepdim=True, unbiased=False)
        feature_std = torch.where(
            feature_std < 1e-6,
            torch.ones_like(feature_std),
            feature_std,
        )
        x[:, :5] = (continuous - feature_mean) / feature_std
""",
)
start = graph_cell.index("    edge_rows = []")
end = graph_cell.index("    ## NN's wissen nicht", start)
replacement = """    edge_pairs, edge_attributes, edge_stats = build_bidirectional_edges(
        matches=current_history_trimmed.to_dict("records"),
        player_to_idx=player_to_idx,
        snapshot_date=t_block,
        config=EDGE_CONFIG,
        alpha_days=alpha_days,
    )

    edge_dim = len(edge_feature_names(EDGE_CONFIG))
    if len(edge_pairs) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(
            edge_pairs, dtype=torch.long
        ).t().contiguous()
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

"""
set_source(cells[13], graph_cell[:start] + replacement + graph_cell[end:])
graph_cell = source(cells[13])
graph_cell = graph_cell.replace(
    "        edge_attr=edge_attr\n",
    "        edge_attr=edge_attr,\n"
    "        raw_bscore_general=torch.nan_to_num(\n"
    "            raw_bscore_general, nan=0.0, posinf=0.0, neginf=0.0\n"
    "        )\n",
)
set_source(cells[13], graph_cell)

model_definition = """class TennisGINE(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        hidden_dim,
        match_context_dim,
        dropout=0.5,
        init_scale=1.0,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)

        def message_mlp():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.conv1 = GINEConv(
            message_mlp(),
            edge_dim=edge_in_dim,
            aggr=MODEL_CONFIG.aggregation,
        )
        self.conv2 = GINEConv(
            message_mlp(),
            edge_dim=edge_in_dim,
            aggr=MODEL_CONFIG.aggregation,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        pair_dim = hidden_dim * 4
        self.match_head = nn.Sequential(
            nn.Linear(pair_dim + match_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.bscore_scale = nn.Parameter(torch.tensor(float(init_scale)))
        nn.init.zeros_(self.match_head[-1].weight)
        nn.init.zeros_(self.match_head[-1].bias)

    @staticmethod
    def pair_features(h_a, h_b):
        return torch.cat(
            [h_a, h_b, h_a - h_b, torch.abs(h_a - h_b)],
            dim=1,
        )

    def pair_score(self, h_a, h_b, match_context):
        pair_z = self.pair_features(h_a, h_b)
        return self.match_head(
            torch.cat([pair_z, match_context], dim=1)
        ).squeeze(-1)

    def forward(self, data, player_a_idx, player_b_idx, match_context):
        h0 = self.node_encoder(data.x)

        message1 = self.norm1(
            F.relu(self.conv1(h0, data.edge_index, data.edge_attr))
        )
        h1 = h0 + message1 if MODEL_CONFIG.residual_connections else message1

        message2 = self.norm2(
            F.relu(self.conv2(h1, data.edge_index, data.edge_attr))
        )
        h = h1 + message2 if MODEL_CONFIG.residual_connections else message2

        h_a = h[player_a_idx]
        h_b = h[player_b_idx]
        correction_ab = self.pair_score(h_a, h_b, match_context)

        if MODEL_CONFIG.antisymmetric_decoder:
            correction_ba = self.pair_score(h_b, h_a, match_context)
            correction = 0.5 * (correction_ab - correction_ba)
        else:
            correction = correction_ab

        if MODEL_CONFIG.direct_bscore_logit:
            bscore_a = data.raw_bscore_general[player_a_idx]
            bscore_b = data.raw_bscore_general[player_b_idx]
            logits = self.bscore_scale * (bscore_a - bscore_b) + correction
        elif MODEL_CONFIG.antisymmetric_decoder:
            # No intercept is allowed here: it would break
            # logit(B, A) = -logit(A, B).
            logits = correction
        else:
            # Exact legacy behaviour retained for current_gnn and edge_full.
            logits = self.bscore_scale + correction

        return logits, (h_a - h_b).detach()
"""
set_source(cells[15], model_definition)

model_cell = source(cells[16]).replace(
    "edge_in_dim=7,",
    "edge_in_dim=len(edge_feature_names(EDGE_CONFIG)),",
)
set_source(cells[16], model_cell)

# Keep the final B-score logistic baseline on exactly the same tournament
# population selected for the GNN.
comparison_cell = source(cells[19])
comparison_cell = comparison_cell.replace(
    'categories = ["G", "M"]',
    """categories = (
    ["G"]
    if TOURNAMENT_SCOPE == "slams"
    else ["G", "M"]
)""",
)
set_source(cells[19], comparison_cell)

training_cell = source(cells[20])
training_cell = training_cell.replace(
    '            "block_idx": block_idx,\n'
    '            "emb_diff": emb_diff[i].tolist(),',
    '            "block_idx": block_idx,\n'
    '            "row_in_block": i,\n'
    '            "emb_diff": emb_diff[i].tolist(),',
)
set_source(cells[20], training_cell)

metadata = notebook.setdefault("metadata", {})
metadata["edge_feature_ablation"] = {
    "source_notebook": str(SOURCE.relative_to(HERE.parent)),
    "generated_by": Path(__file__).name,
}

OUTPUT.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT}")
