"""
GNN-GP for spatio-temporal soil moisture imputation — Cook Agronomy Farm.

Reference:
    T. Rashid, A. Majumder, B. Natarajan (2026). Graph-based spatio-temporal
    soil moisture imputation with uncertainty quantification.
    Smart Agricultural Technology 15, 102455.

What it does (Case 2 of the paper):
    A fraction of sensor locations (30, 50 or 70 %) is held out: every reading
    at those locations is removed from training. The model then imputes the
    volumetric water content (VWC) at the held-out locations, for every depth
    and date, with a predictive mean and standard deviation.

Model (paper Section 3.4, Table 2):
    1. Graph:   k-nearest-neighbour graph (k = 5) on the scaled 3D coordinates
                (x, y, depth) of all training and test points.
    2. GNN:     two GraphSAGE layers (mean aggregation, 8 units, tanh, 20 %
                dropout) turn each point's coordinates into an 8-dim embedding.
    3. GP:      exact Gaussian process with kernel
                    K = alpha * K_crop * K_spatial * K_temporal
                K_spatial  : RBF with 8 length-scales (ARD) on the embedding
                K_temporal : RBF with 1 length-scale on the scaled date
                K_crop     : IndexKernel over the 5 crops (rank 1)
    4. Training: the GNN weights and all GP parameters are optimised together
                with Adam (lr 0.01, 60 epochs, full batch) to maximise the GP
                marginal log-likelihood.

Metrics (paper Section 4.3): MAE, RMSE, PICP and NMPIW of the 95 % interval.

Usage:
    python gnngp_cook.py                      # 30, 50 and 70 % missing, seeds 42 and 43
    python gnngp_cook.py --missing 0.5        # one missing fraction
    python gnngp_cook.py --device cpu         # force CPU (default: GPU if available)
"""

import argparse
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

# ── Settings (paper Table 2) ─────────────────────────────────────────────────
K_NEIGHBOURS = 5        # k-NN graph connectivity
HIDDEN_DIM = 8          # GraphSAGE hidden and output dimension
N_LAYERS = 2            # GraphSAGE layers
DROPOUT = 0.2           # dropout after the hidden layer
EPOCHS = 60
LR = 0.01
SEEDS = [42, 43]
Z_95 = 1.96             # 95 % interval: mean +/- 1.96 std

SPATIAL_COLS = ["x_coordinate", "y_coordinate", "Depth"]   # already scaled to [0, 1]
N_CROPS = 5             # Crop_Code takes the values 0, 0.25, 0.5, 0.75, 1.0

DEFAULT_DATA = Path(__file__).parent / "data" / "cook_2015_springsummer_final.csv"


# ── Data ─────────────────────────────────────────────────────────────────────
def load_cook(path):
    """Read the Cook CSV and drop rows without a VWC reading.

    The CSV already holds x, y, depth and date scaled to [0, 1], and the crop
    as Crop_Code in {0, 0.25, 0.5, 0.75, 1.0}.
    """
    cols = SPATIAL_COLS + ["Date_Normalized", "VWC", "Crop_Code"]
    df = pd.read_csv(path)[cols].dropna(subset=["VWC"])
    n_loc = df[["x_coordinate", "y_coordinate"]].drop_duplicates().shape[0]
    print(f"Cook: {len(df):,} readings at {n_loc} sensor locations")
    return df


def split_by_location(df, missing_frac, seed):
    """Hold out a random fraction of sensor locations (all depths and dates).

    Splitting by location, not by reading, means the test locations are
    never observed in training (paper Section 4.2).
    """
    locations = df[["x_coordinate", "y_coordinate"]].drop_duplicates()
    n_test = max(1, int(len(locations) * missing_frac))
    test_loc = locations.sample(n=n_test, random_state=seed)
    train_loc = locations.drop(test_loc.index)
    on = ["x_coordinate", "y_coordinate"]
    train = df.merge(train_loc, on=on).reset_index(drop=True)
    test = df.merge(test_loc, on=on).reset_index(drop=True)
    return train, test


# ── Graph ────────────────────────────────────────────────────────────────────
def build_knn_graph(train_xyz, test_xyz, k):
    """Edge list for the k-NN graph over training and test points.

    Nodes 0..n_train-1 are training points, the rest are test points.
    Each training point links to its k nearest other training points, and
    each test point links to its k nearest training points (never to other
    test points). Edges are unweighted and made two-way, so every node
    receives messages from its neighbours: PyG passes messages from
    edge_index[0] (source) to edge_index[1] (target).
    """
    n_train = len(train_xyz)
    edges = []

    knn = NearestNeighbors(n_neighbors=k + 1).fit(train_xyz)
    _, nbrs = knn.kneighbors(train_xyz)
    for i, row in enumerate(nbrs):
        for j in row[1:]:                    # row[0] is the point itself
            edges.append((i, j))

    knn = NearestNeighbors(n_neighbors=k).fit(train_xyz)
    _, nbrs = knn.kneighbors(test_xyz)
    for i, row in enumerate(nbrs):
        for j in row:
            edges.append((n_train + i, j))

    edge_index = torch.tensor(edges, dtype=torch.long).T     # shape [2, n_edges]
    return to_undirected(edge_index, num_nodes=n_train + len(test_xyz))


# ── GNN ──────────────────────────────────────────────────────────────────────
class GraphSAGEEncoder(nn.Module):
    """GraphSAGE layers that map (x, y, depth) to an embedding.

    tanh and dropout follow every layer except the last, so the final layer
    gives the embedding that the GP's spatial kernel uses.
    """

    def __init__(self, in_dim=3, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1], aggr="mean") for i in range(n_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.dropout(torch.tanh(x))
        return x


# ── GP ───────────────────────────────────────────────────────────────────────
class CropSpatioTemporalGP(gpytorch.models.ExactGP):
    """Exact GP with kernel alpha * K_crop * K_spatial * K_temporal.

    Input columns: [embedding (emb_dim), date (1), crop index (1)].
    """

    def __init__(self, train_x, train_y, likelihood, emb_dim):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        spatial = gpytorch.kernels.RBFKernel(
            ard_num_dims=emb_dim, active_dims=tuple(range(emb_dim)))
        temporal = gpytorch.kernels.RBFKernel(active_dims=(emb_dim,))
        crop = gpytorch.kernels.IndexKernel(
            num_tasks=N_CROPS, rank=1, active_dims=(emb_dim + 1,))
        self.covar_module = gpytorch.kernels.ScaleKernel(crop * spatial * temporal)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def crop_index(crop_code):
    """Crop_Code {0, 0.25, 0.5, 0.75, 1.0} -> index {0, 1, 2, 3, 4} (as float)."""
    return np.rint(crop_code * (N_CROPS - 1)).astype(np.float32)


# ── Train and predict ────────────────────────────────────────────────────────
def train_and_predict(train, test, seed, dropout=DROPOUT, epochs=EPOCHS,
                      device="cpu"):
    """Fit GNN-GP on the training points; return test mean, std and truth."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def tensor(a):
        return torch.tensor(np.asarray(a), dtype=torch.float, device=device)

    xyz = tensor(np.vstack([train[SPATIAL_COLS].values, test[SPATIAL_COLS].values]))
    t_train = tensor(train["Date_Normalized"].values).unsqueeze(-1)
    t_test = tensor(test["Date_Normalized"].values).unsqueeze(-1)
    c_train = tensor(crop_index(train["Crop_Code"].values)).unsqueeze(-1)
    c_test = tensor(crop_index(test["Crop_Code"].values)).unsqueeze(-1)
    y_train = tensor(train["VWC"].values)
    y_test = test["VWC"].values
    n_train = len(train)

    edge_index = build_knn_graph(train[SPATIAL_COLS].values,
                                 test[SPATIAL_COLS].values, K_NEIGHBOURS).to(device)
    gnn = GraphSAGEEncoder(dropout=dropout).to(device)

    def gp_input(emb, rows, t, c):
        # [embedding | date | crop] for the selected rows
        return torch.cat([emb[rows], t, c], dim=-1)

    # The GP needs initial training inputs; it is re-fed every epoch below.
    with torch.no_grad():
        x0 = gp_input(gnn(xyz, edge_index), slice(0, n_train), t_train, c_train)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    gp = CropSpatioTemporalGP(x0, y_train, likelihood, emb_dim=HIDDEN_DIM).to(device)

    # One optimiser for the GNN weights and the GP parameters together.
    optimiser = torch.optim.Adam(list(gnn.parameters()) + list(gp.parameters()), lr=LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    gnn.train(); gp.train(); likelihood.train()
    for _ in range(epochs):
        optimiser.zero_grad()
        x = gp_input(gnn(xyz, edge_index), slice(0, n_train), t_train, c_train)
        gp.set_train_data(x, y_train, strict=False)     # embeddings changed
        loss = -mll(gp(x), y_train)
        loss.backward()
        optimiser.step()

    gnn.eval(); gp.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        emb = gnn(xyz, edge_index)
        gp.set_train_data(gp_input(emb, slice(0, n_train), t_train, c_train),
                          y_train, strict=False)
        pred = likelihood(gp(gp_input(emb, slice(n_train, None), t_test, c_test)))
    return pred.mean.cpu().numpy(), pred.stddev.cpu().numpy(), y_test


# ── Metrics ──────────────────────────────────────────────────────────────────
def metrics(mean, std, truth):
    """MAE, RMSE, PICP and NMPIW of the 95 % interval (paper Section 4.3)."""
    lower, upper = mean - Z_95 * std, mean + Z_95 * std
    return {
        "MAE": float(np.mean(np.abs(mean - truth))),
        "RMSE": float(np.sqrt(np.mean((mean - truth) ** 2))),
        "PICP": float(np.mean((truth >= lower) & (truth <= upper))),
        "NMPIW": float(np.mean(upper - lower) / (truth.max() - truth.min())),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GNN-GP on the Cook dataset")
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="path to cook_2015_springsummer_final.csv")
    parser.add_argument("--missing", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                        help="fractions of sensor locations held out")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    df = load_cook(args.data)
    names = ["MAE", "RMSE", "PICP", "NMPIW"]
    print(f"\n{'missing':>8} {'seed':>5} " + " ".join(f"{n:>7}" for n in names))
    for frac in args.missing:
        runs = []
        for seed in args.seeds:
            train, test = split_by_location(df, frac, seed)
            m = metrics(*train_and_predict(train, test, seed, dropout=args.dropout,
                                           epochs=args.epochs, device=args.device))
            runs.append(m)
            print(f"{frac:>8.0%} {seed:>5} " + " ".join(f"{m[n]:>7.3f}" for n in names),
                  flush=True)
        avg = {n: np.mean([r[n] for r in runs]) for n in names}
        print(f"{frac:>8.0%} {'mean':>5} " + " ".join(f"{avg[n]:>7.3f}" for n in names))


if __name__ == "__main__":
    main()
