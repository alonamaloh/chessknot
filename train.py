#!/usr/bin/env python3
"""Train a simple MLP to evaluate Chessnot positions.

Architecture: 45 -> 32 (ReLU) -> 1 (tanh)
Input: 45 class counts normalized by /64
Output: tanh, target is game outcome in {-1, 0, +1}
Loss: MSE

Usage: python train.py [data_glob] [output_model.bin] [epochs]
"""

import glob
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn


class ChessnotMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(45, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_data(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching: {pattern}")
    all_counts, all_outcome = [], []
    for path in paths:
        with h5py.File(path, "r") as f:
            all_counts.append(f["class_counts"][:].astype(np.float32) / 64.0)
            all_outcome.append(f["outcome"][:].astype(np.float32))
        print(f"  {path}: {len(all_counts[-1])} positions")
    return np.concatenate(all_counts), np.concatenate(all_outcome)


def export_model(model, path):
    """Export weights as raw float32 in layer order: w1, b1, w2, b2, w3, b3."""
    with open(path, "wb") as f:
        for layer in model.net:
            if isinstance(layer, nn.Linear):
                w = layer.weight.detach().cpu().numpy().astype(np.float32)
                b = layer.bias.detach().cpu().numpy().astype(np.float32)
                f.write(w.tobytes())
                f.write(b.tobytes())
    size = sum(p.numel() for p in model.parameters())
    print(f"Exported {size} parameters to {path}")


def main():
    data_glob = sys.argv[1] if len(sys.argv) > 1 else "training_data.h5"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "model.bin"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    print(f"Loading {data_glob}...")
    X, y = load_data(data_glob)
    print(f"  {len(X)} positions, outcome distribution: "
          f"+1:{(y > 0).sum()}, 0:{(y == 0).sum()}, -1:{(y < 0).sum()}")

    # Train/validation split (90/10)
    n = len(X)
    perm = np.random.RandomState(42).permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = perm[:split], perm[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load everything onto GPU
    X_train = torch.from_numpy(X[train_idx]).to(device)
    y_train = torch.from_numpy(y[train_idx]).to(device)
    X_val = torch.from_numpy(X[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)

    model = ChessnotMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 4096
    n_train = len(X_train)

    print(f"Training on {device} for {epochs} epochs ({n_train} train, {len(X_val)} val)...")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_train[idx])
            loss = loss_fn(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(idx)
        train_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    print(f"Best validation loss: {best_val_loss:.6f}")
    model.load_state_dict(best_state)
    export_model(model, model_path)


if __name__ == "__main__":
    main()
