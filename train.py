#!/usr/bin/env python3
"""Train a simple MLP to evaluate Chessnot positions.

Architecture: 45 -> 128 (ReLU) -> 64 (ReLU) -> 1 (tanh)
Input: 45 class counts normalized by /64
Output: tanh, target is game outcome in {-1, 0, +1}
Loss: MSE

Usage: python train.py [training_data.h5] [output_model.bin] [epochs]
"""

import struct
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ChessnotMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(45, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_data(path):
    with h5py.File(path, "r") as f:
        counts = f["class_counts"][:].astype(np.float32) / 64.0
        outcome = f["outcome"][:].astype(np.float32)
    return counts, outcome


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
    data_path = sys.argv[1] if len(sys.argv) > 1 else "training_data.h5"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "model.bin"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    print(f"Loading {data_path}...")
    X, y = load_data(data_path)
    print(f"  {len(X)} positions, outcome distribution: "
          f"+1:{(y > 0).sum()}, 0:{(y == 0).sum()}, -1:{(y < 0).sum()}")

    # Train/validation split (90/10)
    n = len(X)
    perm = np.random.RandomState(42).permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = perm[:split], perm[split:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessnotMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"Training on {device} for {epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = loss_fn(val_pred, y_val.to(device)).item()

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
