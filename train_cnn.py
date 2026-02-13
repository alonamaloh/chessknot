#!/usr/bin/env python3
"""Train a CNN to evaluate Chessnot positions and predict moves.

Architecture:
  Input: 2x8x8 (white/black bitboard planes)
  Trunk: Conv 2→32 + 3 residual blocks (32 channels)
  Heads: value (→tanh), move_from (64-way), move_to (64-way)
  Loss: MSE(value) + CE(from) + CE(to)

Usage: python train_cnn.py [data_glob] [output_model.pt] [epochs] [max_samples]

Only loads HDF5 files that contain move_from/move_to fields.
"""

import glob
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_SHIFTS = None  # lazily initialized on correct device


def _get_shifts(device):
    global _SHIFTS
    if _SHIFTS is None or _SHIFTS.device != device:
        _SHIFTS = torch.arange(64, device=device, dtype=torch.int64)
    return _SHIFTS


def unpack_bitboards(white, black):
    """Convert (B,) int64 bitboard tensors to (B, 2, 8, 8) float32 on GPU."""
    shifts = _get_shifts(white.device)
    w = ((white.unsqueeze(1) >> shifts) & 1).float().reshape(-1, 8, 8)
    b = ((black.unsqueeze(1) >> shifts) & 1).float().reshape(-1, 8, 8)
    return torch.stack([w, b], dim=1)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ChessnotCNN(nn.Module):
    def __init__(self, channels=32, num_blocks=3):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # Value head: conv 1x1→1, flatten(64), linear→1, tanh
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_fc = nn.Linear(64, 1)

        # From head: conv 1x1→1, flatten(64) → 64 logits
        self.from_conv = nn.Conv2d(channels, 1, 1)

        # To head: conv 1x1→1, flatten(64) → 64 logits
        self.to_conv = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        trunk = self.trunk(self.input_conv(x))

        # Value
        v = F.relu(self.value_conv(trunk))
        v = torch.tanh(self.value_fc(v.view(v.size(0), -1))).squeeze(-1)

        # From logits
        f = self.from_conv(trunk).view(trunk.size(0), -1)

        # To logits
        t = self.to_conv(trunk).view(trunk.size(0), -1)

        return v, f, t


def load_data(pattern):
    """Load raw bitboards (as int64 for torch) plus labels from HDF5 files."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching: {pattern}")
    all_white, all_black, all_outcome, all_from, all_to = [], [], [], [], []
    skipped = 0
    for path in paths:
        with h5py.File(path, "r") as f:
            if "move_from" not in f or "move_to" not in f:
                print(f"  {path}: skipped (no move data)")
                skipped += 1
                continue
            # Store as int64 (torch has no uint64); bit patterns are preserved
            all_white.append(f["white"][:].view(np.int64))
            all_black.append(f["black"][:].view(np.int64))
            all_outcome.append(f["outcome"][:].astype(np.float32))
            all_from.append(f["move_from"][:].astype(np.int64))
            all_to.append(f["move_to"][:].astype(np.int64))
            print(f"  {path}: {len(all_outcome[-1])} positions")
    if not all_white:
        raise FileNotFoundError(f"No files with move data found (skipped {skipped})")
    return (
        np.concatenate(all_white),
        np.concatenate(all_black),
        np.concatenate(all_outcome),
        np.concatenate(all_from),
        np.concatenate(all_to),
    )


def main():
    data_glob = sys.argv[1] if len(sys.argv) > 1 else "training_data*.h5"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "model_cnn.pt"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    print(f"Loading {data_glob}...")
    white, black, outcome, move_from, move_to = load_data(data_glob)
    print(f"  {len(white)} positions, outcome distribution: "
          f"+1:{(outcome > 0).sum()}, 0:{(outcome == 0).sum()}, -1:{(outcome < 0).sum()}")

    # Train/validation split (90/10)
    n = len(white)
    perm = np.random.RandomState(42).permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = perm[:split], perm[split:]

    if max_samples > 0:
        max_train = int(max_samples * 0.9)
        max_val = max_samples - max_train
        if len(train_idx) > max_train:
            train_idx = train_idx[:max_train]
        if len(val_idx) > max_val:
            val_idx = val_idx[:max_val]
        print(f"  Subsampled to {len(train_idx)} train, {len(val_idx)} val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Store raw int64 bitboards on GPU (16 bytes/pos vs 512 for float32 planes)
    w_train = torch.from_numpy(white[train_idx]).to(device)
    b_train = torch.from_numpy(black[train_idx]).to(device)
    y_train = torch.from_numpy(outcome[train_idx]).to(device)
    f_train = torch.from_numpy(move_from[train_idx]).to(device)
    t_train = torch.from_numpy(move_to[train_idx]).to(device)

    w_val = torch.from_numpy(white[val_idx]).to(device)
    b_val = torch.from_numpy(black[val_idx]).to(device)
    y_val = torch.from_numpy(outcome[val_idx]).to(device)
    f_val = torch.from_numpy(move_from[val_idx]).to(device)
    t_val = torch.from_numpy(move_to[val_idx]).to(device)

    model = ChessnotCNN().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()
    ce_fn = nn.CrossEntropyLoss()

    batch_size = 4096
    n_train = len(w_train)

    print(f"Training on {device} for {epochs} epochs ({n_train} train, {len(w_val)} val)...")
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm_t = torch.randperm(n_train, device=device)
        train_vloss = 0.0
        train_floss = 0.0
        train_tloss = 0.0
        train_fcorrect = 0
        train_tcorrect = 0

        for i in range(0, n_train, batch_size):
            idx = perm_t[i:i + batch_size]
            xb = unpack_bitboards(w_train[idx], b_train[idx])
            val_pred, from_logits, to_logits = model(xb)

            loss_v = mse_fn(val_pred, y_train[idx])
            loss_f = ce_fn(from_logits, f_train[idx])
            loss_t = ce_fn(to_logits, t_train[idx])
            loss = loss_v + loss_f + loss_t

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = len(idx)
            train_vloss += loss_v.item() * bs
            train_floss += loss_f.item() * bs
            train_tloss += loss_t.item() * bs
            train_fcorrect += (from_logits.argmax(1) == f_train[idx]).sum().item()
            train_tcorrect += (to_logits.argmax(1) == t_train[idx]).sum().item()

        train_vloss /= n_train
        train_floss /= n_train
        train_tloss /= n_train
        train_facc = train_fcorrect / n_train
        train_tacc = train_tcorrect / n_train

        # Validation
        model.eval()
        n_val = len(w_val)
        val_vloss = 0.0
        val_floss = 0.0
        val_tloss = 0.0
        val_fcorrect = 0
        val_tcorrect = 0

        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                xb = unpack_bitboards(w_val[i:i + batch_size], b_val[i:i + batch_size])
                val_pred, from_logits, to_logits = model(xb)

                bs = len(xb)
                val_vloss += mse_fn(val_pred, y_val[i:i + bs]).item() * bs
                val_floss += ce_fn(from_logits, f_val[i:i + bs]).item() * bs
                val_tloss += ce_fn(to_logits, t_val[i:i + bs]).item() * bs
                val_fcorrect += (from_logits.argmax(1) == f_val[i:i + bs]).sum().item()
                val_tcorrect += (to_logits.argmax(1) == t_val[i:i + bs]).sum().item()

        val_vloss /= n_val
        val_floss /= n_val
        val_tloss /= n_val
        val_facc = val_fcorrect / n_val
        val_tacc = val_tcorrect / n_val
        val_total = val_vloss + val_floss + val_tloss

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}  "
                  f"train[mse={train_vloss:.4f} from={train_facc:.3f} to={train_tacc:.3f}]  "
                  f"val[mse={val_vloss:.4f} from={val_facc:.3f} to={val_tacc:.3f}]")

    print(f"\nBest validation total loss: {best_val_loss:.6f}")
    model.load_state_dict(best_state)
    torch.save(best_state, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
