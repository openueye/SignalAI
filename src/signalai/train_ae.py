"""遥测异常检测自编码器训练。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import TensorDataset, windows_from_batch
from .models import TelemetryAE

WINDOW = 64
STRIDE = 16


def standardize(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


def train_autoencoder(
    telemetry: dict,
    art_dir: Path,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_val, X_test, mean, std = standardize(
        telemetry["X_train"], telemetry["X_val"], telemetry["X_test"]
    )
    np.savez_compressed(
        art_dir / "telemetry_scaler.npz", mean=mean, std=std
    )

    win_train, _ = windows_from_batch(X_train, None, WINDOW, STRIDE)

    train_loader = DataLoader(
        TensorDataset(win_train, np.zeros(len(win_train), dtype=np.int64)),
        batch_size=batch_size,
        shuffle=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TelemetryAE(channels=win_train.shape[-1], window=WINDOW).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history: list[dict] = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_seen = 0.0, 0
        for xb, _ in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            n_seen += len(xb)
        history.append({"epoch": epoch, "train_mse": total / n_seen})
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"[ae] epoch {epoch:02d}/{epochs} mse={total / n_seen:.5f} "
                f"({time.time() - start:.0f}s)"
            )

    torch.save(model.state_dict(), art_dir / "telemetry_ae.pt")

    meta = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "history": history,
        "n_train_windows": int(len(win_train)),
    }
    (art_dir / "ae_train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
