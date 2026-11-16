"""调制识别模型训练。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import TensorDataset, train_val_split
from .models import ModulationCNN
from .simulate import MODULATIONS


def train_modulation(
    train_data: dict,
    art_dir: Path,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = train_data["X"].astype(np.float32)
    y = train_data["y"].astype(np.int64)
    (X_tr, y_tr), (X_val, y_val) = train_val_split(X, y)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModulationCNN(n_classes=len(MODULATIONS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    history: list[dict] = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n_seen += len(xb)
        scheduler.step()

        model.eval()
        correct, n_val = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                n_val += len(xb)
        val_acc = correct / max(n_val, 1)
        history.append(
            {"epoch": epoch, "train_loss": total_loss / n_seen, "val_acc": val_acc}
        )
        print(
            f"[mod] epoch {epoch:02d}/{epochs} loss={total_loss / n_seen:.4f} "
            f"val_acc={val_acc:.4f} ({time.time() - start:.0f}s)"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    torch.save(best_state, art_dir / "mod_cnn.pt")
    meta = {
        "best_val_acc": best_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "n_train": len(X_tr),
        "n_val": len(X_val),
        "history": history,
        "classes": list(MODULATIONS),
    }
    (art_dir / "mod_train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
