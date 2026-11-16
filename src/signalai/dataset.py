"""数据构建、保存与窗口化。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .simulate import (
    build_iq_eval,
    build_iq_train,
    build_telemetry_test,
    build_telemetry_train,
    build_telemetry_val,
)

WINDOW = 64
STRIDE = 16


def ensure_iq_data(
    data_dir: Path,
    n_train_per_class: int = 1200,
    n_eval_per_class: int = 200,
    seed: int = 0,
) -> tuple[dict, dict]:
    train_path = data_dir / "iq_train.npz"
    eval_path = data_dir / "iq_eval.npz"
    if not train_path.exists():
        X, y, snr = build_iq_train(n_train_per_class, seed=seed)
        np.savez_compressed(train_path, X=X, y=y, snr=snr)
    if not eval_path.exists():
        X, y, snr = build_iq_eval(n_eval_per_class, seed=seed + 1)
        np.savez_compressed(eval_path, X=X, y=y, snr=snr)
    return dict(np.load(train_path)), dict(np.load(eval_path))


def ensure_telemetry_data(
    data_dir: Path,
    seed: int = 2,
) -> dict:
    path = data_dir / "telemetry.npz"
    if not path.exists():
        X_train = build_telemetry_train(seed=seed)
        X_val = build_telemetry_val(seed=seed + 1)
        X_test, mask_test, kinds = build_telemetry_test(seed=seed + 2)
        np.savez_compressed(
            path,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            mask_test=mask_test,
            kinds=np.array(kinds),
        )
    return dict(np.load(path, allow_pickle=False))


def window_label(mask: np.ndarray, start: int, end: int) -> int:
    """窗口只要包含任一异常点即标为异常（尖峰类异常点稀疏，不能用覆盖率）。"""
    return int(np.any(mask[start:end]))


def windows_from_batch(
    X: np.ndarray,
    masks: np.ndarray | None = None,
    window: int = WINDOW,
    stride: int = STRIDE,
):
    """把 (N, L, C) 序列切为窗口，返回 (windows, labels)。"""
    n, length, _ = X.shape
    starts = list(range(0, length - window + 1, stride))
    out = np.empty((n * len(starts), window, X.shape[2]), dtype=np.float32)
    labels = np.zeros(n * len(starts), dtype=np.int64)
    i = 0
    for j in range(n):
        for s in starts:
            out[i] = X[j, s:s + window]
            if masks is not None:
                labels[i] = window_label(masks[j], s, s + window)
            i += 1
    return out, labels


class TensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X)
        self.y = torch.as_tensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def train_val_split(
    X: np.ndarray, y: np.ndarray, val_frac: float = 0.1, seed: int = 7
) -> tuple[tuple, tuple]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])
