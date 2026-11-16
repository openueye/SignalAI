"""模型评测与图表输出。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .dataset import windows_from_batch
from .models import ModulationCNN, TelemetryAE
from .simulate import MODULATIONS

WINDOW = 64
STRIDE = 16


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_figure(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_modulation(art_dir: Path, data: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = _device()
    model = ModulationCNN(n_classes=len(MODULATIONS)).to(device)
    model.load_state_dict(torch.load(art_dir / "mod_cnn.pt", map_location=device))
    model.eval()

    X = torch.as_tensor(data["X"], dtype=torch.float32)
    y = data["y"].astype(np.int64)
    snr = data["snr"].astype(np.float64)

    preds = []
    with torch.no_grad():
        for start in range(0, len(X), 512):
            logits = model(X[start:start + 512].to(device))
            preds.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds)

    acc = float(np.mean(pred == y))
    snr_values = sorted(float(s) for s in np.unique(snr))
    acc_by_snr = {
        f"{s:g}": float(np.mean(pred[snr == s] == y[snr == s])) for s in snr_values
    }
    acc_ge_0db = float(np.mean(pred[snr >= 0] == y[snr >= 0]))
    cm = confusion_matrix(y, pred, labels=list(range(len(MODULATIONS))))
    per_class = {
        MODULATIONS[i]: float(cm[i, i] / max(cm[i].sum(), 1))
        for i in range(len(MODULATIONS))
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([float(k) for k in acc_by_snr], list(acc_by_snr.values()), marker="o")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title("Modulation recognition accuracy vs. SNR")
    _save_figure(fig, art_dir / "fig_acc_vs_snr.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(MODULATIONS)), MODULATIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(MODULATIONS)), MODULATIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(MODULATIONS)):
        for j in range(len(MODULATIONS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion matrix (all SNRs)")
    _save_figure(fig, art_dir / "fig_confusion.png")

    metrics = {
        "accuracy": acc,
        "accuracy_ge_0db": acc_ge_0db,
        "accuracy_by_snr": acc_by_snr,
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y)),
        "classes": list(MODULATIONS),
    }
    (art_dir / "metrics_mod.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[eval-mod] overall acc={acc:.4f}  acc(SNR>=0dB)={acc_ge_0db:.4f}\n"
        + "  " + "  ".join(f"{k}dB:{v:.3f}" for k, v in acc_by_snr.items())
    )
    return metrics


def _per_channel_mse(model, windows: np.ndarray, device: torch.device) -> np.ndarray:
    """返回每个窗口、每个通道的平均重建误差，形状 (N, C)。"""
    rows = []
    with torch.no_grad():
        for start in range(0, len(windows), 1024):
            xb = torch.as_tensor(windows[start:start + 1024], dtype=torch.float32).to(device)
            rows.append((model(xb) - xb).pow(2).mean(dim=1).cpu().numpy())
    return np.concatenate(rows)


def evaluate_autoencoder(art_dir: Path, telemetry: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = _device()
    scaler = np.load(art_dir / "telemetry_scaler.npz")
    mean, std = scaler["mean"], scaler["std"]
    X_val = (telemetry["X_val"] - mean) / std
    X_test = (telemetry["X_test"] - mean) / std

    model = TelemetryAE(channels=X_test.shape[-1], window=WINDOW).to(device)
    model.load_state_dict(torch.load(art_dir / "telemetry_ae.pt", map_location=device))
    model.eval()

    val_win, _ = windows_from_batch(X_val, None, WINDOW, STRIDE)
    test_win, y_win = windows_from_batch(X_test, telemetry["mask_test"], WINDOW, STRIDE)

    val_err = _per_channel_mse(model, val_win, device)
    test_err = _per_channel_mse(model, test_win, device)

    # 逐通道标准化重建误差，取最大通道作为窗口分数（异常通常只出现在单通道）
    mu = val_err.mean(axis=0)
    sd = val_err.std(axis=0) + 1e-8
    val_scores = ((val_err - mu) / sd).max(axis=1)
    scores = ((test_err - mu) / sd).max(axis=1)

    threshold = float(np.quantile(val_scores, 0.995))
    pred = scores > threshold

    auroc = float(roc_auc_score(y_win, scores))
    precision = float(precision_score(y_win, pred, zero_division=0))
    recall = float(recall_score(y_win, pred, zero_division=0))
    f1 = float(f1_score(y_win, pred, zero_division=0))

    # 序列级：一条遥测序列取窗口分数的最大值，阈值由正常验证序列的 95 分位标定
    n_starts = (telemetry["X_test"].shape[1] - WINDOW) // STRIDE + 1
    n_seq = len(telemetry["X_test"])
    assert len(scores) == n_seq * n_starts
    seq_scores = scores.reshape(n_seq, n_starts).max(axis=1)
    val_seq_scores = val_scores.reshape(len(telemetry["X_val"]), -1).max(axis=1)
    seq_threshold = float(np.quantile(val_seq_scores, 0.95))
    flagged = seq_scores > seq_threshold

    kinds = np.asarray(telemetry["kinds"]).astype(str)
    seq_anomaly = kinds != "normal"
    seq_auroc = float(roc_auc_score(seq_anomaly, seq_scores))
    false_alarm_rate = float(flagged[~seq_anomaly].mean())
    detection_rate_by_kind = {
        k: float(flagged[kinds == k].mean()) for k in sorted(set(kinds) - {"normal"})
    }

    window_kinds = np.repeat(kinds, n_starts)
    win_detection_by_kind = {
        k: float(pred[(window_kinds == k) & (y_win == 1)].mean())
        for k in sorted(set(kinds) - {"normal"})
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # dropout 等异常的分数可达数千，用对数刻度展示完整分布
    s_norm = np.log10(1.0 + np.clip(scores[y_win == 0], 0, None))
    s_anom = np.log10(1.0 + np.clip(scores[y_win == 1], 0, None))
    axes[0].hist(s_norm, bins=60, alpha=0.7, label="normal", density=True)
    axes[0].hist(s_anom, bins=60, alpha=0.7, label="anomaly", density=True)
    axes[0].axvline(
        np.log10(1.0 + threshold), color="r", linestyle="--", label="threshold (q99.5)"
    )
    axes[0].set_xlabel("log10(1 + window score), score clipped at 0")
    axes[0].set_ylabel("density")
    axes[0].set_title("Anomaly score distribution (log scale)")
    axes[0].legend()

    fpr, tpr, _ = roc_curve(y_win, scores)
    axes[1].plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[1].set_xlabel("false alarm rate")
    axes[1].set_ylabel("detection rate")
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right")
    axes[1].set_title("Window-level ROC")
    _save_figure(fig, art_dir / "fig_ae_anomaly.png")

    metrics = {
        "window_level": {
            "threshold": threshold,
            "auroc": auroc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_windows": int(len(y_win)),
            "anomaly_ratio": float(np.mean(y_win)),
        },
        "sequence_level": {
            "threshold": seq_threshold,
            "auroc": seq_auroc,
            "false_alarm_rate": false_alarm_rate,
            "detection_rate_by_kind": detection_rate_by_kind,
        },
        "window_detection_rate_by_kind": win_detection_by_kind,
    }
    (art_dir / "metrics_ae.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[eval-ae] window AUROC={auroc:.4f} P={precision:.3f} R={recall:.3f} F1={f1:.3f}\n"
        f"  sequence AUROC={seq_auroc:.4f} false_alarm={false_alarm_rate:.3f} "
        f"detection={ {k: round(v, 3) for k, v in detection_rate_by_kind.items()} }"
    )
    return metrics
