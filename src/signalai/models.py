"""模型定义：IQ 调制识别 CNN 与遥测自编码器。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModulationCNN(nn.Module):
    """原始 IQ 序列端到端分类器（RadioML 风格的浅层 CNN）。"""

    def __init__(self, n_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


class TelemetryAE(nn.Module):
    """多通道遥测窗口自编码器，重建误差作为异常分数。"""

    def __init__(self, channels: int = 6, window: int = 64, latent: int = 24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (window // 8), latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64 * (window // 8)),
            nn.ReLU(),
            nn.Unflatten(1, (64, window // 8)),
            nn.ConvTranspose1d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, channels, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, window, channels) -> Conv1d 需要 (B, channels, window)
        x = x.transpose(1, 2)
        return self.decoder(self.encoder(x)).transpose(1, 2)
