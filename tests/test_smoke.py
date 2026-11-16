"""冒烟测试：仿真形状、模型前向、标签构造。"""

import numpy as np
import torch

from signalai.dataset import window_label, windows_from_batch
from signalai.models import ModulationCNN, TelemetryAE
from signalai.simulate import (
    MODULATIONS,
    N_SAMPLES,
    generate_modulated,
    generate_telemetry_normal,
    inject_anomaly,
)


def test_modulation_shapes_and_power():
    rng = np.random.default_rng(0)
    for mod in MODULATIONS:
        x = generate_modulated(mod, 10.0, rng)
        assert x.shape == (2, N_SAMPLES)
        assert x.dtype == np.float32
        assert np.isfinite(x).all()
        power = float(np.mean(x[0] ** 2 + x[1] ** 2))
        assert abs(power - 1.0) < 0.05, (mod, power)


def test_modulation_cnn_forward():
    model = ModulationCNN(n_classes=len(MODULATIONS))
    out = model(torch.zeros(4, 2, N_SAMPLES))
    assert out.shape == (4, len(MODULATIONS))


def test_telemetry_ae_roundtrip():
    model = TelemetryAE(channels=6, window=64)
    x = torch.randn(4, 64, 6)
    out = model(x)
    assert out.shape == x.shape


def test_anomaly_injection_and_window_labels():
    rng = np.random.default_rng(1)
    seq = generate_telemetry_normal(1024, rng)
    bad, mask, kind = inject_anomaly(seq, rng)
    assert bad.shape == seq.shape
    assert mask.any()
    assert kind in {"spike", "drift", "stuck", "dropout", "step"}
    assert window_label(mask, 0, 64) in (0, 1)
    x, y = windows_from_batch(bad[None], mask[None])
    assert x.shape[1:] == (64, 6)
    assert y.ndim == 1
