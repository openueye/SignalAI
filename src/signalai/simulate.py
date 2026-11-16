"""信号与遥测数据仿真。

通信侧：8 类调制信号的基带 IQ 仿真，含随机载波频偏/相偏、定时偏移、
相位噪声与 AWGN，覆盖 SNR -10~20 dB。
测控侧：6 通道遥测时序（姿态角、母线电压、温度、下行电平），
并注入 spike / drift / stuck / dropout / step 五类异常。

全部使用 numpy 随机数生成器，固定种子可完全复现。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import upfirdn

SPS = 8                 # 每符号采样点数
N_SAMPLES = 1024        # 单样本 IQ 长度
N_SYMBOLS = N_SAMPLES // SPS

MODULATIONS = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "GMSK", "AM-DSB", "FM")

TELEMETRY_CHANNELS = (
    "roll",
    "pitch",
    "yaw",
    "bus_voltage",
    "temperature",
    "downlink_rssi",
)

ANOMALY_KINDS = ("spike", "drift", "stuck", "dropout", "step")


# ---------------------------------------------------------------------------
# 通信信号仿真
# ---------------------------------------------------------------------------

def _rrc_filter(sps: int = SPS, beta: float = 0.35, span: int = 6) -> np.ndarray:
    """根升余弦成形滤波器（单位能量）。"""
    n = span * sps
    t = np.arange(-(n // 2), n // 2 + 1) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-9:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(
                np.pi * ti * (1 + beta)
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


_RRC = _rrc_filter()


def _pulse_shape(symbols: np.ndarray) -> np.ndarray:
    """符号上采样并经过 RRC 成形，返还与符号数等长的 N_SAMPLES。"""
    up = np.zeros(len(symbols) * SPS, dtype=complex)
    up[::SPS] = symbols
    out = upfirdn(_RRC, up)
    delay = (len(_RRC) - 1) // 2
    return out[delay:delay + len(up)]


def _psk_symbols(order: int, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, order, n)
    return np.exp(2j * np.pi * idx / order)


def _qam_symbols(order: int, n: int, rng: np.random.Generator) -> np.ndarray:
    side = int(np.sqrt(order))
    levels = np.arange(-(side - 1), side, 2)
    norm = np.sqrt(np.mean(levels ** 2) * 2.0)
    return (rng.choice(levels, n) + 1j * rng.choice(levels, n)) / norm


def _gauss_kernel(sps: int = SPS, bt: float = 0.3, span: int = 4) -> np.ndarray:
    n = span * sps
    t = np.arange(-(n // 2), n // 2 + 1) / sps
    alpha = np.sqrt(np.log(2.0)) / (2.0 * np.pi * bt)
    h = np.exp(-(t ** 2) / (2.0 * alpha ** 2))
    return h / h.sum()


def _gmsk(rng: np.random.Generator) -> np.ndarray:
    bits = rng.integers(0, 2, N_SYMBOLS)
    nrz = np.repeat(2.0 * bits - 1.0, SPS)
    freq = np.convolve(nrz, _gauss_kernel(), mode="same")
    phase = 2 * np.pi * 0.5 * np.cumsum(freq) / SPS
    return np.exp(1j * phase)


def _tone_message(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(N_SAMPLES, dtype=float)
    msg = np.zeros(N_SAMPLES)
    for _ in range(3):
        f = rng.uniform(0.003, 0.02)
        ph = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.0)
        msg += amp * np.sin(2 * np.pi * f * t + ph)
    return msg / (np.max(np.abs(msg)) + 1e-12)


def _am_dsb(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(N_SAMPLES, dtype=float)
    msg = _tone_message(rng)
    fc = rng.uniform(0.05, 0.15)
    return (1.0 + 0.7 * msg) * np.exp(2j * np.pi * fc * t)


def _fm(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(N_SAMPLES, dtype=float)
    msg = _tone_message(rng)
    fc = rng.uniform(0.05, 0.15)
    return np.exp(1j * (2 * np.pi * fc * t + 2 * np.pi * 0.05 * np.cumsum(msg)))


def _impair(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """随机频偏、相偏、定时偏移、相位噪声与幅度起伏。"""
    n = len(x)
    frac = rng.uniform(0.0, 1.0)
    grid = np.arange(n)
    xr = np.interp(grid + frac, grid, x.real)
    xi = np.interp(grid + frac, grid, x.imag)
    x = xr + 1j * xi

    f_off = rng.uniform(-0.004, 0.004)
    phi = rng.uniform(0, 2 * np.pi)
    phase_noise = np.cumsum(rng.normal(0.0, 5e-4, n))
    x = x * np.exp(1j * (2 * np.pi * f_off * grid + phi + phase_noise))
    return x * rng.uniform(0.5, 1.5)


def _awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = np.mean(np.abs(x) ** 2)
    noise_power = p / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))
    )
    return x + noise


def generate_modulated(mod: str, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """生成一个 (2, N_SAMPLES) 的 float32 IQ 样本。"""
    if mod == "BPSK":
        x = _pulse_shape(_psk_symbols(2, N_SYMBOLS, rng))
    elif mod == "QPSK":
        x = _pulse_shape(_psk_symbols(4, N_SYMBOLS, rng))
    elif mod == "8PSK":
        x = _pulse_shape(_psk_symbols(8, N_SYMBOLS, rng))
    elif mod == "16QAM":
        x = _pulse_shape(_qam_symbols(16, N_SYMBOLS, rng))
    elif mod == "64QAM":
        x = _pulse_shape(_qam_symbols(64, N_SYMBOLS, rng))
    elif mod == "GMSK":
        x = _gmsk(rng)
    elif mod == "AM-DSB":
        x = _am_dsb(rng)
    elif mod == "FM":
        x = _fm(rng)
    else:
        raise ValueError(f"unknown modulation: {mod}")

    x = _impair(x, rng)
    x = _awgn(x, snr_db, rng)
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    return np.stack([x.real, x.imag]).astype(np.float32)


def build_iq_train(
    n_per_class: int,
    snr_range: tuple[float, float] = (-5.0, 20.0),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """训练集：每类随机 SNR，返回 (X, y, snr)。"""
    rng = np.random.default_rng(seed)
    total = n_per_class * len(MODULATIONS)
    X = np.empty((total, 2, N_SAMPLES), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    snr = np.empty(total, dtype=np.float32)
    i = 0
    for label, mod in enumerate(MODULATIONS):
        for _ in range(n_per_class):
            s = rng.uniform(*snr_range)
            X[i] = generate_modulated(mod, s, rng)
            y[i] = label
            snr[i] = s
            i += 1
    return X, y, snr


EVAL_SNRS = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def build_iq_eval(
    n_per_class_per_snr: int = 200,
    snrs: tuple[float, ...] = EVAL_SNRS,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """评测集：每类每 SNR 固定条数。"""
    rng = np.random.default_rng(seed)
    total = n_per_class_per_snr * len(MODULATIONS) * len(snrs)
    X = np.empty((total, 2, N_SAMPLES), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    snr = np.empty(total, dtype=np.float32)
    i = 0
    for label, mod in enumerate(MODULATIONS):
        for s in snrs:
            for _ in range(n_per_class_per_snr):
                X[i] = generate_modulated(mod, s, rng)
                y[i] = label
                snr[i] = s
                i += 1
    return X, y, snr


# ---------------------------------------------------------------------------
# 遥测时序仿真
# ---------------------------------------------------------------------------

_TELEMETRY_SCALES = np.array([2.0, 1.5, 3.0, 0.2, 0.5, 1.0])


def generate_telemetry_normal(length: int, rng: np.random.Generator) -> np.ndarray:
    """生成一段正常遥测序列，形状 (length, 6)。"""
    t = np.arange(length, dtype=float)
    ch = np.empty((length, len(TELEMETRY_CHANNELS)))
    ch[:, 0] = 8.0 * np.sin(2 * np.pi * t / 600 + rng.uniform(0, 2 * np.pi)) + 0.5 * np.sin(
        2 * np.pi * t / 97
    )
    ch[:, 1] = 5.0 * np.sin(2 * np.pi * t / 420 + rng.uniform(0, 2 * np.pi))
    ch[:, 2] = 12.0 * np.sin(2 * np.pi * t / 300 + rng.uniform(0, 2 * np.pi))
    ch[:, 3] = 28.0 + 0.15 * np.sin(2 * np.pi * t / 250 + rng.uniform(0, 2 * np.pi))
    ch[:, 4] = 35.0 + 1.5 * np.sin(2 * np.pi * t / 800 + rng.uniform(0, 2 * np.pi))
    ch[:, 5] = -85.0 + 2.0 * np.sin(2 * np.pi * t / 500 + rng.uniform(0, 2 * np.pi))
    ch += rng.normal(0.0, 1.0, ch.shape) * _TELEMETRY_SCALES
    return ch


def inject_anomaly(
    seq: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, str]:
    """注入一段单通道异常，返回 (异常序列, 异常掩码, 异常类型)。"""
    seq = seq.copy()
    n = seq.shape[0]
    kind = str(rng.choice(ANOMALY_KINDS))
    ch = int(rng.integers(0, seq.shape[1]))
    start = int(rng.integers(200, max(201, n - 400)))
    dur = int(rng.integers(60, 200))
    end = min(n, start + dur)
    mask = np.zeros(n, dtype=bool)
    std = float(np.std(seq[:, ch])) + 1e-6

    if kind == "spike":
        count = max(3, dur // 10)
        idx = rng.choice(np.arange(start, end), size=count, replace=False)
        seq[idx, ch] += rng.choice([-1.0, 1.0], count) * rng.uniform(6, 12) * std
        mask[idx] = True
    elif kind == "drift":
        seq[start:end, ch] += np.linspace(0.0, rng.uniform(4, 8) * std, end - start)
        mask[start:end] = True
    elif kind == "stuck":
        seq[start:end, ch] = seq[start, ch]
        mask[start:end] = True
    elif kind == "dropout":
        seq[start:end, ch] = 0.0
        mask[start:end] = True
    else:  # step
        seq[start:end, ch] += rng.choice([-1.0, 1.0]) * rng.uniform(3, 6) * std
        mask[start:end] = True
    return seq, mask, kind


def build_telemetry_train(
    n_sequences: int = 300,
    length: int = 4096,
    seed: int = 2,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([generate_telemetry_normal(length, rng) for _ in range(n_sequences)])


def build_telemetry_val(
    n_sequences: int = 60,
    length: int = 4096,
    seed: int = 3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([generate_telemetry_normal(length, rng) for _ in range(n_sequences)])


def build_telemetry_test(
    n_normal: int = 120,
    n_anomaly: int = 180,
    length: int = 4096,
    seed: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """返回 (序列, 掩码, 异常类型)；正常序列掩码全 0。"""
    rng = np.random.default_rng(seed)
    seqs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    kinds: list[str] = []

    for _ in range(n_normal):
        seqs.append(generate_telemetry_normal(length, rng))
        masks.append(np.zeros(length, dtype=bool))
        kinds.append("normal")
    for _ in range(n_anomaly):
        seq, mask, kind = inject_anomaly(generate_telemetry_normal(length, rng), rng)
        seqs.append(seq)
        masks.append(mask)
        kinds.append(kind)
    return np.stack(seqs), np.stack(masks), kinds
