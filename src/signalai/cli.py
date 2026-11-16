"""SignalAI 命令行入口。

用法（在项目根目录）：
    python -m signalai gen-data
    python -m signalai train-mod --epochs 15
    python -m signalai train-ae --epochs 30
    python -m signalai eval
    python -m signalai all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import ensure_iq_data, ensure_telemetry_data
from .evaluate import evaluate_autoencoder, evaluate_modulation
from .train_ae import train_autoencoder
from .train_mod import train_modulation

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "artifacts" / "data"
ART_DIR = ROOT / "artifacts"


def _prepare() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)


def cmd_gen_data(args: argparse.Namespace) -> None:
    _prepare()
    train, ev = ensure_iq_data(
        DATA_DIR,
        n_train_per_class=args.n_train_per_class,
        n_eval_per_class=args.n_eval_per_class,
        seed=args.seed,
    )
    tel = ensure_telemetry_data(DATA_DIR, seed=args.seed + 2)
    print(f"IQ train: {train['X'].shape}, eval: {ev['X'].shape}")
    print(
        f"Telemetry train: {tel['X_train'].shape}, val: {tel['X_val'].shape}, "
        f"test: {tel['X_test'].shape}"
    )


def cmd_train_mod(args: argparse.Namespace) -> None:
    _prepare()
    train, _ = ensure_iq_data(DATA_DIR, seed=args.seed)
    meta = train_modulation(
        train,
        ART_DIR,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    print(f"best val acc: {meta['best_val_acc']:.4f}")


def cmd_train_ae(args: argparse.Namespace) -> None:
    _prepare()
    tel = ensure_telemetry_data(DATA_DIR, seed=args.seed + 2)
    meta = train_autoencoder(
        tel, ART_DIR, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed
    )
    print(f"anomaly threshold: {meta['threshold']:.5f}")


def cmd_eval(args: argparse.Namespace) -> None:
    _prepare()
    _, ev = ensure_iq_data(DATA_DIR, seed=args.seed)
    evaluate_modulation(ART_DIR, ev)
    tel = ensure_telemetry_data(DATA_DIR, seed=args.seed + 2)
    evaluate_autoencoder(ART_DIR, tel)


def cmd_all(args: argparse.Namespace) -> None:
    cmd_gen_data(args)
    cmd_train_mod(args)
    cmd_train_ae(args)
    cmd_eval(args)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train-per-class", type=int, default=1200)
    p.add_argument("--n-eval-per-class", type=int, default=200)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signalai", description="测控通信信号智能分析原型")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, func in (
        ("gen-data", cmd_gen_data),
        ("train-mod", cmd_train_mod),
        ("train-ae", cmd_train_ae),
        ("eval", cmd_eval),
        ("all", cmd_all),
    ):
        p = sub.add_parser(name)
        _add_common(p)
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "train-ae" and args.epochs == 15:
        args.epochs = 30
    args.func(args)


if __name__ == "__main__":
    main()
