"""Train a model and benchmark it against the baselines, in one command.

This is the entry point you run on the GPU box. It wires:
    load_dataset -> model.fit -> model.simulate -> benchmark_table (vs gray-box).

Examples
--------
    # synthetic sample, GRU reference model (no real data needed)
    python -m hydrophysics.train --model gru --epochs 30

    # real data, physics-informed UDE, on the GPU
    export HYDROMIND_GW_DATA=/path/to/data
    python -m hydrophysics.train --model ude \\
        --baseline /path/to/gw_fit_results.csv --out results/ude --epochs 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .baselines import (
    climatology_prediction,
    last_value_prediction,
    load_graybox_baseline,
)
from .config import Config, default_config
from .data import load_dataset
from .eval import benchmark_table, evaluate_predictions


def pick_device(requested: str | None = None) -> str:
    """cuda > mps > cpu, unless overridden."""
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def build_model(name: str, device: str, epochs: int, rain_memory: tuple = (),
                rollout: str = "loop", amp: bool = False, capture: str = "none"):
    if name == "gru":
        from .models.gru import GlobalGRU
        return GlobalGRU(device=device, epochs=epochs)
    if name == "ude":
        from .models.ude import PhysicsUDE
        return PhysicsUDE(device=device, epochs=epochs, rain_memory=rain_memory,
                          rollout=rollout, amp=amp, capture=capture)
    if name == "ude_nemo":
        from .models.ude_physicsnemo import PhysicsNeMoUDE
        return PhysicsNeMoUDE(device=device, epochs=epochs, rain_memory=rain_memory,
                              rollout=rollout, amp=amp, capture=capture)
    if name == "pinn":
        from .models.pinn_field import SpatialPINN
        return SpatialPINN(device=device, epochs=epochs)
    raise ValueError(f"unknown model '{name}' (choose: gru, ude, ude_nemo, pinn)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train + benchmark a groundwater model")
    ap.add_argument("--model", default="gru", choices=["gru", "ude", "ude_nemo", "pinn"])
    ap.add_argument("--data", default=None, help="data dir (else HYDROMIND_GW_DATA / sample)")
    ap.add_argument("--baseline", default=None, help="gray-box gw_fit_results.csv")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (auto if unset)")
    ap.add_argument("--rain-memory", default="", metavar="D1,D2,...",
                    help="(ude/ude_nemo) comma-separated recharge-memory decays, e.g. "
                         "0.99,0.95,0.85 -- lifts simulation KGE 0.591 -> 0.704. Empty = off.")
    ap.add_argument("--et", action="store_true",
                    help="use net recharge (rain - ET0) instead of rain, with FAO-56 ET0 "
                         "from Open-Meteo (cached). With --rain-memory, lifts 0.704 -> 0.754.")
    ap.add_argument("--rollout", default="loop", choices=["loop", "scan", "adjoint"],
                    help="(ude/ude_nemo) integrator backend: loop = reference sequential "
                         "semi-implicit step, scan = same recurrence chunk-parallel (GPU-"
                         "efficient), adjoint = torchdiffeq.odeint_adjoint (constant-memory).")
    ap.add_argument("--amp", action="store_true",
                    help="(ude/ude_nemo) bf16 autocast for the training step (CUDA only)")
    ap.add_argument("--capture", default="none", choices=["none", "cudagraph", "compile"],
                    help="(ude/ude_nemo) eliminate per-step launch overhead: cudagraph = "
                         "PhysicsNeMo StaticCaptureTraining (needs --model ude_nemo), "
                         "compile = torch.compile. CUDA only.")
    ap.add_argument("--et-coef", type=float, default=1.0, help="ET0 subtraction coefficient")
    ap.add_argument("--et-cache", default="results/et/openmeteo_et0_2012_2022.npz",
                    help="ET0 .npz cache (fetched from Open-Meteo if absent)")
    ap.add_argument("--out", default=None, help="output dir for predictions/tables")
    args = ap.parse_args(argv)
    rain_memory = tuple(float(x) for x in args.rain_memory.split(",") if x.strip())

    cfg = (
        Config(
            data_dir=Path(args.data) if args.data else default_config().data_dir,
            baseline_results=Path(args.baseline) if args.baseline else None,
        )
        if (args.data or args.baseline)
        else default_config()
    )

    data = load_dataset(cfg)
    print(data.summary())

    if args.et:
        from .et import net_recharge
        data = net_recharge(data, coef=args.et_coef, cache_path=args.et_cache)
        print(f"ET driver: rainfall -> net recharge (rain - {args.et_coef}*ET0)")

    device = pick_device(args.device)
    print(f"device: {device} | model: {args.model} | epochs: {args.epochs}")

    model = build_model(args.model, device, args.epochs, rain_memory=rain_memory,
                        rollout=args.rollout, amp=args.amp, capture=args.capture)
    model.fit(data)
    pred = model.simulate(data)

    sim_preds = {
        model.name: pred,
        "climatology": climatology_prediction(data),
        "last_value": last_value_prediction(data),
    }
    graybox = None
    try:
        graybox = load_graybox_baseline(cfg)
    except FileNotFoundError:
        print("(no gray-box baseline configured)")

    table = benchmark_table(data, sim_preds, graybox=graybox, period="val")
    pd.set_option("display.width", 120)
    print("\n=== SIMULATION-mode benchmark (validation) ===")
    print(table.round(3).to_string())

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(out / "benchmark.csv")
        evaluate_predictions(data, pred, period="val").round(4).to_csv(
            out / f"per_well_{model.name}.csv"
        )
        print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
