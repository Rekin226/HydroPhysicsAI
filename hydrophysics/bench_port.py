"""Pre-port vs post-port benchmark for the PhysicsUDE on the NVIDIA stack.

Trains the same operator on the same data under each configuration and reports
training wall-clock, peak GPU memory, and validation skill (median KGE/NSE/RMSE),
so the "does the NVIDIA stack help" question is answered by one table:

  - ude / loop            : the pre-port reference (plain PyTorch, sequential rollout)
  - ude_nemo / loop       : the PhysicsNeMo port, same integrator (parity check)
  - ude / scan            : GPU-parallel chunked-scan rollout (same recurrence)
  - ude / loop + amp      : bf16 autocast training
  - ude_nemo / scan + amp : the full post-port configuration

The adjoint backend is benchmarked separately at the micro level (see the audit
report): constant-memory backprop works, but at this scale (61 wells, ~4000 days,
~10 MiB autograd graph) adaptive continuous-time integration is orders of magnitude
slower, so it is not part of the training table.

    export HYDROMIND_GW_DATA=/path/to/data
    python -m hydrophysics.bench_port --epochs 1500 --out results/bench_port
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .baselines import climatology_prediction
from .config import default_config
from .data import load_dataset
from .eval import benchmark_table
from .train import build_model, pick_device

CONFIGS = [
    # (label, model, rollout, amp, capture)
    ("pre_port__ude_loop", "ude", "loop", False, "none"),
    ("port__ude_nemo_loop", "ude_nemo", "loop", False, "none"),
    ("post__ude_scan", "ude", "scan", False, "none"),
    ("post__ude_loop_amp", "ude", "loop", True, "none"),
    ("post__ude_nemo_scan_amp", "ude_nemo", "scan", True, "none"),
    # CUDA-graph capture (PhysicsNeMo StaticCaptureTraining) vs the plain-PyTorch
    # control (torch.compile), on both the launch-bound loop and the already
    # launch-thrifty scan: does the framework's capture utility buy anything on top
    # of restructuring the integrator?
    ("graph__ude_nemo_loop_cudagraph", "ude_nemo", "loop", False, "cudagraph"),
    ("graph__ude_nemo_scan_cudagraph", "ude_nemo", "scan", False, "cudagraph"),
    ("graph__ude_scan_compile", "ude", "scan", False, "compile"),
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Pre/post NVIDIA-stack training benchmark")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/bench_port")
    ap.add_argument("--rain-memory", default="", metavar="D1,D2,...",
                    help="optional recharge-memory decays, applied to every config")
    ap.add_argument("--only", default="", metavar="SUBSTR,...",
                    help="run only configs whose label contains one of these substrings "
                         "(e.g. --only graph, to add capture rows to an existing table)")
    ap.add_argument("--csv-name", default="port_benchmark.csv",
                    help="output filename under --out")
    ap.add_argument("--baseline-s", type=float, default=0.0, metavar="SECONDS",
                    help="pre-port wall-clock to compute speedup against when the "
                         "pre_port config is not part of this run (see --only)")
    args = ap.parse_args(argv)
    only = [x for x in args.only.split(",") if x.strip()]
    rain_memory = tuple(float(x) for x in args.rain_memory.split(",") if x.strip())

    import torch

    data = load_dataset(default_config())
    print(data.summary())
    device = pick_device(args.device)
    is_cuda = str(device).startswith("cuda")
    print(f"device: {device} | epochs: {args.epochs}")

    rows = []
    preds = {}
    for label, model_name, rollout, amp, capture in CONFIGS:
        if only and not any(sub in label for sub in only):
            continue
        if (amp or capture != "none") and not is_cuda:
            print(f"skip {label} (needs cuda)")
            continue
        model = build_model(model_name, device, args.epochs,
                           rain_memory=rain_memory, rollout=rollout, amp=amp,
                           capture=capture)
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model.fit(data)
        if is_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if is_cuda else float("nan")

        pred = model.simulate(data)
        table = benchmark_table(data, {model.name: pred,
                                       "climatology": climatology_prediction(data)},
                                period="val")
        skill = table.loc[model.name]
        rows.append({
            "config": label, "model": model_name, "rollout": rollout, "amp": amp,
            "capture": capture,
            "train_s": round(dt, 1), "peak_gpu_mb": round(peak_mb, 1),
            "kge_median": skill["kge_median"], "nse_median": skill["nse_median"],
            "rmse_median": skill["rmse_median"],
        })
        preds[label] = pred
        print(f"{label:26s} {dt:7.1f}s  peak {peak_mb:8.1f} MiB  "
              f"KGE {skill['kge_median']:.4f}  NSE {skill['nse_median']:.4f}  "
              f"RMSE {skill['rmse_median']:.4f}", flush=True)

    df = pd.DataFrame(rows).set_index("config")
    # Speedups are always quoted against the pre-port reference. When --only runs a
    # subset that excludes it, --baseline-s carries its wall-clock over from the full run
    # so the added rows stay comparable with the published table.
    pre = [i for i in df.index if i.startswith("pre_port__")]
    base = df.loc[pre[0], "train_s"] if pre else args.baseline_s
    df["speedup_vs_pre"] = (base / df["train_s"]).round(2) if base else float("nan")
    print("\n=== pre/post NVIDIA-stack benchmark ===")
    print(df.to_string())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / args.csv_name)
    print(f"\nwrote -> {out}/{args.csv_name}")


if __name__ == "__main__":
    main()
