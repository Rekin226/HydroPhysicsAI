"""GPU throughput benchmark for the forecaster: CPU vs CUDA vs CUDA+bf16-AMP.

Trains GlobalForecastLSTM for a fixed number of epochs on a fixed synthetic dataset
under each backend and reports wall-clock, training throughput (windows x epochs / s),
peak GPU memory, and speedups. Synthetic data keeps the benchmark reproducible anywhere
and isolates compute (no real data needed).

    python -m hydrophysics.bench                       # default 40 wells, 8 epochs
    python -m hydrophysics.bench --wells 60 --epochs 12 --out results/bench

Writes ``gpu_benchmark.csv`` + ``gpu_benchmark.png`` and prints a markdown table.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import Config
from .data import load_dataset
from .sample import write_sample


def _time_fit(data, device, amp, epochs, hidden, batch, seed=0):
    """Wall-clock + peak GPU memory for one fit() under a backend."""
    import torch

    from .models.forecast_lstm import GlobalForecastLSTM

    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model = GlobalForecastLSTM(hidden=hidden, epochs=epochs, batch=batch,
                               amp=amp, device=device, seed=seed)
    # Build windows once (CPU) so the timed region is dominated by the train loop;
    # the build cost is identical across backends.
    t0 = time.perf_counter()
    model.fit(data)
    if is_cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if is_cuda else float("nan")
    return dt, peak_mb, model


def _n_train_windows(data, lookback=90, horizon=30):
    import numpy as np
    finite = np.isfinite(data.target)
    tm = data.train_mask
    W, T = data.target.shape
    count = 0
    for i in range(W):
        for t0 in range(lookback - 1, T - horizon):
            if not tm[t0]:
                continue
            if finite[i, t0 - lookback + 1:t0 + horizon + 1].all():
                count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Forecaster GPU throughput benchmark")
    ap.add_argument("--wells", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default="results/bench")
    args = ap.parse_args(argv)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("bench needs torch: pip install -e '.[gpu]'") from exc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tmp = out / "_bench_data"
    d = write_sample(tmp, n_wells=args.wells, seed=0, start="2014-01-01", end="2019-12-31")
    data = load_dataset(Config(data_dir=d))
    n_win = _n_train_windows(data)
    print(data.summary())
    print(f"training windows: {n_win} | epochs: {args.epochs} | "
          f"hidden: {args.hidden} | batch: {args.batch}\n")

    backends = [("CPU", "cpu", False)]
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        backends += [(f"CUDA fp32 ({name})", "cuda", False),
                     (f"CUDA bf16-AMP ({name})", "cuda", True)]
    else:
        print("(CUDA not available; CPU only)")

    rows = []
    for label, dev, amp in backends:
        dt, peak, _ = _time_fit(data, dev, amp, args.epochs, args.hidden, args.batch)
        thr = n_win * args.epochs / dt
        rows.append({"backend": label, "device": dev, "amp": amp,
                     "seconds": dt, "windows_per_s": thr, "peak_gpu_mb": peak})
        print(f"{label:34s} {dt:7.2f}s  {thr:12.0f} win/s  "
              f"peak {peak:8.1f} MB" if dev != "cpu" else
              f"{label:34s} {dt:7.2f}s  {thr:12.0f} win/s")

    import pandas as pd
    df = pd.DataFrame(rows)
    base = df.loc[df["device"] == "cpu", "seconds"]
    if not base.empty:
        df["speedup_vs_cpu"] = base.iloc[0] / df["seconds"]
    df.round(2).to_csv(out / "gpu_benchmark.csv", index=False)

    # --- markdown + plot ---
    print("\n=== markdown ===")
    cols = ["backend", "seconds", "windows_per_s", "peak_gpu_mb", "speedup_vs_cpu"]
    have = [c for c in cols if c in df.columns]
    print("| " + " | ".join(have) + " |")
    print("|" + "|".join(["---"] * len(have)) + "|")
    for _, r in df.iterrows():
        cells = []
        for c in have:
            v = r[c]
            cells.append(f"{v:.1f}" if isinstance(v, float) and c != "speedup_vs_cpu"
                         else (f"{v:.2f}x" if c == "speedup_vs_cpu" else str(v)))
        print("| " + " | ".join(cells) + " |")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3.5))
        labels = [r.split(" (")[0] for r in df["backend"]]
        colors = ["0.6", "#5b9bd5", "#76b900"][:len(df)]
        ax.bar(labels, df["windows_per_s"], color=colors, edgecolor="k", linewidth=0.5)
        for x, (v, s) in enumerate(zip(df["windows_per_s"], df.get("speedup_vs_cpu", []),
                                       strict=False)):
            tag = f"{v/1e6:.2f}M/s" + (f"\n{s:.1f}x" if s == s else "")
            ax.text(x, v, tag, ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("training throughput (windows/s)")
        ax.set_title("Forecaster training throughput by backend")
        ax.margins(y=0.18)
        fig.tight_layout()
        fig.savefig(out / "gpu_benchmark.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass

    # tidy the throwaway dataset
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
