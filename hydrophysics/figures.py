"""Generate the README figures end-to-end.

    python -m hydrophysics.figures                 # bundled synthetic sample (default)
    python -m hydrophysics.figures --out results/figures

By default it runs on the *synthetic* sample so the figures are reproducible anywhere
(CI, no GPU, no real data) and publish no agency time series. Point it at real data with
``--data`` / ``HYDROMIND_GW_DATA`` to regenerate the same figures on the real wells.

Figures written:
  - simulation_hydrographs.png : observed vs free-running PhysicsUDE for example wells
  - forecast_fan.png           : probabilistic LSTM 30-day forecast with 90% interval
  - simulation_benchmark.png   : median-KGE bars (UDE vs climatology / last-value)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .baselines import climatology_prediction, last_value_prediction
from .config import Config, default_config
from .data import load_dataset
from .eval import evaluate_predictions
from .viz import plot_benchmark_bars, plot_forecast_fan, plot_simulation_hydrographs


def _pick_wells(data, pred, k=3):
    """Best / median / worst well by validation KGE, for an honest spread."""
    per = evaluate_predictions(data, pred, period="val")["kge"].dropna().sort_values()
    if per.empty:
        return list(data.well_ids[:k])
    order = per.index.tolist()
    picks = {order[-1], order[len(order) // 2], order[0]}
    return [w for w in order[::-1] if w in picks][:k]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate README figures")
    ap.add_argument("--data", default=None, help="data dir (else HYDROMIND_GW_DATA / sample)")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sim-epochs", type=int, default=400)
    ap.add_argument("--fc-epochs", type=int, default=40)
    args = ap.parse_args(argv)

    cfg = (Config(data_dir=Path(args.data)) if args.data else default_config())
    data = load_dataset(cfg)
    print(data.summary())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from .train import pick_device
    device = pick_device(args.device)
    print(f"device: {device}")

    # --- simulation: PhysicsUDE free-running hindcast --------------------------------
    from .models.ude import PhysicsUDE
    ude = PhysicsUDE(device=device, epochs=args.sim_epochs).fit(data)
    sim = ude.simulate(data)
    clim = climatology_prediction(data)

    wells = _pick_wells(data, sim)
    print(f"hydrograph wells (best/median/worst): {wells}")
    plot_simulation_hydrographs(
        data, {"PhysicsUDE": sim, "climatology": clim}, wells,
        out / "simulation_hydrographs.png",
    )

    scores = {
        "PhysicsUDE": evaluate_predictions(data, sim, "val")["kge"].median(),
        "climatology": evaluate_predictions(data, clim, "val")["kge"].median(),
        "last-value": evaluate_predictions(
            data, last_value_prediction(data), "val")["nse"].median(),
    }
    plot_benchmark_bars(
        {"PhysicsUDE": scores["PhysicsUDE"], "climatology": scores["climatology"]},
        out / "simulation_benchmark.png",
        title="Simulation benchmark (median KGE, validation)",
    )

    # --- probabilistic forecast fan -------------------------------------------------
    try:
        from .models.forecast_lstm import GlobalForecastLSTM
        fc = GlobalForecastLSTM(horizon=30, probabilistic=True,
                                device=device, epochs=args.fc_epochs).fit(data)
        mean_cube, sigma_cube = fc.forecast_dist(data)
        # choose a well + origin with a realized future and finite forecast
        v0 = int(np.where(data.val_mask)[0][0])
        t0 = min(v0 + 5, data.n_days - 31)
        wid = max(data.well_ids,
                  key=lambda w: np.isfinite(mean_cube[data.well_index(w), t0]).sum())
        plot_forecast_fan(data, mean_cube, sigma_cube, wid, t0, out / "forecast_fan.png")
        print(f"forecast fan: well {wid} from origin index {t0}")
    except Exception as e:  # forecasting is optional for the figure set
        print(f"(skipped forecast fan: {e})")

    print(f"\nwrote figures -> {out}")
    for p in sorted(out.glob("*.png")):
        print(" -", p)


if __name__ == "__main__":
    main()
