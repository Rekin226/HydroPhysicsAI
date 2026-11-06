"""G0, the synthetic decidability studies of I1.backcast_2010_11.v1 (model only; synthetic
worlds only; no observation of any family is read). The lead runs them after the backcast
panel and the baselines and before the final declaration draft, which records each result
file by sha256 and maps every gate whose G0 fails to report-only.

Every study writes ``<g0 dir>/g0_<name>.json`` once (mode 'x') and prints only its gate
(power, false pass, decidable). Seeds are the declared ``bc11_const.G0_SEEDS``.

``g1``   G1 + G1b: FV.stat.v2's AR(1) generator (stat_fv.anomalies) on the qualifying wells'
         county layout; 2012-22 fully observed (optimistic), 2010-11 observed in the
         availability look's available months (an upper bound on QC'd months); E04's
         recipe refitted per world exactly as scored. Power = mean over the 9 AR cells
         (rho 0.80/0.86/0.93 x share 0/0.15/0.30) of P(G1 and G1b); false pass = max over the
         3 rho = 0 cells. Report-only: per-well 2010-11 level offsets (sd 0.5, 1.0).
``g2``   G2 per county: 2011 truth = the page model's F1 run + AR noise (power) or the
         model's own 2012-22 climatology + AR noise (null: no 2011 information); 2012-22 =
         the page forward + the same noise; noise sd per well = the 2012-22 physics residual
         sd (baselines; spent heads). Power = mean over the 9 cells, false = max.
``g3``   G3 (pooled): truth = F1 + noise (power) or F0 + noise (null).
``yr``   the Yunlin arm (R8.6c): chx_lib.world on the panel's Yunlin contrast (N2C's
         declared noise and confounder, heterogeneous loading, no trend), the availability
         mask; power = P(pass | k 1, rho -0.83), false = max over rho in {0, -0.83} of
         P(pass | k 0).

    python dataprep_staging/bc11/g0_bc11.py g1 --out <g0 dir>
    python dataprep_staging/bc11/g0_bc11.py g2|g3|yr --panel <npz> --baselines <npz> --out <dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "ch_x"), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
#: CPU entry point: this checkout's hydrophysics (the hash-bound copy), not an editable install
if str(HERE.parents[1]) not in sys.path[:1]:
    sys.path.insert(0, str(HERE.parents[1]))

import bc11_const as K  # noqa: E402
import bc11_lib as BL  # noqa: E402
import chx_io as IO  # noqa: E402

NAMES = ("g1", "g2", "g3", "yr")


def avail() -> dict:
    p = IO.rp(K.AVAIL_OUT)
    a = json.loads(p.read_text())
    if not a["feasibility"]["passed"]:
        raise SystemExit("REFUSED: the availability stop rule did not pass")
    return {"a": a, "sha256": IO.sha256(p)}


def window_mask(sids, months_by_well: dict, months=BL.WIN_M) -> np.ndarray:
    """(W, len(months)) True where the availability look found the month available."""
    return np.array([[m in set(months_by_well.get(str(s), [])) for m in months]
                     for s in sids], bool)


def layout(a: dict) -> tuple[list[str], np.ndarray]:
    f = a["feasibility"]
    ids = sorted(map(str, f["eligible_wells"]))
    return ids, np.array([f["per_well"][s]["county"] for s in ids])


def _write_once(out: Path, name: str, rep: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"g0_{name}.json"
    with p.open("x") as f:
        json.dump(rep, f, indent=1, default=float)
    return p


def _stamp(name: str, inputs: dict) -> dict:
    return {"what": f"I1.backcast_2010_11.v1 G0 {name} (synthetic worlds; model only)",
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "seed": K.G0_SEEDS[name], "inputs": inputs,
            "code_sha256": {p: IO.sha_or_missing(HERE.parents[1] / p) for p in (
                f"{K.X}/bc11_lib.py", f"{K.X}/bc11_const.py", f"{K.X}/g0_bc11.py")}}


def _panel(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _sigma(P: dict, base: dict) -> np.ndarray:
    """Per placed well, the 2012-22 physics residual sd (NaN -> the median of the others)."""
    pos = {str(s): i for i, s in enumerate(base["sid"])}
    s = np.array([base["phys_resid_sd"][pos[str(w)]] for w in P["sid"]], float)
    if not np.isfinite(s).any():
        raise SystemExit("no finite physics residual sd")
    return np.where(np.isfinite(s), s, float(np.nanmedian(s)))


def study(name: str, out: Path, panel: Path | None = None, baselines: Path | None = None,
          reps: int | None = None) -> dict:
    av = avail()
    a = av["a"]
    if name == "g1":
        ids, county = layout(a)
        mask = window_mask(ids, a["months_by_well"])
        r = BL.run_g0_g1(county, mask, reps=reps or K.G0_K_G1)
        rep = {**_stamp(name, {"availability": [K.AVAIL_OUT, av["sha256"]]}), **r}
        p = _write_once(out, name, rep)
        g = r["gate"]
        print(f"G0 g1: power {g['power']:.3f}, false pass {g['false_pass']:.3f}, decidable "
              f"{g['decidable']} -> {p}")
        return rep
    if panel is None:
        raise SystemExit(f"{name} needs --panel")
    P = _panel(panel)
    inputs = {"availability": [K.AVAIL_OUT, av["sha256"]], "panel": [str(panel),
                                                                     IO.sha256(panel)]}
    s_mask = window_mask(P["sid"], a["months_by_well"], BL.SCORED_M)
    if name == "yr":
        sys.path.insert(0, str(HERE.parent / "n2c_reading"))
        import synth_n2c as SN

        par = {k: float(SN.PARAMS[k]["value"]) for k in ("sigma_tot_m", "block_share", "phi",
                                                          "kappa_m")}
        mask = window_mask(P["sid_yl"], a["months_by_well"])
        r = BL.run_g0_yr(P, mask, par, reps=reps or K.G0_K_YR)
        rep = {**_stamp(name, inputs), "params": par, "params_from": K.G0_PARAMS_FROM, **r}
    else:
        if baselines is None:
            raise SystemExit(f"{name} needs --baselines")
        from baselines_bc11 import load as load_base

        base = load_base(baselines)
        inputs["baselines"] = [str(baselines), IO.sha256(baselines)]
        sig = _sigma(P, base)
        county = np.asarray(P["county"]).astype(str)
        mf = np.asarray(P["native_fit"], float)
        F1 = np.asarray(P["F1"], float)[:, BL.N_PRE:]
        if name == "g2":
            r = BL.run_g0_g2(mf, F1, sig, county, s_mask, reps=reps or K.G0_K_G2)
        else:
            F0 = np.asarray(P["F0"], float)[:, BL.N_PRE:]
            r = BL.run_g0_g3(mf, F1, F0, sig, county, s_mask, reps=reps or K.G0_K_G3)
        rep = {**_stamp(name, inputs), **r}
    p = _write_once(out, name, rep)
    print(f"G0 {name}: {json.dumps(rep['gate'], default=float)} -> {p}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", choices=NAMES)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--panel", type=Path)
    ap.add_argument("--baselines", type=Path)
    a = ap.parse_args(argv)
    study(a.name, a.out, a.panel, a.baselines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
