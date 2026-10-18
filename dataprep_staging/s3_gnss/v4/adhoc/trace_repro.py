"""WP S3 fix round of the S3r repair (2026-10-18): reproduce into evidence files the ad-hoc
traces quoted in ``results/dataprep/reports/S3.repair.md`` 7.5 and 7.6, with ledger rows.

    dataprep_staging/tools/cpu_slot.sh python -I -B \
        dataprep_staging/s3_gnss/v4/adhoc/trace_repro.py OUT_DIR [--ledger]

* HLES (and KTES) around the January 2023 outage: weekly medians of the neighbours' pass-2
  residuals and of the detection common mode as the plain median (v3 rule), the withdrawn R7
  (re-implemented below exactly as it ran, as a trace only) and R7' (``pipeline``), each from
  its own full-variant family run.
* KTES around 2024-04-25 under R7': neighbours reporting and the common mode (plain vs R7').
No injection and no score.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))   # dataprep_staging
sys.path.insert(0, str(HERE.parent))       # v4 modules
import pipeline as P  # noqa: E402
import smoke_v4 as S  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import REPO  # noqa: E402


def imputed_median_r7(X: np.ndarray, half: int, min_n: int, passes: int = 3
                      ) -> tuple[np.ndarray, np.ndarray]:
    """The withdrawn R7 (as it ran in the first smoke check): offsets against the current
    common mode, iterated."""
    X = np.asarray(X, dtype=float)
    present = np.isfinite(X)
    n_imp = np.zeros(len(X), dtype=int)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        c = np.nanmedian(np.where(present, X, np.nan), axis=1)
        for _ in range(passes):
            D = pd.DataFrame(X - c[:, None])
            dl = D.rolling(2 * half + 1, center=True, min_periods=min_n).median()
            dl = dl.interpolate(method="linear", limit_area="inside").ffill().bfill()
            dv = dl.to_numpy()
            lev = np.nanmedian(np.where(present & np.isfinite(dv), X - dv, np.nan), axis=1)
            imp = lev[:, None] + dv
            use_imp = ~present & np.isfinite(imp)
            V = np.where(present, X, np.where(use_imp, imp, np.nan))
            c = np.nanmedian(V, axis=1)
            n_imp = use_imp.sum(axis=1)
    return c, n_imp


def weekly(run: dict, s: str, t0: str, t1: str) -> pd.DataFrame:
    R = run["R2"].sort_index()
    R = R.reindex(pd.date_range(R.index.min(), R.index.max(), freq="D"))
    base = [m for m in run["members"] if m in R]
    nb = P.knn_neighbours(s, base, run["xy"])
    win = R.loc[t0:t1, nb + [s]]
    T = 1000 * win
    T["cm_plain"] = 1000 * R[nb].median(axis=1).reindex(win.index)
    T["cm_run"] = 1000 * run["cm"][s].reindex(win.index)
    T["n_present"] = win[nb].notna().sum(axis=1)
    return T.resample("7D").median().round(1)


def main(argv: list[str]) -> int:
    od = Path(argv[0])
    od.mkdir(parents=True, exist_ok=True)
    p, _ = S.load_tuned("full")
    r7b = S.wra_unrolled(p, None)                       # R7' (current pipeline)
    orig = P.imputed_median
    P.imputed_median = imputed_median_r7                # the withdrawn R7, trace only
    try:
        r7 = S.wra_unrolled(p, None)
    finally:
        P.imputed_median = orig
    out, rows = {}, []
    for tag, run in (("r7_withdrawn", r7), ("r7b", r7b)):
        for s in ("HLES", "KTES"):
            W = weekly(run, s, "2022-11-15", "2023-03-20")
            W.to_csv(od / f"trace_{s}_{tag}_2022-11_2023-03.csv")
            jan = W.loc["2023-01-10":"2023-01-31"]
            out[f"{s}.{tag}.jan2023_weekly_median_mm"] = {
                "cm_plain": float(jan.cm_plain.median()), "cm_run": float(jan.cm_run.median())}
            out[f"{s}.{tag}.dec2022_and_feb2023_mm"] = {
                "cm_run_dec": float(W.loc["2022-12-06":"2022-12-27"].cm_run.median()),
                "cm_run_feb": float(W.loc["2023-02-14":"2023-02-28"].cm_run.median())}
    K = weekly(r7b, "KTES", "2024-02-15", "2024-07-15")
    K.to_csv(od / "trace_KTES_r7b_2024-02_2024-07.csv")
    k = K.loc["2024-04-11":"2024-05-09"]
    out["KTES.r7b.2024-04-11..05-09"] = {
        "n_present_min": int(k.n_present.min()),
        "max_abs_cm_plain_minus_r7b_mm": float((k.cm_plain - k.cm_run).abs().max())}
    (od / "trace_repro.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    rel = lambda q: Path(q).resolve().relative_to(REPO).as_posix()   # noqa: E731
    for nid in out:
        rows.append({"number_id": f"S3.fix.trace.{nid}", "source": rel(od / "trace_repro.json"),
                     "role": "input", "product_version": "gnss_clean/v4 smoke traces (no "
                     "product; S3r fix round)", "note": "adhoc/trace_repro.py"})
    if "--ledger" in argv:
        p_ = ledger.append_rows("S3", rows)
        print(f"ledger: {len(rows)} rows -> {p_}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
