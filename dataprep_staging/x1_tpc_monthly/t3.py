"""Frozen model code of the DRAFT test ``X1.hybrid_tf_tpc.prospective_2026-11.v1`` (T3).

T3 is the registered hybrid ARX layer (``i1.hybrid_tf``, imported read-only, never edited)
with ONE change: its two electricity drivers (3 km disc and in-grid county) are read from
the X1 product ``electricity_x1_tpc/full/v1`` (census where the census records the month,
TPC-downscaled otherwise) and extended past that product's last month with the FROZEN X1
model (``parameters.json`` of v1, ``extend.apply``) on later archived TPC d007012 vintages.
Every other driver, rule, hyperparameter, well and baseline is the registered I1 one.

The swap is done at run time: :func:`patched` replaces ``i1.hybrid_tf.prospective.
driver_panel`` by :func:`driver_panel_x1` inside a ``with`` block and restores it after. The
hybrid package's files are not modified (their sha256 stay those of the I1 registration).

Electricity month t counts as recorded (T3 driver rule) if
* the X1 v1 county row has flag 0 or 5, or
* t is after the v1 product's last month and an archived d007012 vintage (a dated directory
  with a valid SOURCE.json, fetched on or before 2028-01-31) carries own-county 農林漁牧 for
  t, t-1 and t-2: the frozen X1 model is applied (missing secondary features take their
  month-of-year training mean, ``extend``); discs are county x frozen month-of-year share.
Any other month takes the hybrid layer's frozen climatology rule. Values are GWh x 1e6 (the
hybrid layer's kWh), so recorded census months are numerically the I1 driver.

Nothing in this module reads a head value: :func:`fit` and :func:`predict` delegate to
``i1.hybrid_tf.prospective`` (which reads heads through the guard) and are run only by the
registrar (fit, once, heads to 2026-09) and by the scorer (after the window).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import extend as X  # noqa: E402
import panel  # noqa: E402
from common.guard import open_product  # noqa: E402
from common.paths import REPO  # noqa: E402
from common.source import validate_source  # noqa: E402

sys.path.insert(0, str(REPO))
from i1.hybrid_tf import prospective as P  # noqa: E402

X1_DIR = REPO / "results/dataprep/electricity_x1_tpc/full/v1"
X1_USE = "statistical_layer_history"
TPC_ROOT = panel.L0 / "tpc_d007012"
NAT_ROOT = panel.L0 / "moeaea_agri_electricity"
DRIVER_DEADLINE = "2028-01-31"
ET0_GLOBS = ("results/dataprep/model_ready/release_*/full/et0/v*",
             "results/dataprep/et0/full/v*", "results/dataprep/et0/v*")


def tpc_vintages(deadline: str = DRIVER_DEADLINE) -> list[Path]:
    """Archived d007012 files (dated directories <= deadline with a valid SOURCE.json)."""
    out = []
    for d in sorted(TPC_ROOT.glob("20??-??-??")):
        if d.name > deadline or not (d / "SOURCE.json").exists():
            continue
        rec = json.loads((d / "SOURCE.json").read_text(encoding="utf-8"))
        if validate_source(rec, d):
            continue
        out += [d / f["file"] for f in rec["files"] if f["file"].endswith(".csv")]
    return out


def nat_vintages(deadline: str = DRIVER_DEADLINE) -> list[Path]:
    """Energy Administration sets 256 then 236, newest valid dated directory last."""
    out = []
    for d in sorted(NAT_ROOT.glob("20??-??-??")):
        if d.name > deadline or not (d / "SOURCE.json").exists():
            continue
        if validate_source(json.loads((d / "SOURCE.json").read_text(encoding="utf-8")), d):
            continue
        out += [d / "set_256.csv", d / "set_236.csv"]
    return [p for p in out if p.exists()]


def weather_ext() -> pd.DataFrame:
    """Fan-mean monthly rain and ET0 (as ``panel.weather``), ET0 from every product found
    under ``ET0_GLOBS`` that grants statistical_layer_history (newest value per month)."""
    parts = []
    for g in ET0_GLOBS:
        for d in sorted(REPO.glob(g)):
            try:
                tp = open_product(d, X1_USE)
            except Exception:  # noqa: BLE001 - a product without the use is skipped
                continue
            for o in (o for o in tp.outputs if o.endswith(".npz")):
                with np.load(tp.path(o)) as z:
                    if "et0" not in z.files:
                        continue
                    et = pd.DataFrame(np.asarray(z["et0"], float).T,
                                      index=pd.to_datetime(z["dates"]))
                m = et.mean(axis=1).resample("MS").mean()
                n = et.notna().all(axis=1).resample("MS").sum()
                m[n.to_numpy() < np.asarray(n.index.days_in_month)] = np.nan
                parts.append(m)
    et0 = pd.concat(parts, axis=1).ffill(axis=1).iloc[:, -1] if parts else pd.Series(
        dtype=float)
    rf = pd.read_csv(panel.RAIN, parse_dates=["date time"]).set_index("date time")
    rf_m = rf.mean(axis=1).resample("MS").mean()
    cnt = rf.groupby(rf.index.to_period("M")).size()
    ok = cnt.reindex(rf_m.index.to_period("M")).to_numpy() >= np.asarray(
        rf_m.index.days_in_month)
    rf_m[~ok] = np.nan
    w = pd.DataFrame({"et0": et0, "rain": rf_m})
    w.index = pd.DatetimeIndex(w.index, name="date")
    return w


def x1_electricity(sids, dd: pd.DatetimeIndex, county_of: dict) -> tuple[dict, dict, dict]:
    """Electricity drivers per well (kWh) and recorded masks from X1 v1 + frozen extension."""
    pr = open_product(X1_DIR, X1_USE)
    cty = pr.read_table("county_monthly.parquet")
    dsc = pr.read_table("discs_monthly.parquet")
    params = json.loads(pr.path("parameters.json").read_text(encoding="utf-8"))
    ym = [d.strftime("%Y-%m") for d in dd]
    last = max(cty["date"])
    ext = [d for d in dd if d.strftime("%Y-%m") > last]
    ext_c, ext_n = {}, {}
    if ext:
        tpc = panel.tpc_wide(tpc_vintages())
        nat = panel.national_ag(nat_vintages())
        wx = weather_ext()
        idx = pd.DatetimeIndex(ext)
        for c in ("changhua", "yunlin"):
            a = X.apply(params["county"][c], tpc, nat, wx, idx)
            ext_c[c] = a["value"].to_numpy()
            ext_n[c] = a["n_missing_features"].to_numpy()
    T = len(dd)
    vals = {k: np.full((len(sids), T), np.nan) for k in ("local", "county")}
    rec = {k: np.zeros((len(sids), T), bool) for k in ("local", "county")}
    cpiv = {c: g.set_index("date") for c, g in cty.groupby("county")}
    dpiv = dsc.pivot(index="date", columns="sid", values="value")
    dflag = dsc.pivot(index="date", columns="sid", values="flag")
    moy = np.asarray(dd.month - 1)
    is_ext = np.array([m > last for m in ym])
    for i, s in enumerate(sids):
        c = county_of[s]
        g = cpiv[c].reindex(ym)
        okc = g["flag"].isin([0, 5]).to_numpy()
        vc = np.where(okc, g["value"].to_numpy(float), np.nan)
        dv = dpiv[s].reindex(ym).to_numpy(float)
        okd = dflag[s].reindex(ym).isin([0, 5]).to_numpy()
        vd = np.where(okd, dv, np.nan)
        if ext:
            sh = np.asarray(params["disc_share"][s])[moy[is_ext]]
            vc[is_ext] = ext_c[c]
            vd[is_ext] = ext_c[c] * sh
            okc[is_ext] = okd[is_ext] = np.isfinite(ext_c[c])
        vals["county"][i], rec["county"][i] = vc * 1e6, okc & np.isfinite(vc)
        vals["local"][i], rec["local"][i] = vd * 1e6, okd & np.isfinite(vd)
    info = {"x1_product": str(X1_DIR.relative_to(REPO)), "x1_last_month": last,
            "extension_months": [d.strftime("%Y-%m") for d in ext],
            "extension_missing_features": {c: v.tolist() for c, v in ext_n.items()}}
    return vals, rec, info


_INFO: dict = {}


def driver_panel_x1(src: dict, sids, dd: pd.DatetimeIndex):
    """``prospective.driver_panel`` with the two electricity drivers taken from X1."""
    vals, recm, attrs = _ORIG_DRIVER_PANEL(src, sids, dd)
    county_of = dict(zip(list(sids), attrs["county"], strict=True))
    ev, er, info = x1_electricity(list(sids), dd, county_of)
    for k, kk in (("electricity_local_3km", "local"), ("electricity_county", "county")):
        recm[k] = er[kk]
        vals[k] = np.where(er[kk], ev[kk], np.nan)
    _INFO.clear()
    _INFO.update(info)
    return vals, recm, attrs


_ORIG_DRIVER_PANEL = P.driver_panel


@contextlib.contextmanager
def patched():
    P.driver_panel = driver_panel_x1
    try:
        yield
    finally:
        P.driver_panel = _ORIG_DRIVER_PANEL


def code_hashes() -> dict:
    names = ("t3.py", "extend.py", "model.py", "panel.py")
    return {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in names}


def fit(fit_end: str = "2026-09", out_dir: Path | None = None) -> dict:
    """Registrar only: fit T3's frozen parameters (reads heads 2012-02..fit_end)."""
    with patched():
        params = P.fit("full", fit_end, None)
    params["schema"] = "x1-t3-frozen/1"
    params["x1"] = dict(_INFO)
    params["x1_code_sha256"] = code_hashes()
    params["x1_product_manifest_sha256"] = hashlib.sha256(
        (X1_DIR / "MANIFEST.json").read_bytes()).hexdigest()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=False)
        data = json.dumps(params, indent=1, ensure_ascii=False, default=float).encode()
        (out_dir / "parameters.json").write_bytes(data)
        params["parameters_sha256"] = hashlib.sha256(data).hexdigest()
    return params


def predict(params: dict, end: str, y_prev: np.ndarray | None, sids=None) -> dict:
    """Scorer only: frozen predictions with X1 electricity (see ``prospective.predict``)."""
    with patched():
        out = P.predict(params, "full", end, y_prev, sids)
    out["x1"] = dict(_INFO)
    return out
