"""FE.encompass_2026-12.v1 (round-8 rank 9): constants and pure functions.

Does the frozen physics forecast E02 (results/twin/prospective_2026-11, forward cf2c7154) add
information beyond the frozen statistical layer E04 (results/twin/prospective_2026-11_hybrid)
on the wells they share, over 2026-12..2027-10?

Two-contrast encompassing model (review B1, lead's decision). For every scored well-month

    y   = obs - STAT
    x_P = PHYS - STAT          (STAT = E04 predicted_m, PHYS = E02 predicted_m)
    x_C = CLIM - STAT          (CLIM = E04 baseline_m, the layer's own climatology)

    y = lambda_P x_P + lambda_C x_C + e        (within wells: each well's window means removed)

x_C = -(the layer's persisted anomaly), so lambda_C absorbs an over- or under-persisting
layer; lambda_P is the information the physics forecast adds beyond BOTH the climatology and
the persisted anomaly. Readings use the encompassing coefficient lambda_P only; lambda_C, the
one-contrast lambda (x_P alone) and the raw-level (not demeaned) fit are report-only.

Interval: a well x quarter (pigeonhole) block bootstrap. Each replicate draws the wells with
replacement AND the four window quarters (2026-12..02, 03..05, 06..08, 09..10) with
replacement; a row's weight is (copies of its well) x (copies of its quarter); the within-well
means are recomputed in every replicate; both coefficients come from the same replicate. The
90 % interval is the 5th..95th percentile of the replicates.

Readings (review S1, lead's decision): 'physics_informative' if L_P > 0; 'stat_encompasses'
if U_P < DELTA = 0.25 (a fixed, declared margin); otherwise undecided.

Nothing here opens a file.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- identities
ID = "FE.encompass_2026-12.v1"
WP = "FE"
TEST = "FEP.lambda_wd_2026-12"
UNIT = "FE:heads_2026-12_2027-10_E02xE04"
DATA_SCOPE = "heads:2026-12..2027-10"
F = "dataprep_staging/fe_encompass"
LIB = f"{F}/fe_lib.py"
DECIDE = f"{F}/decide_fe.py"
SCORER = f"{F}/score_fe.py"
PREREG = f"{F}/prereg_fe.py"
TESTS = "tests/test_fe_encompass.py"
REPORT_DIR = f"results/dataprep/reports/{ID}"
REPORT = f"{REPORT_DIR}/report.json"
DECIDABILITY = f"{REPORT_DIR}/decidability.json"
ATTEMPTS_NAME = "attempts.jsonl"

E02_DIR = "results/twin/prospective_2026-11"
E04_DIR = "results/twin/prospective_2026-11_hybrid"
E02_PRED_SHA = "16ac3c3783cea9f3c94560448d06c64ec8dc8fbb450effa28b465469a931cc32"
E02_PROTOCOL_SHA = "8a96c0ea9330fd79432eb17a666640ad526728c838b62bd8459cd41de0d908b9"
E02_MODEL_SHA = "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"
E04_PRED_SHA = "982f4bfa91280539bd47147285dadaee322940c8f140f9fbba3405783ad3abed"
E04_PROTOCOL_SHA = "18a87f270c47853c7f07bf700df2b3b8bf484394a3c548b6ee45f03593d42264"
E04_MANIFEST_SHA = "8e28ac21a7d590da93875bb02a639399ab9b196661e779aa9dff3538b777b512"
E02C_ID = "E02c.prospective_2026-11.v1"
E02D_ID = "E02d.decomposed_prospective_2026-12.v1"
#: the canonical E02c report FE binds (E02c's documented --out; a lead-written not-scorable
#: record for E02c goes to this path too)
E02C_REPORT = "results/dataprep/reports/E02c/score/report.json"
FV_STAT_V2_ID = "FV.stat.v2"
FV_STAT_V2_DECL = f"results/dataprep/preregistration/{FV_STAT_V2_ID}.json"
FV_STAT_V2_DECL_SHA = "a15e468d1292bb9d170fa981d96ee52bfadc80051c867496dbdb94ff2c8de7e6"
FV_STAT_V2_REPORT = f"results/dataprep/reports/{FV_STAT_V2_ID}/report.json"
FV_STAT_V2_REPORT_SHA = "6145a225e4171d39253e6d34c2b48871b42ca751e3f317acae5568b681a25963"
FV_STAT_V2_ERRATUM = f"results/dataprep/reports/{FV_STAT_V2_ID}/erratum_county_layout.json"
FV_STAT_V2_ERRATUM_SHA = "ce2c01012a3d8e211d4a7376e544c6801e3deaff04858df5044f0fee85ced578"
# FV.decidability.v2's forward truth worlds (physics arm, review B2) and their pinned inputs
FV2_ID = "FV.decidability.v2_postfailure"
FV2_DECL = f"results/dataprep/preregistration/{FV2_ID}.json"
FV2_DECL_SHA = "f7e35a17482cce307c6be84013f9a2919f16f4be66e9a01cd7eeacbd08fb49d4"
FV2_REPORT = f"results/dataprep/reports/{FV2_ID}/report_worlds.json"
FV2_REPORT_SHA = "320ac928059ca78d343b5c3a1b8b32054a0c1ce15d5633b1e5060dfa7fbf9dae"
FV2_ATTEMPTS = "results/twin_runs/fv2/attempts.jsonl"
FV1_DECL = "results/dataprep/preregistration/FV.decidability.v1.json"
FV1_DECL_SHA = "cf2adf0d0c9b9fe5f70f5e6ee6cd1bc133964811c2b944eaf2cce344b4d3f61d"
FV1_INPUTS = "results/dataprep/reports/FV.decidability.v1/inputs"
NOISE_CSV = f"{FV1_INPUTS}/noise_ar1.csv"            # pre-origin scale (S4), window_end 2022-12
NOISE_SHA = "73965a08914149af2132c96f617268701277b49411cf17cfbf57d41fbeaa9d55"
MASK_CSV = f"{FV1_INPUTS}/obs_mask.csv"              # E02's datum months (availability only)
MASK_SHA = "605a35de5d07cab40cef3a1e86fae719502ff3afcd0f6ea36310ccf5425d0c89"
WELLS_CSV = "results/dataprep/reports/E02c/baselines/wells_158.csv"   # cell / layer per well
WELLS_SHA = "8ea086330f8a9ecf5ff76313c9dc3e617e8e500775d70d7cc180f077340ac7d3"
ZR0_NPZ = "results/twin_forward/fv_zr0.npz"          # FV's same-code page-recipe reference
ZR0_SHA = "a1833d5f861817223ebafb5a651444239c8877d21a4d4752e91dbbbd356172cf"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
RULINGS_SHA = "ec610b692835d955141e9261e6d6046ee1984f39891bcf1c2e77591aad1c69eb"
RULING_ID = "R8.7"
N_SHARED = 147                    # wells in both E02 (158) and E04 (147): checked in code

# ----------------------------------------------------------------------------- calendar
FIRST, LAST = "2026-12", "2027-10"
MONTHS = pd.date_range(f"{FIRST}-01", f"{LAST}-01", freq="MS")          # 11 months
QUARTERS = ("2026-12..2027-02", "2027-03..2027-05", "2027-06..2027-08", "2027-09..2027-10")
QIDX = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
REGISTER_BY = "2026-11-30T16:00:00+00:00"
#: review B2 hard rule: the gating physics arm must be built, reviewed and merged by
#: 2026-11-25; a registration created after this instant makes FE report-only
GATING_ARM_BY = "2026-11-25T23:59:59+00:00"
NOT_BEFORE = "2028-01-15T00:00:00+00:00"
NO_LATER_THAN = "2028-01-31"
NO_LATER_THAN_END = "2028-02-01T00:00:00+00:00"   # end of the no-later-than day (UTC)

# ----------------------------------------------------------------------------- the rule
MIN_MONTHS = 6                    # a well is scored with >= 6 window months (E02's rule)
MIN_WELLS = 100                   # fewer scored wells -> not_evaluable (E02's min_stations)
BOOT_B = 2000
CI = (0.05, 0.95)                 # 90 % percentile interval
MAX_BAD_REPLICATES = 0.01         # share of degenerate replicates above this -> not_evaluable
X_RMS_MIN_M = 0.05                # partial physics contrast RMS below this -> not_informative
DELTA = 0.25                      # fixed encompassing margin: U_P < DELTA -> stat_encompasses
COND_MIN = 1e-8                   # det(A) / prod(diag A) below this -> degenerate replicate
Z90 = 1.6448536269514722
COUNTY = {"07": "Changhua", "08": "Nantou", "09": "Yunlin"}
NAMES = ("P", "C")                # lambda_P (physics contrast), lambda_C (climatology contrast)

if len(MONTHS) != len(QIDX):
    raise ImportError("FE's quarter map does not cover its window")


class EngineeringFailure(RuntimeError):
    """An input or run is inconsistent: no verdict, nothing written (fix, log, rerun)."""


class Refused(RuntimeError):
    """The runner refuses (registration, date, order or write-once rules); nothing read."""


def county_of(sid) -> str:
    return COUNTY.get(str(sid)[:2], "other")


def check_seed(seed, refused: dict | None = None) -> int:
    """A seed is a non-negative int chosen by the lead; there is never a default."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise EngineeringFailure(f"seed must be a non-negative int, got {seed!r}")
    if refused and int(seed) in refused:
        raise EngineeringFailure(f"seed {seed} refused: {refused[int(seed)]}")
    return int(seed)


# ----------------------------------------------------------------------------- panel
def panel(frame: pd.DataFrame, min_months: int = MIN_MONTHS) -> dict:
    """Rows (station_id, date, observed_m, stat_m, phys_m, clim_m) -> the scored panel: wells x
    the 11 window months with x_P = PHYS - STAT, x_C = CLIM - STAT, y = obs - STAT and the
    mask m; only months 2026-12..2027-10 and wells with >= ``min_months`` months are kept."""
    f = frame.copy()
    f["station_id"] = f["station_id"].astype(str)
    f["_m"] = pd.to_datetime(f["date"]).dt.to_period("M").dt.to_timestamp()
    f = f[f["_m"].isin(MONTHS)]
    for c in ("observed_m", "stat_m", "phys_m", "clim_m"):
        f = f[np.isfinite(f[c].to_numpy(dtype=float))]
    if f.duplicated(["station_id", "_m"]).any():
        raise EngineeringFailure("duplicate station-months in the FE frame")
    n = f.groupby("station_id")["_m"].nunique()
    ids = sorted(n[n >= int(min_months)].index)
    f = f[f["station_id"].isin(ids)]
    W, T = len(ids), len(MONTHS)
    xp, xc, y = np.zeros((W, T)), np.zeros((W, T)), np.zeros((W, T))
    m = np.zeros((W, T), bool)
    r = f["station_id"].map({s: i for i, s in enumerate(ids)}).to_numpy()
    c = f["_m"].map({t: j for j, t in enumerate(MONTHS)}).to_numpy()
    st = f["stat_m"].to_numpy(float)
    xp[r, c] = f["phys_m"].to_numpy(float) - st
    xc[r, c] = f["clim_m"].to_numpy(float) - st
    y[r, c] = f["observed_m"].to_numpy(float) - st
    m[r, c] = True
    return {"ids": ids, "xp": xp, "xc": xc, "y": y, "m": m, "rows": int(m.sum())}


def min_months_mask(m: np.ndarray, min_months: int = MIN_MONTHS) -> np.ndarray:
    """Drop (mask out) wells with fewer than ``min_months`` observed months (FE's rule)."""
    keep = m.sum(axis=1) >= int(min_months)
    return m & keep[:, None]


# ----------------------------------------------------------------------------- estimator
def coefs(cq: np.ndarray, cw: np.ndarray, xs, y: np.ndarray, m: np.ndarray, *,
          demean: bool = True) -> np.ndarray:
    """(B, k) least-squares coefficients of y on the k regressors ``xs`` per replicate.
    ``cq``: (B, T) month weights (copies of each month's quarter); ``cw``: (B, W) copies of
    each well. ``demean`` removes each well's (weighted) window means; otherwise no intercept.
    NaN rows mark degenerate replicates (no variation, or collinear regressors)."""
    mm = m.astype(float)
    Z = [np.where(m, v, 0.0) for v in (*xs, y)]
    k = len(xs)
    s1 = cq @ mm.T
    sums = [cq @ z.T for z in Z]
    safe = np.where(s1 > 0, s1, 1.0)

    def cross(i, j):
        sij = cq @ (Z[i] * Z[j]).T
        if not demean:
            return sij
        return np.where(s1 > 0, sij - sums[i] * sums[j] / safe, 0.0)

    B = cq.shape[0]
    A = np.empty((B, k, k))
    r = np.empty((B, k))
    for i in range(k):
        r[:, i] = (cw * cross(i, k)).sum(axis=1)
        for j in range(i, k):
            A[:, i, j] = A[:, j, i] = (cw * cross(i, j)).sum(axis=1)
    diag = np.einsum("bii->bi", A)
    ok = (diag > 1e-12).all(axis=1)
    if k > 1:
        det = np.linalg.det(A)
        with np.errstate(invalid="ignore", divide="ignore"):
            ok &= det / np.prod(np.where(diag > 0, diag, 1.0), axis=1) > COND_MIN
    out = np.full((B, k), np.nan)
    if ok.any():
        out[ok] = np.linalg.solve(A[ok], r[ok][..., None])[..., 0]
    return out


def lambdas(cq, cw, x, y, m, *, demean: bool = True) -> np.ndarray:
    """One-regressor coefficient per replicate (report-only one-contrast form)."""
    return coefs(cq, cw, [x], y, m, demean=demean)[:, 0]


def boot_weights(n_wells: int, B: int, rng: np.random.Generator) -> tuple[np.ndarray,
                                                                          np.ndarray]:
    """The well x quarter (pigeonhole) draws: (cq (B, 11) month weights, cw (B, W))."""
    cw = rng.multinomial(n_wells, np.full(n_wells, 1.0 / n_wells), size=B).astype(float)
    q = rng.multinomial(len(QUARTERS), np.full(len(QUARTERS), 1.0 / len(QUARTERS)),
                        size=B).astype(float)
    return q[:, QIDX], cw


def estimate(xs, y: np.ndarray, m: np.ndarray, *, names=NAMES, rng: np.random.Generator,
             B: int = BOOT_B, demean: bool = True, weights=None) -> dict:
    """Point coefficients and their 90 % well x quarter block-bootstrap intervals, one dict per
    regressor name ({lambda, lower, upper}), plus the degenerate-replicate share. ``weights``
    reuses pre-drawn (cq, cw) (common draws across arms and forms of one world)."""
    W, T = y.shape
    names = tuple(names)[:len(xs)]
    pt = coefs(np.ones((1, T)), np.ones((1, W)), xs, y, m, demean=demean)[0]
    cq, cw = weights if weights is not None else boot_weights(W, B, rng)
    cb = coefs(cq, cw, xs, y, m, demean=demean)
    ok = np.isfinite(cb).all(axis=1)
    out = {"B": int(len(cb)), "bad_share": 1.0 - float(ok.mean()) if len(cb) else 1.0}
    for i, n in enumerate(names):
        if ok.any() and np.isfinite(pt[i]):
            lo, hi = np.quantile(cb[ok, i], CI)
            out[n] = {"lambda": float(pt[i]), "lower": float(lo), "upper": float(hi)}
        else:
            out[n] = {"lambda": float(pt[i]), "lower": None, "upper": None}
    return out


def partial_rms(xp: np.ndarray, xc: np.ndarray, m: np.ndarray) -> float:
    """RMS of the within-well demeaned x_P after removing its pooled projection on the
    within-well demeaned x_C (model output only): the part of the physics contrast lambda_P
    can be identified from."""
    mm = m.astype(float)
    n = mm.sum(axis=1)

    def dm(v):
        mean = np.where(n > 0, np.where(m, v, 0.0).sum(axis=1) / np.where(n > 0, n, 1.0), 0.0)
        return np.where(m, v - mean[:, None], 0.0)

    p, c = dm(xp), dm(xc)
    tot = mm.sum()
    if tot == 0:
        return float("nan")
    cc = (c * c).sum()
    r = p - ((p * c).sum() / cc) * c if cc > 1e-12 else p
    return float(math.sqrt((r * r).sum() / tot))


def contrast_rms(x: np.ndarray, m: np.ndarray) -> float:
    """RMS of the within-well demeaned x over scored rows (report-only)."""
    return partial_rms(x, np.zeros_like(x), m)


# ----------------------------------------------------------------------------- readings
READINGS = ("physics_informative", "stat_encompasses", "undecided", "not_informative",
            "not_evaluable")
READING_RULE = (
    "Applied once to the encompassing coefficient lambda_P of the within-well two-contrast fit "
    "y = lambda_P x_P + lambda_C x_C + e and its 90 % well x quarter block-bootstrap interval "
    "[L_P, U_P]: not_evaluable if < 100 scored wells (>= 6 months each), no window product "
    "granting prospective_scorer for 2026-12..2027-10, or > 1 % degenerate replicates; else "
    "not_informative if the partial physics contrast RMS (x_P within wells, net of x_C) over "
    "the scored rows is < 0.05 m (model output only); else physics_informative if L_P > 0; "
    "else stat_encompasses if U_P < 0.25 (the fixed margin delta); otherwise undecided. If "
    "L_P > 0 and U_P < 0.25 both hold the reading is physics_informative with the clause "
    "'small'. lambda_C, the one-contrast lambda, the raw-level fit and the per-county fits "
    "are report-only and set no reading.")


def reading(est_p: dict | None, *, bad_share: float = 0.0, delta: float = DELTA,
            informative: bool, evaluable: bool) -> dict:
    """The declared reading on lambda_P (see :data:`READING_RULE`)."""
    if not evaluable or est_p is None or est_p.get("lower") is None:
        return {"reading": "not_evaluable", "small": False}
    if bad_share > MAX_BAD_REPLICATES:
        return {"reading": "not_evaluable", "small": False}
    if not informative:
        return {"reading": "not_informative", "small": False}
    lo, hi = est_p["lower"], est_p["upper"]
    below = hi < delta
    if lo > 0:
        return {"reading": "physics_informative", "small": bool(below)}
    if below:
        return {"reading": "stat_encompasses", "small": False}
    return {"reading": "undecided", "small": False}


# Page text (under ruling R8.7 only). {lam}, {lo}, {hi}, {d}, {n} are filled at scoring.
WORDING = {
    "before_scoring": {
        "en": "Whether the frozen physics forecast adds anything to the statistical layer's "
              "first-year forecast (E04, the forecast of record) is decided by a registered "
              "encompassing test (FE) on the 147 wells both share, 2026-12..2027-10, scored "
              "not before 2028-01-15.",
        "zh": "凍結的物理預報是否能為統計層的第一年預報（E04，即記錄中的預報）增加資訊，由一項登錄"
              "的涵蓋檢驗（FE）判定：以兩者共有的 147 口井、2026-12 至 2027-10 為範圍，最早於 "
              "2028-01-15 計分。"},
    "physics_informative": {
        "en": "Registered encompassing test (FE, prospective): over 2026-12..2027-10 the "
              "frozen physics forecast added information to the statistical layer, beyond its "
              "climatology and persisted anomaly, on the within-well head anomalies at {n} "
              "wells (encompassing coefficient lambda {lam:.2f}, 90 % interval {lo:.2f} to "
              "{hi:.2f}, above 0). E04 stays the forecast of record; this number defines no "
              "blend of the two forecasts.",
        "zh": "登錄涵蓋檢驗（FE，前瞻）：2026-12 至 2027-10 期間，凍結的物理預報在 {n} 口井的井內"
              "水位異常上，於統計層的氣候值與延續異常之外增加了資訊（涵蓋係數 λ {lam:.2f}，90 % "
              "區間 {lo:.2f} 至 {hi:.2f}，高於 0）。E04 仍為記錄中的預報；此數值不定義兩個預報的"
              "任何混合。"},
    "physics_informative_small": {
        "en": " The whole interval lies below the declared margin {d:.2f}, so the addition is "
              "small.",
        "zh": " 整個區間低於事先宣告的界限 {d:.2f}，故增加的資訊很小。"},
    "stat_encompasses": {
        "en": "Registered encompassing test (FE, prospective): over 2026-12..2027-10 the "
              "frozen physics forecast added no detectable information to the statistical "
              "layer on the within-well head anomalies at {n} wells (encompassing coefficient "
              "lambda {lam:.2f}, 90 % interval up to {hi:.2f}, below the declared margin "
              "{d:.2f}). The statistical layer encompasses the physics forecast for "
              "first-year anomalies; in this model the physics head levels are an "
              "average-weather scenario, not a forecast.",
        "zh": "登錄涵蓋檢驗（FE，前瞻）：2026-12 至 2027-10 期間，凍結的物理預報在 {n} 口井的井內"
              "水位異常上未能為統計層增加可偵測的資訊（涵蓋係數 λ {lam:.2f}，90 % 區間上限 "
              "{hi:.2f}，低於事先宣告的界限 {d:.2f}）。就第一年異常而言，統計層涵蓋了物理預報；"
              "在本模型中，物理模式的水位為平均天氣情境，而非預報。"},
    "undecided": {
        "en": "Registered encompassing test (FE, prospective): over 2026-12..2027-10 the "
              "test could not decide whether the frozen physics forecast adds to the "
              "statistical layer (encompassing coefficient lambda {lam:.2f}, 90 % interval "
              "{lo:.2f} to {hi:.2f}). E04 stays the forecast of record.",
        "zh": "登錄涵蓋檢驗（FE，前瞻）：2026-12 至 2027-10 期間，此檢驗無法判定凍結的物理預報"
              "是否能為統計層增加資訊（涵蓋係數 λ {lam:.2f}，90 % 區間 {lo:.2f} 至 {hi:.2f}）。"
              "E04 仍為記錄中的預報。"},
    "not_informative": {
        "en": "Registered encompassing test (FE): not informative (beyond the statistical "
              "layer's own terms the two frozen forecasts barely differ within wells over the "
              "scored months); E04 stays the forecast of record.",
        "zh": "登錄涵蓋檢驗（FE）：不具資訊性（扣除統計層本身的項目後，兩個凍結預報在計分月份的"
              "井內幾乎沒有差異）；E04 仍為記錄中的預報。"},
    "not_evaluable": {
        "en": "Registered encompassing test (FE): not evaluable ({why}); E04 stays the "
              "forecast of record.",
        "zh": "登錄涵蓋檢驗（FE）：無法評估（{why}）；E04 仍為記錄中的預報。"},
    "report_only_prefix": {
        "en": "Report-only (decidability {why}): ",
        "zh": "僅供報告（可判定性：{why}）："},
    "never": {"en": ["validated", "calibrated", "combined forecast", "weight",
                     "no-information reference", "the physics model is right",
                     "the physics model is wrong"],
              "zh": ["校準", "驗證", "權重", "組合預報", "無資訊參考值"]},
}


def wording(rd: dict, est_p: dict | None, *, n_wells: int, role: str, why: str = "",
            delta: float = DELTA) -> dict:
    """Filled page text for a reading; report-only readings carry the declared prefix."""
    key = rd["reading"]
    e = est_p or {}
    vals = {"lam": e.get("lambda") or 0.0, "lo": e.get("lower") or 0.0,
            "hi": e.get("upper") or 0.0, "d": delta, "n": n_wells}
    out = {}
    for lang in ("en", "zh"):
        if key == "not_evaluable":
            t = WORDING[key][lang].format(why=why or "see report")
        else:
            t = WORDING[key][lang].format(**vals)
        if key == "physics_informative" and rd.get("small"):
            t += WORDING["physics_informative_small"][lang].format(**vals)
        if role != "gating" and key != "not_evaluable":
            t = WORDING["report_only_prefix"][lang].format(why=why or "not measured") + t
        out[lang] = t
    return out
