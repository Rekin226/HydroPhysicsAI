"""E02z's statistic (pure) and small shared helpers.

The level rule is E02c's, imported read-only from the hash-bound
dataprep_staging/e02c/score_e02c.py (never copied, never edited): rows with finite CL and CLT
at blocked wells with >= 6 such months (``with_baselines``), equal-well RMSE (``_rmse3``) and
the paired 5 km block bootstrap of MSE(ZR) - MSE(baseline) (``paired_bootstrap``, B 2000,
central 90 %; its month-resampled interval is the declared report-only secondary). The only
change is the unit of decision: per county (Yunlin, Changhua), each with its own wells and
blocks, >= 20 wells, on E02z's registered seed. The scorer and the decidability engine call
the same ``county_level``, so a synthetic world is scored by exactly the registered code path.
Review S5 (lead): a county also needs >= 10 distinct 5 km blocks (only tightens the rule).

Nothing here reads a file.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"),
           str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02z_const as K  # noqa: E402
import score_e02c as S  # noqa: E402  (E02c's frozen scorer, read-only)

if (S.MIN_MONTHS, S.BOOT_B, S.BOOT_CI, S.BLOCK_KM, S.MIN_K_COUNTY) != (
        K.MIN_MONTHS, K.BOOT_B, K.BOOT_CI, K.BLOCK_KM, K.MIN_WELLS_COUNTY):
    raise ImportError("E02z's copy of E02c's thresholds differs from score_e02c.py")

LEVEL_VERDICTS = ("level_supported", "level_not_supported", "level_not_evaluable",
                  "engineering_failure_after_look")


class Refused(RuntimeError):
    """A script refuses to run (registration, seed, date, order or provenance rule)."""


class EngineeringFailure(Refused):
    """An input or run is inconsistent: nothing is written (fix, log, rerun)."""


# ----------------------------------------------------------------------------- helpers
def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rp(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_utc(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_once(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, sort_keys=True, default=jsonable)
        f.write("\n")


def log_attempt(path: Path, rec: dict) -> None:
    """Append one attempt record (never rewritten). No statistic goes into this log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": K.ID, "time": now_utc().isoformat(timespec="seconds"),
                             **rec}, sort_keys=True, ensure_ascii=False, default=jsonable)
                 + "\n")


def attempt_no(path: Path) -> int:
    path = Path(path)
    if not path.is_file():
        return 1
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return 1 + sum(r.get("event") == "start" for r in rows)


def check_seed(seed) -> int:
    """A registration's seed: a non-negative int chosen by the lead (no default anywhere),
    never one previewed, smoked or registered elsewhere."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise Refused(f"seed must be a non-negative int chosen at registration, got {seed!r}")
    if int(seed) in K.REFUSED_SEEDS:
        raise Refused(f"seed {seed} refused: {K.REFUSED_SEEDS[int(seed)]}")
    return int(seed)


def deadline_problem(created: str | None) -> str | None:
    """Review S6: a registration created at or after REG_DEADLINE is refused."""
    if not created:
        return "the registration has no 'created' time"
    if parse_utc(created) >= parse_utc(K.REG_DEADLINE):
        return f"the registration was created {created}, not before {K.REG_DEADLINE}"
    return None


def window_months(win: str):
    return S.window_months(*K.WINDOWS[win])


# ----------------------------------------------------------------------------- statistic
def county_level(f, county: str, blocks: dict, seed: int) -> dict:
    """E02c's strict level rule on one county's rows ``f`` (already through
    ``S.with_baselines``: columns station_id, date, county, predicted_m, observed_m, cl_m,
    clt_m). Supported iff RMSE(ZR) < RMSE(CL) AND RMSE(ZR) < RMSE(CLT) AND both block-bootstrap
    90 % intervals of the MSE difference lie below 0."""
    g = f[f["county"] == county]
    n_wells = int(g["station_id"].nunique())
    out = {"county": county, "n_wells": n_wells, "n_rows": int(len(g)),
           "min_wells": K.MIN_WELLS_COUNTY, "min_months": K.MIN_MONTHS}
    n_blocks = len({blocks[w] for w in g["station_id"].unique()})
    out.update({"n_blocks": n_blocks, "min_blocks": K.MIN_BLOCKS_COUNTY})
    if n_wells < K.MIN_WELLS_COUNTY:
        out.update({"verdict": "level_not_evaluable",
                    "why": f"fewer than {K.MIN_WELLS_COUNTY} {county} wells with >= "
                           f"{K.MIN_MONTHS} window months and finite CL/CLT ({n_wells})"})
        return out
    if n_blocks < K.MIN_BLOCKS_COUNTY:
        out.update({"verdict": "level_not_evaluable",
                    "why": f"fewer than {K.MIN_BLOCKS_COUNTY} distinct 5 km blocks among "
                           f"{county}'s scored wells ({n_blocks})"})
        return out
    rmse = S._rmse3(g)
    vs_cl = S.paired_bootstrap(g, "predicted_m", "cl_m", blocks, seed=seed)
    vs_clt = S.paired_bootstrap(g, "predicted_m", "clt_m", blocks, seed=seed)
    beat_cl = bool(rmse["ZR"] < rmse["CL"] and vs_cl["a_better_block"])
    beat_clt = bool(rmse["ZR"] < rmse["CLT"] and vs_clt["a_better_block"])
    out.update({"rmse_equal_well_m": rmse, "vs_CL": vs_cl, "vs_CLT": vs_clt,
                "beat_cl": beat_cl, "beat_clt": beat_clt,
                "verdict": "level_supported" if beat_cl and beat_clt
                else "level_not_supported"})
    return out


def window_level(frame, baselines, blocks: dict, seed: int, win: str) -> dict:
    """Both counties on one window's annotated rows (``S.annotate`` output): the rows are
    restricted to the window months, joined to the frozen CL/CLT (``S.with_baselines``), and
    each county is decided on its own."""
    months = window_months(win)
    fr = frame[frame["date"].isin(months)].reset_index(drop=True)
    f = S.with_baselines(fr, baselines, blocks)
    return {"window": win, "months": [str(m.date())[:7] for m in (months[0], months[-1])],
            "n_rows_window": int(len(fr)), "n_rows_with_baselines": int(len(f)),
            "counties": {c: county_level(f, c, blocks, seed) for c in K.COUNTIES}}


def wording(verdict: str, county: str, win: str, *, why: str = "",
            decide: dict | None = None) -> str:
    """The page/ledger line for one county and window, with the decidability suffix."""
    first, last = K.WINDOWS[win]
    base = K.WORDING[verdict].format(county=county, first=first, last=last, why=why)
    d = decide or {"reading": "not_run"}
    r = d.get("reading") or "not_run"
    suf = K.DECIDE_SUFFIX.get(r, K.DECIDE_SUFFIX["not_run"])
    return base + suf.format(p=float(d.get("p", float("nan"))), why=d.get("why", ""))
