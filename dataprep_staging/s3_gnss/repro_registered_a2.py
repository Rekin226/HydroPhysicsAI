"""WP S3 resume check: the gate's A2 under the registered ``validate.py``.

The pre-registration ``S3.gnss_clean_acceptance.v1`` hashed ``validate.py`` at sha256
461cd6a1...; the gate run (``validation_preorigin.json``) used that file. ``validate.py`` was
changed once afterwards (``monthly()`` sorts its month index), so both gnss_clean manifests
record the later file (5a06c7ac...). No copy of the registered file was archived.

This script
1. recovers the registered file from the current one by undoing that one change, and refuses
   to go on unless the result has the registered sha256 (it is then archived as evidence);
2. logs a ``reproduce`` look under the same pre-registration id (A2 units, A2 data scope);
3. runs ``test_2021_extra`` with the registered and with the current file on the PUBLISHED
   pre-origin tables (``gnss_clean/preorigin/v1``), and compares both with the gate run's A2.

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/repro_registered_a2.py

Writes ``S3_evidence/acceptance/registered_code/validate.py`` and
``S3_evidence/acceptance/repro_registered_a2.json``. Changes no product.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from harness import prereg  # noqa: E402

REPO = HERE.parents[1]
PRODUCT = REPO / "results/dataprep/gnss_clean/preorigin/v1"
EVID = REPO / "results/dataprep/reports/S3_evidence/acceptance"
PREREG_ID = "S3.gnss_clean_acceptance.v1"
A2_SCOPE = "gnss_wra+rings+piles:2020-01..2021-12:extra_2021"
A2_UNITS = ["s3.pair.GFES-光復國小", "s3.pair.HLES-宏崙國小", "s3.pair.HNES-湖南國小",
            "s3.pair.JMES-拯民國小", "s3.pair.KTES-客厝國小", "s3.pair.SJES-溪州國小",
            "s3.pair.SSES-新生國小", "s3.pair.STES-秀潭國小", "s3.pair.TKJS-土庫國中",
            "s3.pair.YCES-元長國小", "s3.pair.GFES-NTUH_282_29", "s3.pair.SJES-SJES",
            "s3.pair.STES-STES", "s3.pair.TKJS-TKJS_281_29"]

NEW_BLOCK = '''    g = g[g["count"] >= min_days]["median"].unstack("station")
    # unstack does not guarantee a sorted month index (stations starting at different
    # months); label slicing in ols_rate needs it sorted (fixed 2026-10-12 after the
    # pre-registration: the full-variant NGL scoring raised on an unsorted index; label
    # look-ups in A2 never depended on the order)
    return g.sort_index()
'''
OLD_BLOCK = '''    g = g[g["count"] >= min_days]["median"].unstack("station")
    return g
'''


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def registered_sha() -> str:
    p, _ = prereg.load_prereg(PREREG_ID)      # verifies the file against INDEX and ledger
    return next(i["sha256"] for i in p["inputs"]
                if i["path"] == "dataprep_staging/s3_gnss/validate.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare(a: dict, b: dict) -> dict:
    """max |difference| over the numeric fields of two A2 results (pairs and summaries)."""
    diffs, n = [], 0
    pa = pd.DataFrame(a["pairs"]).set_index(["kind", "gnss", "reference"]).sort_index()
    pb = pd.DataFrame(b["pairs"]).set_index(["kind", "gnss", "reference"]).sort_index()
    same_keys = list(pa.index) == list(pb.index) and list(pa.columns) == list(pb.columns)
    if same_keys:
        num = pa.select_dtypes("number").columns
        x, y = pa[num].to_numpy(float), pb[num].to_numpy(float)
        both_nan = np.isnan(x) & np.isnan(y)
        nan_mismatch = int((np.isnan(x) ^ np.isnan(y)).sum())
        d = np.where(both_nan, 0.0, np.abs(x - y))
        diffs.append(float(np.nanmax(d)))
        n += int(d.size)
    else:
        nan_mismatch = -1
    for k, v in a.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                w = b.get(k, {}).get(kk)
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    diffs.append(abs(float(vv) - float(w)))
                    n += 1
                elif vv != w:
                    diffs.append(float("inf"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            w = b.get(k)
            ok = (np.isnan(v) and np.isnan(w)) if isinstance(v, float) and np.isnan(v) \
                else abs(float(v) - float(w)) == 0 or abs(float(v) - float(w)) < 1e-12
            diffs.append(0.0 if ok else abs(float(v) - float(w)))
            n += 1
        elif isinstance(v, bool) and v != b.get(k):
            diffs.append(float("inf"))
    return {"pair_keys_identical": same_keys, "nan_pattern_mismatches": nan_mismatch,
            "n_values_compared": n, "max_abs_diff": max(diffs) if diffs else None}


def main() -> int:
    cur_path = HERE / "validate.py"
    cur = cur_path.read_bytes()
    want = registered_sha()
    text = cur.decode("utf-8")
    if text.count(NEW_BLOCK) != 1:
        raise SystemExit("current validate.py does not contain the post-registration block")
    reg = text.replace(NEW_BLOCK, OLD_BLOCK).encode("utf-8")
    if sha(reg) != want:
        raise SystemExit(f"recovered file sha256 {sha(reg)} != registered {want}")
    out_code = EVID / "registered_code" / "validate.py"
    out_code.parent.mkdir(parents=True, exist_ok=True)
    if out_code.exists() and out_code.read_bytes() != reg:
        raise SystemExit(f"{out_code} exists with other content; not overwriting")
    out_code.write_bytes(reg)

    look = prereg.record_look(
        A2_UNITS, A2_SCOPE, "S3", "reproduce", prereg_id=PREREG_ID,
        note="resume check: A2 recomputed on the published pre-origin tables with the "
             "registered validate.py (recovered, sha256 461cd6a1...) and the current file "
             "(5a06c7ac...); no new test, no threshold change")

    wd = pd.read_parquet(PRODUCT / "wra" / "daily.parquet")
    cms = pd.read_parquet(PRODUCT / "common_mode.parquet")
    cms.index = pd.DatetimeIndex(pd.to_datetime(cms.pop("date")), name="date")
    st = pd.read_csv(PRODUCT / "stations.csv")

    v_reg = load_module("validate_registered", out_code)
    v_cur = load_module("validate_current", cur_path)
    r_reg = v_reg.test_2021_extra(wd, cms, st)
    r_cur = v_cur.test_2021_extra(wd, cms, st)

    gate = json.loads((EVID / "validation_preorigin.json").read_text())["A2_2021_extra"]

    def js(r: dict) -> dict:     # same rounding/cleaning path as the gate run's JSON
        return json.loads(json.dumps(r, default=str).replace("NaN", "null"),
                          parse_constant=lambda c: None)

    def nanify(r: dict) -> dict:
        r = json.loads(json.dumps(r))
        for p in r["pairs"]:
            for k, v in p.items():
                if v is None:
                    p[k] = float("nan")
        for k, v in list(r.items()):
            if v is None:
                r[k] = float("nan")
        return r

    g, a, b = nanify(gate), nanify(js(r_reg)), nanify(js(r_cur))
    summ = {k: r_reg[k] for k in ("n_pairs", "n_evaluable", "pass",
                                  "ratio_cmr_to_uncorrected_pooled")}
    summ["adopted"] = r_reg["cm_removed_adopted_wra_hf"]
    res = {
        "purpose": "resume check of the gate's A2 provenance (no new test)",
        "prereg_id": PREREG_ID,
        "look_logged": {"time": look["time"], "purpose": "reproduce"},
        "registered_validate_sha256": want,
        "recovered_validate_sha256": sha(reg),
        "current_validate_sha256": sha(cur),
        "recovered_file": str(out_code.relative_to(REPO)),
        "change_after_registration": "monthly(): 'return g' -> comment (4 lines) + "
                                     "'return g.sort_index()'",
        "inputs": {str(p.relative_to(REPO)): sha(p.read_bytes()) for p in
                   (PRODUCT / "wra/daily.parquet", PRODUCT / "common_mode.parquet",
                    PRODUCT / "stations.csv", EVID / "validation_preorigin.json")},
        "registered_vs_gate_run": compare(a, g),
        "current_vs_gate_run": compare(b, g),
        "registered_vs_current": compare(a, b),
        "a2_summary_registered_file": summ,
    }
    out = EVID / "repro_registered_a2.json"
    out.write_text(json.dumps(res, indent=1, default=str, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("registered_vs_gate_run", "current_vs_gate_run",
                                          "registered_vs_current",
                                          "a2_summary_registered_file")},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
