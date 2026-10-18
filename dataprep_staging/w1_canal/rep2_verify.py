"""Post-publication checks of W1 repair round 2 (surface_water_v4 preorigin/v2, full/v4).

    cpu_slot.sh python -I dataprep_staging/w1_canal/rep2_verify.py PROBE_DIR

* the A0 checker re-hashes every surface_water_v4 manifest and the CODIS SOURCE.json;
* interval-label scan of every output of every version: an interval-bearing column (sd,
  *_p05/_p95, lo/hi) or array (sd, draws_*) counts as labelled if a row-level
  ``interval_calibration`` is non-empty wherever the interval is, or (npz) a per-month
  ``interval_calibration`` array exists, or the manifest's ``interval_labels`` names the output;
* identity: values, flags, members and grid values unchanged against the previous versions;
  sd changed only in the structural-zero months of the pre-origin product;
* loader-guard probes (prior_width_input now refused for the pre-origin product);
* the W1 registrations still load with unchanged inputs; ledger conflicts across fragments;
* forward-archive probes: the WRA real-time endpoints (status codes) and a search of the
  reachable Civil IoT SensorThings mirror for discharge or stage at the composite gauges
  (responses are untrusted data, saved under PROBE_DIR, parsed as JSON only).
Writes results/dataprep/reports/W1_evidence/rep2_verify.json.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import check, ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, L0_NEW, from_repo, rel_to_repo  # noqa: E402
from common.source import check_source_file  # noqa: E402
from harness import prereg2  # noqa: E402

from w1_canal import config as C  # noqa: E402
from w1_canal.build import jsonable  # noqa: E402
from w1_canal.ext_verify import PROBES, http_probe  # noqa: E402
from w1_canal.verify import probe, try_path, try_table  # noqa: E402

EVIDENCE = DATAPREP / "reports" / "W1_evidence"
BASE = DATAPREP / C.PRODUCT_BASE
VERSIONS = [("preorigin", "v1"), ("preorigin", "v2"), ("full", "v1"), ("full", "v2"),
            ("full", "v3"), ("full", "v4")]
IDS = ["W1_janmay_benchmark_2026-10-12", "W1_janmay_benchmark_2026-10-12.record",
       "W1.ext2026_rain_runoff.v1", "W1.ext2026_uncertainty.r1",
       "W1.ext2026_prospective_obs.v1"]
INTERVAL_COL = re.compile(r"^(sd|.*_sd|.*p05|.*p95|lo|hi)$")
STA = "https://sta.colife.org.tw/STA_WaterResource_v2/v1.0"
STATION_NAMES = ["寶石橋", "內茅埔", "水里", "集集", "濁水溪", "陳有蘭"]


# ----------------------------------------------------------------------------- label scan
def scan_version(d: Path) -> dict:
    m = json.loads((d / "MANIFEST.json").read_text())
    inv = m.get("interval_labels") or {}
    res = {"outputs": {}, "unlabelled": [], "interval_values": 0, "labelled_values": 0}
    for o in m["outputs"]:
        f = o["path"]
        p = d / f
        rec = {}
        if p.suffix in (".csv", ".parquet"):
            t = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
            cols = [c for c in t.columns if INTERVAL_COL.match(c)]
            row_lab = (t["interval_calibration"].fillna("").astype(str).str.len() > 0
                       if "interval_calibration" in t.columns else None)
            for c in cols:
                nn = t[c].notna()
                n = int(nn.sum())
                if row_lab is not None:
                    lab = int((nn & row_lab).sum())
                    how = "row column"
                elif f in inv:
                    lab, how = n, "manifest interval_labels"
                else:
                    lab, how = 0, "none"
                rec[c] = {"non_null": n, "labelled": lab, "how": how}
                res["interval_values"] += n
                res["labelled_values"] += lab
                if lab < n:
                    res["unlabelled"].append(f"{f}:{c} ({n - lab} of {n})")
        elif p.suffix == ".npz":
            z = np.load(p, allow_pickle=True)
            keys = [k for k in z.files if k == "sd" or k.startswith("draws_")]
            has = "interval_calibration" in z.files
            T = len(z["dates"]) if "dates" in z.files else None
            ok_len = has and T is not None and len(z["interval_calibration"]) == T
            for k in keys:
                n = int(np.isfinite(np.asarray(z[k], float)).sum())
                if ok_len:
                    lab, how = n, "per-month array"
                elif f in inv:
                    lab, how = n, "manifest interval_labels"
                else:
                    lab, how = 0, "none"
                rec[k] = {"values": n, "labelled": lab, "how": how}
                res["interval_values"] += n
                res["labelled_values"] += lab
                if lab < n:
                    res["unlabelled"].append(f"{f}:{k} ({n - lab} of {n})")
        if rec:
            res["outputs"][f] = rec
    res["n_unlabelled_columns"] = len(res["unlabelled"])
    return res


# ----------------------------------------------------------------------------- identity
def identity(prev: Path, new: Path, pre: bool) -> dict:
    out = {}
    a = pd.read_csv(prev / "district_monthly.csv")
    b = pd.read_csv(new / "district_monthly.csv")
    out["rows_equal"] = len(a) == len(b)
    out["values_identical"] = bool(np.array_equal(a.value.to_numpy(), b.value.to_numpy()))
    out["flags_identical"] = bool(np.array_equal(a.flag.to_numpy(), b.flag.to_numpy()))
    sz = a.structural_zero.astype(bool).to_numpy()
    out["sd_identical_outside_structural"] = bool(np.array_equal(a.sd.to_numpy()[~sz],
                                                                 b.sd.to_numpy()[~sz]))
    out["sd_structural_max"] = float(np.abs(b.sd.to_numpy()[sz]).max())
    if pre:
        out["structural_alt_equals_v1_sd"] = bool(np.array_equal(
            a.sd.to_numpy()[sz], b.structural_alt_value.to_numpy()[sz]))
    else:
        out["sd_identical_all"] = bool(np.array_equal(a.sd.to_numpy(), b.sd.to_numpy()))
    for f in ("district_draws.npz", "grid_sw_v4.npz"):
        za = np.load(prev / f, allow_pickle=True)
        zb = np.load(new / f, allow_pickle=True)
        keys = [k for k in za.files if k not in ("note", "sd")]
        out[f] = {k: bool(np.array_equal(za[k], zb[k])) for k in keys}
        if f == "grid_sw_v4.npz":
            if pre:
                gs = np.array([a[a.date == t].structural_zero.all()
                               for t in pd.to_datetime(za["dates"]).strftime("%Y-%m-%d")])
                out[f]["sd_identical_outside_structural_max_abs"] = float(
                    np.abs(za["sd"][:, ~gs] - zb["sd"][:, ~gs]).max())
                out[f]["sd_structural_max"] = float(np.abs(zb["sd"][:, gs]).max())
            else:
                out[f]["sd"] = bool(np.array_equal(za["sd"], zb["sd"]))
    comp_a = pd.read_parquet(prev / "inflow_composite_daily.parquet")
    comp_b = pd.read_parquet(new / "inflow_composite_daily.parquet")
    out["composite_value_sd_flag_identical"] = all(
        np.array_equal(comp_a[c].to_numpy(), comp_b[c].to_numpy()) for c in ("value", "sd", "flag"))
    return out


# ----------------------------------------------------------------------------- probes
def curl_json(url: str, params: dict, dest: Path) -> dict | None:
    args = ["curl", "-s", "-m", "60", "-A", "Mozilla/5.0", "-G", "-o", str(dest)]
    for k, v in params.items():
        args += ["--data-urlencode", f"{k}={v}"]
    r = subprocess.run(args + [url], capture_output=True, text=True)
    if r.returncode != 0 or not dest.exists():
        return None
    try:
        return json.loads(dest.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def colife_search(probe_dir: Path) -> dict:
    probe_dir.mkdir(parents=True, exist_ok=True)
    out = {"service": STA, "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "things": {}, "flow_or_stage_datastreams_in_basin": None}
    for i, nm in enumerate(STATION_NAMES):
        j = curl_json(f"{STA}/Things", {"$filter": f"substringof('{nm}',description)",
                                        "$top": "100", "$select": "@iot.id,description"},
                      probe_dir / f"things_{i}.json")
        if j is None:
            out["things"][nm] = "no JSON"
            continue
        names = [re.sub(r".*stationName=([^,]*).*", r"\1", v.get("description", ""))
                 for v in j.get("value", [])]
        out["things"][nm] = names
    flt = ("(substringof('流量',name) or name eq '水位') and (substringof('濁水',Thing/description)"
           " or substringof('陳有蘭',Thing/description) or substringof('水里',Thing/description) "
           "or substringof('集集',Thing/description) or substringof('寶石',Thing/description) "
           "or substringof('內茅埔',Thing/description))")
    j = curl_json(f"{STA}/Datastreams", {"$filter": flt, "$top": "200",
                                         "$select": "@iot.id,name,description"},
                  probe_dir / "datastreams_basin.json")
    if j is not None:
        out["flow_or_stage_datastreams_in_basin"] = [
            re.sub(r".*Datastream_FullName=([^,]*).*", r"\1", v.get("description", ""))
            for v in j.get("value", [])]
    ops = curl_json(f"{STA}/ObservedProperties", {"$select": "name", "$top": "1000"},
                    probe_dir / "observed_properties.json")
    if ops is not None:
        names = pd.Series([v.get("name") for v in ops.get("value", [])]).value_counts()
        out["observed_property_names_first_1000"] = names.to_dict()
    return out


def main(probe_dir: str):
    out = {"run_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    out["checker"] = {f"{v}/{n}": check.check_manifest(BASE / v / n / "MANIFEST.json")
                      for v, n in VERSIONS}
    out["checker"]["cwa_codis_rain_upper_choushui SOURCE.json"] = {
        "status": check_source_file(L0_NEW / "cwa_codis_rain_upper_choushui"
                                    / "SOURCE.json")["status"]}
    out["label_scan"] = {f"{v}/{n}": scan_version(BASE / v / n) for v, n in VERSIONS}
    out["identity"] = {
        "preorigin_v2_vs_v1": identity(BASE / "preorigin" / "v1", BASE / "preorigin" / "v2",
                                       True),
        "full_v4_vs_v3": identity(BASE / "full" / "v3", BASE / "full" / "v4", False)}
    pre, full = BASE / "preorigin" / "v2", BASE / "full" / "v4"
    out["guard_probes"] = {
        "preorigin/v2": [
            probe(pre, "forcing_frozen", try_path("grid_sw_v4.npz")),
            probe(pre, "prior", try_table("district_monthly.csv")),
            probe(pre, "prior", try_path("grid_sw_v4.npz")),
            probe(pre, "selection", try_table("district_monthly.csv")),
            probe(pre, "model_design"), probe(pre, "design_time_evidence"),
            probe(pre, "prior_width_input"), probe(pre, "forcing_recorded_arm"),
            probe(pre, "scoring_target"), probe(pre, "calibration_target"),
            probe(pre, "baseline")],
        "full/v4": [
            probe(full, "forcing_recorded_arm", try_table("district_monthly.csv")),
            probe(full, "forcing_recorded_arm", try_path("district_draws.npz")),
            probe(full, "statistical_layer_history"), probe(full, "after_the_fact_analysis"),
            probe(full, "forcing_frozen"), probe(full, "model_design"), probe(full, "prior"),
            probe(full, "prior_width_input"), probe(full, "selection"),
            probe(full, "scoring_target"), probe(full, "calibration_target")]}
    regs = {}
    legacy = json.loads((DATAPREP / "preregistration" / f"{IDS[0]}.json").read_text())
    changed = [r["path"] for r in legacy["inputs"] + legacy["code"]
               if sha256_file(from_repo(r["path"])) != r["sha256"]]
    regs[IDS[0]] = {"ok": not changed, "legacy_w1_writer": True,
                    "n_hashes": len(legacy["inputs"]) + len(legacy["code"]),
                    "changed": changed, "note": "predates the A1 tool; indexed through "
                                                f"{IDS[1]}"}
    for i in IDS[1:]:
        try:
            doc, sha = prereg2.load_prereg(i)
            regs[i] = {"ok": True, "sha256": sha, "changed_inputs": prereg2.changed_inputs(doc)}
        except Exception as e:  # noqa: BLE001
            regs[i] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    out["registrations"] = regs
    frag = ledger.fragment_path(C.WP)
    with open(frag, encoding="utf-8") as fh:
        n_rows = sum(1 for _ in fh) - 1
    out["ledger"] = {"fragment": rel_to_repo(frag), "rows": n_rows,
                     "conflicts_all_fragments": len(ledger.conflicts(ledger.read_all()))}
    out["forward_archive_probes"] = [http_probe(u) for u in PROBES]
    out["forward_archive_alternative"] = colife_search(Path(probe_dir))
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with open(EVIDENCE / "rep2_verify.json", "w") as f:
        json.dump(jsonable(out), f, indent=1, ensure_ascii=False)
    for k, v in out["checker"].items():
        print("checker", k, v if isinstance(v, str) else v.get("status"),
              v.get("n_hashed") if isinstance(v, dict) else "")
    for k, v in out["label_scan"].items():
        print("labels", k, v["labelled_values"], "/", v["interval_values"], v["unlabelled"][:6])
    print(json.dumps(jsonable(out["identity"]))[:1500])
    for k, v in out["guard_probes"].items():
        for g in v:
            print(k, g["use"], g.get("opened"), g.get("action", g.get("error")))
    print(out["registrations"])
    print(out["ledger"])
    for p in out["forward_archive_probes"]:
        print(p["http_code"], p["url"][:70])
    print(json.dumps(out["forward_archive_alternative"], ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main(sys.argv[1])
