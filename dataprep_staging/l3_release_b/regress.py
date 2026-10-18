"""Compare release_20261012b with release_20261012 and check the repaired components.

    PYTHONPATH=<repo>/dataprep_staging python -m l3_release_b.regress NEW_RELEASE_DIR --json OUT

Reads both releases (never writes into them). For every component present in both, every data
output is compared: byte copies by sha256; re-packed npz array by array (the key
``release_component`` names the release and is expected to differ); tables by sha256, and for
the two tables the repair changes on purpose (pre-origin piles and stations), row by row against
the old table. Then the repaired content is checked: no pre-origin pile value published after
2022-12, no post-origin station, narrowed pile uses, and every other byte unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import hashing
from common.paths import ORIGIN, REPO

from l3_release_b import config as C
from l3_release_b import io_det

OLD = REPO / C.RELEASE_DIR / C.SUPERSEDES
INTENDED = {"preorigin/piles_printed/v1/printed_monthly.parquet",
            "preorigin/heads_observed/v1/stations.parquet"}
NAME_KEYS = {"release_component"}


def _npz_equal(a: Path, b: Path) -> tuple[bool, list[str]]:
    xa, sa = io_det.read_npz(a)
    xb, sb = io_det.read_npz(b)
    diff = sorted((set(xa) ^ set(xb)) | set(sa) ^ set(sb))
    for k in sorted(set(xa) & set(xb)):
        if k in NAME_KEYS:
            continue
        u, v = xa[k], xb[k]
        same = (u.shape == v.shape and u.dtype == v.dtype
                and np.array_equal(u, v, equal_nan=u.dtype.kind == "f"))
        if not same:
            diff.append(k)
    return not diff, diff


def compare(new: Path) -> dict:
    old_rel = json.loads((OLD / "RELEASE.json").read_text(encoding="utf-8"))
    new_rel = json.loads((new / "RELEASE.json").read_text(encoding="utf-8"))
    old_out = {o["path"]: o for v in old_rel["variants"].values() for c in v["components"]
               for o in c["outputs"]}
    rows = []
    for v in new_rel["variants"].values():
        for c in v["components"]:
            for o in c["outputs"]:
                p = o["path"]
                rec = {"path": p, "mode": o["mode"]}
                if p not in old_out:
                    rec["status"] = "new_file"
                elif o["sha256"] == old_out[p]["sha256"]:
                    rec["status"] = "identical"
                elif p.endswith(".npz"):
                    ok, d = _npz_equal(OLD / p, new / p)
                    rec["status"] = "equal_except_release_name" if ok else "DIFFERENT"
                    rec["differing_keys"] = d
                else:
                    rec["status"] = "changed"
                rec["intended_change"] = p in INTENDED
                rows.append(rec)
    unexpected = [r for r in rows if r["status"] in ("DIFFERENT", "new_file")
                  or (r["status"] == "changed" and not r["intended_change"])]
    gp = new / "grid/v1/grid.npz"
    grid_same = _npz_equal(OLD / "grid/v1/grid.npz", gp)[0]
    return {"files": rows, "unexpected": unexpected, "grid_equal": grid_same,
            "counts": {s: sum(r["status"] == s for r in rows) for s in
                       ("identical", "equal_except_release_name", "changed", "DIFFERENT",
                        "new_file")}}


def piles_check(new: Path) -> dict:
    o = pd.read_parquet(OLD / "preorigin/piles_printed/v1/printed_monthly.parquet")
    n = pd.read_parquet(new / "preorigin/piles_printed/v1/printed_monthly.parquet")
    key = ["sensor_id", "month"]
    dropped = o.merge(n[key], on=key, how="left", indicator=True)
    dropped = dropped[dropped["_merge"] == "left_only"]
    m = o.merge(n, on=key, suffixes=("_o", "_n"))
    same_value = bool(np.array_equal(m["value_o"].to_numpy(), m["value_n"].to_numpy(),
                                     equal_nan=True))
    same_flag = bool((m["flag_o"] == m["flag_n"]).all())
    rp_removed = int((m["reprint_value_o"].notna() & m["reprint_value_n"].isna()).sum())
    rp_kept = int(m["reprint_value_n"].notna().sum())
    cols_changed = sorted(c[:-2] for c in m.columns if c.endswith("_o") and not (
        m[c].astype(str) == m[c[:-2] + "_n"].astype(str)).all())
    man = json.loads((new / "preorigin/piles_printed/v1/MANIFEST.json").read_text(
        encoding="utf-8"))
    return {
        "rows_old": int(len(o)), "rows_new": int(len(n)), "rows_dropped": int(len(dropped)),
        "dropped_by_month": dropped.groupby("month").size().astype(int).to_dict(),
        "dropped_reports": sorted(set(dropped["report"])),
        "kept_rows_value_identical": same_value, "kept_rows_flag_identical": same_flag,
        "reprints_removed": rp_removed, "reprints_kept": rp_kept,
        "columns_changed_on_kept_rows": cols_changed,
        "latest_report_published": str(n["report_published"].max()),
        "latest_reprint_published": str(n["reprint_published"].dropna().max()),
        "latest_vintage_month": str(n["vintage_month"].max()),
        "any_published_after_origin": bool(
            (n["report_published"] > ORIGIN).any()
            or (n["reprint_published"].dropna() > ORIGIN).any()),
        "allowed_uses": man["allowed_uses"], "fit_window_end": man["fit_window_end"],
        "exemptions": man["exemptions"],
        "fitting_uses_granted": sorted(set(man["allowed_uses"]) & {
            "calibration_target", "datum_fit", "baseline"}),
    }


def stations_check(new: Path) -> dict:
    o = pd.read_parquet(OLD / "preorigin/heads_observed/v1/stations.parquet")
    n = pd.read_parquet(new / "preorigin/heads_observed/v1/stations.parquet")
    gone = sorted(set(o["sid"]) - set(n["sid"]))
    og = o[o["sid"].isin(gone)]
    keep_o = o[~o["sid"].isin(gone)].reset_index(drop=True)
    same_rest = bool(keep_o.equals(n.reset_index(drop=True)))
    return {"stations_old": int(len(o)), "stations_new": int(len(n)), "dropped": len(gone),
            "dropped_all_established_after_origin": bool(og["established_after_origin"].all()),
            "dropped_head_months": int(og["n_months_in_table"].sum()),
            "max_establish_date_new": int(n["EstablishDate"].astype("int64").max()),
            "remaining_rows_identical": same_rest}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("new")
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    new = Path(a.new)
    res = {"old": f"{C.RELEASE_DIR}/{C.SUPERSEDES}",
           "old_tree_digest_unchanged": None,
           "compare": compare(new), "piles": piles_check(new), "stations": stations_check(new)}
    rb = json.loads((OLD / "REBUILD_CHECK.json").read_text(encoding="utf-8"))
    files = {p.relative_to(OLD).as_posix(): hashing.sha256_file(p)
             for p in hashing.iter_files(OLD) if p.name != "REBUILD_CHECK.json"}
    res["old_tree_digest_unchanged"] = files == {f["path"]: f["sha256"] for f in rb["files"]}
    pl, st, cp = res["piles"], res["stations"], res["compare"]
    res["passed"] = bool(
        not cp["unexpected"] and cp["grid_equal"] and res["old_tree_digest_unchanged"]
        and pl["rows_dropped"] == 35 and pl["rows_new"] == 231
        and not pl["any_published_after_origin"] and pl["kept_rows_value_identical"]
        and pl["kept_rows_flag_identical"] and not pl["fitting_uses_granted"]
        and st["dropped"] == 13 and st["dropped_all_established_after_origin"]
        and st["dropped_head_months"] == 0 and st["remaining_rows_identical"]
        and st["max_establish_date_new"] <= 20221231)
    Path(a.json).write_text(io_det.dumps(res), encoding="utf-8")
    print(f"compare {cp['counts']}; unexpected {len(cp['unexpected'])}; piles {pl['rows_old']}"
          f"->{pl['rows_new']} (reprints removed {pl['reprints_removed']}); stations "
          f"{st['stations_old']}->{st['stations_new']}; old release unchanged "
          f"{res['old_tree_digest_unchanged']}; passed {res['passed']}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
