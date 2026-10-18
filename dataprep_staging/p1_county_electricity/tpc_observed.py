"""WP P1 round 2: ``county_electricity/tpc_observed/v<N>`` - the observed TPC table as an L1
product with its own integrity gate (the validator role of the revised plan).

Why a separate product. The plan revision of 2026-10-12 (afternoon) makes P1 "a validator of
census coverage". In v1 the observed d007012 rows sit inside ``county_electricity/{preorigin,
full}/v1``, whose gates (source separation, backcast, interval coverage) failed, so the loader
guard refuses the ``validator`` use of the observed rows too, although none of those failures
concerns the published table itself. This product carries only observed rows (flag 0): the
d007012 county x sector monthly sales as published, and the Energy Administration national
agricultural series, gated by the pre-registered integrity checks of
``P1r2.tpc_observed_gate`` (G1 archive hashes, G2 completeness, G3 internal shares; G4 the
re-fetch, reported). No value is fitted, so the rows are time-sliceable: a pre-origin consumer
opens it with ``window_end <= 2022-12``.

Usage: python -I tpc_observed.py <fetch_log_r2.json> [--dry]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import lib  # noqa: E402
from common import manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402
from common.uses import USES  # noqa: E402

L0 = REPO / "data_fetch_api/new_measurements"
TPC_DIR = L0 / "tpc_d007012/2026-10-12"
NAT_DIR = L0 / "moeaea_agri_electricity/2026-10-12"
TPC = TPC_DIR / "d007012.csv"
NAT = {"236": NAT_DIR / "set_236.csv", "256": NAT_DIR / "set_256.csv"}
PREREG = REPO / "results/dataprep/preregistration/P1r2.tpc_observed_gate.json"
PRODUCT = "county_electricity/tpc_observed"
COUNTIES = (lib.CH, lib.YL)
TPC_MONTHS = pd.date_range("2016-01-01", "2026-08-01", freq="MS")
NAT_MONTHS = pd.date_range("2007-01-01", "2026-08-01", freq="MS")
ORIGIN_LAST = pd.Timestamp("2022-12-01")
CODE = [HERE / "tpc_observed.py", HERE / "lib.py", HERE.parent / "common/manifest.py",
        HERE.parent / "common/hashing.py"]
ALLOWED = ["validator", "after_the_fact_analysis", "statistical_layer_history",
           "design_time_evidence"]


def read_national_sets() -> dict[str, pd.Series]:
    out = {}
    for k, p in NAT.items():
        d = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
        s = pd.Series(pd.to_numeric(d["電力消費_農業部門(數值)"], errors="coerce").values,
                      index=pd.to_datetime(d.iloc[:, 0].str.strip(), format="%Y%m"))
        out[k] = s.sort_index()
    return out


def g1_hashes() -> dict:
    res = {}
    for d, files in ((TPC_DIR, ["d007012.csv"]), (NAT_DIR, ["set_236.csv", "set_256.csv"])):
        rec = json.loads((d / "SOURCE.json").read_text(encoding="utf-8"))
        want = {f["file"]: f["sha256"] for f in rec["files"]}
        for f in files:
            got = sha256_file(d / f)
            res[rel_to_repo(d / f)] = {"sha256": got, "source_json": want.get(f),
                                       "match": got == want.get(f)}
    return {"files": res, "pass": all(v["match"] for v in res.values())}


def g2_completeness(tpc: pd.DataFrame, sets: dict, last: pd.Timestamp) -> dict:
    t = tpc[tpc.date <= last]
    months = TPC_MONTHS[TPC_MONTHS <= last]
    nmonths = NAT_MONTHS[NAT_MONTHS <= last]
    out = {"counties": {}}
    ok = True
    for c in COUNTIES:
        x = t[t.county == c]
        dup = int(x.duplicated(["sector", "date"]).sum())
        ag = x[x.sector == lib.AG].set_index("date")["kwh"]
        missing = [str(m.date())[:7] for m in months if m not in ag.index]
        nonpos = [str(m.date())[:7] for m in ag.index if not (ag.loc[m] > 0)]
        sectors = x.groupby("date")["sector"].nunique()
        out["counties"][c] = {"duplicate_sector_months": dup, "ag_months_expected": len(months),
                              "ag_months_present": int(ag.index.isin(months).sum()),
                              "ag_missing": missing, "ag_nonpositive": nonpos,
                              "sectors_per_month": {str(int(k)): int(v) for k, v in
                                                    sectors.value_counts().items()},
                              "sector_names": sorted(x.sector.unique().tolist())}
        ok &= dup == 0 and not missing and not nonpos
    merged = pd.concat([sets["256"], sets["236"]]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged[merged.index <= last]
    nat_missing = [str(m.date())[:7] for m in nmonths if m not in merged.index]
    nat_nonpos = int((merged.reindex(nmonths) <= 0).sum())
    ov = sets["236"].index.intersection(sets["256"].index)
    ov = ov[ov <= last]
    rel = (np.abs(sets["236"].loc[ov] - sets["256"].loc[ov])
           / np.abs(sets["256"].loc[ov]).clip(lower=1e-12)) if len(ov) else pd.Series(dtype=float)
    dup_in_sets = {k: int(s.index.duplicated().sum()) for k, s in sets.items()}
    out["national"] = {"months_expected": len(nmonths), "missing": nat_missing,
                       "nonpositive": nat_nonpos, "overlap_months": len(ov),
                       "overlap_max_rel_diff": float(rel.max()) if len(rel) else None,
                       "duplicate_months_within_sets": dup_in_sets}
    ok &= (not nat_missing and nat_nonpos == 0 and (rel.max() <= 1e-9 if len(rel) else True)
           and not any(dup_in_sets.values()))
    out["pass"] = bool(ok)
    return out


def g3_shares(tpc: pd.DataFrame, last: pd.Timestamp) -> dict:
    t = tpc[tpc.date <= last].copy()
    tot = t.groupby(["county", "date"])["kwh"].transform("sum")
    t["pct_calc"] = 100.0 * t["kwh"] / tot
    t["absdiff"] = (t["pct_calc"] - t["pct"]).abs()
    sel = t.county.isin(COUNTIES)
    two = t[sel]
    others = t[~sel]
    res = {"rows_checked": int(len(two)), "pct_missing": int(two["pct"].isna().sum()),
           "max_abs_diff_pp": float(two["absdiff"].max()),
           "rows_over_0.01pp": int((two["absdiff"] > 0.01).sum()),
           "other_counties": {"rows": int(len(others)),
                              "max_abs_diff_pp": float(others["absdiff"].max()),
                              "rows_over_0.01pp": int((others["absdiff"] > 0.01).sum()),
                              "pct_missing": int(others["pct"].isna().sum())}}
    res["pass"] = bool(res["pct_missing"] == 0 and res["max_abs_diff_pp"] <= 0.01)
    return res


def qc_flags(tpc: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    """Flag 9 (suspect, kept) for rows whose kWh contradicts the published shares.

    Rule (fixed after G3 had failed; it changes no gate, only flags): in a county-month where
    any row's kWh share differs from the published share by more than 0.01 pp, each row's
    implied county total kWh / (pct / 100) is compared with the median implied total of the
    other rows of that county-month; a row deviating by more than max(5 %, 0.006 / pct) (the
    second term covers the rounding of small published shares to 0.01 %) is flagged 9
    (method ``qc_share_inconsistent``). If no row is singled out, every row of the
    county-month is flagged 9 (``qc_share_inconsistent_unattributed``).
    """
    t = tpc.copy()
    tot = t.groupby(["county", "date"])["kwh"].transform("sum")
    t["absdiff"] = (100.0 * t["kwh"] / tot - t["pct"]).abs()
    t["impl"] = t["kwh"] / (t["pct"] / 100.0)
    flag = pd.Series(0, index=t.index)
    method = pd.Series("tpc_d007012_observed", index=t.index, dtype=object)
    bad = t.loc[t["absdiff"] > 0.01, ["county", "date"]].drop_duplicates()
    rec = []
    for _, r in bad.iterrows():
        x = t[(t.county == r.county) & (t.date == r.date)]
        hit = []
        for i, row in x.iterrows():
            med = float(np.median(x.drop(i)["impl"]))
            tol = max(0.05, 0.006 / row["pct"]) if row["pct"] > 0 else np.inf
            dev = abs(row["impl"] / med - 1.0)
            if dev > tol:
                hit.append((i, row["sector"], round(dev, 4), round(row["kwh"] / (
                    med * row["pct"] / 100.0), 4)))
        if hit:
            for i, *_ in hit:
                flag.loc[i], method.loc[i] = 9, "qc_share_inconsistent"
        else:
            flag.loc[x.index], method.loc[x.index] = 9, "qc_share_inconsistent_unattributed"
        rec.append({"county": r.county, "month": str(r.date.date())[:7],
                    "flagged": [{"sector": s, "deviation": d, "kwh_over_implied": q}
                                for _, s, d, q in hit] or "all rows (unattributed)"})
    summ = {"failing_county_months": len(bad), "rows_flagged_9": int((flag == 9).sum()),
            "detail": rec,
            "changhua_yunlin_ag_rows_flagged": int(((flag == 9) & t.county.isin(COUNTIES)
                                                    & (t.sector == lib.AG)).sum())}
    return flag, method, summ


def ag_share_diagnostic(tpc: pd.DataFrame, flag: pd.Series) -> dict:
    """Reported only (designed after G3 failed): Changhua / Yunlin 農林漁牧 rows against their
    published share, with the county total of a county-month that holds a flag-9 row rebuilt
    as the median implied total kWh / (pct / 100) of its unflagged rows."""
    t = tpc.copy()
    t["flag"] = flag
    t["impl"] = t["kwh"] / (t["pct"] / 100.0)
    out = {}
    for c in COUNTIES:
        x = t[t.county == c]
        diffs = []
        for dte, g in x.groupby("date"):
            ok = g[g.flag == 0]
            total = g["kwh"].sum() if (g.flag == 0).all() else float(np.median(ok["impl"]))
            ag = g[g.sector == lib.AG].iloc[0]
            diffs.append(abs(100.0 * ag["kwh"] / total - ag["pct"]))
        out[c] = {"months": len(diffs), "max_abs_diff_pp": float(np.max(diffs)),
                  "rows_over_0.01pp": int(np.sum(np.asarray(diffs) > 0.01))}
    out["note"] = "post hoc diagnostic, not a gate; reported for the validator decision"
    return out


def g4_refetch(log_path: Path) -> dict:
    log = json.loads(log_path.read_text(encoding="utf-8"))
    return {"log": rel_to_repo(log_path), "run_utc": log["run_utc"],
            "identical": {s: e["identical"] for s, e in log["sources"].items()},
            "archived_to": {s: e.get("archived_to") for s, e in log["sources"].items()},
            "d007012_last_modified": log["sources"]["tpc_d007012"]["files"]["d007012.csv"]
            .get("last_modified_header"), "reported_only": True}


def tables(tpc: pd.DataFrame, sets: dict, pv: str, flag: pd.Series, method: pd.Series
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = tpc.copy()
    t["value"] = t["kwh"] / 1e6
    t = t.rename(columns={"pct": "pct_of_county_published"})
    t["month"] = t["date"].dt.strftime("%Y-%m")
    t["sd"] = np.nan
    t["flag"] = flag.reindex(t.index).astype("int64")
    t["method"] = method.reindex(t.index)
    t["unit"] = "GWh/month (TPC billed sales)"
    t["product_version"] = pv
    t["source_ids"] = "tpc_d007012/2026-10-12/d007012.csv"
    t = t[["date", "month", "county", "sector", "value", "kwh", "pct_of_county_published",
           "sd", "flag", "method", "unit", "product_version", "source_ids"]]
    n = []
    for k in ("256", "236"):
        s = sets[k]
        n.append(pd.DataFrame({"date": s.index, "value": s.to_numpy(), "set_id": k}))
    nat = pd.concat(n, ignore_index=True)
    nat = nat.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    nat["month"] = nat["date"].dt.strftime("%Y-%m")
    nat["sd"], nat["flag"], nat["method"] = np.nan, 0, "moeaea_national_agri_observed"
    nat["unit"] = "GWh/month (Energy Administration, 電力消費_農業部門)"
    nat["product_version"] = pv
    nat["source_ids"] = np.where(nat["set_id"] == "236",
                                 "moeaea_agri_electricity/2026-10-12/set_236.csv",
                                 "moeaea_agri_electricity/2026-10-12/set_256.csv")
    nat = nat[["date", "month", "value", "sd", "flag", "method", "unit", "set_id",
               "product_version", "source_ids"]]
    return t.sort_values(["county", "sector", "date"]).reset_index(drop=True), nat


CARD = """# county_electricity/tpc_observed/{pv}

Observed Taiwan Power county x sector monthly electricity sales (d007012, data.gov.tw 29935,
archived fetch 2026-10-12, Last-Modified {lm}) and the Energy Administration national monthly
agricultural electricity (sets 236 + 256), as published. WP P1 round 2.

- **Coverage.** d007012: every county, sectors 住宅 / 服務業 / 農林漁牧 / 工業, {t0}..{t1}
  ({n_tpc} rows). National: {n0}..{n1} ({n_nat} months).
- **Flags.** 0 (observed) and 9 (suspect, kept: {n9} rows whose kWh contradicts the
  published shares, rule in `tpc_observed.qc_flags`; Changhua/Yunlin 農林漁牧 rows flagged:
  {n9ag}). No value is filled, corrected or inferred. County names are normalised 臺 -> 台
  (as in P1 v1).
- **Gate.** Pre-registered `P1r2.tpc_observed_gate`: G1 archive hashes {g1}, G2 completeness
  {g2}, G3 internal shares {g3}; gate passed = {gate}. G4 re-fetch identical = {g4}.
  G3 fails on one Changhua county-month (2016-11): the published 住宅 kWh is about 1/19 of
  the value its own published share implies; see validation.json `qc_flag9` and the post hoc
  `ag_share_diagnostic_post_hoc`.
- **Allowed uses.** validator (census coverage, the revised P1 role; refused by the guard
  while the gate is failed), after_the_fact_analysis, statistical_layer_history;
  design_time_evidence only through a time slice ending by 2022-12
  (`guard.open_product(dir, use, window_end="2022-12")`; time column `date`).
- **Known limits.** TPC does not define 農林漁牧: it includes non-pump load (livestock cooling,
  aerators, lighting). Billing month vs consumption month is not documented. d007012 is a
  rolling window (2016-01 on). The national series is a different statistic (Energy
  Administration balance) and is not the sum of the county rows.
- **Licence.** Open Government Data License v1.0 (both sources, per SOURCE.json).
- **Personal data.** None.
"""


def main(fetch_log: str, dry: bool) -> int:
    tpc = lib.read_tpc_d007012(TPC)
    sets = read_national_sets()
    val = {"product": PRODUCT, "prereg": rel_to_repo(PREREG),
           "prereg_sha256": sha256_file(PREREG)}
    full_last = pd.Timestamp("2026-08-01")
    val["G1_archive_hash"] = g1_hashes()
    val["G2_completeness"] = g2_completeness(tpc, sets, full_last)
    val["G3_internal_share"] = g3_shares(tpc, full_last)
    val["G4_refetch"] = g4_refetch(Path(fetch_log))
    val["slice_le_2022_12"] = {"G2": g2_completeness(tpc, sets, ORIGIN_LAST),
                               "G3": g3_shares(tpc, ORIGIN_LAST)}
    gate = {k: val[k]["pass"] for k in ("G1_archive_hash", "G2_completeness",
                                        "G3_internal_share")}
    gate["passed"] = all(gate.values())
    val["gate"] = gate
    flag, method, qc = qc_flags(tpc)
    val["qc_flag9"] = qc
    val["ag_share_diagnostic_post_hoc"] = ag_share_diagnostic(tpc, flag)
    print(json.dumps(val, ensure_ascii=False, indent=1, default=str))
    if dry:
        return 0
    d = manifest.new_version_dir(PRODUCT)
    pv = f"{PRODUCT}/{d.name}"
    t, nat = tables(tpc, sets, pv, flag, method)
    t.to_parquet(d / "tpc_county_sector_monthly_observed.parquet", index=False)
    nat.to_parquet(d / "national_agri_monthly_observed.parquet", index=False)
    (d / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1,
                                                  default=str), encoding="utf-8")
    (d / "PRODUCT_CARD.md").write_text(CARD.format(
        pv=d.name, lm=val["G4_refetch"]["d007012_last_modified"],
        t0=str(t.date.min().date())[:7], t1=str(t.date.max().date())[:7], n_tpc=len(t),
        n0=str(nat.date.min().date())[:7], n1=str(nat.date.max().date())[:7], n_nat=len(nat),
        g1=gate["G1_archive_hash"], g2=gate["G2_completeness"], g3=gate["G3_internal_share"],
        gate=gate["passed"], g4=val["G4_refetch"]["identical"], n9=qc["rows_flagged_9"],
        n9ag=qc["changhua_yunlin_ag_rows_flagged"]), encoding="utf-8")
    forbidden = sorted(u for u in USES if u not in ALLOWED)
    m = manifest.build_manifest(
        d, product=PRODUCT, layer="L1",
        inputs=[TPC, TPC_DIR / "SOURCE.json", NAT["236"], NAT["256"], NAT_DIR / "SOURCE.json"],
        code=CODE,
        parameters={"prereg": rel_to_repo(PREREG), "prereg_sha256": val["prereg_sha256"],
                    "fitted_parameters": "none (observed rows only)",
                    "county_name_normalisation": "臺 -> 台 (lib.read_tpc_d007012)",
                    "national_merge": "sets 256 (to 2017) and 236 (2018 on); on overlap the "
                                      "236 value is kept (overlap equality is checked in G2)",
                    "refetch_log": val["G4_refetch"]["log"]},
        fit_window_end="2026-08", validation_report="validation.json",
        gate={"passed": gate["passed"],
              "criteria": "pre-registration P1r2.tpc_observed_gate: G1 archive hashes, G2 "
                          "completeness, G3 internal shares (G4 reported)",
              **{k: v for k, v in gate.items() if k != "passed"}},
        allowed_uses=ALLOWED, forbidden_uses=forbidden,
        personal_data="none (published county x sector aggregates)",
        time_slicing={"column": "date", "params_fit_window_end": "2007-01"},
        extra={"time_slicing_note": "no fitted parameters: params_fit_window_end is the "
                                    "first data month; both tables carry the datetime column "
                                    "'date' (ISO encoding)",
               "role_note": "observed validator for census coverage (plan revision "
                            "2026-10-12); the inference products county_electricity/"
                            "{preorigin,full}/v1 are unchanged"})
    p = manifest.write_manifest(d, m)
    print("wrote", rel_to_repo(p))
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry"]
    raise SystemExit(main(args[0], "--dry" in sys.argv))
