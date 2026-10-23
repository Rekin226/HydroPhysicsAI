"""WP S4a: evaluate the pre-registered setup acceptance criteria (S4a.setup_acceptance.v1).

Reads the records the other S4a steps wrote under ``results/dataprep/insar_tools/s4a_run/``
and checks the environment and the LiCSBAS checkout directly:

  S4a-1  env/env_record.json, conda-meta channels of the licsbas env, hydro snapshots
         (before / after / final, ``hydro_snapshot.sh``), a fresh LiCSBAS_check_install.py run
  S4a-2  git rev-parse / status / fsck / ls-files / check-ignore of the LiCSBAS checkout
  S4a-3  smoke_result.json                         S4a-4  validation_<primary>.json
  S4a-5  validation_<frame>.json of every fetched frame, .part scan, SOURCE.json per fetch
         directory, checker_licsar.json            S4a-6  spot_checks.json
  S4a-7  transfer_tally.json                       S4a-8  size_estimate.json, listings,
         fetch_decision.json                       S4a-9  validation_<primary>.json (GNSS)

Thresholds are copied from the declaration and never changed here. Writes
``acceptance_result.json`` with one row per criterion (S4a-7 is reported in two parts, the
hard cap and the D1 comparison; the criterion passes only if both do) and the data gate used
by the product manifest (S4a-4, -5, -6, -9: properties of the fetched data).

Usage (hydro env): python -I evaluate_acceptance.py
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # dataprep_staging (common)
sys.path.insert(0, str(HERE))

import network as nw  # noqa: E402
from common.paths import PREREG_DIR, REPO, rel_to_repo  # noqa: E402
from config import FRAMES, IFG_EXTS, LICSBAS_DIR, LICSBAS_ENV, RUN_DIR  # noqa: E402

PRIMARY = "069A_06694_131402"
DECL = PREREG_DIR / "S4a.setup_acceptance.v1.json"
DATA_GATE = ("S4a-4", "S4a-5", "S4a-6", "S4a-9")


def sh(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout + p.stderr).strip()


def jload(name: str) -> dict:
    p = RUN_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


def s4a_1() -> tuple[bool, str]:
    env = jload("env/env_record.json")
    chans = set()
    for f in (LICSBAS_ENV / "conda-meta").glob("*.json"):
        c = (json.loads(f.read_text()).get("channel") or "?").rstrip("/")
        # 'conda-forge' or https://conda.anaconda.org/conda-forge[/<subdir>]
        chans.add("conda-forge" if (c == "conda-forge" or c.startswith(
            "https://conda.anaconda.org/conda-forge")) else c)
    e = RUN_DIR / "env"
    snaps = {}
    for kind in ("conda_meta", "pip_freeze"):
        txt = {t: (e / f"hydro_{kind}_{t}.txt").read_bytes()
               for t in ("before", "after", "final") if (e / f"hydro_{kind}_{t}.txt").exists()}
        snaps[kind] = len(txt) == 3 and len(set(txt.values())) == 1
    # not -I: the check reads PYTHONPATH (set by licsbas_env.sh to the LiCSBAS library); the
    # script is the official LiCSBAS checkout, not downloaded data
    rc, out = sh([str(HERE / "licsbas_env.sh"), "python",
                  str(LICSBAS_DIR / "bin" / "LiCSBAS_check_install.py")])
    (e / "check_install_final.log").write_text(out + "\n")
    install_ok = rc == 0 and "LiCSBAS install is OK" in out
    ok = (LICSBAS_ENV.is_dir() and chans == {"conda-forge"} and all(snaps.values())
          and install_ok)
    n_pkg = len(list((LICSBAS_ENV / "conda-meta").glob("*.json")))
    return ok, (f"env {LICSBAS_ENV} ({n_pkg} packages, channels {sorted(chans)}, python "
                f"{env.get('env', {}).get('python')}, gdal {env.get('env', {}).get('gdal')}); "
                f"hydro conda-meta before/after/final identical: {snaps['conda_meta']}; "
                f"pip list before/after/final identical: {snaps['pip_freeze']}; "
                f"LiCSBAS_check_install.py rc {rc}, 'install is OK': {install_ok}")


def s4a_2() -> tuple[bool, str, dict]:
    d = LICSBAS_DIR
    _, head = sh(["git", "rev-parse", "HEAD"], d)
    _, remote = sh(["git", "remote", "get-url", "origin"], d)
    _, status = sh(["git", "status", "--porcelain"], d)
    rc_fsck, fsck = sh(["git", "fsck", "--no-progress"], d)
    _, tracked = sh(["git", "ls-files", "--", rel_to_repo(d)], REPO)
    rc_ign, ign = sh(["git", "check-ignore", "-q", rel_to_repo(d / "README.md")], REPO)
    rec = {"commit": head, "origin": remote, "worktree_clean": status == "",
           "fsck_rc": rc_fsck, "fsck_output": fsck[:500], "tracked_by_repo": tracked != "",
           "ignored_by_repo": rc_ign == 0, "path": rel_to_repo(d)}
    ok = (remote.rstrip("/").removesuffix(".git").endswith("github.com/comet-licsar/LiCSBAS")
          and len(head) == 40 and rec["worktree_clean"] and rc_fsck == 0 and fsck == ""
          and not rec["tracked_by_repo"] and rec["ignored_by_repo"])
    return ok, (f"{remote} @ {head}; clean worktree {rec['worktree_clean']}; git fsck rc "
                f"{rc_fsck}, output {'none' if not fsck else fsck[:80]}; under gitignored "
                f"{rec['path']} (tracked files 0: {not rec['tracked_by_repo']})"), rec


def main() -> int:
    decl = json.loads(DECL.read_text())
    crit = {c["id"]: c for c in decl["thresholds"]["criteria"]}
    rows: list[dict] = []

    def add(cid: str, ok: bool, measured: str, part: str | None = None) -> None:
        c = crit[cid]
        rows.append({"id": cid + (f" ({part})" if part else ""), "criterion": c["criterion"],
                     "threshold": c["threshold"], "measured": measured, "passed": bool(ok)})

    ok, m = s4a_1()
    add("S4a-1", ok, m)
    ok, m, git_rec = s4a_2()
    add("S4a-2", ok, m)

    sm = jload("smoke_result.json")
    parts3 = []
    for name, r in sm.get("runs", {}).items():
        parts3.append(f"{name}: {r['n_pairs_linked']} pairs ({r['first_pair']}..{r['last_pair']}"
                      f"), {len(r['unw_georef_variants'])} unw georeference variant(s), steps "
                      f"exit 0 {len(r['steps_exit0'])}/5, cum.h5 {r['cum_h5'].get('cum_shape')} "
                      f"(epochs x rows x cols), passed {r['passed']}")
    add("S4a-3", bool(sm.get("passed")),
        "; ".join(parts3) + " (criterion on the single-variant run)" if sm
        else "smoke run missing")

    v = jload(f"validation_{PRIMARY}.json")
    n_av = v["n_pairs_listed"] - v["n_pairs_unavailable"]
    frac = v["n_pairs_fetched"] / n_av if n_av else 0.0
    net = v["network_fetched"]
    avail = [p for p in nw.window_pairs(nw.parse_pairs(
        Path(FRAMES[PRIMARY]["l0_dir"]) / "listings/interferograms.html"),
        FRAMES[PRIMARY]["start"], FRAMES[PRIMARY]["end"])
        if p not in {u["pair"] for u in v["unavailable_pairs"]}]
    na = nw.network_stats(avail)
    spans = (net.get("first_epoch") == na["first_epoch"]
             and net.get("last_epoch") == na["last_epoch"])
    ok4 = frac >= 0.99 and net.get("n_components") == 1 and spans and v["epochs_2022"] >= 25
    add("S4a-4", ok4,
        f"fetched {v['n_pairs_fetched']} / available {n_av} = {frac:.4f} (listed "
        f"{v['n_pairs_listed']}, unavailable on portal {v['n_pairs_unavailable']}); "
        f"components {net.get('n_components')}, {net.get('first_epoch')}..{net.get('last_epoch')}"
        f" (available network {na['first_epoch']}..{na['last_epoch']}, {na['n_epochs']} epochs);"
        f" fetched epochs {net.get('n_epochs')}; epochs in 2022 {v['epochs_2022']}")

    dec = jload("fetch_decision.json")
    fetched_frames = [f for f in FRAMES if (RUN_DIR / f"validation_{f}.json").exists()]
    probs = {f: jload(f"validation_{f}.json")["n_problems"] for f in fetched_frames}
    unfetched_avail = {f: jload(f"validation_{f}.json")["n_pairs_available_not_fetched"]
                       for f in fetched_frames}
    parts = sorted(str(p) for f in FRAMES for p in Path(FRAMES[f]["l0_dir"]).rglob("*.part"))
    src = {f: (Path(FRAMES[f]["l0_dir"]) / "SOURCE.json").is_file() for f in FRAMES}
    ck = jload("checker_licsar.json")
    cks = ck.get("summary", {})
    ok5 = (all(n == 0 for n in probs.values()) and not parts and all(src.values())
           and bool(ck.get("passed_l0")))
    add("S4a-5", ok5,
        f"validation problems {probs}; .part leftovers {len(parts)}; SOURCE.json per fetch "
        f"directory {src}; checker on licsar_*: {cks.get('sources')} SOURCE.json, failed "
        f"{cks.get('sources_failed')}, orphans {cks.get('orphans')}, files re-hashed "
        f"{cks.get('files_hashed')}")

    sc = jload("spot_checks.json")
    raw_batches = set()
    for f in fetched_frames:
        mt = nw.listing_mtimes(Path(FRAMES[f]["l0_dir"]) / "listings/interferograms.html")
        for line in (RUN_DIR / f"fetch_log_{f}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("reader") == "raw_multirange":
                raw_batches.add((f, mt.get(Path(r["file"]).parent.name, "?")[:7]))
    checked = {(c["frame"], c["batch"]) for c in sc.get("checks", [])
               if c["bitwise_equal"]}
    ok6 = bool(raw_batches) and raw_batches <= checked
    add("S4a-6", ok6,
        f"{len(checked)} of {len(raw_batches)} raw-reader batches bitwise equal on the first "
        f"and last 12 window rows ({sorted(b for _, b in raw_batches)}); "
        f"{sum(c['n_raw_files_in_batch'] for c in sc.get('checks', []))} raw-reader files")

    t = jload("transfer_tally.json")
    prim = t["items"].get(f"subsets_{PRIMARY}", {}).get("bytes", 0)
    prim_all = prim + t["items"].get("run1_aborted_clips", {}).get("bytes", 0)
    add("S4a-7", t["total_bytes"] <= 3.0e9,
        f"S4a total {t['total_bytes']/1e9:.3f} GB <= 3.0 GB", part="hard cap")
    add("S4a-7", prim <= 1.5e9,
        f"primary subsets {prim/1e9:.3f} GB ({prim_all/1e9:.3f} GB with the removed run-1 "
        f"clips) against D1's about 1.4 GB (fail above 1.5 GB)", part="D1 comparison")

    est = jload("size_estimate.json")["frames"]
    m8, ok8 = [], True
    for f in ("069A_06632_162010", "105D_06628_232108"):
        l0 = Path(FRAMES[f]["l0_dir"])
        listed = nw.window_pairs(nw.parse_pairs(l0 / "listings/interferograms.html"),
                                 FRAMES[f]["start"], FRAMES[f]["end"])
        pages = [p for p in listed if (l0 / "listings" / "ifg_dirs" / f"{p}.html").exists()]
        n_av8 = sum(all(nw.pair_products(l0 / "listings" / "ifg_dirs" / f"{p}.html", p,
                                         IFG_EXTS).values()) for p in pages)
        e = est.get(f, {})
        d = dec.get("frames", {}).get(f, {})
        fetched = f in fetched_frames
        ok_f = (len(pages) == len(listed) and e.get("expected_smart_clip_bytes") is not None
                and bool(d) and fetched == bool(d.get("fetch")))
        ok8 &= ok_f
        m8.append(f"{f}: {len(listed)} in-window pairs listed, availability read for "
                  f"{len(pages)}, {n_av8} with unw+cc; expected subset "
                  f"{e.get('expected_smart_clip_bytes', 0)/1e9:.3f} GB (full "
                  f"{e.get('expected_full_bytes', 0)/1e9:.2f} GB); rule: "
                  f"{'fetch' if d.get('fetch') else 'list only'}; fetched {fetched}")
    add("S4a-8", ok8, "; ".join(m8))

    g = v.get("gnss_summary", {})
    st = v.get("gnss_stations_in_box", [])
    good = sorted(s["station"] for s in st if s["off_fan"]
                  and s["valid_frac_preorigin_pairs"] >= 0.5)
    add("S4a-9", g.get("n_off_fan_valid50_preorigin", 0) >= 2,
        f"{g.get('n_off_fan_valid50_preorigin')} off-fan NGL stations valid in >= 50 % of "
        f"{g.get('n_sampled_preorigin')} sampled pre-origin pairs: {good} "
        f"({sum(s['off_fan'] for s in st)} of {len(st)} stations in the box are off-fan)")

    by_id: dict[str, bool] = {}
    for r in rows:
        k = r["id"].split(" ")[0]
        by_id[k] = by_id.get(k, True) and r["passed"]
    out = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "declaration": rel_to_repo(DECL), "declaration_created": decl["created"],
           "results": rows, "by_criterion": by_id,
           "passed": all(by_id.values()), "n_passed": sum(by_id.values()),
           "n_criteria": len(by_id),
           "data_gate": {"criteria": list(DATA_GATE),
                         "passed": all(by_id[c] for c in DATA_GATE)},
           "licsbas_git": git_rec,
           "unfetched_available_pairs": unfetched_avail}
    (RUN_DIR / "acceptance_result.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(f"{'PASS' if r['passed'] else 'FAIL'} {r['id']}: {r['measured']}")
    print(f"{out['n_passed']}/{out['n_criteria']} criteria pass; data gate "
          f"{out['data_gate']['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
