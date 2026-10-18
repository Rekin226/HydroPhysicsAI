"""WP S5: hyperparameters by restricted marginal likelihood on PRE-ORIGIN data only (U7).

Registered procedure (S5.subsidence_fusion.v1, procedure H):

* Stage A - leveling 2012-2021 only, knots 2012..2022: ell, sig_v0, q, sig_s, sig_lev, sig_rb
  (Nelder-Mead, log space, from HYPER_START, maxfev 220); one screening pass (|z| > 5 after
  the local offset and random velocity) and a re-fit from the optimum (maxfev 80).
* Stage B - + gnss_clean/preorigin/v2 (flags 0-1), knots 2012..2023: sig_gnss, sig_d, sig_cm
  (maxfev 120); one GNSS screening pass and a re-fit (maxfev 60).
* Stage C - + InSAR run P (pre-origin pairs): sig_ins, sig_aps, ell_aps (maxfev 120).
* Stage D - depth fractions f of rings and piles from the A-C posterior (fit without them),
  then + rings (<= 2021) and printed piles (<= 2022-12): sig_cmp, sig_rc (maxfev 80); one
  screening pass and a re-fit (maxfev 40).

    tools/cpu_slot.sh python -I s5hyper.py main          -> S5_evidence/hyper_pre.json
    tools/cpu_slot.sh python -I s5hyper.py t1            -> S5_evidence/hyper_t1.json
       (T1: stage A again on the training leveling, surveys <= 2019-12-31; B-D kept)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import s5const as C  # noqa: E402
import s5data as D  # noqa: E402
import s5fit as F  # noqa: E402

KNOTS_A = tuple(float(y) for y in range(2012, 2023))     # leveling ends 2021-09


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load(variant: str) -> tuple[dict, dict]:
    data, qc = D.load_all(variant)
    for k in ("lev", "gnss", "cmp"):
        data[k] = data[k].copy()
        data[k]["rid"] = np.arange(len(data[k]))
    return data, qc


def masks_from(data: dict, flagged: dict, extra: dict | None = None) -> dict:
    m = {}
    for fam in ("lev", "gnss", "cmp"):
        keep = ~data[fam]["rid"].isin(flagged.get(fam, [])).to_numpy()
        if extra and fam in extra:
            keep &= np.asarray(extra[fam], bool)
        m[fam] = keep
    if extra and "insar" in extra:
        m["insar"] = extra["insar"]
    return m


def stage_a(data: dict, h: dict, lev_keep: np.ndarray | None = None) -> tuple[dict, dict, list]:
    keep = np.ones(len(data["lev"]), bool) if lev_keep is None else lev_keep
    pr = F.make_problem(data, KNOTS_A, ("lev",), masks={"lev": keep})
    log(f"stage A: {len(pr._lev_sites)} benchmarks, N {pr.N}")
    h, info = F.optimise(pr, h, C.STAGES["A"], ("lev",), C.NM_MAXFEV["A"], log=log)
    post = F.fit(pr, h)
    flagged = F.outliers(pr, post, h, "lev")
    log(f"stage A screening: {len(flagged)} surveys flagged")
    keep2 = keep & ~data["lev"]["rid"].isin(flagged).to_numpy()
    pr2 = F.make_problem(data, KNOTS_A, ("lev",), masks={"lev": keep2})
    h, info2 = F.optimise(pr2, h, C.STAGES["A"], ("lev",), 80, log=log)
    return h, {"first": info, "refit": info2, "n_flagged": len(flagged)}, flagged


def main_run(out: Path) -> dict:
    t0 = time.time()
    data, qc = load("pre")
    h = dict(C.HYPER_START)
    res = {"variant": "pre", "qc": qc, "stages": {}, "flagged": {}}
    h, ia, fl = stage_a(data, h)
    res["stages"]["A"] = ia
    res["flagged"]["lev"] = fl
    log(f"A done {json.dumps({k: round(h[k], 3) for k in C.STAGES['A']})}")
    # ---- B
    m = masks_from(data, res["flagged"])
    pr = F.make_problem(data, C.KNOTS_PRE, ("lev", "gnss"), masks=m)
    h, ib = F.optimise(pr, h, C.STAGES["B"], ("gnss",), C.NM_MAXFEV["B"], log=log)
    post = F.fit(pr, h)
    res["flagged"]["gnss"] = F.outliers(pr, post, h, "gnss")
    log(f"B screening: {len(res['flagged']['gnss'])} GNSS months flagged")
    m = masks_from(data, res["flagged"])
    pr = F.make_problem(data, C.KNOTS_PRE, ("lev", "gnss"), masks=m)
    h, ib2 = F.optimise(pr, h, C.STAGES["B"], ("gnss",), 60, log=log)
    res["stages"]["B"] = {"first": ib, "refit": ib2}
    log(f"B done {json.dumps({k: round(h[k], 3) for k in C.STAGES['B']})}")
    # ---- C
    pr = F.make_problem(data, C.KNOTS_PRE, ("lev", "gnss", "insar"), masks=m)
    h, ic = F.optimise(pr, h, C.STAGES["C"], ("insar",), C.NM_MAXFEV["C"], log=log)
    res["stages"]["C"] = ic
    log(f"C done {json.dumps({k: round(h[k], 3) for k in C.STAGES['C']})}")
    # ---- D
    post = F.fit(pr, h)
    fd = F.estimate_f(pr, post, data["cmp"][m["cmp"]])
    fdict = {k: v["f"] for k, v in fd.items()}
    pr = F.make_problem(data, C.KNOTS_PRE, F.FAMILIES, masks=m, fdict=fdict)
    h, idd = F.optimise(pr, h, C.STAGES["D"], ("cmp",), C.NM_MAXFEV["D"], log=log)
    post = F.fit(pr, h)
    res["flagged"]["cmp"] = F.outliers(pr, post, h, "cmp")
    log(f"D screening: {len(res['flagged']['cmp'])} ring/pile values flagged")
    m = masks_from(data, res["flagged"])
    pr = F.make_problem(data, C.KNOTS_PRE, F.FAMILIES, masks=m, fdict=fdict)
    h, idd2 = F.optimise(pr, h, C.STAGES["D"], ("cmp",), 40, log=log)
    res["stages"]["D"] = {"first": idd, "refit": idd2}
    post = F.fit(pr, h)
    res.update({"hyper": h, "f": fd, "fdict": fdict, "m2ll_final": post.m2loglik,
                "N": pr.N, "m_nodes": pr.m, "seconds": round(time.time() - t0, 1),
                "fit_window_end": "2022-12"})
    F.dump(res, out)
    log(f"done in {res['seconds']} s -> {out}")
    return res


def t1_run(out: Path) -> dict:
    """Stage A on the T1 training leveling (surveys <= 2019-12-31, main screening applied)."""
    t0 = time.time()
    base = json.loads((C.EVIDENCE / "hyper_pre.json").read_text())
    data, _ = load("pre")
    keep = ~data["lev"]["rid"].isin(base["flagged"]["lev"]).to_numpy()
    keep &= data["lev"]["date"].to_numpy() <= C.T1_BASE_WIN[1]
    h = dict(C.HYPER_START)
    h.update({k: base["hyper"][k] for k in base["hyper"] if k not in C.STAGES["A"]})
    h, ia, fl = stage_a(data, h, keep)
    res = {"base": "hyper_pre.json", "stage_A": ia, "flagged_lev_t1": fl, "hyper": h,
           "seconds": round(time.time() - t0, 1), "training_leveling_end": C.T1_BASE_WIN[1]}
    F.dump(res, out)
    log(f"T1 stage A done in {res['seconds']} s -> {out}")
    return res


def main(argv=None) -> int:
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["main", "t1"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.what == "main":
        main_run(Path(a.out) if a.out else C.EVIDENCE / "hyper_pre.json")
    else:
        t1_run(Path(a.out) if a.out else C.EVIDENCE / "hyper_t1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
