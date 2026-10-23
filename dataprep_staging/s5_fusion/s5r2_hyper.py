"""WP S5 repair round 2: hyperparameter procedure H2 (restricted marginal likelihood, data before
each run's origin only; U7). Registered in S5.repair_r2.v1.

Stages (Nelder-Mead in log space from s5r2_const.HYPER_START, first pass + one restart from the
best evaluated point; the best evaluated point is kept):

* A - leveling only: ell, sig_v0, sig_be, sig_ep, sig_s, sig_lev, sig_rb, sig_camp; one |z| > 5
  screening pass and a re-optimisation (restart budget);
* B - + GNSS (v4, gap segments): sig_gnss, sig_d, sig_cm, sig_gc, tau_g; one screening pass and
  a re-optimisation;
* C - + InSAR (run P2): sig_ins, sig_aps, ell_aps, sig_ir (skipped when there is no InSAR);
* D - depth fractions f from the A-C posterior (fit without rings and piles; round-1 rule), then
  + rings and piles: sig_cmp, sig_rc, sig_cc, tau_c; one screening pass and a re-optimisation.

Chains:
    main  pre-origin: leveling 2012-2021, GNSS <= 2022-12, run P2, rings <= 2021, piles <= 2022
          -> hyper_main.json (T1 second look B-D, T2, T3, the product)
    t1    stage A again on the leveling <= 2019-12 (main screening), B-D from main
          -> hyper_t1.json (the registered T1 design, second look)
    t1f   pseudo-origin 2020-01 (FRESH test): leveling <= 2017-12 only (2018-19 not loaded),
          GNSS, rings, piles <= 2019-12, no InSAR, ring interval from 2017-2019 coverage
          -> hyper_t1f.json

    tools/cpu_slot.sh python -I s5r2_hyper.py main|t1|t1f [--dry <dir>] [--cap N]
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
import s5fit as F  # noqa: E402
import s5r2_const as K  # noqa: E402
import s5r2_data as RD  # noqa: E402
import s5r2_model as RM  # noqa: E402

CAP = None          # dry runs: cap on every Nelder-Mead budget


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def budget(stage: str) -> tuple[int, int]:
    a, b = K.NM_MAXFEV[stage], K.NM_RESTART[stage]
    return (min(a, CAP), min(b, CAP)) if CAP else (a, b)


def load(chain: str):
    if chain == "t1f":
        return RD.load_all2("pre", lev_end=K.T1F_TRAIN_END, other_end=K.T1F_OTHER_END,
                            ring_win=("2017-01-01", K.T1F_OTHER_END))
    return RD.load_all2("pre")


def stage_a(data, h, knots, lev_keep=None):
    keep = np.ones(len(data["lev"]), bool) if lev_keep is None else lev_keep
    pr = RM.make_problem2(data, knots, ("lev",), masks={"lev": keep})
    log(f"stage A: {len(pr._lev_sites)} benchmarks, N {pr.N}")
    fa, ra = budget("A")
    h, i1 = RM.optimise2(pr, h, K.STAGES["A"], ("lev",), fa, ra, log=log)
    post = RM.fit2(pr, h)
    fl = RM.outliers2(pr, post, h, "lev")
    log(f"stage A screening: {len(fl)} surveys flagged")
    keep2 = keep & ~data["lev"]["rid"].isin(fl).to_numpy()
    pr2 = RM.make_problem2(data, knots, ("lev",), masks={"lev": keep2})
    h, i2 = RM.optimise2(pr2, h, K.STAGES["A"], ("lev",), ra, 0, log=log)
    return h, {"first": i1, "refit": i2, "n_flagged": len(fl)}, fl


def stages_bcd(data, h, knots, flagged, res, lev_keep=None):
    extra = {"lev": lev_keep} if lev_keep is not None else None
    fb, rb = budget("B")
    m = RD.masks_from(data, flagged, extra)
    pr = RM.make_problem2(data, knots, ("lev", "gnss"), masks=m)
    h, ib = RM.optimise2(pr, h, K.STAGES["B"], ("gnss",), fb, rb, log=log)
    post = RM.fit2(pr, h)
    flagged["gnss"] = RM.outliers2(pr, post, h, "gnss")
    log(f"B screening: {len(flagged['gnss'])} GNSS months flagged")
    m = RD.masks_from(data, flagged, extra)
    pr = RM.make_problem2(data, knots, ("lev", "gnss"), masks=m)
    h, ib2 = RM.optimise2(pr, h, K.STAGES["B"], ("gnss",), rb, 0, log=log)
    res["stages"]["B"] = {"first": ib, "refit": ib2}
    log(f"B done {json.dumps({k: round(h[k], 3) for k in K.STAGES['B']})}")
    fams_ac = ("lev", "gnss")
    if len(data["insar"]["groups"]):
        fams_ac = ("lev", "gnss", "insar")
        fc, rc = budget("C")
        pr = RM.make_problem2(data, knots, fams_ac, masks=m)
        h, ic = RM.optimise2(pr, h, K.STAGES["C"], ("insar",), fc, rc, log=log)
        res["stages"]["C"] = ic
        log(f"C done {json.dumps({k: round(h[k], 3) for k in K.STAGES['C']})}")
    else:
        res["stages"]["C"] = {"skipped": "no InSAR in this chain"}
    pr = RM.make_problem2(data, knots, fams_ac, masks=m)
    post = RM.fit2(pr, h)
    fd = F.estimate_f(pr, post, data["cmp"][m["cmp"]])
    fdict = {k: v["f"] for k, v in fd.items()}
    fdd, rdd = budget("D")
    pr = RM.make_problem2(data, knots, F.FAMILIES, masks=m, fdict=fdict)
    h, idd = RM.optimise2(pr, h, K.STAGES["D"], ("cmp",), fdd, rdd, log=log)
    post = RM.fit2(pr, h)
    flagged["cmp"] = RM.outliers2(pr, post, h, "cmp")
    log(f"D screening: {len(flagged['cmp'])} ring/pile values flagged")
    m = RD.masks_from(data, flagged, extra)
    pr = RM.make_problem2(data, knots, F.FAMILIES, masks=m, fdict=fdict)
    h, idd2 = RM.optimise2(pr, h, K.STAGES["D"], ("cmp",), rdd, 0, log=log)
    res["stages"]["D"] = {"first": idd, "refit": idd2}
    post = RM.fit2(pr, h)
    res.update({"hyper": h, "f": fd, "fdict": fdict, "flagged": flagged,
                "m2ll_final": post.m2loglik, "N": pr.N, "m_nodes": pr.m})
    return h


def run(chain: str, out: Path) -> dict:
    t0 = time.time()
    data, qc = load(chain)
    res = {"chain": chain, "qc": qc, "stages": {}}
    h = dict(K.HYPER_START)
    if chain == "t1":
        base = json.loads((out.parent / "hyper_main.json").read_text())
        keep = ~data["lev"]["rid"].isin(base["flagged"]["lev"]).to_numpy()
        keep &= data["lev"]["date"].astype(str).to_numpy() <= C.T1_BASE_WIN[1]
        h.update({k: base["hyper"][k] for k in base["hyper"] if k not in K.STAGES["A"]})
        h, ia, fl = stage_a(data, h, K.KNOTS_A, keep)
        res.update({"base": "hyper_main.json", "stages": {"A": ia}, "flagged_lev_t1": fl,
                    "hyper": h, "training_leveling_end": C.T1_BASE_WIN[1]})
    else:
        knots_a = K.KNOTS_T1F_A if chain == "t1f" else K.KNOTS_A
        knots = K.KNOTS_T1F if chain == "t1f" else C.KNOTS_PRE
        h, ia, fl = stage_a(data, h, knots_a)
        res["stages"]["A"] = ia
        log(f"A done {json.dumps({k: round(h[k], 3) for k in K.STAGES['A']})}")
        h = stages_bcd(data, h, knots, {"lev": fl}, res)
        res["fit_window_end"] = K.T1F_OTHER_END[:7] if chain == "t1f" else "2022-12"
        res["leveling_end"] = K.T1F_TRAIN_END if chain == "t1f" else C.LEVELING_END
    res["seconds"] = round(time.time() - t0, 1)
    res["at_bound"] = sorted({n for st in res["stages"].values() if isinstance(st, dict)
                              for part in (st.values() if "first" in st else [st])
                              if isinstance(part, dict) for n in part.get("at_bound", [])})
    F.dump(res, out)
    log(f"{chain} done in {res['seconds']} s -> {out}")
    return res


def main(argv=None) -> int:
    global CAP
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("chain", choices=["main", "t1", "t1f"])
    ap.add_argument("--dry", default=None, help="dry run on synthetic values into this dir")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    CAP = a.cap
    if a.dry:
        import s5r2_dryrun as DR
        DR.install()
        out = Path(a.dry) / f"hyper_{a.chain}.json"
    else:
        out = K.EVID / f"hyper_{a.chain}.json"
        if out.exists():
            raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    run(a.chain, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
