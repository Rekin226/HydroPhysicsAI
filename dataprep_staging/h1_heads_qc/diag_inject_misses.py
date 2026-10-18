"""After-the-fact diagnosis of the injected events that the H1a detector missed (reported only;
no parameter or threshold is changed on its result). Pre-origin data only.

For each missed injection in ``injection_O.json`` the injection is replayed exactly as in
``s5_inject.py`` (same samples shifted, base-run barometric loading held fixed) and the
detector's jumps within +-1 day of the injected jump and reversal are listed with their class
(offset / spike / neither / absent), their size and the persistence statistics, so that the
miss is attributed to: jump below 1 m, failed persistence, failed pairing, or other.

Usage: diag_inject_misses.py WORKDIR INJECTION_JSON OUT_JSON
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h1_heads_qc import baro, pipeline, rawio, screens  # noqa: E402
from h1_heads_qc import config as C  # noqa: E402

DAY = 86400


def tsec(x) -> int:
    return int(pd.Timestamp(x).value // 10**9)


def near(jumps, t0):
    return [{"t": str(screens.ts(J["t"])), "j": round(J["j"], 3), "cls": J["cls"],
             "post_minus_pre": round(J["post"] - J["pre"], 3)}
            for J in jumps if abs(J["t"] - t0) <= DAY]


def raw_jump(t, v, tau):
    """Raw sample difference at the injected time (non-sentinel samples)."""
    good = np.isfinite(v) & (np.abs(v) < C.SENTINEL_ABS)
    tt, vv = t[good], v[good]
    i = np.searchsorted(tt, tau)
    if i == 0 or i >= len(tt):
        return None
    return {"dv": float(vv[i] - vv[i - 1]), "dt_min": float((tt[i] - tt[i - 1]) / 60)}


def main():
    work, inj_json, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    d = json.loads(inj_json.read_text())
    R = pd.DataFrame(d["rows"])
    miss = R[(R.status == "injected") & (R.recovered.astype(float) == 0)]
    meta = pd.read_parquet(work / "station_meta_clean.parquet").set_index("sid")
    P = baro.load_pressure()
    out = []
    for _, e in miss.iterrows():
        res = pickle.loads((work / "stage1" / f"{e.sid}.pkl").read_bytes())
        segs = [{"start": x["start"], "end": x["end"], "a": x.get("a")}
                for x in res.get("be", {}).get("segments", [])]
        sh, _ = rawio.load_well(e.sid, keep_sentinels=True)
        t, v = screens.to_arrays(sh)
        tau0 = tsec(e.tau0)
        tau1 = tsec(e.tau1) if isinstance(e.tau1, str) else None
        good = np.isfinite(v) & (np.abs(v) < C.SENTINEL_ABS)
        m = good & (t >= tau0)
        if tau1 is not None:
            m &= t < tau1
        v2 = v.copy()
        v2[m] += e["size"]
        mrow = meta.loc[e.sid].to_dict()
        for key in ("WellElevation", "WellDepth"):
            mrow[key] = float(mrow[key]) if mrow.get(key) is not None and \
                np.isfinite(mrow.get(key)) else None
        o = pipeline.process_well(e.sid, mrow, P, pre_only=True,
                                  hist_override=pd.Series(v2, index=sh.index),
                                  be_override={"segments": segs,
                                               "p_ref_hpa": res.get("p_ref_hpa")}, o1=False)
        cat = o["pre_cat"]
        j0, j1 = near(cat["jumps"], tau0), near(cat["jumps"], tau1) if tau1 else []
        r0, r1 = raw_jump(t, v2, tau0), raw_jump(t, v2, tau1) if tau1 else None

        def why(js, rj):
            if any(x["cls"] == "offset" for x in js):
                return "offset_detected"
            if js:
                return "classified_" + "_".join(sorted({x["cls"] for x in js}))
            if rj is None:
                return "no_sample"
            if rj["dt_min"] > C.JUMP_MAX_DT_S / 60:
                return "gap_gt_70min"
            if abs(rj["dv"]) <= C.JUMP_M:
                return "raw_jump_le_1m"
            return "dead_time_or_persistence_window_short"
        w0, w1 = why(j0, r0), (why(j1, r1) if tau1 else None)
        if e.kind == "paired":
            if w0 == "offset_detected" and w1 == "offset_detected":
                cause = "both_jumps_offsets_but_not_paired"
            elif w0 != "offset_detected":
                cause = "first_jump_" + w0
            else:
                cause = "return_jump_" + w1
        else:
            cause = "jump_" + w0
        out.append({"real": int(e.real), "sid": e.sid, "kind": e.kind,
                    "size": round(float(e["size"]), 3), "tau0": e.tau0, "tau1": e.tau1,
                    "raw_jump_at_tau0": r0, "raw_jump_at_tau1": r1,
                    "jumps_near_tau0": j0, "jumps_near_tau1": j1,
                    "first": w0, "return": w1, "cause": cause})
        print(e.real, e.sid, e.kind, round(float(e["size"]), 2), cause, flush=True)
    causes = pd.Series([x["cause"] for x in out]).value_counts().to_dict()
    Path(out_json).write_text(json.dumps({"note": "after-the-fact diagnosis; reported only, "
                                                  "no parameter changed",
                                          "n_missed": len(out), "causes": causes,
                                          "rows": out}, indent=1, default=str))
    print(json.dumps(causes, indent=1))


if __name__ == "__main__":
    main()
