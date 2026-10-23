"""Design-time exploration for the P2b repair (look [P2-r3-explore-2025], logged before this
runs). Full variant only (C0 full v3), all counties. No product value depends on this script;
it informs the masking rule registered in P2abc.r3.repair.v1.

Per county, billing group (A = code M2 & detected phase A, B = code M1 & detected phase B,
monthly = detected phase monthly) and billing pair (or month) 2012..2026:
  total      pair-sum energy of the group (GWh)
  zshare     share of zero bills among active meters (previous and next bill positive)
  next_ratio for those zero-bill meters: sum of their next bill / sum of their own mean of the
             same next bill over the three previous years (about 1 if the zero is idling or a
             lost bill, about 2 if the missed bill was merged into the next one)
  next_ratio_nz the same ratio for the active meters with a positive bill

Usage: python -I x1_explore_2025.py <out_json>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from p3lib import L  # noqa: E402


def pair_table(X: np.ndarray, starts: np.ndarray, width: int) -> dict:
    """X (n, T) zero-filled; bills of ``width`` months starting at ``starts``."""
    B = np.stack([X[:, s:s + width].sum(1) for s in starts], axis=1)     # (n, K)
    out = {}
    for k in range(3, len(starts) - 1):
        s = starts[k]
        prev_, cur, nxt = B[:, k - 1], B[:, k], B[:, k + 1]
        act = (prev_ > 0) & (nxt > 0)
        z = act & (cur == 0)
        # own mean of the same next bill over the three previous years
        per_year = 12 // width
        hist = [k + 1 - per_year * j for j in (1, 2, 3)]
        if min(hist) < 0:
            nr = nr_nz = np.nan
        else:
            hm = np.mean(B[:, hist], axis=1)
            nr = float(nxt[z].sum() / hm[z].sum()) if z.any() and hm[z].sum() > 0 else np.nan
            nzm = act & (cur > 0)
            nr_nz = float(nxt[nzm].sum() / hm[nzm].sum()) if hm[nzm].sum() > 0 else np.nan
        out[L.ym(s)] = {"total_GWh": float(cur.sum() / 1e6),
                        "n_active": int(act.sum()),
                        "zshare": float(z.sum() / max(act.sum(), 1)),
                        "next_ratio_zero": nr, "next_ratio_nonzero": nr_nz}
    return out


def main(out_json: str) -> None:
    prod = L.open_c0("full")
    res = {}
    for c in L.COUNTIES:
        cd = L.CountyData("full", c, prod=prod)
        T = cd.T
        X = np.nan_to_num(cd.M[:, :T - 1].astype(np.float64))     # drop partial 2026-07
        code, ph = cd.code, cd.phase
        groups = {"A": (code == "M2") & (ph == "A"), "B": (code == "M1") & (ph == "B"),
                  "monthly_M0": (code == "M0") & (ph == "monthly"),
                  "monthly_M9": (code == "M9") & (ph == "monthly")}
        r = {"n": {g: int(s.sum()) for g, s in groups.items()}}
        t12 = L.tidx("2012-01")
        for g, sel in groups.items():
            if g in ("A", "B"):
                s0 = 0 if g == "A" else 1
                st = np.arange(s0, T - 2, 2)
                st = st[st >= t12 - 8]
                tab = pair_table(X[sel], st, 2)
            else:
                st = np.arange(t12 - 40, T - 1)
                tab = pair_table(X[sel], st, 1)
            r[g] = {k: v for k, v in tab.items() if k >= "2012-01"}
        # monthly-meter month ratios
        for g in ("monthly_M0", "monthly_M9"):
            Z = X[groups[g]].sum(0)
            r[g + "_ratios"] = {str(y): {"jul_over_jun": float(Z[L.tidx(f"{y}-07")]
                                                                / Z[L.tidx(f"{y}-06")]),
                                         "aug_over_jul": float(Z[L.tidx(f"{y}-08")]
                                                               / Z[L.tidx(f"{y}-07")]),
                                         "sep_over_aug": float(Z[L.tidx(f"{y}-09")]
                                                               / Z[L.tidx(f"{y}-08")])}
                                for y in range(2012, 2026)}
        # C0 method counts by month 2025-01..2026-07 for group A/B
        r["c0_methods_2025_26"] = {}
        for g in ("A", "B"):
            sel = groups[g]
            r["c0_methods_2025_26"][g] = {
                L.ym(t): {L.METHODS[int(m)]: int((cd.Mm[sel, t] == m).sum())
                          for m in np.unique(cd.Mm[sel, t])}
                for t in range(L.tidx("2025-01"), T)}
        res[c] = r
        print(c, r["n"], flush=True)
    L.save_json(res, out_json)


if __name__ == "__main__":
    main(sys.argv[1])
