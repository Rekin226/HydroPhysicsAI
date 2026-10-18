"""Amendment of the repair-round pre-registration of WP P2abc (A1 tool ``harness.prereg3``):
P2abc.r3.repair.v2 supersedes P2abc.r3.repair.v1 before any of v1's tests was scored.

Why: building the zero tables after v1 (values only; look [P2-r3-zeros-build-review]) showed
that the registered classifier's window-end rule (a trailing run must end at the window's last
month) makes every trailing run interior for a meter whose last window month is not an
observation. In the pre-origin product that month (2022-12) is C0's flag-4 fill for the
Dec-Jan-paired meters (about 44k Yunlin and 51k Changhua meters), so their terminations would
be classed long idle (Yunlin pre-origin terminated zero-months 128,907 against round 2's
260,542). Fix (zeros3.classify_core): a meter's window is trimmed to its first..last observed
month inside the window before runs are classified; identical to the pilot's rule where the edge
months are observations. Nothing else changes: the tests, metrics, thresholds and gates are
v1's, under new test ids (a changed design needs new ids).

Usage: python -I prereg_r3b.py [--dry-run]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import p3lib as P3  # noqa: E402
import prereg_r3 as R3  # noqa: E402
import prereg_v2 as P  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg2 import read_chained  # noqa: E402
from p3lib import L  # noqa: E402

PREREG_ID = "P2abc.r3.repair.v2"
V1 = "P2abc.r3.repair.v1"
RENAME = {"P2r3.c1.": "P2r3b.c1.", "P2r3.k1.": "P2r3b.k1.", "P2r3.b2.": "P2r3b.b2."}
UNIT_RENAME = {"P2r3:zero_shipped_yunlin": "P2r3b:zero_shipped_yunlin",
               "P2r3:assembled3_preorigin": "P2r3b:assembled_preorigin",
               "P2r3:assembled3_full": "P2r3b:assembled_full",
               **{f"P2r3:july_member_units_{c}": f"P2r3b:july_member_units_{c}"
                  for c in L.COUNTIES}}


def _ren(s: str) -> str:
    for a, b in RENAME.items():
        s = s.replace(a, b)
    return s


def tests() -> list[dict]:
    out = []
    for t in R3.tests():
        t = copy.deepcopy(t)
        t["test_id"] = _ren(t["test_id"])
        t["gating"] = _ren(t["gating"])
        t["fresh_units"] = [UNIT_RENAME[u] for u in t["fresh_units"]]
        ext = {}
        for u, m in t["external_units"].items():
            nu = UNIT_RENAME.get(u, u)
            m = dict(m)
            if nu != u:
                m["description"] = m["description"].replace("this registration",
                                                            "P2abc.r3.repair.v2")
            ext[nu] = m
        t["external_units"] = ext
        if t["test_id"].startswith("P2r3b.k1."):
            t["design_time_units"] = t.get("design_time_units", []) + [
                "P2r3:zero_tables_products"]
            t["external_units"]["P2r3:zero_tables_products"] = P.none_unit(
                "first build of the repair-round zero tables (classifier of v1), read in look "
                "[P2-r3-zeros-build-review]; rebuilt after this registration")
        if t["test_id"].startswith("P2r3b.c1."):
            t["what"] = t["what"].replace(
                "(zeros3.classify_core,", "(zeros3.classify_core with the window-edge "
                "amendment: a meter's window trimmed to its first..last observed month,")
        out.append(t)
    return out


def doc() -> dict:
    d = R3.doc()
    d["id"] = PREREG_ID
    d["title"] = "Amendment (before scoring) of " + V1 + ": " + d["title"]
    d["tests"] = tests()
    d["fresh_units"] = sorted({u for t in d["tests"] for u in t["fresh_units"]})
    d["thresholds"] = {t["test_id"]: t["threshold"] for t in d["tests"]}
    d["gate_scores"] = {"not_applicable": _ren(d["gate_scores"]["not_applicable"])}
    d["primary_model"] = d["primary_model"].replace(
        "(billing pilot s05(e) run kinds on the observed matrix;",
        "(billing pilot s05(e) run kinds on the observed matrix, each meter's window trimmed "
        "to its first..last observed month;")
    v1 = [r for r in read_chained(prereg3.PREREG_DIR / "INDEX.jsonl") if r["id"] == V1]
    scored = [lk for lk in read_chained(prereg3.PREREG_DIR / "looks.jsonl")
              if lk.get("prereg_id") == V1 and lk.get("purpose") == "score"]
    d["interpretation"]["amendment"] = {
        "supersedes": V1, "v1_sha256": v1[0]["sha256"] if v1 else None,
        "v1_tests_scored": len(scored),
        "why": "after v1, building the zero tables (values only) showed that the window-end "
               "rule makes every trailing run interior for a meter whose last window month is "
               "not an observation; in the pre-origin product 2022-12 is C0's flag-4 fill for "
               "the Dec-Jan-paired meters (Yunlin pre-origin terminated zero-months 128,907 vs "
               "round 2's 260,542; look [P2-r3-zeros-build-review])",
        "change": "zeros3.classify_core: a meter's window is trimmed to its first..last "
                  "observed month inside the window (leading: starts at the first observed "
                  "month and the first use lies after it; trailing: ends at the last observed "
                  "month and the last use lies before it); identical to the pilot's rule where "
                  "both edge months are observations",
        "unchanged": "tests, metrics, thresholds, gates, units' definitions and every other "
                     "rule of v1; new test ids P2r3b.* and fresh unit ids P2r3b:*",
        "known_before_this_registration": "the first build's class counts and screened months "
                                          "per county and variant (zeros3 build log); no "
                                          "check of v1 or v2 has been computed"}
    d["interpretation"]["code_at_registration"] = P3.v3_code()
    return d


def main(dry: bool) -> None:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("\n".join(errs))
    if dry:
        print(json.dumps({k: (v if k != "tests" else [(t["test_id"], t["fresh_units"])
                                                      for t in v])
                          for k, v in d.items() if k not in ("inputs", "primary_model")},
                         ensure_ascii=False, indent=1)[:5000])
        return
    path, sha = prereg3.write_prereg(d)
    print(path, sha)


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
