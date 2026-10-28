"""The one exploratory follow-up of X5.shutdown_sdid.v1 after its synthetic gate failed.

Label: post-failure, exploratory; never validation; cannot promote.

Fixed after the synthetic failure and BEFORE the real-data report existed: the synthetic
check showed the block-placebo se too small (scenario A0: sd of tau over panels / mean
placebo se = k). Here every reported T2/T1 estimate is re-read with se * k and a normal
reference: p_one = Phi(tau / (k se)), p_two = 2 Phi(-|tau| / (k se)). Nothing is re-estimated.

Usage: python explore_x5.py   (writes OUT_DIR/exploratory_calibrated_inference.json once)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x5const as K  # noqa: E402


def main() -> int:
    out = K.OUT_DIR / "exploratory_calibrated_inference.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the follow-up runs once")
    syn = json.loads((K.OUT_DIR / "synthetic/synthetic_check.json").read_text())
    rep = json.loads((K.REPORT_DIR / "report.json").read_text())
    k = syn["scenario_A0"]["tau_sd"] / syn["scenario_A"]["se_mean"]

    def recal(r: dict) -> dict:
        se = r["se_placebo"] * k
        return {"tau": r["tau"], "se_calibrated": se,
                "ci90": [r["tau"] - K.Z90 * se, r["tau"] + K.Z90 * se],
                "p_one_sided_less_sinking": float(norm.cdf(r["tau"] / se)),
                "p_two_sided": float(2 * norm.cdf(-abs(r["tau"]) / se))}

    T2, T1 = rep["T2"], rep["T1"]
    res = {"label": "post-failure, exploratory; never validation; cannot promote",
           "k_se_inflation": k,
           "k_source": "synthetic_check.json: scenario_A0.tau_sd / scenario_A.se_mean",
           "T2_primary": recal(T2["primary"]), "T2_lead_2018": recal(T2["lead_placebo_2018"]),
           "T2_post_2019_2020": recal(T2["post_2019_2020"]),
           "T1_post_2019_2021": recal(T1["post_2019_2021"]),
           "T1_lead_2018": recal(T1["lead_placebo_2018"])}
    p, ld = res["T2_primary"], res["T2_lead_2018"]
    res["would_meet_declared_rule_with_calibrated_se"] = bool(
        p["tau"] < 0 and p["p_one_sided_less_sinking"] < K.P_SUCCESS
        and ld["p_two_sided"] >= K.P_LEAD_NULL)
    txt = json.dumps(res, indent=1)
    print(txt)
    with open(out, "x", encoding="utf-8") as f:
        f.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
