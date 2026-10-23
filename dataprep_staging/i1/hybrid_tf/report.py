"""Task 4 helper: render the tables of ``report.md`` and write ``report.json`` from the JSON
outputs of :mod:`.inventory`, :mod:`.evaluate` and :mod:`.declaration` (no number is typed by
hand). Writes ``tables.md`` and ``report.json`` in ``results/dataprep/reports/I1_hybrid_tf/``.

Run from the repository root:
    PYTHONPATH=dataprep_staging dataprep_staging/tools/cpu_slot.sh \
        ~/miniforge3/envs/hydro/bin/python -m i1.hybrid_tf.report
"""
from __future__ import annotations

import json

from . import config as C

LABEL = {
    "clim": "climatology", "persist": "persistence (origin head)",
    "persist1": "persistence (previous month)", "stat_ar1": "AR(1) layer (existing)",
    "stat_ar1_1": "AR(1) layer (existing)", "phys": "R0 physics", "phys_ar": "R0 + origin residual",
    "phys_ar1": "R0 + previous-month residual", "tf": "TF, all drivers recorded",
    "phys_tf": "R0 + TF, all drivers recorded", "tf_fc": "TF, forecast drivers",
    "phys_tf_fc": "R0 + TF, forecast drivers (optimistic*)",
    "tf_fc_elec": "TF, forecast drivers + electricity recorded",
    "tf_fc_hold": "TF, forecast drivers, hold-last fill",
    "tf_ar1": "TF + residual, all recorded", "phys_tf_ar1": "R0 + TF + residual, all recorded",
    "arx1": "ARX, all drivers recorded", "arx_fc1_1": "ARX, forecast drivers (rain recorded)",
    "arx_fc_1": "ARX, forecast drivers (nothing recorded)",
    "arx_fc1elec_1": "ARX, forecast drivers + electricity recorded",
}
KEY = {"seasonal": ("clim", "persist", "stat_ar1", "phys", "phys_ar", "tf", "phys_tf", "tf_fc",
                    "phys_tf_fc", "tf_fc_elec", "tf_fc_hold"),
       "onestep": ("clim", "persist1", "stat_ar1_1", "phys_ar1", "tf_ar1", "phys_tf_ar1",
                   "arx1", "arx_fc1_1", "arx_fc_1", "arx_fc1elec_1")}
SPLIT = {"loyo": "leave one year out", "fwd": "forward chaining (forecast mimic)"}
SETTING = {"seasonal": "frozen at the origin, no head after it",
           "onestep": "one month ahead, previous head known"}


def skill_table(design: dict, split: str, setting: str, models) -> str:
    bc = design["splits"][split][setting]["by_county"]
    ref = {"seasonal": "stat_ar1", "onestep": "stat_ar1_1"}[setting]
    out = [f"**{SPLIT[split]}, {SETTING[setting]}** (Changhua {bc['changhua']['clim']['n_wells']}"
           f" wells, {bc['changhua']['clim']['n_months']} well-months; Yunlin "
           f"{bc['yunlin']['clim']['n_wells']} wells, {bc['yunlin']['clim']['n_months']})", "",
           "| model | CH median R2 | CH RMSE (m) | CH ratio to clim | CH ratio to AR(1) "
           "| YL median R2 | YL RMSE (m) | YL ratio to clim | YL ratio to AR(1) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        cells = [LABEL.get(m, m) + f" (`{m}`)"]
        for c in ("changhua", "yunlin"):
            r, s = bc[c][m], bc[c][ref]
            cells += [f"{r['median_r2']:.2f}", f"{r['pooled_rmse_m']:.2f}",
                      f"{r['rmse_ratio_vs_clim']:.2f}",
                      f"{r['rmse_ratio_vs_clim'] / s['rmse_ratio_vs_clim']:.2f}"]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def availability_table(av: dict) -> str:
    out = ["| driver | source | allowed uses | last month present | last month recorded "
           "| months missing before 2026-11 |", "|---|---|---|---|---|---|"]
    for r in av["drivers"]:
        uses = r["allowed_uses"] if isinstance(r["allowed_uses"], list) else [r["allowed_uses"]]
        out.append(f"| {r['driver']} | {r['product']} | {', '.join(uses)} | "
                   f"{r['last_present']} | {r['last_recorded'] or 'not usable'} | "
                   f"{r['months_missing_before_2026_11'] if r['last_recorded'] else '-'} |")
    h = av["heads"]
    out.append(f"| heads (last fetch) | {h['source']} ({h['dataset']}) | timestamps only | "
               f"{h['last_month']} | {h['last_month']} ({h['wells_with_data_in_last_month']} "
               f"of {h['n_files']} wells) | - |")
    return "\n".join(out)


def looks() -> list[dict]:
    """The looks this study logged (harness looks.jsonl, notes naming I1_hybrid_tf)."""
    out = []
    for line in (C.REPO / "results/dataprep/preregistration/looks.jsonl").read_text(
            encoding="utf-8").splitlines():
        r = json.loads(line)
        if "I1_hybrid_tf" in r.get("note", ""):
            out.append({k: r[k] for k in ("time", "wp", "units", "data_scope", "purpose")})
    return out


def main() -> None:
    design = json.loads((C.OUT / "design_skill.json").read_text())
    av = json.loads((C.OUT / "availability.json").read_text())
    parts = ["## Driver availability", "", availability_table(av), ""]
    for split in ("fwd", "loyo"):
        for setting in ("seasonal", "onestep"):
            parts += [skill_table(design, split, setting, KEY[setting]), ""]
    parts += ["## All models", ""]
    for split in ("fwd", "loyo"):
        for setting in ("seasonal", "onestep"):
            models = list(design["splits"][split][setting]["by_county"]["changhua"])
            parts += [skill_table(design, split, setting, models), ""]
    (C.OUT / "tables.md").write_text("\n".join(parts), encoding="utf-8")
    summary = {}
    for split in ("fwd", "loyo"):
        for setting in ("seasonal", "onestep"):
            bc = design["splits"][split][setting]["by_county"]
            ref = {"seasonal": "stat_ar1", "onestep": "stat_ar1_1"}[setting]
            summary[f"{split}:{setting}"] = {
                c: {m: {"median_r2": round(bc[c][m]["median_r2"], 3),
                        "pooled_rmse_m": round(bc[c][m]["pooled_rmse_m"], 3),
                        "ratio_to_clim": round(bc[c][m]["rmse_ratio_vs_clim"], 3),
                        "ratio_to_ar1": round(bc[c][m]["rmse_ratio_vs_clim"]
                                              / bc[c][ref]["rmse_ratio_vs_clim"], 3),
                        "share_wells_beat_clim": round(bc[c][m]["share_wells_beat_clim"], 3)}
                    for m in KEY[setting]} for c in ("changhua", "yunlin")}
    draft = sorted(C.DRAFT.glob("*.draft.json"))
    rec = {"schema": "i1-hybrid-tf-report/1", "date": "2026-10-20",
           "role": "design-time evidence and an inventory; the prospective test is a draft, "
                   "not registered",
           "availability": {r["driver"]: {"last_recorded": r["last_recorded"],
                                          "months_missing_before_2026_11":
                                              r["months_missing_before_2026_11"],
                                          "allowed_uses": r["allowed_uses"]}
                            for r in av["drivers"]},
           "heads_last_fetched_month": av["heads"]["last_month"],
           "design_skill": summary,
           "draft_declaration": str(draft[0].relative_to(C.REPO)) if draft else None,
           "looks_logged": looks(),
           "files": ["availability.json", "availability.csv", "design_skill.json",
                     "design_wells.csv", "data_roles.csv", "tables.md",
                     "draft_declaration/"]}
    (C.OUT / "report.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote tables.md and report.json")


if __name__ == "__main__":
    main()
