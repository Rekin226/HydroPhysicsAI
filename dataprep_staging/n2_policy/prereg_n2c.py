"""Write N2C.changhua_response.prospective_2026-11.v1 (harness.prereg3) BEFORE the window opens.

The prospective test of R1c's Changhua HEAD response to RECORDED pumping anomalies of both
counties, 2026-11..2027-10 (plan S3 of the changhua_volume gap plan, 2026-10-28, revised
by its review's must-fix list). The page model carries no Changhua pumping, so N2 holds the
74 Changhua wells report-only and allowed a Changhua arm only by a separate registration
written before 2026-10-31 16:00 UTC. User rulings 2026-10-29: R1c, which FAILED its Changhua
head gate (I1.R1c) and the G2 fresh-partition gates, may be the PRIMARY model of this test,
with its failures disclosed below; the N2 'companion' is dropped (this is not it).

Frozen here (nothing is fitted): R1c's ZR predictions registered with ``twin.prospective
register`` on the existing forward ``datum_gate_ccz.npz`` (N2's forward flags), and the
report-only arms q0c and x5r1 likewise (results/twin/prospective_2026-11_n2c_<arm>/); the
both-county per-cell multiplier builder ``build_r_both.py``; the forward command; the
decision rule (N2's beta_response rule on the Changhua wells).

Every input is pinned by the sha256 it had when this draft was reviewed (FROZEN_SHA256,
CODE_SHA256 below); the script refuses to register if any current hash differs, if an input
is added or dropped, or if its own sha256 is not the one the lead passes.

Usage: python -I prereg_n2c.py                                   (validate and dry-run)
       python -I prereg_n2c.py --register --expect-self-sha <sha256 of this file>
       (the lead registers -- this script never logs a look)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
ID = "N2C.changhua_response.prospective_2026-11.v1"
TEST_ID = "N2CP.heads_changhua_response"
UNIT = "N2C:heads_changhua_prospective_2026-11_2027-10"
DEADLINE = dt.datetime(2026, 10, 31, 16, 0, tzinfo=dt.timezone.utc)
HERE = Path(__file__).resolve().parent
N2 = REPO / "results/dataprep/preregistration/N2.policy_response.prospective_2026-11.v1.json"
N2_ZR = REPO / "results/twin/prospective_2026-11/predictions.csv"
REH = REPO / "results/dataprep/reports/N2C_policy/rehearsal"
CODE_COMMIT = "1b08fb6"                     # main, committed (merge of PR #56)
CODE_FILES = ("hydrophysics/twin/forward.py", "hydrophysics/twin/recorded_pumping.py",
              "hydrophysics/twin/scenario.py", "hydrophysics/twin/prospective.py",
              "hydrophysics/twin/coverage.py", "hydrophysics/twin/zone_county.py",
              "hydrophysics/twin/inputs.py", "tests/test_twin_recorded_pumping.py")
X1 = "results/dataprep/electricity_x1_tpc/full/v1"
COUNTY_MAP = "results/dataprep/zone_county_map/v2/zone_county_map.npz"
ARMS = {  # arm -> (theta dir, forward npz, role)
    "r1c": ("results/twin_runs/stage3_datum_gate_ccz", "results/twin_forward/datum_gate_ccz.npz",
            "primary"),
    "q0c": ("results/twin_runs/stage3_datum_gate_q0c", "results/twin_forward/datum_gate_q0c.npz",
            "report-only"),
    "x5r1": ("results/twin_runs/stage3_datum_gate_x5r1",
             "results/twin_forward/datum_gate_x5r1.npz", "report-only"),
}
EVIDENCE = {
    "i1_r1c_prereg": "results/dataprep/preregistration/I1.R1c.county_zones.v1.json",
    "g2_report": "results/dataprep/reports/I1_G2_fresh_partition/report.json",
    "step_c_results": "results/dataprep/reports/N1_policy/step_c/results.json",
    "units_registry": "results/dataprep/validation_units/v3/units.json",
}
# Draft-time sha256 of every input in inputs_detail (path -> sha256), frozen 2026-10-29 after
# review (the 27 hashes the review re-checked, the 18 it listed as unreported, and the 2 R1c
# ensemble theta files the ensemble secondary needs). --register
# refuses if a current hash differs or if the set of inputs is not exactly this set.
FROZEN_SHA256 = {
    "dataprep_staging/n2_policy/build_r.py":
        "9ed2793fb562c8c58d73ea70facdfce12b1d654921ada39013f718f90cd8ab13",
    "dataprep_staging/n2_policy/build_r_both.py":
        "7d498b9c8b4054b9813a4e7460ae9fe3e8fefd9a6e30dc683fc89fb89feac734",
    "dataprep_staging/n2_policy/rehearse_n2c.py":
        "9df712c17c4eb8f23041e6e31386ba5ca6ae13bd0d1a5cd3acfe65a90f908bc8",
    "results/dataprep/electricity_x1_tpc/full/v1/MANIFEST.json":
        "921b9193b42d48d859e10d14145398b4ed5714e539309360087f4c2cdd1945d4",
    "results/dataprep/electricity_x1_tpc/full/v1/county_monthly.parquet":
        "75a423a0bd2263ab6af50a4298c00888e2b5192db2dd1cb338c8270e8850a5d5",
    "results/dataprep/preregistration/I1.R1c.county_zones.v1.json":
        "0686d88478b6e4e5d56bc192eb77ac3e347777056090999e0be1ee078c5abf76",
    "results/dataprep/preregistration/N2.policy_response.prospective_2026-11.v1.json":
        "bfbbd6b3b5e0580c8b4e9025c18db966a7f728420f5a2d07fb4ccb047cff6b0c",
    "results/dataprep/reports/I1_G2_fresh_partition/report.json":
        "e94d4240e488bbd5ab7911f1ba52cca709f91a0780e1d3ab8a90e0f2a71fe855",
    "results/dataprep/reports/N1_policy/step_c/results.json":
        "edcd50314c959fedaaaa1a120b97d5bb02f7d63a0350880faddcc577a1e9b063",
    "results/dataprep/reports/N2C_policy/rehearsal/ENGINEERING_LOG.txt":
        "aae0194cb0da00c42b7d33506e89a7155a950b1502ec33faf05b28518fdb3615",
    "results/dataprep/reports/N2C_policy/rehearsal/forward_pr_q0c.members.csv":
        "fcfa4baa00860018e77ce55f282efac31fd1abb897845fea9711a1674512ece2",
    "results/dataprep/reports/N2C_policy/rehearsal/forward_pr_q0c.npz":
        "ab32b5597a182bfe6cd2b823c489a1b1cfcda6f46ee28cab849fb1caed604261",
    "results/dataprep/reports/N2C_policy/rehearsal/forward_pr_q0c.summary.csv":
        "dfb9d7020a1da3b5ba618347d5dabb6ce7b0706cfead6c9af1d5fe84519c6104",
    "results/dataprep/reports/N2C_policy/rehearsal/forward_pr_r1c.members.csv":
        "52f3b5091379d57697d4dcaba47998a8d6775220313f2cbd9c1a1e721a50f31c",
    "results/dataprep/reports/N2C_policy/rehearsal/forward_pr_r1c.npz":
        "a1f889a54fa52cee3f6650788f1722ca193a48921f0c27514ee2140f52f7ff0f",
    "results/dataprep/reports/N2C_policy/rehearsal/forward_pr_r1c.summary.csv":
        "f2ee7b93d93f8260f9bad3bb5370ef76616e9791310b6bedbf43449e0d77e597",
    "results/dataprep/reports/N2C_policy/rehearsal/forward_pr_x5r1.members.csv":
        "b72c49405202c353f383732050886ea12e3dca93a3a2ad8d519b98feb6c07928",
    "results/dataprep/reports/N2C_policy/rehearsal/forward_pr_x5r1.npz":
        "15bd8eea91831d6535c97a44f74d3d55ec8fbd92efe0d6e80bdcf562bea8d13a",
    "results/dataprep/reports/N2C_policy/rehearsal/forward_pr_x5r1.summary.csv":
        "d23ab1805cbb37567e0fe9a9f3da681c9b5e99e4c5f7932771394af556a35487",
    "results/dataprep/reports/N2C_policy/rehearsal/r_both_2023-01_2026-08.json":
        "3d60df30fa94e9f2a335d78703c1c2f544597f093107e238ba22a4e4a0228dc2",
    "results/dataprep/reports/N2C_policy/rehearsal/r_both_2023-01_2026-08.npz":
        "f9fd01a18e11c52cf5467e4e5a4934d576bd7142613672c9c43ed6d89531087b",
    "results/dataprep/reports/N2C_policy/rehearsal/r_changhua_2023-01_2026-08.json":
        "222affb6e3fd92b907f529af9adb887dd167cb0b493a7b82eda6dc685c0c958d",
    "results/dataprep/reports/N2C_policy/rehearsal/r_changhua_2023-01_2026-08.npz":
        "adc03a5fd1eb044fa8be5c83a12720a88a711ce99237f4312ca41c09332854e7",
    "results/dataprep/reports/N2C_policy/rehearsal/readout.json":
        "5bb20dd2e3f3ee155caf4da13b8510625bae599ef40ebc7ab33c110ce0ff8259",
    "results/dataprep/validation_units/v3/units.json":
        "3df1b5dc690afad0674fc49d7123903953cb27265c97847ce227b083a0671310",
    "results/dataprep/zone_county_map/v2/zone_county_map.npz":
        "e97fac553d6b1b6cb416e2de915189a5fe23330e7c0b2112cc78eacae7aeb424",
    "results/twin/prospective_2026-11/predictions.csv":
        "16ac3c3783cea9f3c94560448d06c64ec8dc8fbb450effa28b465469a931cc32",
    "results/twin/prospective_2026-11_n2c_q0c/manifest.json":
        "ef4222d1dff9119d6edf7d7dfa113ab62b248449ccb771aa40073163c98473f5",
    "results/twin/prospective_2026-11_n2c_q0c/predictions.csv":
        "100a7c6f94d53a8b8246c1426dfd3e3cc623c30dba5640c4f6c60f1bfc5d2e5c",
    "results/twin/prospective_2026-11_n2c_q0c/protocol.json":
        "f22d54e0a39b9e55730fe286a5b7712fedc0a9f390fad4fe3ee297e3083e68d7",
    "results/twin/prospective_2026-11_n2c_r1c/manifest.json":
        "e0575fea28032b92f0f698d5acf84dade4c14ebb42ce4196c30a58588677ea6b",
    "results/twin/prospective_2026-11_n2c_r1c/predictions.csv":
        "58f938062d99194add8430a5ade62820db5fa4e5a6e645df8c895cc6f4a0c2ed",
    "results/twin/prospective_2026-11_n2c_r1c/protocol.json":
        "7a7a6dcbbc392a32723143a604fd83ae4cedbcafcef11a65f587519a2a100d58",
    "results/twin/prospective_2026-11_n2c_x5r1/manifest.json":
        "c22837bf062ab63ecec51f7dcd4aadaae3b6fd4fd52fdad00adbcf2c6e7eac1e",
    "results/twin/prospective_2026-11_n2c_x5r1/predictions.csv":
        "7e7448149da2bd0f9be6425013932bff610f20d7ebf566caaf467649e5e76c05",
    "results/twin/prospective_2026-11_n2c_x5r1/protocol.json":
        "3a8352362026bc4840cf16afc258de9242722fe73c243b0ad6cb2b3d04c002a4",
    "results/twin_forward/datum_gate_ccz.npz":
        "a3763e198cbcfea412a75db1e4c7e7c9bb3dd6bef86569d0a652b8b136e311d9",
    "results/twin_forward/datum_gate_q0c.npz":
        "974bebc2fdfcb7d13b47bdb883f41d82676e8e44736a9ac1ca06a447a6ff4a13",
    "results/twin_forward/datum_gate_x5r1.npz":
        "ed03d7b05b5b2488968a0e8dbdda992a4dc6151a9b36cd4663e41022a8c90a1c",
    "results/twin_runs/stage3_datum_gate_ccz/coupled_leveling/vep_zonal_leveling.json":
        "63e53cfd69786c45fa70edc8d50374f93e8e4fdad3ac5d561c5a0844e1e94bff",
    "results/twin_runs/stage3_datum_gate_ccz/stage3_theta.json":
        "03aa7563caa5fab3de0cbcce87118be5221de44d024c31305af071704743d98b",
    "results/twin_runs/stage3_datum_gate_ccz/stage3_fold_thetas.json":
        "965e627c04057db2f248b291d20109e9360802c38dbc083ea9e6afc3070ac63f",
    "results/twin_runs/stage3_datum_gate_ccz/stage3_posterior_tn.json":
        "4d2f480bcb906742858ced719bbb9bd011ac6a07f5e7c1b5bed37478cda9c39a",
    "results/twin_runs/stage3_datum_gate_q0c/coupled_leveling/vep_zonal_leveling.json":
        "ace0dcf9161be3b2b82b7c4733f05e8cffa207f4e3c0ec0a607767d1584acf51",
    "results/twin_runs/stage3_datum_gate_q0c/stage3_theta.json":
        "52026bd8d4372bf13e4491eeae62d573356d0fff4dceb26408f5c007eb702a3a",
    "results/twin_runs/stage3_datum_gate_x5r1/coupled_leveling/vep_zonal_leveling.json":
        "7db612ebf908d3a570e47a0e99459a29a6f5663d7fcc40dffe858828aca0b571",
    "results/twin_runs/stage3_datum_gate_x5r1/stage3_theta.json":
        "cf200f99be7a9e222f51ddd54fe8fd3ef5fc123c64d95ae3327a101c1e70b856",
}
# sha256 of the code files at CODE_COMMIT (git show), frozen with the draft.
CODE_SHA256 = {
    "hydrophysics/twin/forward.py":
        "9f7a533593546d0588faceb0d1715f35f5f24aa8c4bd8b808ada479646742fc6",
    "hydrophysics/twin/recorded_pumping.py":
        "4a8c99e47c902496ac8959ca6c76ac3320ade27ccaa3793063e3854ce0c36ecb",
    "hydrophysics/twin/scenario.py":
        "c6f250fcce5a3a146091c3d531a847bca1da3503dc6e5146a001e5aa064e7997",
    "hydrophysics/twin/prospective.py":
        "6a53e43487c03cfb1ec3e3046522f3ea2580b6ba368351f99e99a6ce249bf7cf",
    "hydrophysics/twin/coverage.py":
        "bfbd225a05dbc9a4f6ee8e8fcb9a9726de9eb284517db2acd8573bb114c6936c",
    "hydrophysics/twin/zone_county.py":
        "09a8590c22ee0844be65cc883238c422e9e6e630c0d42d1099a697a33e074175",
    "hydrophysics/twin/inputs.py":
        "4823a17bb58e3203ca4f39920c8d97d9550405f04d61f4c71d35e74e664bd118",
    "tests/test_twin_recorded_pumping.py":
        "118fa5b81372ff78dd9a6137abeb928bcb59222ba52eb5f06d01494ee298b58d",
}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_sha(path: str) -> str:
    blob = subprocess.run(["git", "-C", str(REPO), "show", f"{CODE_COMMIT}:{path}"],
                          check=True, capture_output=True).stdout
    return hashlib.sha256(blob).hexdigest()


def zr_dir(arm: str) -> Path:
    return REPO / f"results/twin/prospective_2026-11_n2c_{arm}"


def forward_cmd(arm: str, r: str = "<r_both.npz>", name: str = "PR", start: str = "2023-01",
                out: str | None = None, ensemble: bool = False) -> str:
    theta, _, _ = ARMS[arm]
    thetas = f"--theta {theta}/stage3_theta.json "
    if ensemble:   # the 18 parameter sets of datum_gate_ccz: main + 5 folds + 12 posterior
        thetas += (f"--theta {theta}/stage3_fold_thetas.json "
                   f"--theta {theta}/stage3_posterior_tn.json ")
    out = out or f"<scoring dir>/forward_pr_{arm}"
    return ("HYDROMIND_GW_DATA=<repo>/chou-shui-data/data PYTHONPATH=<checkout of 1b08fb6> "
            "python -P -m hydrophysics.twin.forward " + thetas +
            f"--vep-json {theta}/coupled_leveling/vep_zonal_leveling.json "
            "--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation "
            "--restart-taper-km 5 --apex-hold calibrated --column-heads free "
            "--column-hpc0-fast-days 365 --device cpu --save-members none "
            f"--recorded-pumping '{name}={r}@{start}' --out {out}")


def builder_cmd(out: str, first: str = "2023-01", extra: str = "") -> str:
    return ("python -I dataprep_staging/n2_policy/build_r_both.py --x1 <X1 version>/"
            f"county_monthly.parquet --first {first}{extra} --out {out}")


def changhua_ids() -> list[str]:
    ids = sorted(pd.read_csv(N2_ZR, dtype={"station_id": str})["station_id"].unique())
    ch = [s for s in ids if s.startswith("07")]
    n2 = json.loads(N2.read_text(encoding="utf-8"))
    n2_ids = n2["tests"][0]["external_units"]["N2:heads_prospective_2026-11_2027-10"]["ids"]
    if sorted(n2_ids) != ids or len(ch) != 74:
        raise SystemExit(f"N2 unit ids do not match its ZR predictions, or {len(ch)} != 74 "
                         "Changhua wells")
    for arm in ARMS:
        got = set(pd.read_csv(zr_dir(arm) / "predictions.csv",
                              dtype={"station_id": str})["station_id"])
        if not set(ch) <= got:
            raise SystemExit(f"{arm}: ZR predictions lack Changhua wells "
                             f"{sorted(set(ch) - got)[:5]}")
    return ch


def doc() -> dict:
    ch = changhua_ids()
    zr = {arm: {p.name: {"path": rel_to_repo(p), "sha256": sha(p)}
                for p in sorted(zr_dir(arm).iterdir())} for arm in ARMS}
    models = {}
    for arm, (theta, fwd, role) in ARMS.items():
        models[arm] = {
            "role": role,
            "theta": {"path": f"{theta}/stage3_theta.json",
                      "sha256": sha(REPO / theta / "stage3_theta.json")},
            "column": {"path": f"{theta}/coupled_leveling/vep_zonal_leveling.json",
                       "sha256": sha(REPO / theta / "coupled_leveling/vep_zonal_leveling.json")},
            "zr_forward": {"path": fwd, "sha256": sha(REPO / fwd)},
            "zr_predictions": zr[arm],
            "pr_forward_command": forward_cmd(arm)}
    reh = sorted(p for p in REH.iterdir() if p.is_file()) if REH.is_dir() else []
    inputs_detail = {
        **{f"{arm}_{k}": v for arm, m in models.items()
           for k, v in (("theta", m["theta"]), ("column", m["column"]),
                        ("zr_forward", m["zr_forward"]))},
        **{f"{arm}_zr_{n.split('.')[0]}": v for arm, files in zr.items()
           for n, v in files.items()},
        "x1_county_monthly": {"path": f"{X1}/county_monthly.parquet",
                              "sha256": sha(REPO / X1 / "county_monthly.parquet")},
        "x1_manifest": {"path": f"{X1}/MANIFEST.json", "sha256": sha(REPO / X1 / "MANIFEST.json")},
        "county_map": {"path": COUNTY_MAP, "sha256": sha(REPO / COUNTY_MAP)},
        "n2_registration": {"path": rel_to_repo(N2), "sha256": sha(N2)},
        "n2_zr_predictions": {"path": rel_to_repo(N2_ZR), "sha256": sha(N2_ZR)},
        "r_builder_n2": {"path": rel_to_repo(HERE / "build_r.py"), "sha256": sha(HERE / "build_r.py")},
        "r_builder_n2c": {"path": rel_to_repo(HERE / "build_r_both.py"),
                          "sha256": sha(HERE / "build_r_both.py")},
        "rehearsal_readout_script": {"path": rel_to_repo(HERE / "rehearse_n2c.py"),
                                     "sha256": sha(HERE / "rehearse_n2c.py")},
        **{f"r1c_ensemble_{n.split('.')[0]}": {"path": f"{ARMS['r1c'][0]}/{n}",
                                               "sha256": sha(REPO / ARMS["r1c"][0] / n)}
           for n in ("stage3_fold_thetas.json", "stage3_posterior_tn.json")},
        **{k: {"path": v, "sha256": sha(REPO / v)} for k, v in EVIDENCE.items()},
        **{f"rehearsal_{p.name}": {"path": rel_to_repo(p), "sha256": sha(p)} for p in reh
           if p.suffix in (".json", ".npz", ".txt", ".csv")},
    }
    decision = {
        "response_supported": "beta 90 % interval excludes 0 AND the point estimate lies in "
                              "[1/1.5, 1.5] (N1 ratio_band) AND RMSE(PR) < RMSE(ZR), over the "
                              "Changhua well-months",
        "direction_supported_size_not": "beta 90 % interval excludes 0 and beta > 0, but the "
                                        "point estimate is outside [1/1.5, 1.5] or RMSE(PR) "
                                        ">= RMSE(ZR)",
        "not_supported": "beta 90 % interval includes 0, or beta <= 0",
        "not_informative": "the informative gate fails",
        "skill_wording": "a claim that R1c forecasts Changhua heads better than climatology "
                         "additionally needs RMSE(PR) < RMSE(CL); otherwise only the response "
                         "verdict may be cited",
        "source": "N2.policy_response.prospective_2026-11.v1 decision_rule, unchanged except "
                  "the wells (Changhua, not Yunlin) and the model (R1c, not the page model)"}
    metric = {
        "name": "beta_response",
        "definition": "OLS slope of e = obs - ZR on c = PR - ZR over the well-months of the "
                      "74 Changhua wells in the window (flags 0-1; ZR = R1c's registered "
                      "predictions with their per-well datum, "
                      "results/twin/prospective_2026-11_n2c_r1c), with well fixed effects "
                      "(e and c demeaned per well), so a level offset of ZR cannot create the "
                      "slope; c = heads_mean[PR] - heads_mean[baseline] of the SAME CPU "
                      "forward run at each well's cell and layer (never against the "
                      "GPU-written datum_gate_ccz.npz); PR prediction = registered ZR + c",
        "interval": "90 % from a 5 km block cluster bootstrap, B 2000, seed 20261101",
        "co_reported": [
            "RMSE of PR, ZR and CL over the Changhua well-months (CL = the registered "
            "baseline_m, identical to N2's CL)",
            "the level component: window-mean e on window-mean c across wells (report only)",
            "R1c's Yunlin wells: the same statistics, report only (PR forces both counties)",
            "report-only arms q0c and x5r1: every statistic above with their own ZR "
            "(prospective_2026-11_n2c_q0c / _x5r1) and contrast; never gate, never "
            "substitute for R1c, never select anything"]}
    informative = {
        "rule": "evaluated from the model and the forcing only, before any window head is "
                "read: RMS of the well-demeaned contrast over Changhua well-months >= 0.05 m "
                "(N1 thresholds.informative_head_m) and >= 5 bootstrap clusters; otherwise "
                "'not informative' and no verdict",
        "ruling_if_x1_missing": "if fewer than 6 window months carry a recorded Changhua r, "
                                "the result is 'not informative'"}
    priors = {
        "ruling": "User ruling 2026-10-29: R1c, which failed its Changhua head gate, may be "
                  "the primary model of a prospective Changhua head test; its failures are "
                  "disclosed here",
        "I1.R1c.county_zones.v1": "FAILED 1 of 4 gates: the Changhua out-of-fold head-anomaly "
                                  "RMSE gate, 1.318 m vs R0 1.074 m (reported diagnostics, "
                                  "not gates: amplitude 1.13, correlation 0.44); passed k-fold R2 0.646, Yunlin 1.820 vs 1.856 m, "
                                  "leveling out-of-fold R2 0.663 vs 0.649",
        "I1.G2.fresh_partition.v1": "R1c passes_all False: fails the k-fold R2 gate (0.646) "
                                    "and the Changhua county-mean RMSE gate (0.643 m); passes "
                                    "Changhua correlation (0.437), Yunlin gates and leveling "
                                    "(0.661); reported Changhua amplitude 1.06",
        "N1.policy_response.v1 step C": "design-time masked 2016-18 refit of the R1c recipe "
                                        "(stage3_datum_gate_ccz_m1618, NOT this theta): "
                                        "Changhua heads obs-on-model slope 0.85 [0.63, 1.05] "
                                        "(59 wells, 24 sites); RMSE 1.07 m vs CL 1.19 and ZR "
                                        "1.21 (model better); Changhua leveling lost to CL "
                                        "(not used here). This slope is of observed on modelled "
                                        "heads over masked years, not of e on a recorded-pumping "
                                        "contrast, so it is a prior, not a prediction of beta",
        "other_known_changhua_weaknesses": {
            "county_mean_anomaly_design_time": "R1c fits the Changhua county-mean head "
                                               "anomaly WORSE than R0: RMSE 0.600 vs 0.492 m, "
                                               "correlation 0.70 vs 0.78 (design-time folds; "
                                               "results/dataprep/reports/"
                                               "I1_changhua_interannual/report.md, "
                                               "'Correction'); the earlier claim that the "
                                               "census removed the regional Changhua misfit "
                                               "is withdrawn",
            "county_mean_anomaly_fresh_partition": "G2 fresh partition: Changhua county-mean "
                                                   "RMSE 0.643 vs R0 0.515 m (failed gate); "
                                                   "county-mean correlation 0.680 vs R0 0.736; "
                                                   "single-well RMSE 1.340 vs R0 1.100 m "
                                                   "(reported); R1c amplitude 1.06 vs R0 0.42 "
                                                   "(I1_G2_fresh_partition/report.json)",
            "explained_fraction": "median-well out-of-fold R2 of the Changhua head anomaly: "
                                  "R1c -0.21 vs R0 0.14 (Yunlin 0.31 vs 0.29), against 0.56 "
                                  "for a per-well ridge on recorded drivers "
                                  "(I1_changhua_drivers/report.md): R1c explains LESS than its "
                                  "own wells' mean",
            "amplitude_and_correlation": "design-time predicted/observed Changhua amplitude "
                                         "1.13 at correlation 0.44 (Yunlin 0.83 and 0.74) "
                                         "(I1_changhua_local/report.md); on clean wells R0 "
                                         "0.960 vs R1c 1.207 m",
            "weak_observed_coupling": "model-free: Changhua heads correlate with local pump "
                                      "electricity at -0.30 (Yunlin -0.63) and swing 0.33 m "
                                      "seasonally at mid-fan (Yunlin 0.74 m) "
                                      "(I1_changhua_local/report.md); a pumping-response test "
                                      "in Changhua has a low ceiling",
            "leveling_and_column": "G2 Changhua leveling R2 0.175 (R0 0.007) is subsidence and "
                                   "is not used here; R1c's six-zone county column was "
                                   "suspended as a subsidence fallback by user ruling "
                                   "2026-10-21 (I1_G2_fresh_partition/deviation.json) and is "
                                   "the source of the post-2022 uplift of every R1c-family "
                                   "rung (I1_R4Q0/report.json); no subsidence lever follows "
                                   "from N2C",
            "ladder": "every Changhua-census rung (R1, R1b, R1c, R1d, R1e, R4, R4p, R*) failed "
                      "at least one Changhua head gate (R4 and R4p the Changhua county-mean "
                      "RMSE, I1_R4Q0/report.json; R* passed the Changhua correlation gate, "
                      "0.454, but failed the Changhua county-mean RMSE gate, 0.855 m, "
                      "I1_Rstar/report.json); R0 stays the deliverable"},
        "expectation": "under the decision rule the [1/1.5, 1.5] band applies to the POINT "
                       "estimate. If step C carried over (point near 0.85), the point would "
                       "lie inside the band and the verdict would turn on the other two "
                       "conditions: 'response supported' needs the 90 % interval to exclude 0 "
                       "AND RMSE(PR) < RMSE(ZR). R1c's failed Changhua head-anomaly RMSE gate "
                       "(1.318 vs 1.074 m), its negative explained fraction (-0.21) and the "
                       "weak observed head-electricity coupling make RMSE(PR) >= RMSE(ZR), "
                       "a point below 1/1.5, or an interval including 0 plausible, so "
                       "'direction supported, size not' and 'not supported' are both "
                       "plausible outcomes; no outcome is expected with confidence",
        "rehearsal": "results/dataprep/reports/N2C_policy/rehearsal (2023-01..2026-08 recorded "
                     "months; engineering only; X1 is forbidden for selection, model_design "
                     "and design_time_evidence, so no rehearsal number is a reason for any "
                     "design choice here)"}
    window = ["2026-11", "2027-10"]
    unit_detail = {
        "id": UNIT, "window": window,
        "description": "monthly heads 2026-11..2027-10 (flags 0-1) at the 74 Changhua wells "
                       "(station code 07) of N2's eligible-well definition (the 158 wells of "
                       "the registered prospective forecast; N2 holds these 74 report-only)",
        "shared_with": [
            "N2.policy_response.prospective_2026-11.v1 (its Changhua report-only statistics)",
            "twin.prospective free-running head test results/twin/prospective_2026-11",
            "I1.hybrid_tf.prospective_2026-11.v1 T1 and T2",
            "X1.hybrid_tf_tpc.prospective_2026-11.v1 T3a and T3b"],
        "multiplicity": "these heads are scored by several registered tests; each is reported "
                        "with its own decision, none is selected post hoc, no family-wise claim",
        "why_unspent": "no head of the window exists at drafting (the last fetched head month "
                       "is 2026-09); see blocking_looks at registration"}
    d = {
        "id": ID, "wp": "N2", "kind": "test",
        "title": "Prospective Changhua head response of R1c (a failed-rung alternative model) "
                 "to recorded pumping anomalies of both counties, 2026-11..2027-10",
        "why": "N2's page model carries no Changhua pumping, so its Changhua levers answer "
               "about 0 and N2 holds the Changhua wells report-only. Every Changhua head unit "
               "to 2026-09 is spent; the only route left to decide the Changhua head lever "
               "is a prospective contrast on a model that carries Changhua pumping. R1c's "
               "recipe, refitted with 2016-18 masked, beat CL and ZR on Changhua heads in N1 "
               "step C (design-time); R1c itself failed its Changhua gates (priors).",
        "tests": [{
            "test_id": TEST_ID, "data_scope": "heads:2026-11..2027-10",
            "fresh_units": [UNIT],
            "external_units": {UNIT: {"description": unit_detail["description"], "ids": ch,
                                      "window": "2026-11..2027-10"}},
            "what": "Do the observed Changhua head anomalies of 2026-11..2027-10 follow R1c's "
                    "response to the RECORDED pumping anomalies of both counties (PR - ZR)?",
            "metric": metric, "decision_rule": decision, "informative_gate": informative,
            "seed": 20261101,
            "label": "prospective: the window's heads did not exist at registration"}],
        "arms": {
            "ZR": "zero-response arm: R1c's registered free-running forecast "
                  "(results/twin/prospective_2026-11_n2c_r1c/predictions.csv, written "
                  "2026-10-29 by twin.prospective register on datum_gate_ccz.npz, unchanged): "
                  "climatology pumping after the 2022-12 origin",
            "PR": "policy-response arm: ZR + c(w, t), c = H_PR - H_baseline of one CPU forward "
                  "run of R1c's main parameter set from the same origin state, identical "
                  "recharge and other forcings, differing only in pumping energy",
            "CL": "the registered prospective test's baseline: month-of-year climatology of "
                  "each well's fitted months before the origin"},
        "forcing_rule_frozen_now": {
            "PR_energy": "E_PR(cell, t) = E_ZR(cell, t) x r_county(cell)(t), months "
                         "2023-01..2027-10: r_CH on the 943 Changhua cells of "
                         "zone_county_map/v2 (flags sha 6ca78404, the map R1c was calibrated "
                         "on), r_YL on every other cell",
            "ratio": "r_county(t) = X1 in-grid county value of month t (product "
                     "electricity_x1_tpc/full; flags 0 census or 5 TPC-downscaled) / its "
                     "2012-2022 mean for that month of year (N2's rule, per county; r_YL is "
                     "N2's r(t) bit for bit)",
            "per_cell_file": "built by dataprep_staging/n2_policy/build_r_both.py --x1 <X1 "
                             "county_monthly.parquet> --first 2023-01 --out <r_both.npz> "
                             "(npz: month, multiplier (n_months, 2148), n_active, start)",
            "unrecorded": "a month not published in X1 (or a later X1 version whose manifest "
                          "grants forcing_recorded_arm) by the scoring deadline takes r = 1 "
                          "for that county",
            "x1_version": "electricity_x1_tpc/full/v1 or the first later version extending it "
                          "with the same frozen model (parameters.json unchanged); the version "
                          "used is hashed at scoring",
            "why_both_counties": "the ONLY reason for the both-county primary (review of plan "
                                 "S3, 2026-10-28, must-fix b): drought years move both "
                                 "counties' pumping anomalies together, so in a Changhua-only "
                                 "arm any effect of the Yunlin anomaly on Changhua heads "
                                 "(cross-county flow) would stay in e = obs - ZR but not in "
                                 "c, while correlated with c, and would bias beta; forcing "
                                 "both counties puts that term in c. No "
                                 "rehearsal number (X1-forced, post-origin) entered this "
                                 "choice"},
        "declared_secondary": {
            "changhua_only": "PR with r_CH on Changhua cells and 1 elsewhere; reported beside "
                             "the primary as the size of the omitted Yunlin-anomaly term, "
                             "never in its place. Commands: "
                             + builder_cmd("<scoring dir>/r_changhua.npz",
                                           extra=" --scope changhua") + " ; "
                             + forward_cmd("r1c", r="<scoring dir>/r_changhua.npz", name="PRCH",
                                           out="<scoring dir>/forward_prch_r1c"),
            "window_only": "r(t) applied only from 2026-11 (r = 1 for 2023-01..2026-10). "
                           "Commands: "
                           + builder_cmd("<scoring dir>/r_both_window.npz", first="2026-11")
                           + " ; "
                           + forward_cmd("r1c", r="<scoring dir>/r_both_window.npz",
                                         name="PRW", start="2026-11",
                                         out="<scoring dir>/forward_prw_r1c"),
            "x1_band": "PR with both counties' r(t) at the X1 lo90 and hi90 bounds. Commands: "
                       + builder_cmd("<scoring dir>/r_both_lo90.npz", extra=" --column lo90")
                       + " (and --column hi90 -> r_both_hi90.npz) ; "
                       + forward_cmd("r1c", r="<scoring dir>/r_both_lo90.npz", name="PRLO",
                                     out="<scoring dir>/forward_prlo_r1c")
                       + " (and PRHI on r_both_hi90.npz)",
            "ensemble": "the contrast from R1c's 18-set ensemble mean (main + 5 folds + 12 "
                        "posterior sets, as datum_gate_ccz) instead of the main set. Command: "
                        + forward_cmd("r1c", ensemble=True,
                                      out="<scoring dir>/forward_pr_r1c_ensemble"),
            "detrended": "e and c each detrended per well (linear in time) before the slope "
                         "(post-2022 trend confounding); no extra run"},
        "report_only_arms": {
            "q0c": "Q0 lineage plus the Changhua census (par27); same statistics with its own "
                   "ZR and contrast; report only",
            "x5r1": "WRA energy-to-volume conversion arm (par27); carried report-only "
                    "UNCONDITIONALLY (no post-origin X1 result selects arms, review must-fix)",
            "models": models},
        "scoring": {
            "not_before": "2028-01-15 (N2's not_before): window closed, heads fetched by the "
                          "same feed as N2, X1 month 2027-10 published",
            "no_later_than": "2028-01-31 (the scoring deadline of I1.hybrid_tf and "
                             "X1.hybrid_tf_tpc, which share the session)",
            "same_session": "accepted by user ruling 2026-10-29 (second set). N2C and every "
                            "statistic on the 74 Changhua wells of the window from "
                            "N2.policy_response.prospective_2026-11.v1 (its Changhua "
                            "report-only statistics), the free-running test "
                            "prospective_2026-11, I1.hybrid_tf.prospective_2026-11.v1 T1/T2 "
                            "and X1.hybrid_tf_tpc.prospective_2026-11.v1 T3a/T3b are computed "
                            "in ONE scoring session between 2028-01-15 and 2028-01-31, in "
                            "this order: (1) prereg3.assert_can_score(N2C) succeeds; (2) N2C's "
                            "primary and declared statistics are computed and its report is "
                            "written; (3) N2's Changhua report-only statistics, the "
                            "prospective_2026-11 Changhua statistics, I1 T1/T2 and X1 T3a/T3b "
                            "Changhua statistics. Statistics of those tests on other wells "
                            "(N2's Yunlin primary included) are not restricted by this clause. "
                            "The scorer code for beta_response and its 5 km block bootstrap "
                            "is one implementation shared with N2, hashed and logged before "
                            "step (1)",
            "if_order_breached": "if any statistic of the window on the 74 Changhua wells is "
                                 "computed or read before step (1), or step (1) falls after "
                                 "2028-01-31, N2C is still scored ONCE and its verdict is "
                                 "reported, with the breach (time, test, statistic) listed "
                                 "beside its post_registration_looks and the label 'scored "
                                 "after a look at its wells' window'; a breached 'response "
                                 "supported' does NOT put the Changhua head lever on the page "
                                 "(it is reported, not shown)",
            "engineering": "the forward runs are CPU, frozen code checkout 1b08fb6"},
        "page_mapping": {
            "before_scoring": "from registration until N2C is scored, the Changhua head lever "
                              "is withdrawn from the page; wherever the page or its evidence "
                              "mentions it, it is labelled 'untested (N2C registered, scored "
                              "not before 2028-01-15)'. Registration is not support",
            "response_supported": "ONLY this verdict (and only with the session order kept) "
                                  "puts R1c's Changhua head lever on the page, labelled with "
                                  "R1c's earlier gate failures (I1.R1c Changhua head-anomaly "
                                  "RMSE 1.318 vs 1.074 m; G2 k-fold R2 0.646 and Changhua "
                                  "county-mean RMSE 0.643 m) and with the N2C result (verdict, "
                                  "beta and its 90 % interval, RMSE of PR, ZR and CL); shown "
                                  "as R1c's relative lever, never as the page model's answer",
            "direction_supported_size_not": "the lever is NOT shown; the evidence reports the "
                                            "direction and beta with its interval",
            "not_supported": "the lever is NOT shown; reported as not supported",
            "not_informative": "the lever is NOT shown; labelled 'undecided (N2C not "
                               "informative)'",
            "never": "no Changhua subsidence lever, no absolute pumping volume, no claim "
                     "about the page model follows from any verdict"},
        "attempt_policy": "User rule 2026-10-23 ('trying another way must never threaten the "
                          "test'): engineering failures (crash, bug, missing file) may be "
                          "fixed and re-run before the scoring look, each logged; the test is "
                          "scored ONCE after the window closes; a failed, inconclusive or "
                          "not-informative result is reported as such and never re-scored on "
                          "the same units; the declared secondaries and report-only arms are "
                          "fixed now and reported beside the primary, never in its place; "
                          "after a failure at most ONE follow-up registration (new test ids, "
                          "label 'post-failure, exploratory', never validation), which cannot "
                          "reuse this window; the final report lists every attempt",
        "u7": "2023+ never gates: only the fresh 2026-11..2027-10 Changhua heads gate; "
              "2023-2026 electricity enters only as recorded forcing of the PR arm (X1 "
              "allowed use forcing_recorded_arm); the rehearsal sets no model, arm or "
              "threshold",
        "rulings": {
            "2026-10-29 user (2)": "R1c may be the primary model of a prospective Changhua "
                                   "head test; its failures are disclosed (priors)",
            "2026-10-29 user (3)": "the N2 companion is dropped",
            "2026-10-29 user, second set (page)": "R1c's Changhua head lever is shown only if "
                                                  "N2C returns 'response supported', labelled "
                                                  "with R1c's earlier gate failures and the "
                                                  "N2C result; before scoring it is "
                                                  "'untested' (page_mapping)",
            "2026-10-29 user, second set (session)": "the same-session clause is accepted, "
                                                     "N2C first (scoring.same_session)",
            "2026-10-29 default (scope)": "both-county per-cell PR arm primary, Changhua-only "
                                          "a declared secondary "
                                          "(forcing_rule_frozen_now.why_both_counties)",
            "U4 extension (user, 2026-10-24)": "recorded post-origin pumping may feed a "
                                               "SEPARATE counterfactual arm of a frozen "
                                               "physics model; the registered forecast (ZR) "
                                               "is unchanged",
            "N2 changhua": "N2 allowed a Changhua arm only by a separate registration written "
                           "before 2026-10-31 16:00 UTC"},
        "code": {
            "commit": CODE_COMMIT, "branch": "main",
            "files": {f: git_sha(f) for f in CODE_FILES},
            "forward_command": forward_cmd("r1c"),
            "zr_register_command": "PYTHONPATH=<checkout of 1b08fb6> python -P -m "
                                   "hydrophysics.twin.prospective register --forward "
                                   "<datum_gate_ccz|q0c|x5r1.npz> --stations "
                                   "AMP_V2/data/fan_stations.parquet --historical-wells "
                                   "AMP_V2/data/wells --out "
                                   "results/twin/prospective_2026-11_n2c_<arm> --test-start "
                                   "2026-11-01 --test-end 2027-10-31 (the same command "
                                   "reproduces N2's ZR, sha 16ac3c37, and R0's, sha 730cc156, "
                                   "byte for byte)",
            "r_builder": {"path": rel_to_repo(HERE / "build_r_both.py"),
                          "sha256": sha(HERE / "build_r_both.py"),
                          "imports": rel_to_repo(HERE / "build_r.py")},
            "contrast": "c = heads_mean[PR] - heads_mean[baseline] of the SAME forward run"},
        "rehearsal": {"dir": rel_to_repo(REH),
                      "what": "PR and Changhua-only arms on 2023-01..2026-08 for R1c, q0c and "
                              "x5r1 (GPU, engineering); contrast exactly 0 to the origin"},
        "zones": "R1c's own (county-split mid zone, six-zone county column)",
        "priors": priors,
        "priors_disclosed": priors,
        "burn_in": NA("no fit"), "folds": NA("prospective window"),
        "fresh_units": [UNIT],
        "thresholds": {"ratio_band": 1.5, "informative_head_m": 0.05, "min_clusters": 5,
                       "min_months": 6, "level": 0.90},
        "gate_scores": NA("no gate; the decision rule is reported whichever way it falls"),
        "primary_model": {
            "name": "R1c: stage3_datum_gate_ccz (county zones, Changhua census v1) + its "
                    "coupled_leveling column, forward datum_gate_ccz",
            "theta_sha256": models["r1c"]["theta"]["sha256"],
            "column_sha256": models["r1c"]["column"]["sha256"],
            "forward_sha256": models["r1c"]["zr_forward"]["sha256"],
            "member_for_the_contrast": "stage3_datum_gate_ccz (main parameter set), CPU "
                                       "rollouts; the 18-set ensemble contrast is secondary",
            "status": "failed rung (I1.R1c, G2), primary here by user ruling 2026-10-29"},
        "unit_detail": unit_detail,
        "inputs": sorted({v["path"] for v in inputs_detail.values()}
                         | {rel_to_repo(Path(__file__))}),
        "inputs_detail": inputs_detail,
        "interpretation": "R1c's Changhua head response to real pumping anomalies on fresh "
                          "months. Nothing about the page model, Changhua subsidence, absolute "
                          "pumping volume, or R1c's failed gates (I1.R1c Changhua head-anomaly "
                          "RMSE; G2 k-fold R2 and Changhua county-mean RMSE), which stay "
                          "failed.",
        "what_a_pass_supports": "R1c's Changhua head response to recorded pumping anomalies "
                                "(a relative lever), on fresh months; nothing else",
        "written_by": rel_to_repo(Path(__file__)),
    }
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--register", action="store_true",
                    help="write the pre-registration (the lead only)")
    ap.add_argument("--expect-self-sha", default=None,
                    help="sha256 of this script as reviewed; required with --register")
    a = ap.parse_args(argv)
    d = doc()
    errs = prereg3.validate_prereg(d)
    print("validate:", errs or "ok")
    missing = [p for p in d["inputs"] if not (REPO / p).exists()]
    print("missing inputs:", missing or "none")
    # the real draft-time check: today's hashes (computed in doc()) against the frozen ones
    now_sha = {v["path"]: v["sha256"] for v in d["inputs_detail"].values()}
    bad = sorted(p for p in set(now_sha) | set(FROZEN_SHA256)
                 if now_sha.get(p) != FROZEN_SHA256.get(p))
    bad += sorted(f"{f} @ {CODE_COMMIT}" for f, h in CODE_SHA256.items()
                  if d["code"]["files"].get(f) != h)
    if set(d["code"]["files"]) != set(CODE_SHA256):
        bad.append("code file list differs from CODE_SHA256")
    print(f"inputs changed, added or dropped since drafting ({len(FROZEN_SHA256)} pinned + "
          f"{len(CODE_SHA256)} code):", bad or "none")
    self_sha = sha(Path(__file__))
    print("this script sha256:", self_sha)
    if a.register and a.expect_self_sha != self_sha:
        bad.append(f"--expect-self-sha {a.expect_self_sha!r} != this script {self_sha}")
        print("REFUSED: --register needs --expect-self-sha equal to this script's sha256")
    now = dt.datetime.now(dt.timezone.utc)
    idx = prereg3.membership_index()
    ext = d["tests"][0]["external_units"][UNIT]
    idx.members.setdefault(idx.canon(UNIT), prereg3._declared_members(ext))
    block = prereg3.blocking_looks(d["tests"], None, idx=idx, test_time=now)
    print("blocking looks:", block or "none")
    late = now >= DEADLINE
    if late:
        print(f"DEADLINE PASSED ({DEADLINE:%Y-%m-%d %H:%M} UTC): do not register for the "
              "2026-11 window; a fallback window needs new ZR registrations and new ids")
    print("n Changhua wells:", len(ext["ids"]), "| inputs:", len(d["inputs"]))
    if not a.register or errs or missing or bad or block or late:
        return 1 if (errs or missing or bad or block) else 0
    path, h = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
