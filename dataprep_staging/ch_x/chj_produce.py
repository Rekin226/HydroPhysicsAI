"""CH-J panel producer (I1.CHX.v1 secondary): chj_panel.npz, chj_metrics.json and
chj_panel.json, what ``score_chx.chj_block`` / ``chj_provenance`` read and
``chj_report.py`` scores. PREPARED, NOT RUN: every input comes from the GPU jobs of
``run_chj.sh``, after the engineering record (which names THIS file's sha256 as
``producer_sha256``) is frozen.

Subcommands (run_chj.sh calls them; nothing here prints a model or observed value):

``hindcast --seed S``  (model only, in the hook worktree; CPU or GPU)
    CH-J's seed-S full-fit theta (``stage3_datum_gate_chj_s<S>``) rebuilt as
    ``hindcast.R1cRun`` rebuilds R1c (forward.build_model, its own Changhua-only canal
    field), two free runs over 2012-01..2022-12 exactly as the primary's arms: A recorded
    pumping, B the Changhua cells (CH-J's county map, verified equal to the CHX panel's) at
    their 2012 monthly pattern from 2013-01 (``chx_lib.pattern_energy``). Heads at the CHX
    panel's screens (the 23 Changhua externals, the Yunlin externals; their layer and cell
    from chx_panel.npz, re-derived on the grid and checked) over the CHX window ->
    ``hindcast/chj_hindcast_s<S>.npz`` (+ .json). Arm A is also rebuilt through
    ``policy_lib.TwinModel`` (theta + CH-J's own column) and must agree to 1e-9 m, as the
    primary's ``--check-twinmodel``.
``hindcast-q0``  (model only)
    Q0's free run (``policy_lib.TwinModel`` on Q0's registered theta, the CH-J full fit's
    parent; Q0 has no county split, which ``R1cRun`` requires) at the same screens (cells
    re-derived on Q0's grid) -> ``hindcast/q0_hindcast.npz``: the ``q0_ch`` comparator of
    the county-mean correlation gate.
``check-hook --run DIR --seed S [--require-folds N --q0-fold-dir DIR]``  (CPU)
    run_chj.sh's hook_applied (``check_hook_applied``): the hook ran, CH-J's fold K ran from
    Q0's own fold-K theta and held out exactly Q0's fold K.
``assemble``  (CPU; after every job)
    chj_panel.npz (``chjA_ch_<seed>``, ``chjB_ch_<seed>``, ``chjA_yl_<seed>``, ``q0_ch``),
    chj_metrics.json and chj_panel.json (``score_chx.PANEL_SUMMARY_KEYS``), each written
    once (mode 'x').

Metric definitions (chj_metrics.json ``definitions``): every one is the code that computed
Q0's, R1c's and the X5CS / X5CW rungs' own numbers, on the same files of each run --
``dataprep_staging/i1/score_g2.metrics`` (county_corr, leveling R2, county assignment),
``dataprep_staging/x5_energy_volume/score_rung_x5cs.column`` (Changhua leveling R2),
``score_rung_b.plausibility`` (the forward series and the Dec 2022 / Dec 2025 months) and
``uplift.uplift_share`` at ``prereg_x5cs.RATE_TOL`` (rising share). The one quantity no
earlier rung reported per county is the Yunlin k-fold: the pooled R2 of
``calibrate_flow._r2`` (the formula of stage3_flow.csv ``r2_kfold``) over the Yunlin
held-out entries of stage3_per_entry.npz, county by score_g2's assignment; the fan-wide
``r2_kfold`` is reported beside it. Yunlin cells for the forward: the CHX panel's
``yunlin_cells`` (I1.CHX.v1's placebo rule: centroid in 雲林縣, not in R1c's Changhua map).

Baselines ("Q0", lead decision before any fit): BOTH seeds' Q0 baselines are rerun fresh
at the hook commit -- Q0's recipe on the seed-S fold partition (run_chj.sh q0folds S ->
stage3_datum_gate_q0_f<S>; its folds are also CH-J's fold parents, ruling R8.1e), that run's
own theta's three-zone column (Q0's column recipe, uncapped) on seed-S leveling folds and its
forward (run_chj.sh q0base S). No archived baseline is used: every baseline (and CH-J) file
must record the engineering record's hook_commit, clean (``check_run_provenance``). The
fresh full fit's departure from Q0's registered theta (the CH-J full fit's parent) is
reported (``q0_refit_max_abs_theta_diff``).

``--seed`` assigns folds only: calibrate_flow's full fit does not read it, so each seed's
full fit (and its Changhua sw fractions) is the same recipe, equal up to GPU
non-determinism. The two seeds are therefore not independent evidence on the sw outcome;
both values and their spread are recorded.

Looks: before ``assemble`` reads any observation (stage3_per_entry.npz heads, the columns'
out-of-fold leveling), it logs ONE design-time disclosure look (``log_noharm_look``, purpose
'design_time') on the calibration wells' heads 2012-01..2022-12 and the 2012-21 leveling
benchmarks (I1:leveling_gates_2012_2021), citing the 2026-10-23 history_disclosure row
(user ruling 2026-10-23: the I1 ladder's rung gates read these units without logged looks).
A rerun re-uses an identical existing row instead of logging it twice. No external screen's
observation is opened here (chj_report.py logs the externals' after_the_fact looks).

Posterior tools: a CH-J theta holds keys fixed that are not parameters (``meta.chj_hook``,
``-1000`` share logits); this producer refuses any command line that hands one to a
posterior tool (``refuse_posterior``), and so does run_chj.sh.

    python dataprep_staging/ch_x/chj_produce.py hindcast --seed 2027 [--device cuda]
    python dataprep_staging/ch_x/chj_produce.py hindcast-q0 [--device cuda]
    python dataprep_staging/ch_x/chj_produce.py assemble
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "i1"),
           str(HERE.parent / "x5_energy_volume")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402
import chx_lib as L  # noqa: E402

PRODUCER = "dataprep_staging/ch_x/chj_produce.py"
VERSION = "1.1.0"
CHJ = "results/twin_runs/chx/chj"
ENG = f"{CHJ}/chj_engineering.json"
HC_DIR = f"{CHJ}/hindcast"
PANEL = f"{CHJ}/chj_panel.npz"
METRICS = f"{CHJ}/chj_metrics.json"
SUMMARY = f"{CHJ}/chj_panel.json"
CHX_PANEL = "results/twin_runs/chx/hindcast/chx_panel.npz"
SW_PROV = "results/twin_runs/chx/chj_sw/chj_sw.json"          # chj_forcing.build_sw
RUNS = "results/twin_runs"
FWD = "results/twin_forward"
TWINMODEL_TOL_M = 1e-9
N_FOLDS = 5                                                   # Q0's recipe --n-folds
#: R8.1c (user_rulings_2026-11-03): the fraction fed to chj_block is the max of the
#: Changhua deliveries [0] and percolation [2] entries of log_sw_scale; both reported
SW_ENTRIES = {"deliveries": 0, "percolation": 2}
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
#: R8.1f: CH-J's columns are capped globally at chx_const.CHJ_COLUMN_CAP; Q0's are not
COLUMN_CAP = {"chj": K.CHJ_COLUMN_CAP, "q0": None}
#: tools that sample or perturb a theta's parameters (CH-J thetas are unsupported there)
POSTERIOR_TOOLS = ("hydrophysics.twin.uncertainty", "twin/uncertainty.py",
                   "n1_gaps/identifiability", "identifiability.py", "stage3_posterior")
#: the earlier rungs' metric code this producer calls (recorded by sha256 in the metrics)
REUSED_CODE = (("i1", "score_g2.py"), ("i1", "score_rung_b.py"),
               ("x5_energy_volume", "score_rung_x5cs.py"), ("x5_energy_volume", "uplift.py"),
               ("x5_energy_volume", "prereg_x5cs.py"))
SEED_NOTE = ("--seed assigns folds only: the full fit (and its Changhua sw fractions) does "
             "not read it, so both seeds' full fits are one recipe, equal up to GPU "
             "non-determinism, and are not independent evidence on the sw outcome")
#: the design-time disclosure look of the no-harm reads (module docstring)
HISTORY_ROW = {"time": "2026-10-23T02:42:44.765773Z", "purpose": "history_disclosure",
               "wp": "I1", "units": ["I1:leveling_gates_2012_2021",
                                     "I1:temporal_screens_2020_2022"]}
LOOK_UNIT_WELLS = "I1CHJ:noharm_calibration_wells_2012_2022"
LOOK_UNIT_LEVELING = "I1:leveling_gates_2012_2021"
LOOK_SCOPE = "heads+leveling:2012-01..2022-12"
LOOK_PURPOSE = "design_time"
LOOK_NOTE = ("CH-J no-harm reads (I1.CHX.v1 chj.gates.noharm, 'spent units, design-time'), "
             "logged before chj_produce.py assemble opens them: held-out calibration-well "
             "heads 2012-01..2022-12 in stage3_per_entry.npz of CH-J's and the fresh Q0 "
             "baselines' k-folds (Yunlin k-fold, county_corr) and the columns' out-of-fold "
             "leveling on the 2012-21 surveys (oof and Changhua leveling). The same "
             "calibration-domain uses that the history_disclosure row of "
             "2026-10-23T02:42:44.765773Z (WP I1, logged by N1; user ruling 2026-10-23) "
             "disclosed for every I1 rung gate; no external screen is read")
LOOKS_NOTE = ("one design_time disclosure look (LOOK_UNIT_WELLS, I1:leveling_gates_2012_2021) "
              "logged before any observation was read, citing the 2026-10-23 "
              "history_disclosure row (user ruling 2026-10-23); no external screen "
              "observation is opened")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def chj_suffix(seed: int) -> str:
    """CH-J's run directory suffix (results/twin_runs/stage3_datum_gate_<suffix>)."""
    return f"chj_s{int(seed)}"


def q0_suffix(seed: int) -> str:
    """Q0's fresh baseline for fold partition ``seed`` (module docstring)."""
    return f"q0_f{int(seed)}"


def run_path(suffix: str) -> Path:
    return IO.rp(RUNS) / f"stage3_datum_gate_{suffix}"


def refuse_posterior(argv) -> None:
    """SystemExit if a command line hands a theta to a posterior tool (finding 10): a CH-J
    theta's fixed keys and -1000 share logits are not parameters."""
    hit = [a for a in map(str, argv) for t in POSTERIOR_TOOLS if t in a]
    if hit:
        raise SystemExit(f"REFUSED: {hit} is a posterior tool; posterior tools are "
                         "unsupported on a CH-J theta (meta.chj_hook)")


def _theta_doc(path) -> dict:
    return json.loads(IO.rp(path).read_text())


def _sha(path) -> str:
    return IO.sha256(IO.rp(path))


def commit_ok(recorded, hook_commit) -> bool:
    """A recorded short or full commit names ``hook_commit`` (either a prefix of the other,
    at least 7 hex digits) and is not marked dirty."""
    r, h = str(recorded or "").strip(), str(hook_commit or "").strip()
    if len(r) < 7 or len(h) < 7 or r.endswith("-dirty"):
        return False
    return h.startswith(r) or r.startswith(h)


def check_hook_theta(doc: dict, q0_sha: str) -> None:
    """A CH-J full-fit theta: frozen at Q0 (sha256), the shares and the sw split on, and
    the posterior marker present."""
    m = doc.get("meta") or {}
    probs = []
    if m.get("freeze_from_sha256") != q0_sha:
        probs.append("not frozen at Q0's theta")
    if m.get("pump_layer_shares_county") != list(K.CHJ_SHARES_CHANGHUA):
        probs.append("not the X3 shares")
    if not m.get("sw_county_split"):
        probs.append("no --sw-county-split")
    if not (m.get("chj_hook") or {}).get("posterior_tools_unsupported"):
        probs.append("no chj_hook marker")
    if probs:
        raise SystemExit("REFUSED: CH-J theta " + "; ".join(probs))


def check_hook_applied(run, q0_theta, seed: int, require_folds: int | None = None,
                       q0_fold_dir=None) -> dict:
    """run_chj.sh's ``hook_applied`` (findings 8, R8.1e): the full fit is frozen at Q0 with
    the shares, the sw split and the chj_hook marker, its pinned keys equal Q0's; and every
    fold file of ``run`` (a) ran from the parent of ITS fold and seed, (b) whose own theta
    equals Q0's fold-K theta (``record.theta`` of the Q0 fold file the parent names, which
    must be unchanged since) on every pinned key -- a parent rebuilt from another theta is
    refused --, (c) holds out exactly the entries and sids of that Q0 fold and (d) keeps its
    pinned keys at the parent's values; stage3_fold_thetas.json (if merged) agrees. With
    ``require_folds`` N, folds 0..N-1 must all be present; with ``q0_fold_dir``, every
    parent's Q0 fold file must be that directory's. SystemExit on any failure."""
    run = IO.rp(run)
    doc = json.loads((run / "stage3_theta.json").read_text())
    check_hook_theta(doc, _sha(q0_theta))
    m, th = doc["meta"], doc["theta"]
    if "pump_layer_logits_ch" not in th:
        raise SystemExit("REFUSED: the theta carries no county shares")
    pinned = [k for k in m["freeze_partition"]["pinned"] if not k.startswith("log_sw_scale[")]
    p0 = _theta_doc(q0_theta)["theta"]
    if any(th[k] != p0[k] for k in pinned):
        raise SystemExit("REFUSED: a pinned key of the full fit moved off Q0's value")
    parents = {}
    for f in sorted(run.glob("stage3_fold[0-9]*.json")):
        fd = json.loads(f.read_text())
        k = int(fd["fold"])
        cp = fd.get("chj_parent") or {}
        if not cp or _sha(cp["path"]) != cp["sha256"]:
            raise SystemExit(f"REFUSED: fold {k}: parent missing or changed")
        par = _theta_doc(cp["path"])
        fp = (par.get("meta") or {}).get("chj_fold_parent") or {}
        if not (fp and int(fp["fold"]) == k and int(fp["seed"]) == int(seed)
                == int(fd["seed"])):
            raise SystemExit(f"REFUSED: fold {k} (seed {fd['seed']}) ran from the parent of "
                             f"fold {fp.get('fold')} / seed {fp.get('seed')}")
        src = IO.rp(fp["source"])
        if q0_fold_dir is not None and src.resolve() != (
                IO.rp(q0_fold_dir) / f"stage3_fold{k}.json").resolve():
            raise SystemExit(f"REFUSED: fold {k}: its parent comes from {fp['source']}, not "
                             f"the fresh Q0 folds of {q0_fold_dir}")
        if _sha(src) != fp["source_sha256"]:
            raise SystemExit(f"REFUSED: fold {k}: Q0's fold file changed since the parent")
        q = json.loads(src.read_text())
        if any(par["theta"].get(x) != q["record"]["theta"].get(x) for x in pinned):
            raise SystemExit(f"REFUSED: fold {k}: the parent's theta is not Q0's fold-{k} "
                             "theta on the pinned keys (ruling R8.1e)")
        for key in ("n_folds", "seed", "sids"):
            if q[key] != fd[key]:
                raise SystemExit(f"REFUSED: fold {k}: {key} differs from Q0's fold {k}")
        if q["record"]["entry"] != fd["record"]["entry"]:
            raise SystemExit(f"REFUSED: fold {k} holds out other entries than Q0's fold {k}")
        if any(fd["record"]["theta"][x] != par["theta"][x] for x in pinned):
            raise SystemExit(f"REFUSED: fold {k}: a pinned key moved off its parent's value")
        parents[k] = par["theta"]
    if require_folds is not None and sorted(parents) != list(range(int(require_folds))):
        raise SystemExit(f"REFUSED: folds {sorted(parents)} checked, "
                         f"{list(range(int(require_folds)))} required")
    ft = run / "stage3_fold_thetas.json"
    if ft.exists():
        recs = json.loads(ft.read_text())
        if sorted(int(r["fold"]) for r in recs) != sorted(parents):
            raise SystemExit("REFUSED: stage3_fold_thetas.json and the fold files differ")
        for r in recs:
            t = r["theta"]
            if "pump_layer_logits_ch" not in t or any(
                    t[x] != parents[int(r["fold"])][x] for x in pinned):
                raise SystemExit(f"REFUSED: merged fold {r['fold']}: shares or pinned keys")
    return {"ok": True, "folds_checked": sorted(parents)}


def check_run_provenance(suffix: str, seed: int, hook_commit: str, role: str) -> dict:
    """A CH-J run (role 'chj') or a fresh Q0 baseline (role 'q0') was fitted, merged,
    column-fitted and forwarded at the hook commit (lead decision: no archived baseline):
    the theta meta (git_commit, git_commit_full, git_dirty False), stage3_flow.csv and every
    fold file record ``hook_commit``; the column names this run's theta, its leveling folds
    are ``seed`` and its Ske/Skv cap is R8.1f's (CH-J 0.1 globally, Q0 uncapped); the
    forward npz was rolled out from this very theta (its ``gate`` is the theta's meta, its
    member label the run). SystemExit otherwise. Reads metadata only."""
    import pandas as pd

    d = run_path(suffix)
    theta = d / "stage3_theta.json"
    meta = json.loads(theta.read_text())["meta"]
    probs = []
    if not (commit_ok(meta.get("git_commit"), hook_commit)
            and commit_ok(meta.get("git_commit_full"), hook_commit)
            and meta.get("git_dirty") is False):
        probs.append(f"theta fitted at {meta.get('git_commit')} (dirty "
                     f"{meta.get('git_dirty')}), not the hook commit")
    flow = pd.read_csv(d / "stage3_flow.csv", dtype={"git_commit": str}).iloc[0]
    if not commit_ok(flow.get("git_commit"), hook_commit) or int(flow["seed"]) != int(seed):
        probs.append(f"stage3_flow.csv: commit {flow.get('git_commit')}, seed {flow['seed']}")
    for k in range(N_FOLDS):
        f = d / f"stage3_fold{k}.json"
        if not f.is_file():
            probs.append(f"fold file {k} missing")
            continue
        cfg = json.loads(f.read_text())["cfg"]
        if not commit_ok(cfg.get("git_commit"), hook_commit):
            probs.append(f"fold {k} run at {cfg.get('git_commit')}")
    col = json.loads((d / "coupled_leveling" / "vep_zonal_leveling.json").read_text())
    if IO.rp(str(col.get("theta", ""))).resolve() != theta.resolve():
        probs.append(f"the column was fitted on {col.get('theta')}, not this run's theta")
    if int(col.get("fold_seed", 0)) != int(seed):
        probs.append(f"column leveling fold seed {col.get('fold_seed')}")
    cap = COLUMN_CAP[role]
    got = col.get("ske_skv_max")
    if (cap is None and got is not None) or (cap is not None and (
            got is None or abs(float(got) - cap) > 1e-12)):
        probs.append(f"column Ske/Skv cap {got}, R8.1f requires {cap} for {role}")
    z = np.load(IO.rp(FWD) / f"datum_gate_{suffix}.npz", allow_pickle=False)
    if json.loads(str(z["gate"])) != meta:
        probs.append("the forward npz was not rolled out from this theta (gate != meta)")
    if [str(v) for v in z["member_labels"]] != [d.name]:
        probs.append(f"forward members {list(z['member_labels'])}")
    if probs:
        raise SystemExit(f"REFUSED: {suffix}: " + "; ".join(probs))
    return {"git_commit": meta.get("git_commit"), "column_ske_skv_max": got,
            "theta_sha256": IO.sha256(theta)}


def check_sw(doc: dict, eng: dict) -> str:
    """The CH-J theta's canal field is the one the engineering record names (``sw_sha256``)
    and build_sw produced (chj_sw.json ``output_sha256``)."""
    got = (doc.get("meta") or {}).get("sw_recharge_sha256")
    want = eng.get("sw_sha256")
    probs = []
    if not want:
        probs.append("the engineering record names no sw_sha256")
    elif got != want:
        probs.append("the theta's sw_recharge sha256 differs from the engineering record's")
    prov = IO.rp(SW_PROV)
    if prov.is_file() and json.loads(prov.read_text()).get("output_sha256") != got:
        probs.append("the theta's sw_recharge sha256 differs from chj_sw.json")
    if probs:
        raise SystemExit("REFUSED: " + "; ".join(probs))
    return got


# ----------------------------------------------------------------------------- sw outcome
def sw_fractions(theta: dict) -> dict:
    """R8.1c: the Changhua deliveries [0] and percolation [2] fractions of log_sw_scale
    (``[c0 Changhua, c0 rest, c1 Changhua, c1 rest]``) and the fed value, their max."""
    v = np.ravel(np.asarray(theta["log_sw_scale"], dtype="float64"))
    if v.size != 4:
        raise SystemExit(f"log_sw_scale has {v.size} entries; CH-J's county split has 4")
    out = {k: float(math.exp(v[i])) for k, i in SW_ENTRIES.items()}
    out["fed"] = max(out["deliveries"], out["percolation"])
    out["rule"] = ("R8.1c: fed = max(deliveries [0], percolation [2]); 'co-located recharge "
                   "rejected' only when both sit at the 0.01 bound")
    return out


# ----------------------------------------------------------------------------- hindcast
def screens(P: dict) -> dict:
    """The CHX panel's screens: {'ch'|'yl': (sid, x, y, layer, cell)}."""
    return {n: tuple(np.asarray(P[f"{k}_{n}"]) for k in ("sid", "x", "y", "layer", "cell"))
            for n in ("ch", "yl")}


def check_cells(m, P: dict) -> None:
    for n, (_, x, y, _, cel) in screens(P).items():
        if not np.array_equal(m.grid_cells(x, y), np.asarray(cel, int)):
            raise SystemExit(f"ENGINEERING: the {n} screens' cells differ from the CHX panel")


class _GridCells:
    """``grid_cells`` (hindcast.R1cRun's rule) on any grid, e.g. Q0's TwinModel's."""

    def __init__(self, grid):
        self.grid = grid

    def grid_cells(self, x, y) -> np.ndarray:
        out = []
        for a, b in zip(np.asarray(x, float), np.asarray(y, float), strict=True):
            i = self.grid.active_index(float(a), float(b))
            out.append(-1 if i is None else int(i))
        return np.array(out, int)


def check_axis(dates) -> None:
    """The model's input axis is the CHX run axis: 2012-01..2022-12 (132 months)."""
    run = L.month_list(*K.RUN_MONTHS)
    got = [str(v)[:7] for v in dates]
    if len(got) != len(run) or got[0] != run[0]:
        raise SystemExit(f"the input axis starts {got[:1]} with {len(got)} months, not "
                         f"{run[0]} with {len(run)}")


def chj_arms(m, P: dict) -> dict:
    """Arms A (recorded) and B (Changhua at its PATTERN_YEAR pattern) of runner ``m``
    (``hindcast.R1cRun``'s interface: E (A, T), changhua (A,), heads(E), grid_cells) at the
    CHX panel's screens, over the CHX window. Returns {'A_ch', 'B_ch', 'A_yl', 'B_yl'}."""
    run = L.month_list(*K.RUN_MONTHS)
    if [str(v) for v in P["run_months"]] != run:
        raise SystemExit("the CHX panel's run axis is not 2012-01..2022-12")
    ch = np.asarray(m.changhua, bool)
    if not np.array_equal(ch, np.asarray(P["changhua_cells"], bool)):
        raise SystemExit("ENGINEERING: CH-J's county map differs from the CHX panel's")
    check_cells(m, P)
    H = {"A": m.heads(m.E), "B": m.heads(L.pattern_energy(m.E, run, ch, K.PATTERN_YEAR))}
    if not all(np.isfinite(h).all() for h in H.values()):
        raise SystemExit("ENGINEERING: non-finite heads")
    ws = L.window_slice(run, *K.CHX_WINDOW)
    out = {}
    for n, (_, _, _, lay, cel) in screens(P).items():
        for a, h in H.items():
            out[f"{a}_{n}"] = L.screen_series(h, lay, cel)[:, ws]
    out["_H_A"] = H["A"]
    return out


def _write_once(path: Path, arrays: dict, summary: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fh:
        np.savez(fh, **arrays)
    summary["npz"] = str(path)
    summary["npz_sha256"] = IO.sha256(path)
    with path.with_suffix(".json").open("x") as f:
        json.dump(summary, f, indent=1)
    return summary


def _chx_panel() -> dict:
    p = IO.rp(CHX_PANEL)
    summ = json.loads((p.parent / "chx_panel.json").read_text())
    if IO.sha256(p) != summ["panel_sha256"]:
        raise SystemExit("the CHX panel differs from chx_panel.json")
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files} | {"_sha256": summ["panel_sha256"]}


def hindcast(seed: int, device: str) -> dict:
    import hindcast as H
    from n1_policy.policy_lib import TwinModel

    out = IO.rp(HC_DIR) / f"chj_hindcast_s{int(seed)}.npz"
    if out.exists():
        raise SystemExit(f"{out} exists (archive it first)")
    run = run_path(chj_suffix(seed))
    theta = run / "stage3_theta.json"
    check_hook_theta(_theta_doc(theta), IO.sha256(IO.rp(K.Q0_THETA)))
    column = run / "coupled_leveling" / "vep_zonal_leveling.json"
    P = _chx_panel()
    m = H.R1cRun(theta=str(theta), device=device)
    check_axis(m.inp.dates)
    arms = chj_arms(m, P)
    d = float(np.max(np.abs(TwinModel(str(theta), str(column), device=device).heads()
                            - arms.pop("_H_A"))))
    if d > TWINMODEL_TOL_M:
        raise SystemExit(f"ENGINEERING: arm A differs from policy_lib.TwinModel by {d} m")
    summ = {"what": f"CH-J seed {seed} hindcast arms A/B at the CHX screens (model only)",
            "created": _now(), "seed": int(seed), "theta": str(theta),
            "theta_sha256": IO.sha256(theta), "column_sha256": IO.sha256(column),
            "code_head": H.code_head(), "device": device, "chx_panel_sha256": P["_sha256"],
            "twinmodel_max_abs_diff_m": d,
            "arms": {"A": "recorded", "B": f"Changhua cells at their {K.PATTERN_YEAR} monthly "
                     f"pattern from {K.PATTERN_YEAR + 1}-01"},
            "observed_heads_read": False, "producer_sha256": IO.sha256(Path(__file__))}
    return _write_once(out, arms, summ)


def hindcast_q0(device: str) -> dict:
    """Q0's free run: Q0's registered theta (the CH-J full fit's parent) and, for
    TwinModel's construction only (heads do not depend on it), the fresh seed-2027 Q0
    column."""
    import hindcast as H
    from n1_policy.policy_lib import TwinModel

    out = IO.rp(HC_DIR) / "q0_hindcast.npz"
    if out.exists():
        raise SystemExit(f"{out} exists (archive it first)")
    P = _chx_panel()
    column = run_path(q0_suffix(2027)) / "coupled_leveling" / "vep_zonal_leveling.json"
    tm = TwinModel(str(IO.rp(K.Q0_THETA)), str(column), device=device)
    check_axis(tm.inp.dates)
    check_cells(_GridCells(tm.inp.grid), P)               # the screens on Q0's own grid
    Hq = tm.heads()
    if not np.isfinite(Hq).all():
        raise SystemExit("ENGINEERING: non-finite heads")
    ws = L.window_slice([str(v) for v in P["run_months"]], *K.CHX_WINDOW)
    arrays = {f"A_{n}": L.screen_series(Hq, lay, cel)[:, ws]
              for n, (_, _, _, lay, cel) in screens(P).items()}
    summ = {"what": "Q0 free-run hindcast (arm A) at the CHX screens (model only)",
            "created": _now(), "theta": K.Q0_THETA,
            "theta_sha256": IO.sha256(IO.rp(K.Q0_THETA)), "column": str(column),
            "code_head": H.code_head(), "device": device, "chx_panel_sha256": P["_sha256"],
            "observed_heads_read": False, "producer_sha256": IO.sha256(Path(__file__))}
    return _write_once(out, arrays, summ)


# ----------------------------------------------------------------------------- looks
def _look_key(rec: dict) -> str:
    """The fields that make two look rows the same look (record_look stores members as {}
    when none are given)."""
    k = {f: rec.get(f) for f in ("wp", "purpose", "test_id", "prereg_id", "data_scope", "note")}
    k["units"] = sorted(map(str, rec.get("units") or []))
    k["members"] = rec.get("members") or {}
    return json.dumps(k, sort_keys=True, ensure_ascii=False, default=str)


def find_identical_look(rec: dict, prereg_dir=None) -> dict | None:
    """An existing look row identical to ``rec`` on everything but its times (so a rerun
    re-uses it instead of logging the same look twice), or None."""
    from harness import prereg3

    d = Path(prereg3.PREREG_DIR if prereg_dir is None else prereg_dir)
    want = _look_key(rec)
    for lk in prereg3.read_chained(d / "looks.jsonl"):
        if _look_key(lk) == want:
            return lk
    return None


def log_look_once(units, data_scope: str, purpose: str, *, members=None, test_id=None,
                  prereg_id=None, note: str = "", prereg_dir=None, rows_dir=None) -> dict:
    """``prereg3.record_look`` unless an identical row exists (then that row)."""
    from harness import prereg3

    rec = {"wp": K.WP, "purpose": purpose, "test_id": test_id, "prereg_id": prereg_id,
           "units": list(units), "data_scope": data_scope, "members": members, "note": note}
    old = find_identical_look(rec, prereg_dir)
    if old is not None:
        return old
    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir, "rows_dir": rows_dir}
    return prereg3.record_look(list(units), data_scope, K.WP, purpose, members=members,
                               test_id=test_id, prereg_id=prereg_id, note=note, **kw)


def noharm_sids(suffixes) -> list[str]:
    """The calibration wells of the runs (stage3_fit_summary.json ``sids``, metadata)."""
    out = set()
    for s in suffixes:
        out |= set(map(str, json.loads((run_path(s) / "stage3_fit_summary.json")
                                       .read_text())["sids"]))
    return sorted(out)


def log_noharm_look(sids, prereg_dir=None, rows_dir=None) -> dict:
    """The design-time disclosure look of the no-harm reads, before any is opened."""
    members = {LOOK_UNIT_WELLS: {
        "description": "the calibration wells of CH-J's and the fresh Q0 baselines' k-folds "
                       "(held-out heads in stage3_per_entry.npz)",
        "ids": list(sids), "window": "2012-01..2022-12"}}
    return log_look_once([LOOK_UNIT_WELLS, LOOK_UNIT_LEVELING], LOOK_SCOPE, LOOK_PURPOSE,
                         members=members, test_id=K.CHJ_TEST, prereg_id=K.CHX_ID,
                         note=LOOK_NOTE, prereg_dir=prereg_dir, rows_dir=rows_dir)


# ----------------------------------------------------------------------------- metrics
def _mods():
    import score_g2 as S
    import score_rung_b as B
    import score_rung_x5cs as X
    from prereg_x5cs import RATE_TOL
    from uplift import uplift_share

    return S, B, X, RATE_TOL, uplift_share


def yunlin_kfold(suffix: str, st) -> dict:
    """The Yunlin k-fold: calibrate_flow._r2 (stage3_flow.csv r2_kfold's pooled formula)
    over the Yunlin held-out entries of stage3_per_entry.npz, county by score_g2's
    assignment (stage3_kfold_wells.csv sid -> CountyName)."""
    import pandas as pd

    from hydrophysics.twin.calibrate_flow import _r2

    S = _mods()[0]
    d = S.REPO / f"results/twin_runs/stage3_datum_gate_{suffix}"
    z = np.load(d / "stage3_per_entry.npz", allow_pickle=True)
    w = pd.read_csv(d / "stage3_kfold_wells.csv", dtype={"sid": str}).set_index("entry")
    county = st.loc[w.loc[z["entry"], "sid"].to_numpy(), "CountyName"].to_numpy()
    sel = county == S.COUNTIES["yunlin"]
    return {"r2": float(_r2(np.asarray(z["pred"], float)[sel],
                            np.asarray(z["obs"], float)[sel])),
            "n_entries": int(sel.sum())}


def noharm_metrics(suffix: str, seed: int, st) -> dict:
    """{yunlin_kfold, yunlin_county_corr, oof_leveling, changhua_leveling} of one run on
    fold partition ``seed`` (score_g2.metrics checks the flow seed and the leveling fold
    seed), and what was reported beside them."""
    S, _, X, _, _ = _mods()
    g = S.metrics(suffix, st, int(seed))
    col = X.column(suffix, int(seed))
    yk = yunlin_kfold(suffix, st)
    return {"yunlin_kfold": yk["r2"], "yunlin_county_corr": g["county_corr:yunlin"],
            "oof_leveling": g["leveling_r2"], "changhua_leveling": col["leveling_changhua"],
            "reported": {"kfold_r2_fan": g["kfold_r2"], "n_yunlin_entries": yk["n_entries"],
                         "county_corr_changhua": g["county_corr:changhua"],
                         "leveling_rest": col["leveling_rest"],
                         "ske_skv_max": col["ske_skv_max"],
                         "ske_skv_by_zone": col["ske_skv_by_zone"]}}


def county_subsidence_m(suffix: str, cells: np.ndarray) -> float:
    """County-mean forward subsidence (m, + = sinking) Dec 2022 -> Dec 2025 on ``cells``,
    from the series and months score_rung_b.plausibility uses."""
    import pandas as pd

    B = _mods()[1]
    z = np.load(B.FWD / f"datum_gate_{suffix}.npz")
    d = pd.DatetimeIndex(z["dates"])
    s = (z["subs_mean_by_rheology"][0][0] if "subs_mean_by_rheology" in z.files
         else z["subs_mean"][0])
    if s.shape[0] != np.asarray(cells).shape[0]:
        raise SystemExit(f"datum_gate_{suffix}: {s.shape[0]} cells, the mask "
                         f"{np.asarray(cells).shape[0]}")
    i22 = np.where(d <= pd.Timestamp("2022-12-31"))[0][-1]
    i25 = np.where(d <= pd.Timestamp("2025-12-31"))[0][-1]
    c = np.asarray(cells, bool)
    return float((s[c, i25] - s[c, i22]).mean())


def forward_metrics(suffix: str, yunlin_cells: np.ndarray) -> dict:
    """The forward gates' quantities; the npz's provenance (this run's theta) is checked
    by ``check_run_provenance`` before."""
    _, B, _, tol, uplift_share = _mods()
    ch = np.load(B.CMAP)["changhua"].astype(bool)
    up = uplift_share(suffix, tol)
    return {"subsidence_2023_25_changhua_m": county_subsidence_m(suffix, ch),
            "subsidence_2023_25_yunlin_m": county_subsidence_m(suffix, yunlin_cells),
            "rising_share_changhua": up["uplift_share_ch"],
            "reported": {"rising_share_rest": up["uplift_share_rest"],
                         "rate_tol_cm_per_yr": tol, **B.plausibility(suffix)}}


def theta_diff(a: dict, b: dict) -> float:
    """Max |a - b| over the numeric keys both thetas carry (report only)."""
    out = 0.0
    for k in set(a) & set(b):
        try:
            x = np.ravel(np.asarray(a[k], dtype="float64"))
            y = np.ravel(np.asarray(b[k], dtype="float64"))
        except (TypeError, ValueError):
            continue
        if x.shape == y.shape and x.size:
            out = max(out, float(np.max(np.abs(x - y))))
    return out


def seed_metrics(seed: int, st, yunlin_cells: np.ndarray, q0_sha: str) -> dict:
    """One seed's block of chj_metrics.json (the keys chj_block reads, plus reported)."""
    cs, qs = chj_suffix(seed), q0_suffix(seed)
    doc = _theta_doc(run_path(cs) / "stage3_theta.json")
    check_hook_theta(doc, q0_sha)
    sw = sw_fractions(doc["theta"])
    q0_refit = _theta_doc(run_path(qs) / "stage3_theta.json")["theta"]
    return {"noharm": noharm_metrics(cs, seed, st), "q0": noharm_metrics(qs, seed, st),
            "forward": forward_metrics(cs, yunlin_cells),
            "forward_q0": forward_metrics(qs, yunlin_cells),
            "sw_fraction_changhua": sw["fed"], "sw_fractions_changhua": sw,
            "runs": {"chj": cs, "q0": qs},
            "q0_refit_max_abs_theta_diff": theta_diff(
                q0_refit, _theta_doc(K.Q0_THETA)["theta"])}


DEFINITIONS = {
    "yunlin_kfold": "calibrate_flow._r2 over the Yunlin held-out entries of "
                    "stage3_per_entry.npz (score_g2 county assignment)",
    "yunlin_county_corr": "score_g2.metrics county_corr:yunlin",
    "oof_leveling": "score_g2.metrics leveling_r2 (vep_zonal_leveling.json r2_outoffold)",
    "changhua_leveling": "score_rung_x5cs.column leveling_changhua "
                         "(r2_outoffold_by_zone county:changhua)",
    "subsidence_2023_25_<county>_m": "county-mean forward subsidence Dec 2022 -> Dec 2025 "
                                     "(m, + sinking), score_rung_b.plausibility's series; "
                                     "Changhua = zone_county_map v2, Yunlin = the CHX "
                                     "panel's yunlin_cells",
    "rising_share_changhua": "uplift.uplift_share(suffix, prereg_x5cs.RATE_TOL) "
                             "uplift_share_ch",
    "sw_fraction_changhua": "R8.1c: max of exp(log_sw_scale[0]) (deliveries) and "
                            "exp(log_sw_scale[2]) (percolation); both in "
                            "sw_fractions_changhua",
    "baselines": "Q0 = q0_f<seed>: Q0's recipe rerun at the hook commit on the seed's fold "
                 "partition, its own theta's three-zone uncapped column and forward (lead "
                 "decision; no archived baseline)",
    "columns_r8_1f": "CH-J's column (six-zone county split) is capped Ske/Skv <= 0.1 in EVERY "
                     "zone (ruling R8.1f, global --ske-skv-max 0.1, the q0_csk01 precedent): "
                     "the Yunlin column zones are capped too, not Changhua's alone; Q0's "
                     "baseline column is uncapped",
    "sw_pinned_values": "pin_frozen (chj_forcing, registered) pins the rest-of-fan "
                        "log_sw_scale entries at the per-component initial values: [1] = "
                        "0.15 where a four-entry calibration starts it at 0.25, [3] = 0.25; "
                        "no numerical effect, their field is identically zero",
}


def build_metrics(seeds=K.CHJ_SEEDS) -> dict:
    import pandas as pd

    S = _mods()[0]
    st = pd.read_parquet(S.STATIONS, columns=["sid", "CountyName"]).set_index("sid")
    P = _chx_panel()
    q0_sha = IO.sha256(IO.rp(K.Q0_THETA))
    out = {"seeds": {str(s): seed_metrics(s, st, P["yunlin_cells"], q0_sha) for s in seeds}}
    fed = [v["sw_fraction_changhua"] for v in out["seeds"].values()]
    out.update({"definitions": DEFINITIONS, "seed_note": SEED_NOTE,
                "sw_fraction_seed_spread": float(max(fed) - min(fed)),
                "baselines": {str(s): q0_suffix(s) for s in seeds}, "looks": LOOKS_NOTE,
                "rulings": RULINGS, "rulings_sha256": IO.sha_or_missing(IO.rp(RULINGS)),
                "chx_panel_sha256": P["_sha256"],
                "code_sha256": {f"dataprep_staging/{d}/{f}": IO.sha_or_missing(
                    HERE.parent / d / f) for d, f in REUSED_CODE}})
    return out


# ----------------------------------------------------------------------------- panel
def _hindcast_summary(f: Path, me: str, hook_commit: str, panel_sha: str) -> dict:
    summ = json.loads(f.with_suffix(".json").read_text())
    probs = []
    if summ.get("npz_sha256") != IO.sha256(f):
        probs.append("npz differs from its summary")
    if summ.get("producer_sha256") != me:
        probs.append("written by another producer")
    if not commit_ok(summ.get("code_head"), hook_commit):
        probs.append(f"built at {summ.get('code_head')}, not the hook commit")
    if summ.get("chx_panel_sha256") != panel_sha:
        probs.append("built on another CHX panel")
    if probs:
        raise SystemExit(f"REFUSED: {f}: " + "; ".join(probs))
    return summ


def build_panel(hook_commit: str, seeds=K.CHJ_SEEDS) -> tuple[dict, dict]:
    """chj_panel.npz's arrays from the hindcast files, after their provenance checks
    (this producer, the hook commit, the current thetas, the CHX panel)."""
    P = _chx_panel()
    me = IO.sha256(Path(__file__))
    hc = IO.rp(HC_DIR)
    prov = {}
    arrays = {"months": np.asarray(P["months"]), "sid_ch": np.asarray(P["sid_ch"]),
              "sid_yl": np.asarray(P["sid_yl"])}
    for s in seeds:
        f = hc / f"chj_hindcast_s{int(s)}.npz"
        summ = _hindcast_summary(f, me, hook_commit, P["_sha256"])
        theta = run_path(chj_suffix(s)) / "stage3_theta.json"
        if summ["theta_sha256"] != IO.sha256(theta):
            raise SystemExit(f"{f}: not the hindcast of the current seed-{s} theta")
        z = np.load(f, allow_pickle=False)
        arrays[f"chjA_ch_{s}"], arrays[f"chjB_ch_{s}"] = z["A_ch"], z["B_ch"]
        arrays[f"chjA_yl_{s}"] = z["A_yl"]
        arrays[f"chjB_yl_{s}"] = z["B_yl"]                       # report only
        prov[str(s)] = {"hindcast_sha256": summ["npz_sha256"],
                        "theta_sha256": summ["theta_sha256"], "code_head": summ["code_head"]}
    f = hc / "q0_hindcast.npz"
    summ = _hindcast_summary(f, me, hook_commit, P["_sha256"])
    if summ["theta_sha256"] != IO.sha256(IO.rp(K.Q0_THETA)):
        raise SystemExit(f"{f}: not Q0's registered theta")
    z = np.load(f, allow_pickle=False)
    arrays["q0_ch"], arrays["q0_yl"] = z["A_ch"], z["A_yl"]
    prov["q0"] = {"hindcast_sha256": summ["npz_sha256"], "theta_sha256": summ["theta_sha256"],
                  "code_head": summ["code_head"]}
    check_panel_shapes(arrays, P)
    return arrays, prov


def check_panel_shapes(arrays: dict, P: dict) -> None:
    """Every model array, the per-seed arm keys included, has the CHX panel's
    (screens, window months) shape for its county."""
    T = len(P["months"])
    want = {"ch": (len(P["sid_ch"]), T), "yl": (len(P["sid_yl"]), T)}
    for k, v in arrays.items():
        if k in ("months", "sid_ch", "sid_yl"):
            continue
        parts = k.split("_")
        county = parts[1] if len(parts) > 1 else None
        if county not in want or np.shape(v) != want[county]:
            raise SystemExit(f"{k} has shape {np.shape(v)}, the CHX panel "
                             f"{want.get(county)}")


def _engineering() -> tuple[Path, dict]:
    import score_chx as SC

    eng_p = IO.rp(ENG)
    if not eng_p.is_file():
        raise SystemExit(f"{ENG} missing: the lead freezes it before any CH-J fit")
    eng = json.loads(eng_p.read_text())
    miss = [k for k in SC.ENGINEERING_KEYS if k not in eng]
    if miss:
        raise SystemExit(f"{ENG} lacks {miss}")
    if eng["producer_sha256"] != IO.sha256(Path(__file__)):
        raise SystemExit("this producer's sha256 differs from the engineering record's")
    return eng_p, eng


def assemble(seeds=K.CHJ_SEEDS, prereg_dir=None, rows_dir=None) -> dict:
    """Write chj_panel.npz, chj_metrics.json and chj_panel.json once. Order: the
    engineering record (this producer); metadata-only provenance (the hook on every fold
    of both seeds, the fresh Q0 baselines and CH-J's runs at the hook commit, R8.1f's
    column caps, the forwards' thetas, the canal field); the model-only panel; THEN the
    design-time look; only then the observation-derived no-harm metrics."""
    eng_p, eng = _engineering()
    me = eng["producer_sha256"]
    panel, metrics, summary = IO.rp(PANEL), IO.rp(METRICS), IO.rp(SUMMARY)
    for p in (panel, metrics, summary):
        if p.exists():
            raise SystemExit(f"{p} exists; the CH-J panel is produced once")
    prov_runs = {}
    for s in seeds:
        check_hook_applied(run_path(chj_suffix(s)), K.Q0_THETA, s, require_folds=N_FOLDS,
                           q0_fold_dir=run_path(q0_suffix(s)))
        check_sw(_theta_doc(run_path(chj_suffix(s)) / "stage3_theta.json"), eng)
        prov_runs[chj_suffix(s)] = check_run_provenance(chj_suffix(s), s, eng["hook_commit"],
                                                        "chj")
        prov_runs[q0_suffix(s)] = check_run_provenance(q0_suffix(s), s, eng["hook_commit"],
                                                       "q0")
    arrays, prov = build_panel(eng["hook_commit"], seeds)
    look = log_noharm_look(noharm_sids([chj_suffix(s) for s in seeds]
                                       + [q0_suffix(s) for s in seeds]),
                           prereg_dir, rows_dir)          # before any observation is read
    met = build_metrics(seeds)
    met.update({"producer": PRODUCER, "producer_version": VERSION, "producer_sha256": me,
                "created": _now(), "hindcasts": prov, "run_provenance": prov_runs,
                "noharm_look": {k: look.get(k) for k in ("time", "purpose", "units",
                                                         "test_id", "prereg_id")}})
    with panel.open("xb") as fh:
        np.savez(fh, **arrays)
    with metrics.open("x") as f:
        json.dump(met, f, indent=1, default=float)
    summ = {"engineering_sha256": IO.sha256(eng_p), "producer_sha256": me,
            "chj_panel_sha256": IO.sha256(panel), "chj_metrics_sha256": IO.sha256(metrics),
            "created": _now(), "producer": PRODUCER, "seeds": list(seeds),
            "keys": {k: list(np.shape(v)) for k, v in arrays.items()}}
    with summary.open("x") as f:
        json.dump(summ, f, indent=1)
    return summ


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    refuse_posterior(argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hindcast")
    h.add_argument("--seed", type=int, required=True, choices=list(K.CHJ_SEEDS))
    h.add_argument("--device", default="cpu")
    q = sub.add_parser("hindcast-q0")
    q.add_argument("--device", default="cpu")
    sub.add_parser("assemble")
    c = sub.add_parser("check-hook", help="run_chj.sh's hook_applied")
    c.add_argument("--run", required=True)
    c.add_argument("--seed", type=int, required=True, choices=list(K.CHJ_SEEDS))
    c.add_argument("--q0-theta", default=K.Q0_THETA)
    c.add_argument("--require-folds", type=int, default=None)
    c.add_argument("--q0-fold-dir", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "check-hook":
        r = check_hook_applied(a.run, a.q0_theta, a.seed, a.require_folds, a.q0_fold_dir)
        print(f"hook applied: OK (folds {r['folds_checked']})")
        return 0
    if a.cmd == "hindcast":
        s = hindcast(a.seed, a.device)
    elif a.cmd == "hindcast-q0":
        s = hindcast_q0(a.device)
    else:
        s = assemble()
        print(f"wrote {PANEL} sha256 {s['chj_panel_sha256']}; {METRICS} sha256 "
              f"{s['chj_metrics_sha256']}; {SUMMARY}")
        return 0
    print(f"wrote {s['npz']} sha256 {s['npz_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
