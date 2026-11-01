#!/usr/bin/env bash
# Round 7 ranks 4-6 (bands): the lead's steps, in order. Every step refuses on a missing
# predecessor; nothing here registers anything (prereg_design.py / prereg_e02d.py --write do).
#
#   bash dataprep_staging/e02d/run_e02d.sh check          preconditions (CPU, no read)
#   bash dataprep_staging/e02d/run_e02d.sh dr             DR.relative_drift.v1: the ONE design-
#                                                         time read (logs its look; CPU)
#   bash dataprep_staging/e02d/run_e02d.sh thetas         18 per-member theta files (CPU)
#   bash dataprep_staging/e02d/run_e02d.sh members all    36 forwards (GPU, tmux; MPS fine)
#   bash dataprep_staging/e02d/run_e02d.sh members <label> [<label> ...]
#   bash dataprep_staging/e02d/run_e02d.sh extract        member gate + members_wells.npz (CPU)
#   bash dataprep_staging/e02d/run_e02d.sh register-abs   prospective.register: the A check
#                                                         and the year-2/3 absolute rows (CPU)
#   bash dataprep_staging/e02d/run_e02d.sh artefact       sigma_idio for A, B, C (CPU)
#   bash dataprep_staging/e02d/run_e02d.sh power          U1.decomp_power.v1 (CPU, registered)
#   bash dataprep_staging/e02d/run_e02d.sh archive "<reason>" [label ...]
#                                    engineering retry: move member outputs aside, unopened
#   DRY=1 bash ... members <label>   print the forward command lines only
#
# Order (deadline 2026-11-30T16:00Z for E02d):
#   prereg_design.py dr --write -> dr -> prereg_design.py power --write -> power
#   thetas -> members all -> extract     (model-only; may run in parallel with dr / power)
#   register-abs (after dr) -> artefact -> prereg_e02d.py e02d --write, e02y --write
#
# Code ladder (members): the forwards run from a clean detached worktree
# $HOME/HydroPhysicsAI-e02drun-<code> through PYTHONPATH (python -P, launched from this
# repository's root). The first code is 4c2723e, where FS.start_invariance.v1's S0 rerun
# reproduced the page run (repro rc 0, 2026-11-01). A logged member-gate failure (extract
# exit 3) moves to 730f65b; after that nothing runs. Every member runs the same code.
#
# Sealed outputs: forward.py prints in-sample hindcast skill (heads and leveling) and the
# forward summary on stdout. Each member's stdout goes to results/twin_runs/e02d/sealed/
# <label>_ic<k>.stdout.SEALED, which nobody opens; only its sha256 is logged in
# results/twin_runs/e02d/attempts.jsonl. stderr (tracebacks) may be read to fix a crash.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
HPY="$HOME/miniforge3/envs/hydro/bin/python"
E=dataprep_staging/e02d
CODES=(4c2723e 730f65b)
R=results/twin_runs
G=$R/stage3_datum_gate
LOG=$R/e02d
SEAL=$LOG/sealed
ATT=$LOG/attempts.jsonl
THETAS=$LOG/thetas
MOUT=results/twin_forward/e02d_members
DRDIR=results/dataprep/reports/DR.relative_drift.v1
MIN_FREE_MIB=${E02D_MIN_FREE_MIB:-4000}
VEP="$G/coupled_leveling/vep_zonal_leveling.json,$G/coupled_leveling_tau30/vep_zonal_leveling.json"
RF=chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv
COMMON="--horizon 120 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --rf-timeseries $RF --device cuda --compile-matvec"
F_MEMBER="--gain 1.0 --ic-sigma 0.5 --seed 0 --update-method innovation --restart-taper-km 5 --save-members none"
FORWARD=results/twin_forward/datum_gate_innov.npz
FORWARD_SHA=cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228
STATIONS=AMP_V2/data/fan_stations.parquet
HWELLS=AMP_V2/data/wells

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

log_attempt() {
    mkdir -p "$LOG"
    "$HPY" -I -c '
import fcntl, json, sys
rec = {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    rec[k] = int(v) if v.lstrip("-").isdigit() and k in ("rc", "attempt", "ic") else v
with open(sys.argv[1], "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(json.dumps(rec, sort_keys=True) + "\n")' "$ATT" "$@"
}

count_records() {
    if [ ! -f "$ATT" ]; then echo 0; return; fi
    "$HPY" -I -c '
import json, sys
want = dict(kv.split("=", 1) for kv in sys.argv[2:])
print(sum(all(str(json.loads(s).get(k)) == v for k, v in want.items())
          for s in open(sys.argv[1]) if s.strip()))' "$ATT" "$@"
}

current_code() {
    k=$(count_records event=extract rc=3)
    [ "$k" -lt "${#CODES[@]}" ] || die "the member gate failed at every declared code" \
        "(${CODES[*]}): E02d is NOT registered (no fallback, declared in DR's" \
        "e02d_commitments); record C14a-r N ('undecidable: the relative member spread" \
        "could not be reproduced')"
    echo "${CODES[$k]}"
}

worktree_check() {
    CODE=$(current_code)
    WT="$HOME/HydroPhysicsAI-e02drun-$CODE"
    [ -d "$WT" ] || die "worktree $WT missing (git worktree add --detach $WT $CODE)"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$CODE" ] || die "$WT is not at $CODE"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    mod=$(PYTHONPATH="$WT" "$HPY" -P -c \
        'import hydrophysics.twin.forward as m; print(m.__file__)' 2> /dev/null) \
        || die "hydrophysics.twin.forward does not import from $WT"
    case $mod in "$WT"/*) ;; *) die "forward imports from $mod, not $WT" ;; esac
}

base_check() {
    [ "$(sha "$FORWARD")" = "$FORWARD_SHA" ] || die "$FORWARD sha256 differs from cf2c7154"
    for f in "$G/stage3_theta.json" "$G/stage3_fold_thetas.json" \
             "$G/stage3_posterior_tn.json" "$RF" "${VEP%%,*}" "${VEP##*,}"; do
        [ -f "$f" ] || die "$f missing"
    done
}

gpu_check() {
    nvidia-smi > /dev/null || die "nvidia-smi failed"
    read -r total used < <(nvidia-smi --query-gpu=memory.total,memory.used \
        --format=csv,noheader,nounits | head -1 | tr -d ',')
    free=$((total - used))
    echo "GPU memory free: $free MiB (need $MIN_FREE_MIB)"
    [ "$free" -ge "$MIN_FREE_MIB" ] || die "not enough free GPU memory ($free MiB)"
}

labels() {
    [ -f "$THETAS/manifest.json" ] || die "run 'thetas' first"
    "$HPY" -I -c 'import json, sys
print(" ".join(json.load(open(sys.argv[1]))["labels"]))' "$THETAS/manifest.json"
}

forward_cmd() {   # label ic worktree
    echo "PYTHONPATH=$3 \$HOME/miniforge3/envs/hydro/bin/python -P -m hydrophysics.twin.forward" \
         "--theta $THETAS/$1/stage3_theta.json $F_MEMBER --ic-members $2 $COMMON" \
         "--vep-json $VEP --out $MOUT/$1_ic$2"
}

do_member_one() {   # label ic
    local lab=$1 ic=$2 out=$MOUT/$1_ic$2
    for s in npz summary.csv members.csv; do
        [ ! -e "$out.$s" ] || die "$out.$s exists (archive it first)"
    done
    mkdir -p "$SEAL" "$MOUT"
    chmod 700 "$SEAL"
    n=$(( $(count_records event=forward label="$lab" ic="$ic") + 1 ))
    sealed=$SEAL/${lab}_ic${ic}.stdout.SEALED
    err=$LOG/${lab}_ic${ic}.stderr.log
    t0=$(utc)
    rc=0
    # shellcheck disable=SC2086  # F_MEMBER and COMMON are lists of plain tokens
    PYTHONPATH="$WT" "$HPY" -P -m hydrophysics.twin.forward \
        --theta "$THETAS/$lab/stage3_theta.json" $F_MEMBER --ic-members "$ic" $COMMON \
        --vep-json "$VEP" --out "$out" > "$sealed" 2> "$err" || rc=$?
    chmod 400 "$sealed"
    hs=missing; [ -e "$out.npz" ] && hs=$(sha "$out.npz")
    log_attempt event=forward attempt="$n" label="$lab" ic="$ic" rc="$rc" start="$t0" \
        time="$(utc)" code="$CODE" worktree_head="$(git -C "$WT" rev-parse HEAD)" \
        npz_sha256="$hs" stdout_sealed_sha256="$(sha "$sealed")" stderr_log="$err"
    echo "member $lab ic$ic: exit $rc (attempt $n, code $CODE; stdout sealed)"
    [ "$rc" = 0 ] || die "member $lab ic$ic failed: read $err only, then archive and rerun"
}

do_members() {
    [ "$#" -ge 1 ] || die "members needs 'all' or labels"
    local list=("$@")
    [ "$1" != all ] || read -r -a list <<< "$(labels)"
    if [ "${DRY:-0}" = 1 ]; then
        for l in "${list[@]}"; do for ic in 1 0; do
            forward_cmd "$l" "$ic" "\$HOME/HydroPhysicsAI-e02drun-<code>"; done; done
        return 0
    fi
    base_check
    worktree_check
    gpu_check
    for l in "${list[@]}"; do for ic in 1 0; do do_member_one "$l" "$ic"; done; done
    echo "done (${list[*]}); do not open the sealed logs"
}

do_thetas() {
    base_check
    worktree_check
    [ ! -e "$THETAS" ] || die "$THETAS exists (archive it first)"
    rc=0
    PYTHONPATH="$WT" "$HPY" -P "$E/members_e02d.py" thetas --worktree "$WT" --out "$THETAS" \
        || rc=$?
    log_attempt event=thetas rc="$rc" time="$(utc)" code="$CODE"
    return "$rc"
}

do_extract() {
    base_check
    worktree_check
    [ -f "$DRDIR/wells_e02d.csv" ] || die "run 'dr' first (the wells table gives the cells)"
    rc=0
    "$HPY" -I "$E/members_e02d.py" extract || rc=$?
    log_attempt event=extract rc="$rc" time="$(utc)" code="$CODE"
    case $rc in
        0) echo "member gate passed at $CODE" ;;
        3) echo "member gate FAILED at $CODE (logged): archive 'gate failed at $CODE' and" \
                "rerun thetas/members at the next code" ;;
        *) echo "engineering failure (exit $rc): fix the cause and rerun extract" ;;
    esac
    return "$rc"
}

REG_SPECS=("results/twin_runs/e02d/register_check_2026-12 2026-12-01 2027-10-31"
           "results/twin/prospective_2027-11_y2 2027-11-01 2028-10-31"
           "results/twin/prospective_2028-11_y3 2028-11-01 2029-10-31")

do_register_abs() {
    base_check
    [ -f "$DRDIR/report.json" ] || die "run 'dr' first (its declaration names the rows' unit)"
    # every output directory must be absent: prospective.register writes predictions.csv
    # BEFORE its date check, so a refused run leaves a partial directory without a manifest
    for spec in "${REG_SPECS[@]}"; do
        read -r out _ _ <<< "$spec"
        [ ! -e "$out/manifest.json" ] || die "$out exists (registered once)"
        [ ! -e "$out/predictions.csv" ] || die "$out/predictions.csv exists without a" \
            "manifest (a partial or refused run): archive the directory, log why, rerun"
    done
    # E02's own pre-origin datum read (use datum_fit), logged once per invocation on DR's
    # declared unit under DR's work package (not a new estimand: E02's registration read)
    "$HPY" -I -c '
import sys
sys.path.insert(0, "dataprep_staging")
from harness import prereg3
prereg3.record_look(["DR:fit_residuals_2012-01_2022-12_wells158"], "heads:2012-01..2022-12",
                    "DR", "datum_fit", prereg_id="DR.relative_drift.v1",
                    note="run_e02d.sh register-abs: twin.prospective.register reads the "
                         "pre-origin heads for the E02 datum and fitted sd (use datum_fit) to "
                         "write the 2026-12 check and the year-2/3 absolute rows; E02 own "
                         "read, not a new estimand")'
    # prospective.register refuses a started window: the E02 check runs on 2026-12..2027-10
    for spec in "${REG_SPECS[@]}"; do
        read -r out a b <<< "$spec"
        rc=0
        "$HPY" -m hydrophysics.twin.prospective register --forward "$FORWARD" \
            --stations "$STATIONS" --historical-wells "$HWELLS" --out "$out" \
            --test-start "$a" --test-end "$b" > /dev/null || rc=$?
        if [ "$rc" != 0 ] || [ ! -f "$out/manifest.json" ]; then
            log_attempt event=register_abs time="$(utc)" out="$out" rc="$rc" \
                partial="$([ -e "$out/predictions.csv" ] && echo yes || echo no)"
            die "register $out failed (exit $rc; a started window is refused): archive" \
                "$out, log why, rerun"
        fi
        log_attempt event=register_abs time="$(utc)" out="$out" rc=0 \
            predictions_sha256="$(sha "$out/predictions.csv")"
        echo "wrote $out (predictions sha256 $(sha "$out/predictions.csv"))"
    done
}

do_archive() {
    reason=${1:-}
    [ -n "$reason" ] || die "archive needs a reason"
    shift
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    dst=$LOG/archive/$stamp
    mkdir -p "$dst"
    local pats=("$@")
    [ "$#" -gt 0 ] || pats=("*")
    moved=()
    for p in "${pats[@]}"; do
        for f in "$MOUT"/${p}_ic*.* "$SEAL"/${p}_ic*.SEALED "$LOG"/${p}_ic*.stderr.log; do
            [ -e "$f" ] || continue
            mv "$f" "$dst/"
            moved+=("$(basename "$f")")
        done
    done
    if [ "$#" = 0 ]; then
        for f in "$THETAS" "$LOG/members_wells.npz" "$LOG/members_gate.json"; do
            [ -e "$f" ] || continue
            mv "$f" "$dst/"
            moved+=("$(basename "$f")")
        done
    fi
    log_attempt event=archive time="$(utc)" reason="$reason" dir="$dst" \
        moved="${moved[*]:-none}"
    echo "archived ${#moved[@]} files to $dst (unopened)"
}

case "${1:-}" in
    check) base_check; worktree_check; echo "ok (code $CODE, worktree $WT)" ;;
    dr) "$HPY" -I "$E/dr_drift.py" run ;;
    thetas) do_thetas ;;
    members) shift; do_members "$@" ;;
    extract) do_extract ;;
    register-abs) do_register_abs ;;
    artefact) "$HPY" -I "$E/build_artefact.py" ;;
    power) "$HPY" -I "$E/power_e02d.py" run ;;
    archive) shift; do_archive "$@" ;;
    *) echo "usage: $0 check|dr|thetas|members <all|label...>|extract|register-abs|artefact|" \
            "power|archive '<reason>' [label ...]" >&2
       exit 2 ;;
esac
