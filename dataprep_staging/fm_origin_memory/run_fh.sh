#!/usr/bin/env bash
# FH.forcing_sensitivity.v1 (model-only declaration): the two new forwards and the one
# scoring.
#
#   bash dataprep_staging/fm_origin_memory/run_fh.sh check        preconditions only (CPU)
#   bash dataprep_staging/fm_origin_memory/run_fh.sh forward C19  free run from 2019-12 under
#                                       the 2012-19 climatology (GPU, tmux)
#   bash dataprep_staging/fm_origin_memory/run_fh.sh forward N19  the page's innovation
#                                       restart at 2019-12 (GPU, tmux; C19 and N19 may run
#                                       at the same time in two tmux windows under MPS)
#   bash dataprep_staging/fm_origin_memory/run_fh.sh forward arms C19 then N19, sequential
#   bash dataprep_staging/fm_origin_memory/run_fh.sh score        score once (CPU)
#   bash dataprep_staging/fm_origin_memory/run_fh.sh archive "<reason>" [arm ...]
#                                       engineering retry: move the arms' outputs aside,
#                                       unopened (refused once the report exists)
#   DRY=1 bash ... forward <arm>        print the forward command line, run nothing
#
# Code: the forwards run from a clean detached worktree $HOME/HydroPhysicsAI-fhrun-<code>
# through PYTHONPATH (python -P, launched from the repository root), at 4c2723e, the code
# FS.start_invariance.v1 ran fs_start_S1 with (the scorer's E1 gate compares the new runs'
# 2012-19 months with fs_start_S1's). Create it once:
#   git worktree add --detach "$HOME/HydroPhysicsAI-fhrun-4c2723e" 4c2723e
# No hydrophysics/twin edit in that worktree, ever (code freeze while GPU jobs run).
#
# Recipe: the FS recipe of S1 (C19) and S0 (N19) with --record-end 2019-12 and the baseline
# scenario only; the rain is the vendor cache the page theta was fitted on (as in FS).
# Data dir: forward.main validates the hindcast against leveling whenever its data dir
# (--data, else $HYDROMIND_GW_DATA, else the bundled synthetic hydrophysics/sample_data) is
# an existing directory, after the run and before the npz is written. Unset, the variable
# falls back to sample_data, which has no ls_cache panel, so the forward would crash there
# (FileNotFoundError) after its full GPU run. The forwards therefore run without --data and
# with HYDROMIND_GW_DATA=$NOGW, a path that must not exist (checked before every forward):
# the data dir resolves to None, no leveling file is read and no leveling R2 is printed. The
# page theta has rivers 'none' (the variable's other reader on this path) and the column
# JSONs exist (no column fit), so heads do not depend on it.
#
# Binding: every subcommand refuses unless results/dataprep/preregistration/
# FH.forcing_sensitivity.v1.json exists, is intact (the scorer's step 0) and records in
# frozen_code the sha256 of fm_common.py, score_fh.py and this script as they are now.
#
# Sealed outputs: the forward's stdout carries per-member in-sample hindcast R2 (2012-19)
# and the '=== forward summary ===' block. It goes to results/twin_runs/fh_force/sealed/
# <arm>_forward.stdout.SEALED, which nobody opens, tails or greps before report.json exists;
# its sha256 and those of the outputs are logged in results/twin_runs/fh_force/
# attempts.jsonl. The fh_force CSVs are not opened before the report either. The forward's
# stderr (tracebacks, warnings) goes to <arm>_forward.stderr.log and may be read to
# diagnose a crash. This script prints only exit codes, sha256s and free GPU memory.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
SELF=dataprep_staging/fm_origin_memory/run_fh.sh
SCORER=dataprep_staging/fm_origin_memory/score_fh.py
COMMON_PY=dataprep_staging/fm_origin_memory/fm_common.py
ID=FH.forcing_sensitivity.v1
DECL=results/dataprep/preregistration/$ID.json
CODE=4c2723e
HPY="$HOME/miniforge3/envs/hydro/bin/python"
R=results/twin_runs
G=$R/stage3_datum_gate
LOG=$R/fh_force
NOGW=$LOG/no_gw_data
SEAL=$LOG/sealed
ATT=$LOG/attempts.jsonl
REPORT=results/dataprep/reports/$ID/report.json
MIN_FREE_MIB=${FH_MIN_FREE_MIB:-5000}
PAGE_THETAS="--theta $G/stage3_theta.json --theta $G/stage3_fold_thetas.json --theta $G/stage3_posterior_tn.json"
VEP="$G/coupled_leveling/vep_zonal_leveling.json,$G/coupled_leveling_tau30/vep_zonal_leveling.json"
RF=chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv
COMMON="--horizon 120 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --rf-timeseries $RF --device cuda --compile-matvec"
RECORD_END="--record-end 2019-12"
F_C19="--gain 0.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method innovation --restart-taper-km 5 --save-members none"
F_N19="--gain 1.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method innovation --restart-taper-km 5 --save-members none"
ARMS=(C19 N19)

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# append one JSON record (key=value pairs; values are strings or integers) to attempts.jsonl
log_attempt() {
    mkdir -p "$LOG"
    "$HPY" -I -c '
import fcntl, json, sys
rec = {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    rec[k] = int(v) if v.lstrip("-").isdigit() and k in ("rc", "attempt") else v
with open(sys.argv[1], "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(json.dumps(rec, sort_keys=True) + "\n")' "$ATT" "$@"
}

# count of attempts.jsonl records matching key=value pairs (all must match)
count_records() {
    if [ ! -f "$ATT" ]; then echo 0; return; fi
    "$HPY" -I -c '
import json, sys
want = dict(kv.split("=", 1) for kv in sys.argv[2:])
n = 0
for s in open(sys.argv[1]):
    if s.strip():
        r = json.loads(s)
        n += all(str(r.get(k)) == v for k, v in want.items())
print(n)' "$ATT" "$@"
}

arm_flags() {
    case $1 in
        C19) echo "$PAGE_THETAS $F_C19 $RECORD_END" ;;
        N19) echo "$PAGE_THETAS $F_N19 $RECORD_END" ;;
        *) die "unknown arm $1 (C19 N19)" ;;
    esac
}

outputs() {   # every file an arm's forward writes
    local o=results/twin_forward/fh_force_$1
    echo "$o.npz $o.summary.csv $o.members.csv"
}

preconditions() {
    [ -f "$DECL" ] || die "declare $ID first (prereg_fh.py --write)"
    for f in "$COMMON_PY" "$SCORER" "$SELF"; do
        want=$("$HPY" -I -c 'import json, sys
print(json.load(open(sys.argv[1])).get("frozen_code", {}).get(sys.argv[2], "none"))' "$DECL" "$f")
        [ "$want" = "$(sha "$f")" ] || die "$f sha256 differs from the registered declaration"
    done
    "$HPY" -I "$SCORER" --verify-registration > /dev/null \
        || die "the registered declaration does not verify (INDEX/ledger/frozen code/inputs)"
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; no rerun"
    [ ! -e "$NOGW" ] || die "$NOGW exists; HYDROMIND_GW_DATA must name an absent path"
    WT="$HOME/HydroPhysicsAI-fhrun-$CODE"
    [ -d "$WT" ] || die "worktree $WT missing (git worktree add --detach $WT $CODE)"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$CODE" ] || die "$WT is not at $CODE"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    mod=$(PYTHONPATH="$WT" "$HPY" -P -c \
        'import hydrophysics.twin.forward as m; print(m.__file__)' 2> /dev/null) \
        || die "hydrophysics.twin.forward does not import from $WT"
    case $mod in "$WT"/*) ;; *) die "forward imports from $mod, not $WT" ;; esac
}

gpu_check() {
    nvidia-smi > /dev/null || die "nvidia-smi failed"
    read -r total used < <(nvidia-smi --query-gpu=memory.total,memory.used \
        --format=csv,noheader,nounits | head -1 | tr -d ',')
    free=$((total - used))
    echo "GPU memory free: $free MiB (need $MIN_FREE_MIB; other jobs may hold the rest)"
    [ "$free" -ge "$MIN_FREE_MIB" ] || die "not enough free GPU memory ($free MiB)"
}

forward_cmd() {   # the exact command line for arm $1 at worktree $2 (printed by DRY=1)
    echo "env HYDROMIND_GW_DATA=$NOGW PYTHONPATH=$2 \$HOME/miniforge3/envs/hydro/bin/python -P" \
         "-m hydrophysics.twin.forward $(arm_flags "$1") $COMMON --vep-json $VEP" \
         "--out results/twin_forward/fh_force_$1"
}

do_forward_one() {
    local a=$1 f
    for f in $(outputs "$a"); do
        [ ! -e "$f" ] || die "$f exists (archive it first)"
    done
    mkdir -p "$SEAL"
    chmod 700 "$SEAL"
    n=$(( $(count_records event=forward arm="$a") + 1 ))
    out=results/twin_forward/fh_force_$a
    sealed=$SEAL/${a}_forward.stdout.SEALED
    err=$LOG/${a}_forward.stderr.log
    t0=$(utc)
    rc=0
    # shellcheck disable=SC2046  # arm_flags and COMMON are lists of plain tokens
    [ ! -e "$NOGW" ] || die "$NOGW exists; HYDROMIND_GW_DATA must name an absent path"
    env HYDROMIND_GW_DATA="$NOGW" PYTHONPATH="$WT" "$HPY" -P -m hydrophysics.twin.forward \
        $(arm_flags "$a") $COMMON --vep-json "$VEP" --out "$out" > "$sealed" 2> "$err" || rc=$?
    chmod 400 "$sealed"
    hs=(); for s in npz summary.csv members.csv; do
        if [ -e "$out.$s" ]; then hs+=("$(sha "$out.$s")"); else hs+=(missing); fi
    done
    log_attempt event=forward attempt="$n" arm="$a" rc="$rc" start="$t0" time="$(utc)" \
        code="$CODE" worktree_head="$(git -C "$WT" rev-parse HEAD)" \
        declaration_sha256="$(sha "$DECL")" npz_sha256="${hs[0]}" \
        summary_csv_sha256="${hs[1]}" members_csv_sha256="${hs[2]}" \
        stdout_sealed_sha256="$(sha "$sealed")" stdout_sealed="$sealed" stderr_log="$err"
    echo "forward $a: exit $rc (attempt $n, code $CODE; stdout sealed, sha256 $(sha "$sealed"))"
    [ "$rc" = 0 ] || die "forward $a failed: read $err only, then archive and rerun"
}

do_forward() {
    local which=${1:-}
    [ -n "$which" ] || die "forward needs an arm: C19 N19 or arms"
    local list=("$which")
    [ "$which" != arms ] || list=("${ARMS[@]}")
    if [ "${DRY:-0}" = 1 ]; then
        for a in "${list[@]}"; do forward_cmd "$a" "\$HOME/HydroPhysicsAI-fhrun-$CODE"; done
        return 0
    fi
    preconditions
    gpu_check
    for a in "${list[@]}"; do do_forward_one "$a"; done
    echo "done (${list[*]}); do not open the sealed logs or the fh_force CSVs"
}

do_score() {
    preconditions
    n=$(( $(count_records event=score) + 1 ))
    rc=0
    "$HPY" -I "$SCORER" --attempts "$ATT" --out "$REPORT" || rc=$?
    rs=missing; [ -e "$REPORT" ] && rs=$(sha "$REPORT")
    log_attempt event=score attempt="$n" rc="$rc" time="$(utc)" code="$CODE" \
        report_sha256="$rs" declaration_sha256="$(sha "$DECL")"
    case $rc in
        0) echo "scored once: $REPORT (sha256 $rs); the sealed logs and CSVs may now be read" ;;
        1|2) echo "engineering failure (scorer exit $rc): no F or M was computed;" \
                  "fix the cause (archive and rerun the arms if an output is wrong)" ;;
        3) echo "E1 failed (scorer exit 3): the runs do not share the 2019-12 state; nothing" \
                "more runs under $ID (a v2 must disclose this)" ;;
        *) echo "scorer exit $rc" ;;
    esac
    return "$rc"
}

do_archive() {   # archive '<reason>' [arm ...]  (default: every arm)
    [ -f "$DECL" ] || die "declare $ID first (prereg_fh.py --write)"
    reason=${1:-}
    [ -n "$reason" ] || die "archive needs a reason"
    shift
    local which=("${ARMS[@]}")
    [ "$#" = 0 ] || which=("$@")
    for a in "${which[@]}"; do arm_flags "$a" > /dev/null; done
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; nothing may be rerun"
    [ "$(count_records event=score rc=3)" = 0 ] || die "E1 failed under $ID: no rerun"
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    dst=$LOG/archive/$stamp
    mkdir -p "$dst"
    moved=()
    for a in "${which[@]}"; do
        for f in $(outputs "$a") "$SEAL/${a}_forward.stdout.SEALED" \
                 "$LOG/${a}_forward.stderr.log"; do
            if [ -e "$f" ]; then
                mv "$f" "$dst/"
                moved+=("$(basename "$f")")
            fi
        done
    done
    log_attempt event=archive time="$(utc)" reason="$reason" dir="$dst" \
        arms="${which[*]}" moved="${moved[*]:-none}"
    echo "archived ${#moved[@]} files to $dst (unopened)"
}

case "${1:-}" in
    check) preconditions; echo "preconditions ok (code $CODE, worktree $WT)" ;;
    forward) shift; do_forward "${1:-}" ;;
    score) do_score ;;
    archive) shift; do_archive "$@" ;;
    *) echo "usage: $0 check|forward <C19|N19|arms>|score|archive '<reason>' [arm ...]" >&2
       exit 2 ;;
esac
