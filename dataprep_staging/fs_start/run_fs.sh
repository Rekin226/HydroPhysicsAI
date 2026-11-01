#!/usr/bin/env bash
# FS.start_invariance.v1 (model-only declaration): the forward runs, the S0 reproduction gate
# and the one scoring.
#
#   bash dataprep_staging/fs_start/run_fs.sh check            preconditions only (CPU)
#   bash dataprep_staging/fs_start/run_fs.sh forward S0       the page-recipe rerun (GPU, tmux)
#   bash dataprep_staging/fs_start/run_fs.sh repro            S0 against the page run (CPU)
#   bash dataprep_staging/fs_start/run_fs.sh forward S1|S2|S3|S3ref
#                                    one arm (GPU, tmux; separate windows under MPS are fine)
#   bash dataprep_staging/fs_start/run_fs.sh forward arms     S1 S2 S3 S3ref, sequential
#   bash dataprep_staging/fs_start/run_fs.sh score            score once (CPU)
#   bash dataprep_staging/fs_start/run_fs.sh archive "<reason>" [arm ...]
#                                    engineering retry: move the arms' outputs aside (default
#                                    every arm), unopened (refused once the report exists)
#   DRY=1 bash ... forward <arm>     print the forward command line, run nothing
#
# Code ladder (declared): the forward runs from a clean detached worktree
# $HOME/HydroPhysicsAI-fsrun-<code> through PYTHONPATH (python -P, launched from the
# repository root). The first code is main 4c2723e; a logged S0 reproduction failure
# (repro exit 3) moves to 730f65b, the first commit with --update-method and the last before
# the page run datum_gate_innov.npz was written. fa5cca2, the theta's commit, cannot run the
# page recipe (no --update-method). After 730f65b fails too, nothing runs under v1. Every arm
# of one scoring runs the same code.
#
# Rain: every arm passes --rf-timeseries $RF, the vendor cache the page run read (it was
# written before the 2026-10-11 CWA rebuild replaced the default rf_timeseries.csv).
#
# Binding to the registered design: every subcommand refuses unless the declaration
# results/dataprep/preregistration/FS.start_invariance.v1.json exists, is intact (INDEX.jsonl
# and ledger, via the scorer's step 0), and records in ``frozen_code`` the sha256 of
# score_fs.py and this script as they are now. Step 0 also re-hashes every declared input
# (the forward's files and the head-well directory digest), so drift refuses before any
# forward starts.
#
# Sealed outputs (attempt policy): forward.py writes every scenario's values before the
# scorer runs: stdout carries the '=== forward summary ===' block (fan-mean layer-2 head change
# and subsidence per scenario) and the rheology-axis block, and the run writes
# results/twin_forward/fs_start_<arm>.{npz,summary.csv,members.csv} (S3 also .members.npz).
# This script sends the forward's stdout to results/twin_runs/fs_start/sealed/
# <arm>_forward.stdout.SEALED, which nobody opens, tails or greps before report.json exists,
# and records its sha256 (and those of the outputs) in results/twin_runs/fs_start/
# attempts.jsonl. The forward's stderr (tracebacks and warnings; forward.py prints no value
# there) goes to <arm>_forward.stderr.log and may be read to diagnose a crash. Only `repro`
# reads an arm before the scoring, and only S0's page-published quantities. This script
# prints only exit codes, sha256s and free GPU memory.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
SELF=dataprep_staging/fs_start/run_fs.sh
SCORER=dataprep_staging/fs_start/score_fs.py
ID=FS.start_invariance.v1
DECL=results/dataprep/preregistration/$ID.json
CODES=(4c2723e 730f65b)
HPY="$HOME/miniforge3/envs/hydro/bin/python"
R=results/twin_runs
G=$R/stage3_datum_gate
LOG=$R/fs_start
SEAL=$LOG/sealed
ATT=$LOG/attempts.jsonl
REPORT=results/dataprep/reports/$ID/report.json
MIN_FREE_MIB=${FS_MIN_FREE_MIB:-5000}
PAGE_THETAS="--theta $G/stage3_theta.json --theta $G/stage3_fold_thetas.json --theta $G/stage3_posterior_tn.json"
ONE_THETA="--theta $G/stage3_theta.json"
VEP="$G/coupled_leveling/vep_zonal_leveling.json,$G/coupled_leveling_tau30/vep_zonal_leveling.json"
SCENARIOS=('cut30:irrigation=0.7@2026-01' 'retire_aqua:aquaculture=0@2026-01' 'cut30_2030:irrigation=0.7@2030-01' 'retire_aqua_2030:aquaculture=0@2030-01')
RF=chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv
COMMON="--horizon 120 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --rf-timeseries $RF --device cuda --compile-matvec"
F_S0="--gain 1.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method innovation --restart-taper-km 5 --save-members none"
F_S1="--gain 0.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method innovation --restart-taper-km 5 --save-members none"
F_S2="--gain 1.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method field --save-members none"
F_S3="--gain 1.0 --ic-members 8 --ic-sigma 2.0 --seed 0 --update-method innovation --restart-taper-km 5 --save-members yearly"
F_S3REF="--gain 1.0 --ic-members 0 --update-method innovation --restart-taper-km 5 --save-members none"
ARMS=(S0 S1 S2 S3 S3ref)

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

# the code of the current rung: one step down the ladder per logged S0 reproduction failure
current_code() {
    k=$(count_records event=repro rc=3)
    [ "$k" -lt "${#CODES[@]}" ] || die "S0 failed reproduction at every declared code" \
        "(${CODES[*]}); nothing more runs under $ID (a v2 must disclose the attempts)"
    echo "${CODES[$k]}"
}

arm_flags() {
    case $1 in
        S0) echo "$PAGE_THETAS $F_S0" ;;
        S1) echo "$PAGE_THETAS $F_S1" ;;
        S2) echo "$PAGE_THETAS $F_S2" ;;
        S3) echo "$ONE_THETA $F_S3" ;;
        S3ref) echo "$ONE_THETA $F_S3REF" ;;
        *) die "unknown arm $1 (S0 S1 S2 S3 S3ref)" ;;
    esac
}

outputs() {   # every file an arm's forward writes
    local o=results/twin_forward/fs_start_$1
    echo "$o.npz $o.summary.csv $o.members.csv"
    [ "$1" != S3 ] || echo "$o.members.npz"
}

preconditions() {
    [ -f "$DECL" ] || die "declare $ID first (prereg_fs.py --write)"
    for f in "$SCORER" "$SELF"; do
        want=$("$HPY" -I -c 'import json, sys
print(json.load(open(sys.argv[1])).get("frozen_code", {}).get(sys.argv[2], "none"))' "$DECL" "$f")
        [ "$want" = "$(sha "$f")" ] || die "$f sha256 differs from the registered declaration"
    done
    "$HPY" "$SCORER" --verify-registration > /dev/null \
        || die "the registered declaration does not verify (INDEX/ledger/frozen code)"
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; no rerun"
    CODE=$(current_code)
    WT="$HOME/HydroPhysicsAI-fsrun-$CODE"
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
    local s="" x
    for x in "${SCENARIOS[@]}"; do s="$s --scenario '$x'"; done
    echo "PYTHONPATH=$2 \$HOME/miniforge3/envs/hydro/bin/python -P -m" \
         "hydrophysics.twin.forward $(arm_flags "$1")$s $COMMON --vep-json $VEP" \
         "--out results/twin_forward/fs_start_$1"
}

do_forward_one() {
    local a=$1 f
    for f in $(outputs "$a"); do
        [ ! -e "$f" ] || die "$f exists (archive it first)"
    done
    if [ "$a" != S0 ]; then
        s0=results/twin_forward/fs_start_S0.npz
        [ -e "$s0" ] || die "run S0 and repro first"
        [ "$(count_records event=repro rc=0 code="$CODE" s0_npz_sha256="$(sha "$s0")")" -ge 1 ] \
            || die "no passing reproduction is logged for this S0 file at $CODE (run repro)"
    fi
    mkdir -p "$SEAL"
    chmod 700 "$SEAL"
    n=$(( $(count_records event=forward arm="$a") + 1 ))
    out=results/twin_forward/fs_start_$a
    sealed=$SEAL/${a}_forward.stdout.SEALED
    err=$LOG/${a}_forward.stderr.log
    t0=$(utc)
    rc=0
    # shellcheck disable=SC2046  # arm_flags and COMMON are lists of plain tokens
    PYTHONPATH="$WT" "$HPY" -P -m hydrophysics.twin.forward $(arm_flags "$a") \
        --scenario "${SCENARIOS[0]}" --scenario "${SCENARIOS[1]}" \
        --scenario "${SCENARIOS[2]}" --scenario "${SCENARIOS[3]}" $COMMON \
        --vep-json "$VEP" --out "$out" > "$sealed" 2> "$err" || rc=$?
    chmod 400 "$sealed"
    hs=(); for s in npz summary.csv members.csv members.npz; do
        if [ -e "$out.$s" ]; then hs+=("$(sha "$out.$s")"); else hs+=(missing); fi
    done
    log_attempt event=forward attempt="$n" arm="$a" rc="$rc" start="$t0" time="$(utc)" \
        code="$CODE" worktree_head="$(git -C "$WT" rev-parse HEAD)" \
        declaration_sha256="$(sha "$DECL")" npz_sha256="${hs[0]}" \
        summary_csv_sha256="${hs[1]}" members_csv_sha256="${hs[2]}" \
        members_npz_sha256="${hs[3]}" stdout_sealed_sha256="$(sha "$sealed")" \
        stdout_sealed="$sealed" stderr_log="$err"
    echo "forward $a: exit $rc (attempt $n, code $CODE; stdout sealed, sha256 $(sha "$sealed"))"
    [ "$rc" = 0 ] || die "forward $a failed: read $err only, then archive and rerun"
}

do_forward() {
    local which=${1:-}
    [ -n "$which" ] || die "forward needs an arm: S0 S1 S2 S3 S3ref or arms"
    local list=("$which")
    [ "$which" != arms ] || list=(S1 S2 S3 S3ref)
    if [ "${DRY:-0}" = 1 ]; then
        for a in "${list[@]}"; do forward_cmd "$a" "\$HOME/HydroPhysicsAI-fsrun-<code>"; done
        return 0
    fi
    preconditions
    gpu_check
    for a in "${list[@]}"; do do_forward_one "$a"; done
    echo "done (${list[*]}); do not open the sealed logs or the fs_start CSVs"
}

do_repro() {
    preconditions
    s0=results/twin_forward/fs_start_S0.npz
    [ -e "$s0" ] || die "run forward S0 first"
    s0sha=$(sha "$s0")
    # one decisive repro per S0 file: a second exit-3 record would skip a rung of the ladder
    for r in 0 3; do
        [ "$(count_records event=repro rc=$r s0_npz_sha256="$s0sha")" = 0 ] \
            || die "a repro (exit $r) is already logged for this S0 file"
    done
    rc=0
    "$HPY" "$SCORER" --repro --attempts "$ATT" || rc=$?
    log_attempt event=repro rc="$rc" time="$(utc)" code="$CODE" s0_npz_sha256="$s0sha" \
        declaration_sha256="$(sha "$DECL")"
    case $rc in
        0) echo "S0 reproduces the page run at $CODE; next: forward S1, S2, S3, S3ref" ;;
        3) echo "S0 does not reproduce the page run at $CODE (logged): archive" \
                "'S0 reproduction failed at $CODE', then forward S0 at the next code" ;;
        *) echo "engineering failure (scorer exit $rc): fix the cause and rerun repro" ;;
    esac
    return "$rc"
}

do_score() {
    preconditions
    n=$(( $(count_records event=score) + 1 ))
    rc=0
    "$HPY" "$SCORER" --attempts "$ATT" --out "$REPORT" || rc=$?
    rs=missing; [ -e "$REPORT" ] && rs=$(sha "$REPORT")
    log_attempt event=score attempt="$n" rc="$rc" time="$(utc)" code="$CODE" \
        report_sha256="$rs" declaration_sha256="$(sha "$DECL")"
    case $rc in
        0) echo "scored once: $REPORT (sha256 $rs); the sealed logs and CSVs may now be read" ;;
        1|2|3) echo "engineering failure (scorer exit $rc): no gated value was read;" \
                    "fix the cause, then: run_fs.sh archive '<reason>' and rerun the arms" ;;
        *) echo "scorer exit $rc" ;;
    esac
    return "$rc"
}

do_archive() {   # archive '<reason>' [arm ...]  (default: every arm)
    reason=${1:-}
    [ -n "$reason" ] || die "archive needs a reason"
    shift
    local which=("${ARMS[@]}")
    [ "$#" = 0 ] || which=("$@")
    for a in "${which[@]}"; do arm_flags "$a" > /dev/null; done
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; nothing may be rerun"
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
    repro) do_repro ;;
    score) do_score ;;
    archive) shift; do_archive "$@" ;;
    *) echo "usage: $0 check|forward <arm|arms>|repro|score|archive '<reason>' [arm ...]" >&2
       exit 2 ;;
esac
