#!/usr/bin/env bash
# E02z.level_years23.v1: the decidability forwards, the decidability reading and the scorer.
#
#   bash dataprep_staging/e02z/run_e02z.sh setup          frozen worktree at 5f3af18 + py.sh
#   bash dataprep_staging/e02z/run_e02z.sh check          preconditions only (CPU)
#   bash dataprep_staging/e02z/run_e02z.sh forward [RUN]  GPU, in tmux: ZR0 then R0 Q0 R1c X5R1
#                                                         (or one RUN); baseline only, horizon 82
#   bash dataprep_staging/e02z/run_e02z.sh decide SEED    the decidability reading, once (CPU)
#   bash dataprep_staging/e02z/run_e02z.sh score W ARGS   window B or C (2029 / 2030 sessions):
#                                                         score_e02z.py score --window W ARGS
#   bash dataprep_staging/e02z/run_e02z.sh archive RUN "<reason>"
#                                   engineering retry: move a failed forward aside, unopened
#   DRY=1 bash ... forward [RUN]    print the forward command lines, run nothing
#
# Forwards: five single-scenario (baseline) runs at FV v1's commit 5f3af18 with FV v1's flags at
# --horizon 82 (2023-01..2029-10): ZR0 = the page recipe's 18 parameter sets, each truth one
# parameter set. Each hindcast must equal FV v1's forward of the same run to 0.0 m (gate E in
# decide_e02z.py), so they run on the same device as FV v1 (cuda). Rough cost: well under one
# GPU-hour in total. GPU (T4, sm_75): run in a GPU-idle window, e.g.
#   nvidia-smi && tmux new -d -s e02z 'bash dataprep_staging/e02z/run_e02z.sh forward'
# Code freeze: nothing in the worktree is edited while forwards run; each forward refuses a
# dirty worktree. Nothing reads a post-2022 observation: the record ends at the 2022-12 origin.
#
# Binding: forward / decide / score refuse unless E02z.level_years23.v1 is registered and its
# frozen code (this script included) is intact (decide_e02z.py verify does the check).
#
# Sealed outputs: forward stdout (it prints in-sample hindcast skill) goes to
# results/twin_runs/e02z/sealed/<RUN>.stdout.SEALED, opened by nobody; stderr to
# results/twin_runs/e02z/<RUN>.stderr.log (read only to diagnose a crash). npz values are read
# only by decide_e02z.py. This script prints only exit codes and sha256s.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
ID=E02z.level_years23.v1
DECL=results/dataprep/preregistration/$ID.json
DECIDE=dataprep_staging/e02z/decide_e02z.py
SCORE=dataprep_staging/e02z/score_e02z.py
WT_HEAD=5f3af18
WT="$HOME/HydroPhysicsAI-e02zrun"
RUNS=results/twin_runs/e02z
PY=$RUNS/py.sh
HPY="$HOME/miniforge3/envs/hydro/bin/python"
R=results/twin_runs
SEAL=$RUNS/sealed
ATT=$RUNS/attempts.jsonl
FLAGS="--horizon 82 --gain 1.0 --ic-members 0 --update-method innovation --restart-taper-km 5 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --device cuda --compile-matvec --save-members none"
declare -A THETA=([R0]=stage3_datum_gate_cwa [Q0]=stage3_datum_gate_q0 [R1c]=stage3_datum_gate_ccz [X5R1]=stage3_datum_gate_x5r1)
ZR0_DIR=stage3_datum_gate
ZR0_THETAS=(stage3_theta.json stage3_fold_thetas.json stage3_posterior_tn.json)
RUNS_ORDER=(ZR0 R0 Q0 R1c X5R1)

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

log_attempt() {
    mkdir -p "$RUNS"
    "$HPY" -I -c '
import json, sys
rec = {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    rec[k] = int(v) if v.lstrip("-").isdigit() and k in ("rc", "attempt") else v
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")' "$ATT" "$@"
}

next_attempt() {
    if [ -f "$ATT" ]; then
        "$HPY" -I -c '
import json, sys
n = [json.loads(s).get("attempt", 0) for s in open(sys.argv[1]) if s.strip()]
print(max(n, default=0) + 1)' "$ATT"
    else
        echo 1
    fi
}

out_of() {   # $1 = run -> npz stem (decide_e02z.py reads <stem>.npz)
    if [ "$1" = ZR0 ]; then echo results/twin_forward/e02z_zr0
    else echo "results/twin_forward/e02z_truth_$1"; fi
}

do_setup() {
    if [ ! -d "$WT" ]; then
        git worktree add --detach "$WT" "$WT_HEAD"
    fi
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$WT_HEAD" ] || die "$WT is not $WT_HEAD"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    if [ ! -e "$PY" ]; then
        mkdir -p "$RUNS"
        {
            echo '#!/usr/bin/env bash'
            echo "# Run $WT_HEAD (clean detached worktree) with the main repository as the"
            echo "# working directory ($ID decidability forwards)."
            echo "export PYTHONPATH=$WT"
            echo 'exec "$HOME/miniforge3/envs/hydro/bin/python" -P "$@"'
        } > "$PY"
        chmod 755 "$PY"
    fi
    grep -qx "export PYTHONPATH=$WT" "$PY" || die "$PY does not pin $WT"
    echo "setup ok: $WT at $WT_HEAD; $PY (sha256 $(sha "$PY"))"
}

worktree_ok() {
    [ -x "$PY" ] || die "$PY missing (run setup)"
    grep -qx "export PYTHONPATH=$WT" "$PY" || die "$PY does not pin $WT"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$WT_HEAD" ] || die "$WT is not $WT_HEAD"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
}

registered() {   # $1 = seed (any registered-seed check); 'check' passes the declared seed
    [ -f "$DECL" ] || die "register $ID first (prereg_e02z.py drafts it; the lead registers)"
    "$HPY" -I "$DECIDE" verify --seed "$1" > /dev/null \
        || die "the registration does not verify (INDEX/ledger/frozen code/inputs/seed)"
}

declared_seed() {
    "$HPY" -I -c '
import json, sys
print(json.load(open(sys.argv[1]))["decidability"]["seed"])' "$DECL"
}

forward_one() {   # $1 = ZR0 or a truth tag
    local tag=$1 dir out
    local sealed=$SEAL/${tag}.stdout.SEALED err=$RUNS/${tag}.stderr.log
    out=$(out_of "$tag")
    THETAS=()
    if [ "$tag" = ZR0 ]; then
        dir=$ZR0_DIR
        for t in "${ZR0_THETAS[@]}"; do THETAS+=(--theta "$R/$dir/$t"); done
    else
        dir=${THETA[$tag]}
        THETAS=(--theta "$R/$dir/stage3_theta.json")
    fi
    if [ "${DRY:-0}" = 1 ]; then
        echo "$PY -m hydrophysics.twin.forward ${THETAS[*]} $FLAGS" \
             "--vep-json $R/$dir/coupled_leveling/vep_zonal_leveling.json --out $out"
        return 0
    fi
    for s in npz summary.csv members.csv; do
        [ ! -e "$out.$s" ] || die "$out.$s exists (archive it first)"
    done
    worktree_ok
    local n t0 rc=0
    n=$(next_attempt)
    t0=$(utc)
    # shellcheck disable=SC2086  # FLAGS is a list of plain tokens
    "$PY" -m hydrophysics.twin.forward "${THETAS[@]}" $FLAGS \
        --vep-json "$R/$dir/coupled_leveling/vep_zonal_leveling.json" \
        --out "$out" > "$sealed" 2> "$err" || rc=$?
    chmod 400 "$sealed"
    local hs=missing
    [ -e "$out.npz" ] && hs=$(sha "$out.npz")
    log_attempt event=forward attempt="$n" run="$tag" rc="$rc" start="$t0" time="$(utc)" \
        worktree_head="$(git -C "$WT" rev-parse HEAD)" py_sh_sha256="$(sha "$PY")" \
        declaration_sha256="$(sha "$DECL")" npz_sha256="$hs" \
        stdout_sealed_sha256="$(sha "$sealed")" stderr_log="$err"
    echo "forward $tag: exit $rc (attempt $n; npz sha256 $hs)"
    [ "$rc" = 0 ] || die "forward $tag failed: read $err only, then archive and rerun"
}

do_forward() {
    if [ "${DRY:-0}" != 1 ]; then
        registered "$(declared_seed)"
        worktree_ok
        nvidia-smi > /dev/null || die "nvidia-smi failed"
        mkdir -p "$SEAL"
        chmod 700 "$SEAL"
    fi
    local runs=("${RUNS_ORDER[@]}")
    if [ -n "${1:-}" ]; then
        [ "$1" = ZR0 ] || [ -n "${THETA[$1]:-}" ] || die "unknown run $1 (ZR0 R0 Q0 R1c X5R1)"
        runs=("$1")
    fi
    for t in "${runs[@]}"; do forward_one "$t"; done
    echo "done; do not open the sealed logs; next: decide SEED (after all five runs)"
}

do_decide() {
    local seed=${1:?seed} n rc=0
    registered "$seed"
    for t in "${RUNS_ORDER[@]}"; do
        [ -f "$(out_of "$t").npz" ] || die "forward $t missing"
    done
    n=$(next_attempt)
    "$HPY" -I "$DECIDE" run --seed "$seed" || rc=$?
    log_attempt event=decide attempt="$n" rc="$rc" time="$(utc)"
    [ "$rc" = 0 ] && echo "decidability reading written once" || echo "decide exit $rc"
    return "$rc"
}

do_score() {
    local w=${1:?window B or C} n rc=0
    shift
    n=$(next_attempt)
    "$HPY" -I "$SCORE" score --window "$w" "$@" || rc=$?
    log_attempt event="score_$w" attempt="$n" rc="$rc" time="$(utc)"
    return "$rc"
}

do_archive() {
    local tag=${1:-} reason=${2:-} stamp dst stem moved=()
    [ -n "$tag" ] && [ -n "$reason" ] || die "archive RUN '<reason>'"
    [ ! -e results/dataprep/reports/$ID/decidability/report.json ] \
        || die "the decidability reading is written; nothing may be rerun"
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    dst=$RUNS/archive/$stamp
    mkdir -p "$dst"
    stem=$(out_of "$tag")
    for f in "$stem.npz" "$stem.summary.csv" "$stem.members.csv" \
             "$SEAL/${tag}.stdout.SEALED" "$RUNS/${tag}.stderr.log"; do
        if [ -e "$f" ]; then
            mv "$f" "$dst/"
            moved+=("$(basename "$f")")
        fi
    done
    log_attempt event=archive time="$(utc)" run="$tag" reason="$reason" dir="$dst" \
        moved="${moved[*]:-none}"
    echo "archived ${#moved[@]} files to $dst (unopened); rerun: run_e02z.sh forward $tag"
}

case "${1:-}" in
    setup) do_setup ;;
    check) registered "$(declared_seed)"; worktree_ok; echo "preconditions ok" ;;
    forward) shift; do_forward "${1:-}" ;;
    decide) shift; do_decide "${1:-}" ;;
    score) shift; do_score "$@" ;;
    archive) shift; do_archive "${1:-}" "${2:-}" ;;
    *) echo "usage: $0 setup|check|forward [RUN]|decide SEED|score B|C ARGS|archive RUN R" >&2
       exit 2 ;;
esac
