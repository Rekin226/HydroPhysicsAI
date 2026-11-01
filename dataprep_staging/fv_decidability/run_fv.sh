#!/usr/bin/env bash
# FV.decidability.v1 (model-only declaration): the truth forwards and the one scoring.
#
#   bash dataprep_staging/fv_decidability/run_fv.sh check          preconditions only (CPU)
#   bash dataprep_staging/fv_decidability/run_fv.sh arms           arm CSVs, once (CPU)
#   bash dataprep_staging/fv_decidability/run_fv.sh forward [TAG]  forwards (GPU, tmux; ZR0
#                                   then the four truths in order, or one TAG of ZR0 R0 Q0 R1c
#                                   X5R1 so two can run in parallel under MPS)
#   bash dataprep_staging/fv_decidability/run_fv.sh score          score once (CPU, ~5 min)
#   bash dataprep_staging/fv_decidability/run_fv.sh archive "<reason>"
#                                   engineering retry: move failed outputs aside, unopened
#   DRY=1 bash ... forward [TAG]    print the forward command lines, run nothing
#
# Code: the frozen worktree $HOME/HydroPhysicsAI-x5run at 5f3af18 (main, clean, detached; it
# has forward --recorded-pumping and wrote datum_gate_x5r1.npz, the hindcast reproduction
# reference), launched through results/twin_runs/par28/py.sh from the repository root.
# Recipe: E02's forward options (innovation update, gain 1, restart taper 5 km, apex hold
# calibrated, free column heads) with ONE member (--ic-members 0), horizon 58 (to 2027-10),
# baseline + 58 one-month Yunlin impulse arms + this truth's check arms (--recorded-pumping).
# ZR0 is the same-code reference of the page recipe: the 18 parameter sets of the registered
# run datum_gate_innov (stage3_datum_gate: theta, folds, posterior, in that order), the same
# FLAGS (so no IC member), baseline only, its coupled_leveling column; D_k = truth - ZR0.
# Nothing reads a post-2022 observation: the record ends at the 2022-12 origin and the
# projection is climatology times the arm multipliers.
#
# Binding: every subcommand except check refuses unless the declaration
# results/dataprep/preregistration/FV.decidability.v1.json is registered and intact
# (osse_fv.py verify-registration: INDEX, ledger, frozen_code incl. this script, inputs).
#
# Sealed outputs: the forward stdout (fan summaries, hindcast R2 lines) goes to
# results/twin_runs/fv_decidability/sealed/<TAG>_forward.stdout.SEALED, which nobody opens
# before the report exists; the npz values are read only by the scorer. stderr (tracebacks)
# goes to <TAG>_forward.stderr.log and may be read to diagnose a crash. This script prints
# only exit codes and sha256s.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
OSSE=dataprep_staging/fv_decidability/osse_fv.py
ID=FV.decidability.v1
DECL=results/dataprep/preregistration/$ID.json
WT="$HOME/HydroPhysicsAI-x5run"
WT_HEAD=5f3af18
PY=results/twin_runs/par28/py.sh
HPY="$HOME/miniforge3/envs/hydro/bin/python"
R=results/twin_runs
RUNS=results/twin_runs/fv_decidability
ARMS=results/twin_runs/fv_decidability/arms
SEAL=$RUNS/sealed
ATT=$RUNS/attempts.jsonl
REPORT=results/dataprep/reports/FV.decidability.v1/report.json
FLAGS="--horizon 58 --gain 1.0 --ic-members 0 --update-method innovation --restart-taper-km 5 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --device cuda --compile-matvec --save-members none"
declare -A THETA=([R0]=stage3_datum_gate_cwa [Q0]=stage3_datum_gate_q0 [R1c]=stage3_datum_gate_ccz [X5R1]=stage3_datum_gate_x5r1)
TAGS=(R0 Q0 R1c X5R1)
ZR0_DIR=stage3_datum_gate
ZR0_THETAS=(stage3_theta.json stage3_fold_thetas.json stage3_posterior_tn.json)

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# append one JSON record (key=value pairs; rc and attempt as integers) to attempts.jsonl
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

preconditions() {
    [ -f "$DECL" ] || die "declare $ID first (prereg_fv.py --write)"
    "$HPY" "$OSSE" verify-registration > /dev/null \
        || die "the registered declaration does not verify (INDEX/ledger/frozen code/inputs)"
    [ -x "$PY" ] || die "$PY missing"
    grep -qx "export PYTHONPATH=$WT" "$PY" || die "$PY does not pin $WT"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$WT_HEAD" ] || die "$WT is not $WT_HEAD"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; no rerun"
}

do_arms() {
    preconditions
    [ ! -e "$ARMS" ] || die "$ARMS exists"
    "$HPY" "$OSSE" arms
    log_attempt event=arms time="$(utc)" arms_json_sha256="$(sha "$ARMS/arms.json")" \
        declaration_sha256="$(sha "$DECL")"
}

forward_one() {   # $1 = ZR0 or a truth tag
    local tag=$1 dir out
    local sealed=$SEAL/${tag}_forward.stdout.SEALED err=$RUNS/${tag}_forward.stderr.log
    THETAS=()
    SPECS=()
    if [ "$tag" = ZR0 ]; then
        dir=$ZR0_DIR
        out=results/twin_forward/fv_zr0
        for t in "${ZR0_THETAS[@]}"; do THETAS+=(--theta "$R/$dir/$t"); done
    else
        dir=${THETA[$tag]}
        out=results/twin_forward/fv_truth_$tag
        THETAS=(--theta "$R/$dir/stage3_theta.json")
        if [ -f "$ARMS/arms.json" ]; then
            mapfile -t SPECS < <("$HPY" "$OSSE" specs --truth "$tag" \
                                 | sed 's/^/--recorded-pumping=/')
        fi
    fi
    if [ "${DRY:-0}" = 1 ]; then
        echo "$PY -m hydrophysics.twin.forward ${THETAS[*]} $FLAGS" \
             "--vep-json $R/$dir/coupled_leveling/vep_zonal_leveling.json" \
             "${#SPECS[@]} x --recorded-pumping=... --out $out"
        return 0
    fi
    if [ "$tag" != ZR0 ]; then
        [ "${#SPECS[@]}" -ge 59 ] || die "too few arm specs for $tag (${#SPECS[@]})"
    fi
    for s in npz summary.csv members.csv; do
        [ ! -e "$out.$s" ] || die "$out.$s exists (archive it first)"
    done
    local n t0 rc=0
    n=$(next_attempt)
    t0=$(utc)
    # shellcheck disable=SC2086  # FLAGS is a list of plain tokens
    "$PY" -m hydrophysics.twin.forward "${THETAS[@]}" $FLAGS \
        --vep-json "$R/$dir/coupled_leveling/vep_zonal_leveling.json" \
        ${SPECS[@]+"${SPECS[@]}"} --out "$out" > "$sealed" 2> "$err" || rc=$?
    chmod 400 "$sealed"
    local hs=missing
    [ -e "$out.npz" ] && hs=$(sha "$out.npz")
    log_attempt event=forward attempt="$n" truth="$tag" rc="$rc" start="$t0" time="$(utc)" \
        worktree_head="$(git -C "$WT" rev-parse HEAD)" py_sh_sha256="$(sha "$PY")" \
        declaration_sha256="$(sha "$DECL")" arms_json_sha256="$(sha "$ARMS/arms.json")" \
        npz_sha256="$hs" stdout_sealed_sha256="$(sha "$sealed")" stderr_log="$err"
    echo "forward $tag: exit $rc (attempt $n; npz sha256 $hs)"
    [ "$rc" = 0 ] || die "forward $tag failed: read $err only, then archive and rerun"
}

do_forward() {
    if [ "${DRY:-0}" != 1 ]; then
        preconditions
        [ -f "$ARMS/arms.json" ] || die "run 'arms' first"
        nvidia-smi > /dev/null || die "nvidia-smi failed"
        mkdir -p "$SEAL"
        chmod 700 "$SEAL"
    fi
    if [ -n "${1:-}" ]; then
        [ "$1" = ZR0 ] || [ -n "${THETA[$1]:-}" ] || die "unknown run $1 (ZR0 R0 Q0 R1c X5R1)"
        forward_one "$1"
    else
        for t in ZR0 "${TAGS[@]}"; do forward_one "$t"; done
    fi
    echo "done; do not open the sealed logs; next: score (after ZR0 and all four truths)"
}

do_score() {
    preconditions
    local n rc=0 rs=missing
    n=$(next_attempt)
    "$HPY" "$OSSE" score --out "$REPORT" || rc=$?
    [ -e "$REPORT" ] && rs=$(sha "$REPORT")
    log_attempt event=score attempt="$n" rc="$rc" time="$(utc)" report_sha256="$rs" \
        declaration_sha256="$(sha "$DECL")"
    case $rc in
        0) echo "scored once: $REPORT (sha256 $rs); the sealed logs may now be read" ;;
        1) echo "engineering failure (gate E): nothing written; fix the cause, then" \
                "run_fv.sh archive '<reason>' if a forward must be redone" ;;
        *) echo "scorer exit $rc" ;;
    esac
    return "$rc"
}

do_archive() {
    local reason=${1:-} stamp dst moved=()
    [ -n "$reason" ] || die "archive needs a reason"
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; nothing may be rerun"
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    dst=$RUNS/archive/$stamp
    mkdir -p "$dst"
    for t in ZR0 "${TAGS[@]}"; do
        local stem=results/twin_forward/fv_truth_$t
        [ "$t" = ZR0 ] && stem=results/twin_forward/fv_zr0
        for f in "$stem.npz" "$stem.summary.csv" "$stem.members.csv" \
                 "$SEAL/${t}_forward.stdout.SEALED" "$RUNS/${t}_forward.stderr.log"; do
            if [ -e "$f" ]; then
                mv "$f" "$dst/"
                moved+=("$(basename "$f")")
            fi
        done
    done
    log_attempt event=archive time="$(utc)" reason="$reason" dir="$dst" \
        moved="${moved[*]:-none}"
    echo "archived ${#moved[@]} files to $dst (unopened); rerun: run_fv.sh forward [TAG]"
}

case "${1:-}" in
    check) preconditions; echo "preconditions ok" ;;
    arms) do_arms ;;
    forward) shift; do_forward "${1:-}" ;;
    score) do_score ;;
    archive) shift; do_archive "${1:-}" ;;
    *) echo "usage: $0 check|arms|forward [TAG]|score|archive '<reason>'" >&2; exit 2 ;;
esac
