#!/usr/bin/env bash
# FV.decidability.v2_postfailure (+ riders FV.ext.v1, FV.stat.v1): forwards and scoring.
#
#   bash dataprep_staging/fv_decidability_v2/run_fv2.sh setup         frozen worktree + py.sh
#   bash dataprep_staging/fv_decidability_v2/run_fv2.sh check         preconditions only (CPU)
#   bash dataprep_staging/fv_decidability_v2/run_fv2.sh arms          1000 world CSVs, once
#   bash dataprep_staging/fv_decidability_v2/run_fv2.sh forward [TAG [B]]
#                                   GPU, in tmux: all batches of TAG (or batch B only), or
#                                   every truth in order; 2-3 runs in parallel under MPS
#   bash dataprep_staging/fv_decidability_v2/run_fv2.sh score-worlds  reading 1, once (CPU)
#   bash dataprep_staging/fv_decidability_v2/run_fv2.sh score-ext     FV.ext.v1, once (CPU)
#   bash dataprep_staging/fv_decidability_v2/run_fv2.sh replay-index ARGS...
#                                   index rank 11's replay forwards (metadata only)
#   bash dataprep_staging/fv_decidability_v2/run_fv2.sh score-forcing INDEX
#   bash dataprep_staging/fv_decidability_v2/run_fv2.sh score-forcing --unavailable "<reason>"
#                                   reading 2, once (CPU); --unavailable only on a pre-declared
#                                   ground the scorer checks (FW not evaluable; FW report absent
#                                   on/after 2027-01-31; a logged FV gate-E failure on FW's
#                                   forwards, event forcing_replay_gate_e)
#   bash dataprep_staging/fv_decidability_v2/run_fv2.sh score-stat    FV.stat.v1, once (CPU)
#   bash dataprep_staging/fv_decidability_v2/run_fv2.sh archive TAG B "<reason>"
#                                   engineering retry: move a failed batch aside, unopened
#   DRY=1 bash ... forward [TAG [B]]  print the forward command lines, run nothing
#
# GPU (T4, sm_75): 20 forwards of 51 scenarios x 58 months, one member each (v1: 11-18 min
# per 60 scenarios), about 4-6 GPU-h, 2-3 h wall under MPS. Example:
#   nvidia-smi && nvidia-cuda-mps-control -d
#   tmux new -d -s fv2a 'bash dataprep_staging/fv_decidability_v2/run_fv2.sh forward R0'
#   tmux new -d -s fv2b 'bash dataprep_staging/fv_decidability_v2/run_fv2.sh forward Q0'
#   tmux new -d -s fv2c 'bash dataprep_staging/fv_decidability_v2/run_fv2.sh forward R1c'
#   (then X5R1); echo quit | nvidia-cuda-mps-control when all are done.
# Disk: about 8-9 GB of batch npz under results/twin_forward/fv2_truth_*.
#
# Code: a fresh clean detached worktree $HOME/HydroPhysicsAI-fv2run at 5f3af18, the commit of
# FV v1's ZR0 and truth forwards (so every batch baseline must equal v1's truth forward over
# 2012-22 to 0.0 m), launched through results/twin_runs/fv2/py.sh from the repository root.
# Code freeze: nothing in that worktree is edited while forwards run; each forward refuses a
# dirty worktree. Recipe: E02's forward options with ONE member (--ic-members 0), horizon 58,
# baseline + 50 worlds per forward (--recorded-pumping, Yunlin cells). Nothing reads a
# post-2022 observation: the record ends at the 2022-12 origin.
#
# Binding: every subcommand except setup/check/replay-index refuses unless the relevant
# declaration is registered and intact (verify-registration: INDEX, ledger, frozen_code incl.
# this script, inputs).
#
# Sealed outputs: forward stdout goes to results/twin_runs/fv2/sealed/<TAG>_b<B>.stdout.SEALED,
# which nobody opens before the reports exist; npz values are read only by the scorers.
# stderr goes to <TAG>_b<B>.stderr.log and may be read to diagnose a crash. The forward also
# writes <out>.summary.csv and <out>.members.csv: model outputs, left unopened until scoring.
# This script prints only exit codes and sha256s.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
OSSE=dataprep_staging/fv_decidability_v2/osse_fv2.py
EXT=dataprep_staging/fv_decidability_v2/ext_fv.py
STAT=dataprep_staging/fv_decidability_v2/stat_fv.py
ID=FV.decidability.v2_postfailure
DECL=results/dataprep/preregistration/$ID.json
WT_HEAD=5f3af18
WT="$HOME/HydroPhysicsAI-fv2run"
PY=results/twin_runs/fv2/py.sh
HPY="$HOME/miniforge3/envs/hydro/bin/python"
R=results/twin_runs
RUNS=results/twin_runs/fv2
ARMS=results/twin_runs/fv2/arms
SEAL=$RUNS/sealed
ATT=$RUNS/attempts.jsonl
BATCH=50
NBATCH=5
ZR0_NPZ=results/twin_forward/fv_zr0.npz
REPORT_W=results/dataprep/reports/$ID/report_worlds.json
REPORT_F=results/dataprep/reports/$ID/report_forcing_year.json
FLAGS="--horizon 58 --gain 1.0 --ic-members 0 --update-method innovation --restart-taper-km 5 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --device cuda --compile-matvec --save-members none"
declare -A THETA=([R0]=stage3_datum_gate_cwa [Q0]=stage3_datum_gate_q0 [R1c]=stage3_datum_gate_ccz [X5R1]=stage3_datum_gate_x5r1)
TAGS=(R0 Q0 R1c X5R1)

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
    rec[k] = int(v) if v.lstrip("-").isdigit() and k in ("rc", "attempt", "batch") else v
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
            echo '# working directory (FV.decidability.v2_postfailure).'
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

preconditions() {
    [ -f "$DECL" ] || die "declare $ID first (prereg_fv2.py --decl v2 --write)"
    "$HPY" "$OSSE" verify-registration > /dev/null \
        || die "the registered declaration does not verify (INDEX/ledger/frozen code/inputs)"
    worktree_ok
}

do_arms() {
    preconditions
    for d in FV.ext.v1 FV.stat.v1; do
        if [ ! -f "results/dataprep/preregistration/$d.json" ]; then
            [ -n "${SKIP_RIDERS:-}" ] || die "register $d before any world is drawn, or set" \
                "SKIP_RIDERS='<who, when, why it is not run>' (logged)"
            log_attempt event=rider_skipped time="$(utc)" rider="$d" reason="$SKIP_RIDERS"
        fi
    done
    [ ! -e "$ARMS" ] || die "$ARMS exists"
    "$HPY" "$OSSE" arms
    log_attempt event=arms time="$(utc)" arms_json_sha256="$(sha "$ARMS/arms.json")" \
        declaration_sha256="$(sha "$DECL")"
}

forward_one() {   # $1 = truth tag, $2 = batch
    local tag=$1 b=$2 dir out
    local sealed=$SEAL/${tag}_b${b}.stdout.SEALED err=$RUNS/${tag}_b${b}.stderr.log
    dir=${THETA[$tag]}
    out=results/twin_forward/fv2_truth_${tag}_b${b}
    THETAS=(--theta "$R/$dir/stage3_theta.json")
    SPECS=()
    if [ -f "$ARMS/arms.json" ]; then
        mapfile -t SPECS < <("$HPY" "$OSSE" specs --truth "$tag" --batch "$b" \
                             | sed 's/^/--recorded-pumping=/')
    fi
    if [ "${DRY:-0}" = 1 ]; then
        echo "$PY -m hydrophysics.twin.forward ${THETAS[*]} $FLAGS" \
             "--vep-json $R/$dir/coupled_leveling/vep_zonal_leveling.json" \
             "${#SPECS[@]} x --recorded-pumping=... --out $out"
        return 0
    fi
    [ "${#SPECS[@]}" -ge 1 ] || die "no world specs for $tag batch $b"
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
        ${SPECS[@]+"${SPECS[@]}"} --out "$out" > "$sealed" 2> "$err" || rc=$?
    chmod 400 "$sealed"
    local hs=missing
    [ -e "$out.npz" ] && hs=$(sha "$out.npz")
    log_attempt event=forward attempt="$n" truth="$tag" batch="$b" rc="$rc" start="$t0" \
        time="$(utc)" worktree_head="$(git -C "$WT" rev-parse HEAD)" \
        py_sh_sha256="$(sha "$PY")" declaration_sha256="$(sha "$DECL")" \
        arms_json_sha256="$(sha "$ARMS/arms.json")" npz_sha256="$hs" \
        stdout_sealed_sha256="$(sha "$sealed")" stderr_log="$err"
    echo "forward $tag batch $b: exit $rc (attempt $n; npz sha256 $hs)"
    [ "$rc" = 0 ] || die "forward $tag batch $b failed: read $err only, then archive and rerun"
}

do_forward() {
    if [ "${DRY:-0}" != 1 ]; then
        preconditions
        [ -f "$ARMS/arms.json" ] || die "run 'arms' first"
        nvidia-smi > /dev/null || die "nvidia-smi failed"
        mkdir -p "$SEAL"
        chmod 700 "$SEAL"
    fi
    local tags=("${TAGS[@]}") bs
    if [ -n "${1:-}" ]; then
        [ -n "${THETA[$1]:-}" ] || die "unknown truth $1 (R0 Q0 R1c X5R1)"
        tags=("$1")
    fi
    for t in "${tags[@]}"; do
        if [ -n "${2:-}" ]; then
            bs=("$2")
        else
            mapfile -t bs < <(seq 0 $((NBATCH - 1)))
        fi
        for b in "${bs[@]}"; do forward_one "$t" "$b"; done
    done
    echo "done; do not open the sealed logs; next: score-worlds and score-ext (after all" \
         "$((4 * NBATCH)) batches)"
}

score_with() {   # $1 = label, then the scorer command
    local label=$1 n rc=0
    shift
    n=$(next_attempt)
    "$@" || rc=$?
    log_attempt event="score_$label" attempt="$n" rc="$rc" time="$(utc)"
    case $rc in
        0) echo "scored once: $label" ;;
        1) echo "engineering failure: nothing written; fix the cause (archive a batch if a" \
                "forward must be redone)" ;;
        *) echo "scorer exit $rc" ;;
    esac
    return "$rc"
}

do_archive() {
    local tag=${1:-} b=${2:-} reason=${3:-} stamp dst moved=()
    [ -n "$tag" ] && [ -n "$b" ] && [ -n "$reason" ] || die "archive TAG B '<reason>'"
    [ ! -e "$REPORT_W" ] || die "$REPORT_W exists: reading 1 is scored; nothing may be rerun"
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    dst=$RUNS/archive/$stamp
    mkdir -p "$dst"
    local stem=results/twin_forward/fv2_truth_${tag}_b${b}
    for f in "$stem.npz" "$stem.summary.csv" "$stem.members.csv" \
             "$SEAL/${tag}_b${b}.stdout.SEALED" "$RUNS/${tag}_b${b}.stderr.log"; do
        if [ -e "$f" ]; then
            mv "$f" "$dst/"
            moved+=("$(basename "$f")")
        fi
    done
    log_attempt event=archive time="$(utc)" truth="$tag" batch="$b" reason="$reason" \
        dir="$dst" moved="${moved[*]:-none}"
    echo "archived ${#moved[@]} files to $dst (unopened); rerun: run_fv2.sh forward $tag $b"
}

case "${1:-}" in
    setup) do_setup ;;
    check) preconditions; echo "preconditions ok" ;;
    arms) do_arms ;;
    forward) shift; do_forward "${1:-}" "${2:-}" ;;
    score-worlds) preconditions; score_with worlds "$HPY" "$OSSE" score-worlds ;;
    score-ext) preconditions; score_with ext "$HPY" "$EXT" score ;;
    replay-index) shift; "$HPY" "$OSSE" replay-index "$@" ;;
    score-forcing)
        shift
        preconditions
        if [ "${1:-}" = --unavailable ]; then
            score_with forcing "$HPY" "$OSSE" score-forcing --replay-unavailable "${2:?reason}"
        else
            score_with forcing "$HPY" "$OSSE" score-forcing --replay-index "${1:?index}"
        fi ;;
    score-stat) score_with stat "$HPY" "$STAT" score ;;
    archive) shift; do_archive "${1:-}" "${2:-}" "${3:-}" ;;
    *) echo "usage: $0 setup|check|arms|forward [TAG [B]]|score-worlds|score-ext|" \
            "replay-index ...|score-forcing INDEX|--unavailable R|score-stat|archive T B R" >&2
       exit 2 ;;
esac
