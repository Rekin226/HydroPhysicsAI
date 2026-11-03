#!/usr/bin/env bash
# Round 8 rank 5 (E02x): the lead's steps, in order. Every step refuses on a missing
# predecessor; nothing here registers anything (prereg_e02x.py power|e02x --write do).
#
#   bash dataprep_staging/e02x/run_e02x.sh check            preconditions (CPU, no read)
#   bash dataprep_staging/e02x/run_e02x.sh members all      36 forwards (GPU, tmux; MPS fine)
#   bash dataprep_staging/e02x/run_e02x.sh members split1   the first 9 parameter sets (18 runs)
#   bash dataprep_staging/e02x/run_e02x.sh members split2   the last 9 parameter sets (18 runs)
#   bash dataprep_staging/e02x/run_e02x.sh members <label> [<label> ...]
#   bash dataprep_staging/e02x/run_e02x.sh extract [e02d|rerun]   member gate (CPU); default
#                                                           e02d (lead Q1: 0 GPU, E02d's files)
#   bash dataprep_staging/e02x/run_e02x.sh power <seed>     U1.ext_transfer_power.v1 (CPU,
#                                                           registered; the registered seed)
#   bash dataprep_staging/e02x/run_e02x.sh probe <window>   prereg3 blocking probe (CPU)
#   bash dataprep_staging/e02x/run_e02x.sh sdfit <window>   the ONE logged design-time read
#   bash dataprep_staging/e02x/run_e02x.sh artefact         sigma_idio (CPU)
#   bash dataprep_staging/e02x/run_e02x.sh archive "<reason>" [label ...]
#   DRY=1 bash ... members <label|all|split1|split2>        print the forward command lines
#
# Order (deadline 2026-11-30T16:00Z for the primary window; fallback 2026-12-31T16:00Z):
#   prereg_e02x.py power --write --seed S -> power S
#   extract e02d               (lead decision Q1: no GPU; 'members all' + 'extract rerun' is the
#                               optional GPU reproduction)
#   probe primary -> sdfit primary -> artefact -> prereg_e02x.py e02x --write --window primary
#
# Members: the SAME 18 theta files as E02d (results/twin_runs/e02d/thetas, sha-checked by
# extract), the SAME frozen worktree $HOME/HydroPhysicsAI-e02drun-<code> and command line as
# run_e02d.sh. Code ladder: 4c2723e first (E02d's members ran there, gate passed 2026-11-01);
# a logged member-gate failure (extract exit 3) moves to 730f65b; after that nothing runs
# and E02x is not registered (no fallback interval). Measured cost at 4c2723e (E02d,
# 2026-11-01): 36 runs, 87-163 s each (median 106 s), 1.05 GPU-h in one sequential job.
#
# Sealed outputs: forward.py prints in-sample hindcast skill (heads and leveling) on stdout.
# Each member's stdout goes to results/twin_runs/e02x/sealed/<label>_ic<k>.stdout.SEALED,
# which nobody opens; only its sha256 is logged in results/twin_runs/e02x/attempts.jsonl.
# stderr (tracebacks) may be read to fix a crash. No --data flag is ever passed.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
HPY="$HOME/miniforge3/envs/hydro/bin/python"
E=dataprep_staging/e02x
CODES=(4c2723e 730f65b)
R=results/twin_runs
G=$R/stage3_datum_gate
LOG=$R/e02x
SEAL=$LOG/sealed
ATT=$LOG/attempts.jsonl
THETAS=$R/e02d/thetas
MOUT=results/twin_forward/e02x_members
MIN_FREE_MIB=${E02X_MIN_FREE_MIB:-4000}
VEP="$G/coupled_leveling/vep_zonal_leveling.json,$G/coupled_leveling_tau30/vep_zonal_leveling.json"
RF=chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv
COMMON="--horizon 120 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --rf-timeseries $RF --device cuda --compile-matvec"
F_MEMBER="--gain 1.0 --ic-sigma 0.5 --seed 0 --update-method innovation --restart-taper-km 5 --save-members none"
FORWARD=results/twin_forward/datum_gate_innov.npz
FORWARD_SHA=cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228
LABELS_ALL=(stage3_datum_gate fold0 fold1 fold2 fold3 fold4 foldpost0 foldpost1 foldpost2
            foldpost3 foldpost4 foldpost5 foldpost6 foldpost7 foldpost8 foldpost9 foldpost10
            foldpost11)

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
        "(${CODES[*]}): E02x is NOT registered (no fallback interval); record C14f-r N" \
        "('undecidable: the external member spread could not be reproduced')"
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
    for f in "$THETAS/manifest.json" "$RF" "${VEP%%,*}" "${VEP##*,}"; do
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

expand_labels() {
    case "${1:-}" in
        all) echo "${LABELS_ALL[*]}" ;;
        split1) echo "${LABELS_ALL[*]:0:9}" ;;
        split2) echo "${LABELS_ALL[*]:9:9}" ;;
        *) echo "$*" ;;
    esac
}

do_members() {
    [ "$#" -ge 1 ] || die "members needs 'all', 'split1', 'split2' or labels"
    local list
    read -r -a list <<< "$(expand_labels "$@")"
    for l in "${list[@]}"; do
        case " ${LABELS_ALL[*]} " in *" $l "*) ;; *) die "unknown label $l" ;; esac
    done
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

do_extract() {
    local src=${1:-e02d}
    base_check
    [ "$src" = e02d ] || worktree_check
    rc=0
    "$HPY" -I "$E/members_e02x.py" extract --source "$src" || rc=$?
    log_attempt event=extract source="$src" rc="$rc" time="$(utc)" code="${CODE:-none}"
    case $rc in
        0) echo "member gate passed (source $src)" ;;
        3) echo "member gate FAILED (source $src, logged): archive 'gate failed' and rerun" \
                "members at the next code" ;;
        *) echo "engineering failure (exit $rc): fix the cause and rerun extract" ;;
    esac
    return "$rc"
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
        for f in "$LOG/members_ext.npz" "$LOG/members_gate.json"; do
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
    members) shift; do_members "$@" ;;
    extract) shift; do_extract "$@" ;;
    power) [ -n "${2:-}" ] || die "power needs the registered seed"
           "$HPY" -I "$E/power_e02x.py" run --seed "$2" ;;
    probe) [ -n "${2:-}" ] || die "probe needs a window (primary|fallback)"
           "$HPY" -I "$E/prereg_e02x.py" probe --window "$2" ;;
    sdfit) [ -n "${2:-}" ] || die "sdfit needs a window (primary|fallback)"
           "$HPY" -I "$E/sdfit_e02x.py" run --window "$2" ;;
    artefact) "$HPY" -I "$E/build_artefact_e02x.py" ;;
    archive) shift; do_archive "$@" ;;
    *) echo "usage: $0 check|members <all|split1|split2|label...>|extract [e02d|rerun]|" \
            "power <seed>|probe <window>|sdfit <window>|artefact|archive '<reason>' [label ...]" >&2
       exit 2 ;;
esac
