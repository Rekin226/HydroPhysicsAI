#!/usr/bin/env bash
# SD.cut_response_spread.v1 (model-only declaration, round 7 rank 9): the forwards and the
# one scoring. The same forwards' hindcast months feed X5.gnss_seasonal_column.v1's model
# side (run_gsc.sh), which runs only after this declaration is scored.
#
#   bash dataprep_staging/sd_column/run_sd.sh check             preconditions only (CPU)
#   bash dataprep_staging/sd_column/run_sd.sh forward <key>     one forward (GPU, tmux)
#   bash dataprep_staging/sd_column/run_sd.sh forward all       every forward, sequential
#   bash dataprep_staging/sd_column/run_sd.sh score             score once (CPU)
#   bash dataprep_staging/sd_column/run_sd.sh archive "<reason>" [key ...]
#        engineering retry: move outputs aside, unopened (refused once the report exists)
#   DRY=1 bash ... forward <key|all>    print the forward command lines, run nothing
#
# Keys: page q0 q0cs cwa q0nh ccz q0c x5r1cs. Each forward is ONE parameter set (the theta's
# main set), --ic-members 0, --gain 0.0 (the saved heads are the column's driver heads),
# baseline + cut30, origin 2022-12 (never --extend-record-to), every column of the key on
# one rheology axis. q0nh keeps the hindcast nudging its column was fitted on (gain 1 every
# month through 2022-12 only).
#
# Code: a clean detached worktree $HOME/HydroPhysicsAI-sdrun-4c2723e at main 4c2723e (the
# code FS.start_invariance.v1 reproduced the page run with), imported through PYTHONPATH
# with python -P from the repository root that holds results/ and chou-shui-data/. Create it
# with: git worktree add --detach "$HOME/HydroPhysicsAI-sdrun-4c2723e" 4c2723e
# Code freeze: no hydrophysics/twin edit anywhere while these run. Under MPS, separate tmux
# windows may run different keys at once (one key per window).
#
# Sealed outputs: forward.py prints every scenario's fan means ('=== forward summary ===')
# and, because HYDROMIND_GW_DATA is set, a hindcast-vs-leveling line, and writes
# results/twin_forward/sd_<key>.{npz,summary.csv,members.csv} before the scorer runs. The
# stdout goes to results/twin_runs/sd_column/sealed/<key>_forward.stdout.SEALED (mode 400):
# nobody opens, tails or greps it, the CSVs, or any cut30 slice, before report.json exists.
# stderr (tracebacks, warnings) may be read to diagnose a crash.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${DATAPREP_REPO:-$(cd "$X/../.." && pwd)}
cd "$REPO"
SELF=dataprep_staging/sd_column/run_sd.sh
SCORER=dataprep_staging/sd_column/score_sd.py
ID=SD.cut_response_spread.v1
DECL=results/dataprep/preregistration/$ID.json
CODE=4c2723e
WT="$HOME/HydroPhysicsAI-sdrun-4c2723e"
HPY="$HOME/miniforge3/envs/hydro/bin/python"
LOG=results/twin_runs/sd_column
SEAL=$LOG/sealed
ATT=$LOG/attempts.jsonl
REPORT=results/dataprep/reports/$ID/report.json
KEYS=(page q0 q0cs cwa q0nh ccz q0c x5r1cs)
MIN_FREE_MIB=${SD_MIN_FREE_MIB:-3000}
export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

fwd_cmd() {   # the declared command line of key $1 (sdlib.forward_command)
    "$HPY" -I -c 'import sys; sys.path.insert(0, sys.argv[1])
import sdlib
print(sdlib.forward_command(sys.argv[2]))' "$X" "$1"
}

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

count_records() {
    if [ ! -f "$ATT" ]; then echo 0; return; fi
    "$HPY" -I -c '
import json, sys
want = dict(kv.split("=", 1) for kv in sys.argv[2:])
print(sum(all(str(json.loads(s).get(k)) == v for k, v in want.items())
          for s in open(sys.argv[1]) if s.strip()))' "$ATT" "$@"
}

known_key() { local k; for k in "${KEYS[@]}"; do [ "$k" = "$1" ] && return 0; done; return 1; }

preconditions() {
    [ -f "$DECL" ] || die "declare $ID first (prereg_sd.py --write)"
    for f in "$SCORER" "$SELF" dataprep_staging/sd_column/sdlib.py; do
        want=$("$HPY" -I -c 'import json, sys
print(json.load(open(sys.argv[1])).get("frozen_code", {}).get(sys.argv[2], "none"))' "$DECL" "$f")
        [ "$want" = "$(sha "$f")" ] || die "$f sha256 differs from the registered declaration"
    done
    "$HPY" "$SCORER" verify-registration > /dev/null \
        || die "the registered declaration does not verify (INDEX/ledger/frozen code/inputs)"
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; no rerun"
    [ -d "$WT" ] || die "worktree $WT missing (git worktree add --detach $WT $CODE)"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$CODE" ] || die "$WT is not at $CODE"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    mod=$(PYTHONPATH="$WT" "$HPY" -P -c \
        'import hydrophysics.twin.forward as m; print(m.__file__)' 2> /dev/null) \
        || die "hydrophysics.twin.forward does not import from $WT"
    case $mod in "$WT"/*) ;; *) die "forward imports from $mod, not $WT" ;; esac
    [ -d "$HYDROMIND_GW_DATA" ] || die "HYDROMIND_GW_DATA $HYDROMIND_GW_DATA missing"
}

gpu_check() {
    nvidia-smi > /dev/null || die "nvidia-smi failed"
    read -r total used < <(nvidia-smi --query-gpu=memory.total,memory.used \
        --format=csv,noheader,nounits | head -1 | tr -d ',')
    free=$((total - used))
    echo "GPU memory free: $free MiB (need $MIN_FREE_MIB)"
    [ "$free" -ge "$MIN_FREE_MIB" ] || die "not enough free GPU memory ($free MiB)"
}

do_forward_one() {
    local k=$1 s
    local out=results/twin_forward/sd_$k
    for s in npz summary.csv members.csv; do
        [ ! -e "$out.$s" ] || die "$out.$s exists (archive it first)"
    done
    mkdir -p "$SEAL"
    chmod 700 "$SEAL"
    n=$(( $(count_records event=forward fwd="$k") + 1 ))
    sealed=$SEAL/${k}_forward.stdout.SEALED
    err=$LOG/${k}_forward.stderr.log
    cmd=$(fwd_cmd "$k")
    t0=$(utc)
    rc=0
    eval "PYTHONPATH=\"\$WT\" \"\$HPY\" -P ${cmd#\$PY }" > "$sealed" 2> "$err" || rc=$?
    chmod 400 "$sealed"
    hs=(); for s in npz summary.csv members.csv; do
        if [ -e "$out.$s" ]; then hs+=("$(sha "$out.$s")"); else hs+=(missing); fi
    done
    log_attempt event=forward attempt="$n" fwd="$k" rc="$rc" start="$t0" time="$(utc)" \
        code="$CODE" worktree_head="$(git -C "$WT" rev-parse HEAD)" \
        declaration_sha256="$(sha "$DECL")" npz_sha256="${hs[0]}" \
        summary_csv_sha256="${hs[1]}" members_csv_sha256="${hs[2]}" \
        stdout_sealed_sha256="$(sha "$sealed")" stdout_sealed="$sealed" stderr_log="$err"
    echo "forward $k: exit $rc (attempt $n; stdout sealed, sha256 $(sha "$sealed"))"
    [ "$rc" = 0 ] || die "forward $k failed: read $err only, then archive and rerun"
}

do_forward() {
    local which=${1:-}
    [ -n "$which" ] || die "forward needs a key (${KEYS[*]}) or all"
    local list=("$which")
    [ "$which" != all ] || list=("${KEYS[@]}")
    for k in "${list[@]}"; do known_key "$k" || die "unknown key $k"; done
    if [ "${DRY:-0}" = 1 ]; then
        for k in "${list[@]}"; do fwd_cmd "$k"; done
        return 0
    fi
    preconditions
    gpu_check
    for k in "${list[@]}"; do do_forward_one "$k"; done
    echo "done (${list[*]}); do not open the sealed logs, the sd_* CSVs or any cut30 slice"
}

do_score() {
    preconditions
    n=$(( $(count_records event=score) + 1 ))
    rc=0
    "$HPY" "$SCORER" score --attempts "$ATT" --out "$REPORT" || rc=$?
    rs=missing; [ -e "$REPORT" ] && rs=$(sha "$REPORT")
    log_attempt event=score attempt="$n" rc="$rc" time="$(utc)" code="$CODE" \
        report_sha256="$rs" declaration_sha256="$(sha "$DECL")"
    case $rc in
        0) echo "scored once: $REPORT (sha256 $rs); sealed logs and CSVs may now be read" ;;
        1|2|3) echo "engineering failure (scorer exit $rc): no cut30 quantity was formed; fix," \
                    "then: run_sd.sh archive '<reason>' [key ...] and rerun those forwards" ;;
        *) echo "scorer exit $rc" ;;
    esac
    return "$rc"
}

do_archive() {
    reason=${1:-}
    [ -n "$reason" ] || die "archive needs a reason"
    shift
    local which=("${KEYS[@]}")
    [ "$#" = 0 ] || which=("$@")
    for k in "${which[@]}"; do known_key "$k" || die "unknown key $k"; done
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; nothing may be rerun"
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    dst=$LOG/archive/$stamp
    mkdir -p "$dst"
    moved=()
    for k in "${which[@]}"; do
        for f in results/twin_forward/sd_$k.npz results/twin_forward/sd_$k.summary.csv \
                 results/twin_forward/sd_$k.members.csv "$SEAL/${k}_forward.stdout.SEALED" \
                 "$LOG/${k}_forward.stderr.log"; do
            if [ -e "$f" ]; then mv "$f" "$dst/"; moved+=("$(basename "$f")"); fi
        done
    done
    log_attempt event=archive time="$(utc)" reason="$reason" dir="$dst" \
        keys="${which[*]}" moved="${moved[*]:-none}"
    echo "archived ${#moved[@]} files to $dst (unopened)"
}

case "${1:-}" in
    check) preconditions; echo "preconditions ok (code $CODE, worktree $WT)" ;;
    forward) shift; do_forward "${1:-}" ;;
    score) do_score ;;
    archive) shift; do_archive "$@" ;;
    *) echo "usage: $0 check|forward <key|all>|score|archive '<reason>' [key ...]" >&2
       exit 2 ;;
esac
