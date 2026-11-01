#!/usr/bin/env bash
# X5.n2c_reading.v1 (model-only synthetic declaration, round 5 rank 10): inputs, the CPU
# replay forward of R1c, the panel and the one registered synthetic run.
#
#   bash dataprep_staging/n2c_reading/run_n2c_reading.sh replays   BEFORE registration, CPU:
#        the seven replayed pre-origin anomaly fields (x5_changhua_lever/replay.py)
#   bash dataprep_staging/n2c_reading/run_n2c_reading.sh check     preconditions (registered)
#   bash dataprep_staging/n2c_reading/run_n2c_reading.sh forward   AFTER registration, CPU,
#        tmux: one forward of R1c's main set, baseline + RP2012..RP2018 (N2C's command)
#   bash dataprep_staging/n2c_reading/run_n2c_reading.sh extract   the 74 x 12 contrast panel
#   bash dataprep_staging/n2c_reading/run_n2c_reading.sh run       the registered run (once)
#   bash dataprep_staging/n2c_reading/run_n2c_reading.sh archive "<reason>"
#        engineering retry: move a failed forward attempt aside, unopened
#   DRY=1 bash ... forward    print the forward command line, run nothing
#
# Code: the frozen clean detached worktree $HOME/HydroPhysicsAI-r6ch at 1b08fb6 (N2C's
# registered code commit), launched through results/twin_runs/r6ch/py.sh, which must pin
# PYTHONPATH to it. The forward is N2C's registered primary command (CPU, main parameter set,
# --ic-members 1) with its PR arm replaced by the seven replay arms and --save-members yearly
# (no effect on heads; it lets extract check the unperturbed member against
# datum_gate_ccz.members.npz). The forward's stdout goes to a log that is not opened before
# result.json exists; this script prints exit codes and sha256s only.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
SYNTH=dataprep_staging/n2c_reading/synth_n2c.py
REPLAY=dataprep_staging/x5_changhua_lever/replay.py
ID=X5.n2c_reading.v1
DECL=results/dataprep/preregistration/$ID.json
WT="$HOME/HydroPhysicsAI-r6ch"
WT_HEAD=1b08fb6
PY=results/twin_runs/r6ch/py.sh
HPY="$HOME/miniforge3/envs/hydro/bin/python"
R=results/twin_runs
LOG=$R/n2c_reading
RPL=$LOG/replays
SEAL=$LOG/sealed
ATT=$LOG/attempts.jsonl
OUT=results/twin_forward/n2c_reading_r1c
RESULT=results/dataprep/reports/$ID/result.json
FLAGS="--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation --restart-taper-km 5 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --device cpu --save-members yearly"
YEARS=(2012 2013 2014 2015 2016 2017 2018)
export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

log_attempt() {
    mkdir -p "$LOG"
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
    [ -f "$DECL" ] || die "declare $ID first (prereg_n2c_reading.py --write)"
    "$HPY" -c "import sys; sys.path.insert(0, '$X'); import synth_n2c as S; \
S.registered_declaration()" > /dev/null || die "the registered declaration does not verify"
    [ -x "$PY" ] || die "$PY missing"
    grep -qx "export PYTHONPATH=$WT" "$PY" || die "$PY does not pin $WT"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$WT_HEAD" ] || die "$WT is not $WT_HEAD"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    [ -d "$HYDROMIND_GW_DATA" ] || die "HYDROMIND_GW_DATA $HYDROMIND_GW_DATA missing"
    [ ! -e "$RESULT" ] || die "$RESULT exists: $ID has run; no rerun"
}

fwd_cmd() {
    printf "%s -m hydrophysics.twin.forward --theta %s --vep-json %s %s" "$PY" \
        "$R/stage3_datum_gate_ccz/stage3_theta.json" \
        "$R/stage3_datum_gate_ccz/coupled_leveling/vep_zonal_leveling.json" "$FLAGS"
    for y in "${YEARS[@]}"; do
        printf " --recorded-pumping 'RP%s=%s/rp%s.npz@2023-01'" "$y" "$RPL" "$y"
    done
    printf " --out %s\n" "$OUT"
}

do_replays() {
    [ ! -e "$DECL" ] || die "$ID is registered; the replays are frozen in it"
    "$HPY" -I "$REPLAY" build --out "$RPL" | tail -n 1
}

do_forward() {
    if [ "${DRY:-0}" = 1 ]; then fwd_cmd; return 0; fi
    preconditions
    for s in npz members.npz summary.csv members.csv; do
        [ ! -e "$OUT.$s" ] || die "$OUT.$s exists (archive it first)"
    done
    mkdir -p "$SEAL"
    chmod 700 "$SEAL"
    n=$(next_attempt)
    sealed=$SEAL/r1c_forward.stdout.SEALED
    err=$LOG/r1c_forward.stderr.log
    t0=$(utc)
    rc=0
    eval "$(fwd_cmd)" > "$sealed" 2> "$err" || rc=$?
    chmod 400 "$sealed"
    hs=(); for s in npz members.npz; do
        if [ -e "$OUT.$s" ]; then hs+=("$(sha "$OUT.$s")"); else hs+=(missing); fi
    done
    log_attempt event=forward attempt="$n" rc="$rc" start="$t0" time="$(utc)" \
        worktree_head="$(git -C "$WT" rev-parse HEAD)" py_sh_sha256="$(sha "$PY")" \
        declaration_sha256="$(sha "$DECL")" npz_sha256="${hs[0]}" \
        members_npz_sha256="${hs[1]}" stdout_sealed_sha256="$(sha "$sealed")"
    echo "forward: exit $rc (attempt $n; stdout sealed, sha256 $(sha "$sealed"))"
    [ "$rc" = 0 ] || die "forward failed: read $err only, then archive and rerun"
}

do_extract() {
    preconditions
    "$HPY" "$SYNTH" extract --forward "$OUT.npz" --replays "$RPL/replays.json" \
        --out "$LOG/panel.npz" > "$LOG/extract.log"
    log_attempt event=extract time="$(utc)" panel_sha256="$(sha "$LOG/panel.npz")" \
        npz_sha256="$(sha "$OUT.npz")"
    echo "panel written: sha256 $(sha "$LOG/panel.npz")"
}

do_run() {
    preconditions
    rc=0
    "$HPY" "$SYNTH" run --panel "$LOG/panel.npz" || rc=$?
    rs=missing; [ -e "$RESULT" ] && rs=$(sha "$RESULT")
    log_attempt event=run time="$(utc)" rc="$rc" result_sha256="$rs"
    return "$rc"
}

do_archive() {
    reason=${1:-}
    [ -n "$reason" ] || die "archive needs a reason"
    [ ! -e "$RESULT" ] || die "$RESULT exists: $ID has run; nothing may be rerun"
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    dst=$LOG/archive/$stamp
    mkdir -p "$dst"
    moved=()
    for f in "$OUT.npz" "$OUT.members.npz" "$OUT.summary.csv" "$OUT.members.csv" \
             "$SEAL/r1c_forward.stdout.SEALED" "$LOG/r1c_forward.stderr.log" "$LOG/panel.npz"; do
        if [ -e "$f" ]; then mv "$f" "$dst/"; moved+=("$(basename "$f")"); fi
    done
    log_attempt event=archive time="$(utc)" reason="$reason" dir="$dst" \
        moved="${moved[*]:-none}"
    echo "archived ${#moved[@]} files to $dst (unopened)"
}

case "${1:-}" in
    replays) do_replays ;;
    check) preconditions; echo "preconditions ok" ;;
    forward) do_forward ;;
    extract) do_extract ;;
    run) do_run ;;
    archive) shift; do_archive "${1:-}" ;;
    *) echo "usage: $0 replays|check|forward|extract|run|archive '<reason>'" >&2; exit 2 ;;
esac
