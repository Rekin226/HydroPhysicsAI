#!/usr/bin/env bash
# X5.n2_reading.v1 (model-only synthetic declaration, round 8 rank 4): the replays, the CPU
# replay forward of the page model, the panel and the one registered synthetic run.
#
#   bash dataprep_staging/x5_n2_reading/run_n2_reading.sh replays   BEFORE registration:
#        the seven replayed pre-origin Yunlin anomaly multipliers (pump_electricity/preorigin/v3)
#   bash dataprep_staging/x5_n2_reading/run_n2_reading.sh engineering   the engineering gate
#        (harness vs N2's registered statistic on frozen forwards; prints diffs only)
#   bash dataprep_staging/x5_n2_reading/run_n2_reading.sh check     preconditions (registered)
#   bash dataprep_staging/x5_n2_reading/run_n2_reading.sh forward   AFTER registration, CPU,
#        tmux: one forward of the page model's main set, baseline + RP2012..RP2018 (N2's command)
#   bash dataprep_staging/x5_n2_reading/run_n2_reading.sh extract   the 84 x 12 contrast panel
#   bash dataprep_staging/x5_n2_reading/run_n2_reading.sh run <seed>   the registered run (once)
#   bash dataprep_staging/x5_n2_reading/run_n2_reading.sh archive "<reason>"
#        engineering retry: move a failed forward attempt aside, unopened
#   DRY=1 bash ... forward    print the forward command line, run nothing
#
# Code: a clean detached worktree $HOME/HydroPhysicsAI-n2rd-1fff145 at N2's registered code
# commit 1fff145 (forward.py 088b8bd5, recorded_pumping.py 4a8c99e4), run as N2's
# forward_command runs it (PYTHONPATH=<checkout> python -P). The forward is N2's registered
# command (CPU, page main set, --ic-members 1, --save-members none) with its PR arm replaced by
# the seven replay arms. Its stdout goes to a sealed log that is not opened before report.json
# exists; this script prints exit codes and sha256s only.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
SYNTH=dataprep_staging/x5_n2_reading/synth_n2.py
ID=X5.n2_reading.v1
DECL=results/dataprep/preregistration/$ID.json
N2_CODE=1fff145
WT_TXT='$HOME/HydroPhysicsAI-n2rd-1fff145'
WT="$HOME/HydroPhysicsAI-n2rd-1fff145"
HPY_TXT='$HOME/miniforge3/envs/hydro/bin/python'
HPY="$HOME/miniforge3/envs/hydro/bin/python"
R=results/twin_runs
G=$R/stage3_datum_gate
LOG=$R/n2_reading
RPL=$LOG/replays
SEAL=$LOG/sealed
OUTD=results/dataprep/reports/$ID
ATT=$OUTD/attempts.jsonl
OUT=results/twin_forward/n2_reading_page
RESULT=$OUTD/report.json
FLAGS="--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation --restart-taper-km 5 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --device cpu --save-members none"
YEARS=(2012 2013 2014 2015 2016 2017 2018)
export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

log_attempt() {
    mkdir -p "$OUTD"
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
    [ -f "$DECL" ] || die "declare $ID first (prereg_n2_reading.py --write)"
    "$HPY" "$SYNTH" check > /dev/null || die "the registered declaration does not verify"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$N2_CODE" ] || die "$WT is not $N2_CODE"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    [ -d "$HYDROMIND_GW_DATA" ] || die "HYDROMIND_GW_DATA $HYDROMIND_GW_DATA missing"
    [ ! -e "$RESULT" ] || die "$RESULT exists: $ID has run; no rerun"
}

fwd_cmd() {
    printf "PYTHONPATH=%s %s -P -m hydrophysics.twin.forward --theta %s --vep-json %s %s" \
        "$WT_TXT" "$HPY_TXT" "$G/stage3_theta.json" \
        "$G/coupled_leveling/vep_zonal_leveling.json" "$FLAGS"
    for y in "${YEARS[@]}"; do
        printf " --recorded-pumping 'RP%s=%s/rp%s.csv@2023-01'" "$y" "$RPL" "$y"
    done
    printf " --out %s\n" "$OUT"
}

do_replays() {
    [ ! -e "$DECL" ] || die "$ID is registered; the replays are frozen in it"
    "$HPY" "$SYNTH" replays --out "$RPL" | tail -n 1
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
    sealed=$SEAL/page_forward.stdout.SEALED
    err=$LOG/page_forward.stderr.log
    t0=$(utc)
    rc=0
    eval "$(fwd_cmd)" > "$sealed" 2> "$err" || rc=$?
    chmod 400 "$sealed"
    hs=missing; [ -e "$OUT.npz" ] && hs=$(sha "$OUT.npz")
    log_attempt event=forward attempt="$n" rc="$rc" start="$t0" time="$(utc)" \
        worktree_head="$(git -C "$WT" rev-parse HEAD)" declaration_sha256="$(sha "$DECL")" \
        npz_sha256="$hs" stdout_sealed_sha256="$(sha "$sealed")"
    echo "forward: exit $rc (attempt $n; stdout sealed, sha256 $(sha "$sealed"))"
    [ "$rc" = 0 ] || die "forward failed: read $err only, then archive and rerun"
}

do_extract() {
    preconditions
    "$HPY" "$SYNTH" extract --forward "$OUT.npz" --replays "$RPL/replays.json" \
        --out "$LOG/panel.npz" > "$LOG/extract.log"
    echo "panel written: sha256 $(sha "$LOG/panel.npz")"
}

do_run() {
    seed=${1:-}
    [ -n "$seed" ] || die "run needs the registered seed (no default)"
    preconditions
    "$HPY" "$SYNTH" run --seed "$seed" --prereg-id "$ID" --panel "$LOG/panel.npz"
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
             "$SEAL/page_forward.stdout.SEALED" "$LOG/page_forward.stderr.log" \
             "$LOG/panel.npz"; do
        if [ -e "$f" ]; then mv "$f" "$dst/"; moved+=("$(basename "$f")"); fi
    done
    log_attempt event=archive time="$(utc)" reason="$reason" dir="$dst" \
        moved="${moved[*]:-none}"
    echo "archived ${#moved[@]} files to $dst (unopened)"
}

case "${1:-}" in
    replays) do_replays ;;
    engineering) "$HPY" "$SYNTH" engineering ;;
    check) preconditions; echo "preconditions ok" ;;
    forward) do_forward ;;
    extract) do_extract ;;
    run) shift; do_run "${1:-}" ;;
    archive) shift; do_archive "${1:-}" ;;
    *) echo "usage: $0 replays|engineering|check|forward|extract|run <seed>|archive '<reason>'" >&2
       exit 2 ;;
esac
