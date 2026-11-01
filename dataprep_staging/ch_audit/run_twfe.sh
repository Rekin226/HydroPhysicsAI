#!/usr/bin/env bash
# X5.CH_twfe_mirror.v1 (round 7 rank 16): within-township TWFE mirror on the 74 spent Changhua
# screens. CPU only (no forward, no GPU). Run g0 and score inside tmux.
#
#   bash dataprep_staging/ch_audit/run_twfe.sh prep       BEFORE registration: design.npz
#        (census + station metadata) and the three model panels (stored forwards, baseline
#        hindcast at the screens); prints shas and counts only
#   bash dataprep_staging/ch_audit/run_twfe.sh validate   prereg_twfe.py --validate (dry run)
#   bash dataprep_staging/ch_audit/run_twfe.sh check      preconditions (registered)
#   bash dataprep_staging/ch_audit/run_twfe.sh g0         the synthetic gate, once
#   bash dataprep_staging/ch_audit/run_twfe.sh score      the reading, once (heads read only
#        if G0 passed; the look is logged first)
#   An engineering failure (a crash) writes no g0.json or report.json, so the step is simply
#   rerun; every attempt is logged in attempts.jsonl. A written g0.json or report.json is final:
#   nothing archives or reruns it.
#
# Registration (the lead): prereg_twfe.py --write --expect-self-sha <sha printed by validate>.
#
# Every path is relative to the repository root and needs results/dataprep, results/twin_forward
# and results/twin_runs: run from the main checkout, or from a frozen worktree whose results/
# resolves to the shared results (a build worktree with its own tracked results/ does not).
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
ID=X5.CH_twfe_mirror.v1
DRIVER=dataprep_staging/ch_audit/twfe_mirror.py
PREREG=dataprep_staging/ch_audit/prereg_twfe.py
DECL=results/dataprep/preregistration/$ID.json
LOG=results/twin_runs/ch_twfe
ATT=$LOG/attempts.jsonl
G0=$LOG/g0.json
RESULT=results/dataprep/reports/$ID/report.json
HPY="$HOME/miniforge3/envs/hydro/bin/python"
SLOT=dataprep_staging/tools/cpu_slot.sh
export PYTHONPATH="$REPO"

die() { echo "REFUSED: $*" >&2; exit 1; }
need_results() {
    [ -d results/dataprep/pump_electricity ] && [ -d results/twin_forward ] || \
        die "results/dataprep or results/twin_forward does not resolve here; run from the main" \
            "checkout or a frozen worktree whose results/ is the shared results"
}
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

preconditions() {
    [ -f "$DECL" ] || die "declare $ID first (prereg_twfe.py --write)"
    "$HPY" -c "import sys; sys.path.insert(0, '$X'); import twfe_mirror as M; M.registered()" \
        > /dev/null || die "the registered declaration does not verify"
    [ ! -e "$RESULT" ] || die "$RESULT exists: $ID has been scored; no rerun"
}

do_prep() {
    need_results
    [ ! -e "$DECL" ] || die "$ID is registered; its inputs are frozen"
    rc=0
    "$SLOT" "$HPY" "$DRIVER" design || rc=$?
    log_attempt event=design rc="$rc" time="$(utc)"
    [ "$rc" = 0 ] || die "design failed"
    for m in R1c ccz_m1618 q0c; do
        rc=0
        "$SLOT" "$HPY" "$DRIVER" model-panel --name "$m" || rc=$?
        log_attempt event=model_panel model="$m" rc="$rc" time="$(utc)"
        [ "$rc" = 0 ] || die "model panel $m failed"
    done
}

do_g0() {
    preconditions
    [ ! -e "$G0" ] || die "$G0 exists: G0 has run; its result is final"
    rc=0
    "$SLOT" "$HPY" "$DRIVER" g0 || rc=$?
    gs=missing; [ -e "$G0" ] && gs=$(sha "$G0")
    log_attempt event=g0 rc="$rc" time="$(utc)" g0_sha256="$gs" declaration_sha256="$(sha "$DECL")"
    return "$rc"
}

do_score() {
    preconditions
    [ -e "$G0" ] || die "run g0 first"
    rc=0
    "$SLOT" "$HPY" "$DRIVER" score || rc=$?
    rs=missing; [ -e "$RESULT" ] && rs=$(sha "$RESULT")
    log_attempt event=score rc="$rc" time="$(utc)" report_sha256="$rs"
    return "$rc"
}

case "${1:-}" in
    prep) do_prep ;;
    validate) need_results; "$HPY" "$PREREG" --validate --quiet ;;
    check) preconditions; echo "preconditions ok" ;;
    g0) do_g0 ;;
    score) do_score ;;
    *) echo "usage: $0 prep|validate|check|g0|score" >&2; exit 2 ;;
esac
