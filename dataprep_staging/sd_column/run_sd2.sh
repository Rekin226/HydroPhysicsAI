#!/usr/bin/env bash
# SD.cut_response_spread.v2 (model-only declaration, round 7 rank 9): the one scoring of
# v1's eight sealed forwards. No forward runs under v2: run_sd.sh (v1, unchanged) ran them
# under SD.cut_response_spread.v1, whose scoring stopped at step 2 (rheology labels, scorer
# exit 2) before any cut30 quantity was formed. v2 binds each npz by the sha256 v1's runner
# recorded in attempts.jsonl and maps rheology labels by forward.py's own rule (sd2lib).
# X5.gnss_seasonal_column.v1 (run_gsc.sh) runs only after v2 is scored.
#
#   bash dataprep_staging/sd_column/run_sd2.sh check    preconditions only (CPU)
#   bash dataprep_staging/sd_column/run_sd2.sh score    score once (CPU)
#   DRY=1 bash dataprep_staging/sd_column/run_sd2.sh forward all
#        print the reused forwards' command lines (v1's), run nothing
#
# Sealed outputs: v1's stdout logs (results/twin_runs/sd_column/sealed/, mode 400), the sd_*
# CSVs and every cut30 slice stay unopened until v2's report.json exists. A key whose v1
# forward is not v2's command makes the scorer refuse (score_sd2.bound_npz); rerunning that
# key needs a new declaration. Score attempts go to attempts_v2.jsonl; v1's attempts.jsonl
# is read only.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${DATAPREP_REPO:-$(cd "$X/../.." && pwd)}
cd "$REPO"
SELF=dataprep_staging/sd_column/run_sd2.sh
SCORER=dataprep_staging/sd_column/score_sd2.py
ID=SD.cut_response_spread.v2
V1=SD.cut_response_spread.v1
DECL=results/dataprep/preregistration/$ID.json
CODE=4c2723e
WT="$HOME/HydroPhysicsAI-sdrun-4c2723e"
FORWARD_PY_SHA256=f073eccc2126e908be968c10664d962591f048e3ebf8e5669ff782d6e6b628e6
HPY="$HOME/miniforge3/envs/hydro/bin/python"
LOG=results/twin_runs/sd_column
ATT_V1=$LOG/attempts.jsonl
ATT=$LOG/attempts_v2.jsonl
REPORT=results/dataprep/reports/$ID/report.json
REPORT_V1=results/dataprep/reports/$V1/report.json
export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"

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
    rec[k] = int(v) if v.lstrip("-").isdigit() and k in ("rc", "attempt") else v
with open(sys.argv[1], "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(json.dumps(rec, sort_keys=True) + "\n")' "$ATT" "$@"
}

count_scores() {
    if [ ! -f "$ATT" ]; then echo 0; return; fi
    "$HPY" -I -c '
import json, sys
print(sum(json.loads(s).get("event") == "score" for s in open(sys.argv[1]) if s.strip()))' \
        "$ATT"
}

preconditions() {
    [ -f "$DECL" ] || die "declare $ID first (prereg_sd2.py --write)"
    for f in "$SCORER" "$SELF" dataprep_staging/sd_column/sd2lib.py \
             dataprep_staging/sd_column/score_sd.py dataprep_staging/sd_column/sdlib.py \
             dataprep_staging/sd_column/run_sd.sh; do
        want=$("$HPY" -I -c 'import json, sys
print(json.load(open(sys.argv[1])).get("frozen_code", {}).get(sys.argv[2], "none"))' "$DECL" "$f")
        [ "$want" = "$(sha "$f")" ] || die "$f sha256 differs from the registered declaration"
    done
    "$HPY" "$SCORER" verify-registration > /dev/null \
        || die "the registered declaration does not verify (INDEX/ledger/frozen code/inputs/v1)"
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; no rerun"
    [ ! -e "$REPORT_V1" ] || die "$REPORT_V1 exists: v1 was scored; v2 does not apply"
    [ -f "$ATT_V1" ] || die "$ATT_V1 missing (v1's forward records bind the reused npz)"
    [ -d "$WT" ] || die "worktree $WT missing (git worktree add --detach $WT $CODE)"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$CODE" ] || die "$WT is not at $CODE"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    [ "$(sha "$WT/hydrophysics/twin/forward.py")" = "$FORWARD_PY_SHA256" ] \
        || die "$WT forward.py is not the pinned labelling-rule source"
    [ -d "$HYDROMIND_GW_DATA" ] || die "HYDROMIND_GW_DATA $HYDROMIND_GW_DATA missing"
}

fwd_cmd() {   # the reused forward's command line (sdlib.forward_command, as v1 declared)
    "$HPY" -I -c 'import sys; sys.path.insert(0, sys.argv[1])
import sdlib
print(sdlib.forward_command(sys.argv[2]))' "$X" "$1"
}

do_score() {
    preconditions
    n=$(( $(count_scores) + 1 ))
    rc=0
    "$HPY" "$SCORER" score --attempts "$ATT_V1" --out "$REPORT" || rc=$?
    rs=missing; [ -e "$REPORT" ] && rs=$(sha "$REPORT")
    log_attempt event=score id="$ID" attempt="$n" rc="$rc" time="$(utc)" code="$CODE" \
        report_sha256="$rs" declaration_sha256="$(sha "$DECL")"
    case $rc in
        0) echo "scored once: $REPORT (sha256 $rs); sealed logs and CSVs may now be read" ;;
        1|2|3) echo "engineering failure (scorer exit $rc): no cut30 quantity was formed;" \
                    "a fix to the frozen code or the recipe needs a new declaration" ;;
        *) echo "scorer exit $rc" ;;
    esac
    return "$rc"
}

case "${1:-}" in
    check) preconditions; echo "preconditions ok ($ID on v1's forwards, code $CODE)" ;;
    score) do_score ;;
    forward)
        [ "${DRY:-0}" = 1 ] || die "no forward runs under $ID (v1's eight are reused)"
        [ "${2:-}" = all ] || die "DRY=1 forward all only"
        for k in page q0 q0cs cwa q0nh ccz q0c x5r1cs; do fwd_cmd "$k"; done ;;
    *) echo "usage: $0 check|score|DRY=1 forward all" >&2
       exit 2 ;;
esac
