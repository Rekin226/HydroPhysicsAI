#!/usr/bin/env bash
# X5.gnss_seasonal_column.v1 (registered design-time test, round 7 rank 10): model side,
# G0-OC, registration checks and the single scoring. CPU only; no forward runs here (the
# model side reads the SD forwards' free hindcast months: run under SD.cut_response_spread.v1,
# scored by SD.cut_response_spread.v2, which reuses them after v1's engineering failure).
#
#   bash dataprep_staging/sd_column/run_gsc.sh model-side   BEFORE registration, after SD
#        v2 is scored: freeze the model side (model_side.json + model_series.npz), no GNSS read
#   bash dataprep_staging/sd_column/run_gsc.sh check-model  G-eng: rebuild, byte for byte
#   bash dataprep_staging/sd_column/run_gsc.sh g0oc         BEFORE registration: G0-OC on
#        the frozen model side (synthetic worlds; pass judged on the SCORED rule, both
#        observation scales, T1n characterised too); exit 4 = fail = the test is not run
#   bash dataprep_staging/sd_column/run_gsc.sh check        registered, intact, G0-OC passed
#   bash dataprep_staging/sd_column/run_gsc.sh score        ONCE: assert_can_score logs the
#        looks, then the GNSS daily values are opened (design_time_evidence)
#
# Order: SD v2 scored -> model-side -> check-model -> g0oc (lead logs it as an integrity_check
# look on X5GSC:g0oc_synthetic with g0oc.json's sha256 in the note; prereg_gsc checks it)
# -> prereg_gsc.py --validate / --write -> check -> score. A failure before score is an
# engineering retry (free, logged); score runs once and its report is final.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${DATAPREP_REPO:-$(cd "$X/../.." && pwd)}
cd "$REPO"
ID=X5.gnss_seasonal_column.v1
DECL=results/dataprep/preregistration/$ID.json
SD_REPORT=results/dataprep/reports/SD.cut_response_spread.v2/report.json
OUT=results/dataprep/reports/$ID
MODEL=$OUT/model_side
REPORT=$OUT/report.json
HPY="$HOME/miniforge3/envs/hydro/bin/python"
MS=dataprep_staging/sd_column/gsc_model_side.py
SCORER=dataprep_staging/sd_column/score_gsc.py
LOG=$OUT/attempts.jsonl

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
note() { mkdir -p "$OUT"; printf '{"event": "%s", "rc": %s, "time": "%s"}\n' "$1" "$2" "$(utc)" >> "$LOG"; }

case "${1:-}" in
    model-side)
        [ -f "$SD_REPORT" ] || die "score SD.cut_response_spread.v2 first (its forwards are sealed until then)"
        [ ! -e "$DECL" ] || die "$ID is registered; the model side is frozen in it"
        [ ! -e "$MODEL/model_side.json" ] || die "$MODEL exists (archive it to rebuild)"
        rc=0; "$HPY" "$MS" build --out-dir "$MODEL" || rc=$?; note model_side "$rc"; exit "$rc" ;;
    check-model)
        rc=0; "$HPY" "$MS" check --out-dir "$MODEL" || rc=$?; note check_model "$rc"; exit "$rc" ;;
    g0oc)
        [ ! -e "$DECL" ] || die "$ID is registered; G0-OC is frozen in it"
        [ ! -e "$OUT/g0oc.json" ] || die "$OUT/g0oc.json exists (one G0-OC per model side)"
        rc=0; "$HPY" "$MS" g0oc --out-dir "$MODEL" || rc=$?; note g0oc "$rc"; exit "$rc" ;;
    check)
        [ -f "$DECL" ] || die "register $ID first (prereg_gsc.py --write)"
        "$HPY" "$SCORER" verify-registration ;;
    score)
        [ -f "$DECL" ] || die "register $ID first"
        [ ! -e "$REPORT" ] || die "$REPORT exists: scored once; no rerun"
        "$HPY" "$SCORER" verify-registration > /dev/null || die "registration does not verify"
        rc=0; "$HPY" "$SCORER" score --out "$REPORT" --model-dir "$MODEL" || rc=$?
        note score "$rc"
        [ "$rc" = 0 ] && echo "report sha256 $(sha "$REPORT")"
        exit "$rc" ;;
    *) echo "usage: $0 model-side|check-model|g0oc|check|score" >&2; exit 2 ;;
esac
