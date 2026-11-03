#!/usr/bin/env bash
# X5.gnss_seasonal_family.v1 (round 8 rank 11): model-only gates, registration checks and the
# single scoring. CPU only; no forward runs; no GPU.
#
#   bash dataprep_staging/gnss_family/run_gsf.sh supplement    model-only: every v1 arm at the
#        TKJS(1)/NTUH(1) pile cells, 2016-06..2019-12, from the sealed SD forwards (once)
#   bash dataprep_staging/gnss_family/run_gsf.sh engineering   model-only engineering record
#        (v1 model side 65b927cb, SD v2 5cfc9ea7, supplement rebuild, dedup, cut30) (once)
#   bash dataprep_staging/gnss_family/run_gsf.sh declare <seed>   validate the gates
#        declaration (the LEAD registers it with prereg3.write_prereg before gsep)
#   bash dataprep_staging/gnss_family/run_gsf.sh gsep          G-sep once (refuses without the
#        registered declaration); exit 4 = fail = closure, no GNSS look
#   bash dataprep_staging/gnss_family/run_gsf.sh g0 [workers]  G0 once (refuses unless G-sep
#        passed); exit 4 = fail = closure, no GNSS look
#   bash dataprep_staging/gnss_family/run_gsf.sh validate <seed>  draft_prereg + readiness
#   bash dataprep_staging/gnss_family/run_gsf.sh check         registered and intact
#   bash dataprep_staging/gnss_family/run_gsf.sh score         ONCE, after registration
#
# Order: supplement -> engineering -> declare (lead registers the declaration) -> gsep -> g0
# (lead logs G0 as an integrity_check look on X5GSF:g0_synthetic with g0.json's sha256 in the
# note) -> validate (lead registers X5.gnss_seasonal_family.v1) -> check -> score.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${DATAPREP_REPO:-$(cd "$X/../.." && pwd)}
cd "$REPO"
ID=X5.gnss_seasonal_family.v1
DECL=results/dataprep/preregistration/$ID.json
OUT=results/dataprep/reports/$ID
HPY="$HOME/miniforge3/envs/hydro/bin/python"
G=dataprep_staging/gnss_family

die() { echo "REFUSED: $*" >&2; exit 1; }

case "${1:-}" in
    supplement) "$HPY" "$G/gsf_model.py" supplement ;;
    engineering) "$HPY" "$G/gsf_model.py" engineering ;;
    declare)
        [ -n "${2:-}" ] || die "declare needs the seed (no default)"
        "$HPY" -I "$G/gsf_gates.py" declare --seed "$2" ;;
    gsep)
        [ ! -e "$DECL" ] || die "$ID is registered; the gates are frozen in it"
        "$HPY" "$G/gsf_gates.py" gsep ;;
    g0)
        [ ! -e "$DECL" ] || die "$ID is registered; the gates are frozen in it"
        "$HPY" "$G/gsf_gates.py" g0 --workers "${2:-1}" ;;
    validate)
        [ -n "${2:-}" ] || die "validate needs the scoring seed (no default)"
        "$HPY" -I "$G/prereg_gsf.py" --validate --seed "$2" --quiet ;;
    check)
        [ -f "$DECL" ] || die "register $ID first"
        "$HPY" "$G/score_gsf.py" verify-registration ;;
    score)
        [ -f "$DECL" ] || die "register $ID first"
        [ ! -e "$OUT/report.json" ] || die "$OUT/report.json exists: scored once; no rerun"
        "$HPY" "$G/score_gsf.py" score ;;
    *) echo "usage: $0 supplement|engineering|declare <seed>|gsep|g0 [workers]|validate <seed>|check|score" >&2
       exit 2 ;;
esac
