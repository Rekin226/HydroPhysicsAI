#!/usr/bin/env bash
# WP S4b repair round 2 (S4br), amendment to M1 (S4b.repair.md Part 2b): re-run LiCSBAS15 with
# the no-loop threshold LiCSBAS's batch script documents for runs with --nullify_noloops
# (-i 500; the default "global no-loop count + 1" = 1 masks every pixel with one pixel-level
# no-loop interferogram that step 13 already removed), then LiCSBAS16 on the new mask. Every
# other threshold stays at its default. Run after s4br_run_licsbas.sh, inside one CPU slot:
#   dataprep_staging/tools/cpu_slot.sh dataprep_staging/s4_insar/s4br_remask.sh NAME
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NAME=$1
W="$REPO/results/dataprep/insar_tools/s4br_run/$NAME"
ENVSH="$HERE/licsbas_env.sh"
LOG="$REPO/results/dataprep/insar_tools/s4br_run/${NAME}_licsbas.log"
cd "$W"
step() { echo "== $* ($(date -u +%FT%TZ))"; "$ENVSH" "$@"; echo "== rc=$? $1 ($(date -u +%FT%TZ))"; }
{
  echo "S4br remask $NAME start $(date -u +%FT%TZ)"
  step LiCSBAS15_mask_ts.py -t TS_GEOCml2 -i 500
  step LiCSBAS16_filt_ts.py -t TS_GEOCml2 --n_para 1 --nopngs
  echo "S4br remask $NAME done $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1
