#!/usr/bin/env bash
# WP S4b: LiCSBAS time series for one run (plan WP S4 steps 2-4; D11 = no atmospheric
# correction, so steps 03op/04op/05op are skipped and step 16 is not part of the product).
# Work directory of symlinks into the read-only L0 subsets (make_licsbas_workdir.py, S4a,
# unchanged), then LiCSBAS 02 (2x2 multilook) -> 11 -> 12 -> 13 -> 14 -> 15 with LiCSBAS
# defaults at the pinned commit, single process. Outputs stay under
# results/dataprep/insar_tools/s4b_run/<NAME>/ (gitignored). Run inside one CPU slot:
#   dataprep_staging/tools/cpu_slot.sh dataprep_staging/s4_insar/s4b_run_licsbas.sh \
#       NAME FRAME START END
# NAME   run name (P = pre-origin gate run, F = full product run, D = descending 2025 run)
# FRAME  LiCSAR frame id; START/END yyyymmdd limits on both epochs of a pair.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NAME=$1
FRAME=$2
START=$3
END=$4
RUN="$REPO/results/dataprep/insar_tools/s4b_run"
W="$RUN/$NAME"
ENVSH="$HERE/licsbas_env.sh"
mkdir -p "$RUN"
if [ ! -d "$W/GEOC" ]; then
  "$HOME/miniforge3/envs/licsbas/bin/python" -I "$HERE/make_licsbas_workdir.py" \
    --frame "$FRAME" --out "$W" --start "$START" --end "$END" > "$RUN/${NAME}_workdir.json"
fi
cd "$W"
LOG="$RUN/${NAME}_licsbas.log"
step() { echo "== $* ($(date -u +%FT%TZ))"; "$ENVSH" "$@"; echo "== rc=$? $1 ($(date -u +%FT%TZ))"; }
{
  echo "S4b run $NAME frame $FRAME pairs $START..$END host $(hostname -s) start $(date -u +%FT%TZ)"
  step LiCSBAS02_ml_prep.py -i GEOC -o GEOCml2 -n 2 --n_para 1
  step LiCSBAS11_check_unw.py -d GEOCml2 -t TS_GEOCml2
  step LiCSBAS12_loop_closure.py -d GEOCml2 -t TS_GEOCml2 --n_para 1 --nopngs
  step LiCSBAS13_sb_inv.py -d GEOCml2 -t TS_GEOCml2 --n_para 1 --mem_size 4000 --nopngs
  step LiCSBAS14_vel_std.py -t TS_GEOCml2 --mem_size 4000
  step LiCSBAS15_mask_ts.py -t TS_GEOCml2
  echo "S4b run $NAME done $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1
