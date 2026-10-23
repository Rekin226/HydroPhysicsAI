#!/usr/bin/env bash
# WP S4b repair round 2 (S4br): repaired LiCSBAS time series for one run.
#   M1  LiCSBAS12 --multi_prime --nullify -l pi ; LiCSBAS13 --nullify_noloops
#   M2  LiCSBAS16 spatio-temporal filter (defaults; no deramp, no height term, no NaN fill)
# D11 = no atmospheric correction product (no ERA5/GACOS). Work directory of symlinks into the
# read-only L0 subsets; LiCSBAS writes only inside the work directory (step 12's
# nullification edits GEOCml2, the multilooked copies, never L0). Outputs under
# results/dataprep/insar_tools/s4br_run/<NAME>/ (gitignored). Run inside one CPU slot:
#   dataprep_staging/tools/cpu_slot.sh dataprep_staging/s4_insar/s4br_run_licsbas.sh \
#       NAME FRAME START END
# NAME P2 (pre-origin gate run), F2 (full, primary frame), S2 (full, second frame).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NAME=$1
FRAME=$2
START=$3
END=$4
RUN="$REPO/results/dataprep/insar_tools/s4br_run"
W="$RUN/$NAME"
ENVSH="$HERE/licsbas_env.sh"
PI=3.141592653589793
mkdir -p "$RUN"
if [ ! -d "$W/GEOC" ]; then
  if [ "$FRAME" = "069A_06694_131402" ]; then
    "$HOME/miniforge3/envs/licsbas/bin/python" -I "$HERE/make_licsbas_workdir.py" \
      --frame "$FRAME" --out "$W" --start "$START" --end "$END" > "$RUN/${NAME}_workdir.json"
  else
    "$HOME/miniforge3/envs/licsbas/bin/python" -I "$HERE/s4br_workdir.py" \
      --frame "$FRAME" --out "$W" --start "$START" --end "$END" > "$RUN/${NAME}_workdir.json"
  fi
fi
cd "$W"
LOG="$RUN/${NAME}_licsbas.log"
step() { echo "== $* ($(date -u +%FT%TZ))"; "$ENVSH" "$@"; echo "== rc=$? $1 ($(date -u +%FT%TZ))"; }
{
  echo "S4br run $NAME frame $FRAME pairs $START..$END host $(hostname -s) start $(date -u +%FT%TZ)"
  step LiCSBAS02_ml_prep.py -i GEOC -o GEOCml2 -n 2 --n_para 1
  step LiCSBAS11_check_unw.py -d GEOCml2 -t TS_GEOCml2
  step LiCSBAS12_loop_closure.py -d GEOCml2 -t TS_GEOCml2 -l $PI --multi_prime --nullify \
       --n_para 1 --nopngs
  # LiCSBAS's batch script reruns step 13 once when it fails after nullification ("NaNs in
  # the reference area"); the same single retry here
  step LiCSBAS13_sb_inv.py -d GEOCml2 -t TS_GEOCml2 --nullify_noloops --n_para 1 \
       --mem_size 4000 --nopngs || \
  step LiCSBAS13_sb_inv.py -d GEOCml2 -t TS_GEOCml2 --nullify_noloops --n_para 1 \
       --mem_size 4000 --nopngs
  step LiCSBAS14_vel_std.py -t TS_GEOCml2 --mem_size 4000
  step LiCSBAS15_mask_ts.py -t TS_GEOCml2
  step LiCSBAS16_filt_ts.py -t TS_GEOCml2 --n_para 1 --nopngs
  echo "S4br run $NAME done $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1
