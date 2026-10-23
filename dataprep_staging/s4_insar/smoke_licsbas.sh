#!/usr/bin/env bash
# WP S4a acceptance S4a-3: run LiCSBAS steps 02 -> 11 -> 12 -> 13 -> 14 on a small single-grid
# set of fetched primary-frame pairs (pre-origin dates only), in a work directory of symlinks
# into the read-only L0 subsets. Outputs stay under results/dataprep/insar_tools/s4a_run/smoke/.
# Run inside one CPU slot:
#   dataprep_staging/tools/cpu_slot.sh dataprep_staging/s4_insar/smoke_licsbas.sh [N_PAIRS]
#     [GEOREF_VARIANT NAME]   (optional: only pairs of one unw georeference variant, written to
#                              s4a_run/<NAME>/ with logs <NAME>_licsbas.log, <NAME>_workdir.json)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RUN="$REPO/results/dataprep/insar_tools/s4a_run"
N=${1:-40}
VARIANT=${2:-}
NAME=${3:-smoke}
W="$RUN/$NAME"
FRAME=069A_06694_131402
ENVSH="$HERE/licsbas_env.sh"
"$HOME/miniforge3/envs/licsbas/bin/python" -I "$HERE/make_licsbas_workdir.py" --frame "$FRAME" \
  --out "$W" --start 20200101 --end 20221231 --max-pairs "$N" --georef-variant "$VARIANT" \
  | tee "$RUN/${NAME}_workdir.json"
cd "$W"
step() { echo "== $* ($(date -u +%FT%TZ))"; "$ENVSH" "$@"; echo "== rc=$? $1"; }
{
  step LiCSBAS02_ml_prep.py -i GEOC -o GEOCml2 -n 2 --n_para 1
  step LiCSBAS11_check_unw.py -d GEOCml2 -t TS_GEOCml2
  step LiCSBAS12_loop_closure.py -d GEOCml2 -t TS_GEOCml2 --n_para 1 --nopngs
  step LiCSBAS13_sb_inv.py -d GEOCml2 -t TS_GEOCml2 --n_para 1 --mem_size 2000 --nopngs
  step LiCSBAS14_vel_std.py -t TS_GEOCml2 --mem_size 2000
} > "$RUN/${NAME}_licsbas.log" 2>&1
ls -la TS_GEOCml2/cum.h5 >> "$RUN/${NAME}_licsbas.log"
echo "smoke done"
