#!/usr/bin/env bash
# WP H1a end-to-end run (CPU slots only; never the GPU). Usage:
#   dataprep_staging/h1_heads_qc/run_all.sh WORKDIR
# WORKDIR holds the large intermediates (per-well pickles, about 2 GB); the product goes to
# results/dataprep/heads/L1_clean/v<N>/ and the evidence to results/dataprep/reports/H1a_evidence/.
# The pre-registration (prereg_h1a.py) must exist before s5/s4/s7 run.
set -euo pipefail
W=$1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${PY:-python}
SLOT="$REPO/dataprep_staging/tools/cpu_slot.sh"
E="$REPO/results/dataprep/reports/H1a_evidence"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$W" "$E"
cd "$REPO"
$SLOT $PY -I "$HERE/p0_profile.py" "$E/p0_profile.json"
$SLOT $PY -I "$HERE/s0_meta.py" "$W"
# stage 1, two shards in parallel (two CPU slots)
$SLOT $PY -I "$HERE/s1_screen.py" "$W" 0 2 > "$W/s1_0.log" 2>&1 &
$SLOT $PY -I "$HERE/s1_screen.py" "$W" 1 2 > "$W/s1_1.log" 2>&1 &
wait
$SLOT $PY -I "$HERE/s2_backing.py" "$W" stage1
$SLOT $PY -I "$HERE/s3_assemble.py" "$W" "$W/build" stage1
# leakage probe chain (no post-origin file opened)
$SLOT $PY -I "$HERE/s1_screen.py" "$W" 0 2 --pre-only > "$W/s1p_0.log" 2>&1 &
$SLOT $PY -I "$HERE/s1_screen.py" "$W" 1 2 --pre-only > "$W/s1p_1.log" 2>&1 &
wait
$SLOT $PY -I "$HERE/s2_backing.py" "$W" stage1_pre
$SLOT $PY -I "$HERE/s3_assemble.py" "$W" "$W/build_pre" stage1_pre
$SLOT $PY -I "$HERE/s7_leakprobe.py" "$W/build" "$W/build_pre" "$E/leakprobe.json"
# acceptance tests
$SLOT $PY -I "$HERE/s5_inject.py" "$W" "$W/build" "$E/injection_O.json"
$SLOT $PY -I "$HERE/s4_validate.py" "$W" "$W/build" "$E/validation_full.json"
$SLOT $PY -I "$HERE/s6_build_product.py" "$W" "$W/build" "$E/validation_full.json" \
  "$E/injection_O.json" "$E/leakprobe.json"
