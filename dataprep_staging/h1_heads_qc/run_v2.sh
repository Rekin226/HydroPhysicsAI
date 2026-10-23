#!/usr/bin/env bash
# WP H1a round 2 (v2) end-to-end run (CPU slots only; never the GPU). Usage:
#   dataprep_staging/h1_heads_qc/run_v2.sh WORKDIR REF_WORKDIR [--from STAGE]
# WORKDIR holds the large intermediates (per-well pickles, about 4 GB); REF_WORKDIR is the v1
# run's work directory (stage-1 identity check only). The product goes to
# results/dataprep/heads/L1_clean/v<N>/ and the evidence to
# results/dataprep/reports/H1a_evidence/v2/. The pre-registration H1a.qc_acceptance.v2
# (prereg_h1a_v2.py) must exist before s7/s5/s4 run; looks_h1a_v2.py gates each scoring step.
# Stages: 1 = stage 0/1 (full and pre-only), 2 = v2 decisions, 3 = assembly, 4 = tests,
# 5 = product.
set -euo pipefail
W=$1
REF=$2
FROM=1
if [ "${3:-}" = "--from" ]; then FROM=$4; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${PY:-python}
SLOT="$REPO/dataprep_staging/tools/cpu_slot.sh"
E="$REPO/results/dataprep/reports/H1a_evidence/v2"
V1="$REPO/results/dataprep/heads/L1_clean/v1"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$W" "$E"
cd "$REPO"
if [ "$FROM" -le 1 ]; then
  $SLOT $PY -I "$HERE/s0_meta.py" "$W" > "$W/s0.log" 2>&1
  $SLOT $PY -I "$HERE/s1_screen.py" "$W" 0 2 > "$W/s1_0.log" 2>&1 &
  $SLOT $PY -I "$HERE/s1_screen.py" "$W" 1 2 > "$W/s1_1.log" 2>&1 &
  wait
  $SLOT $PY -I "$HERE/s1_screen.py" "$W" 0 2 --pre-only > "$W/s1p_0.log" 2>&1 &
  $SLOT $PY -I "$HERE/s1_screen.py" "$W" 1 2 --pre-only > "$W/s1p_1.log" 2>&1 &
  wait
fi
if [ "$FROM" -le 2 ]; then
  $SLOT $PY -I "$HERE/s2b_decide_v2.py" "$W" stage1 > "$W/s2b.log" 2>&1 &
  $SLOT $PY -I "$HERE/s2b_decide_v2.py" "$W" stage1_pre > "$W/s2b_pre.log" 2>&1 &
  wait
fi
if [ "$FROM" -le 3 ]; then
  $SLOT $PY -I "$HERE/s3_assemble_v2.py" "$W" "$W/build" stage1 > "$W/s3.log" 2>&1 &
  $SLOT $PY -I "$HERE/s3_assemble_v2.py" "$W" "$W/build_pre" stage1_pre > "$W/s3_pre.log" 2>&1 &
  wait
fi
if [ "$FROM" -le 4 ]; then
  $PY -I "$HERE/looks_h1a_v2.py" score H1a.v2.leakprobe
  $SLOT $PY -I "$REPO/dataprep_staging/h1_heads_qc/s7_leakprobe.py" "$W/build" "$W/build_pre" \
    "$E/leakprobe_v2.json" > "$W/s7.log" 2>&1
  $PY -I "$HERE/looks_h1a_v2.py" score H1a.v2.inject_O.paired H1a.v2.inject_O.step
  $SLOT $PY -I "$HERE/s5_inject_v2.py" "$W" "$W/build" "$E/injection_O_r10-19.json" \
    --reals 10 20 > "$W/s5.log" 2>&1
  $PY -I "$HERE/looks_h1a_v2.py" conformance
  $SLOT $PY -I "$HERE/s4_validate_v2.py" "$W" "$W/build" "$REF" "$V1" \
    "$E/validation_v2.json" > "$W/s4.log" 2>&1
fi
if [ "$FROM" -le 5 ]; then
  $SLOT $PY -I "$HERE/s6_build_product_v2.py" "$W" "$W/build" "$E/validation_v2.json" \
    "$E/injection_O_r10-19.json" "$E/leakprobe_v2.json" > "$W/s6.log" 2>&1
fi
echo "H1a v2 run done"
