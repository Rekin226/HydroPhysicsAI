#!/usr/bin/env bash
# WP H1a repair round (2026-10-13): the registered v2 run (run_v2.sh, stages 1-4 with the same
# commands and the same registered scripts), the after-the-fact verification, the packaging of
# the registered build (provenance only), the post-look review (v3) with the leakage probe and
# conformance checks recomputed on it, and the packaging of the reviewed build. CPU slots
# only; never the GPU. Usage:
#   dataprep_staging/h1_heads_qc/run_v2_repair.sh WORKDIR REF_WORKDIR [--from S] [--to S]
# Stages: 1 = stage 0/1 (full and pre-only), 2 = v2 decisions, 3 = assembly, 4 = registered
# tests (gated by looks_h1a_v2.py) and conformance checks, 5 = after-the-fact verification of
# the v2 build (v2_verify.py, logged by looks_h1a_repair.py diagnose), 6 = product of the
# registered build (s6_build_product_v2b.py --role registered), 7 = v3 review
# (s2c_review_v3.py, assembly, leakage probe, validation and verification on the v3 build in
# WORKDIR_v3, logged by looks_h1a_repair.py review), 8 = product of the reviewed build
# (--role reviewed).
set -euo pipefail
W=$1
REF=$2
shift 2
FROM=1
TO=8
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM=$2; shift 2 ;;
    --to) TO=$2; shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${PY:-python}
SLOT="$REPO/dataprep_staging/tools/cpu_slot.sh"
E="$REPO/results/dataprep/reports/H1a_evidence/v2"
V1="$REPO/results/dataprep/heads/L1_clean/v1"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$W" "$E"
cd "$REPO"
run() { [ "$FROM" -le "$1" ] && [ "$TO" -ge "$1" ]; }
if run 1; then
  $SLOT $PY -I "$HERE/s0_meta.py" "$W" > "$W/s0.log" 2>&1
  $SLOT $PY -I "$HERE/s1_screen.py" "$W" 0 2 > "$W/s1_0.log" 2>&1 &
  $SLOT $PY -I "$HERE/s1_screen.py" "$W" 1 2 > "$W/s1_1.log" 2>&1 &
  wait
  $SLOT $PY -I "$HERE/s1_screen.py" "$W" 0 2 --pre-only > "$W/s1p_0.log" 2>&1 &
  $SLOT $PY -I "$HERE/s1_screen.py" "$W" 1 2 --pre-only > "$W/s1p_1.log" 2>&1 &
  wait
fi
if run 2; then
  $SLOT $PY -I "$HERE/s2b_decide_v2.py" "$W" stage1 > "$W/s2b.log" 2>&1 &
  $SLOT $PY -I "$HERE/s2b_decide_v2.py" "$W" stage1_pre > "$W/s2b_pre.log" 2>&1 &
  wait
fi
if run 3; then
  $SLOT $PY -I "$HERE/s3_assemble_v2.py" "$W" "$W/build" stage1 > "$W/s3.log" 2>&1 &
  $SLOT $PY -I "$HERE/s3_assemble_v2.py" "$W" "$W/build_pre" stage1_pre > "$W/s3_pre.log" 2>&1 &
  wait
fi
if run 4; then
  $PY -I "$HERE/looks_h1a_v2.py" score H1a.v2.leakprobe > "$W/look_leak.log" 2>&1
  $SLOT $PY -I "$REPO/dataprep_staging/h1_heads_qc/s7_leakprobe.py" "$W/build" "$W/build_pre" \
    "$E/leakprobe_v2.json" > "$W/s7.log" 2>&1
  $PY -I "$HERE/looks_h1a_v2.py" score H1a.v2.inject_O.paired H1a.v2.inject_O.step \
    > "$W/look_inject.log" 2>&1
  $SLOT $PY -I "$HERE/s5_inject_v2.py" "$W" "$W/build" "$E/injection_O_r10-19.json" \
    --reals 10 20 > "$W/s5.log" 2>&1
  $PY -I "$HERE/looks_h1a_v2.py" conformance > "$W/look_conf.log" 2>&1
  $SLOT $PY -I "$HERE/s4_validate_v2.py" "$W" "$W/build" "$REF" "$V1" \
    "$E/validation_v2.json" > "$W/s4.log" 2>&1
fi
if run 5; then
  $PY -I "$HERE/looks_h1a_repair.py" diagnose > "$W/look_diag.log" 2>&1
  $SLOT $PY -I "$HERE/v2_verify.py" "$W" "$W/build" "$V1" "$E/injection_O_r10-19.json" \
    "$E/verify_v2.json" > "$W/verify.log" 2>&1
fi
if run 6; then
  $SLOT $PY -I "$HERE/s6_build_product_v2b.py" "$W" "$W/build" "$E/validation_v2.json" \
    "$E/injection_O_r10-19.json" "$E/leakprobe_v2.json" "$E/verify_v2.json" \
    --role registered > "$W/s6.log" 2>&1
fi
W3="${W}_v3"
if run 7; then
  $PY -I "$HERE/s2c_review_v3.py" "$W" "$W3" stage1 > "$W3.s2c.log" 2>&1
  $PY -I "$HERE/s2c_review_v3.py" "$W" "$W3" stage1_pre >> "$W3.s2c.log" 2>&1
  $SLOT $PY -I "$HERE/s3_assemble_v2.py" "$W3" "$W3/build" stage1 > "$W3/s3.log" 2>&1 &
  $SLOT $PY -I "$HERE/s3_assemble_v2.py" "$W3" "$W3/build_pre" stage1_pre > "$W3/s3_pre.log" 2>&1 &
  wait
  $PY -I "$HERE/looks_h1a_repair.py" review > "$W3/look_review.log" 2>&1
  $SLOT $PY -I "$HERE/s7_leakprobe.py" "$W3/build" "$W3/build_pre" "$E/leakprobe_v3.json" \
    > "$W3/s7.log" 2>&1
  $SLOT $PY -I "$HERE/s4_validate_v2.py" "$W3" "$W3/build" "$REF" "$V1" \
    "$E/validation_v3.json" > "$W3/s4.log" 2>&1
  $SLOT $PY -I "$HERE/v2_verify.py" "$W3" "$W3/build" "$V1" "$E/injection_O_r10-19.json" \
    "$E/verify_v3.json" "$E/validation_v3.json" > "$W3/verify.log" 2>&1
fi
if run 8; then
  $SLOT $PY -I "$HERE/s6_build_product_v2b.py" "$W3" "$W3/build" "$E/validation_v3.json" \
    "$E/injection_O_r10-19.json" "$E/leakprobe_v3.json" "$E/verify_v3.json" \
    --role reviewed --registered "$E/validation_v2.json" "$E/leakprobe_v2.json" \
    > "$W3/s6.log" 2>&1
fi
echo "H1a v2 repair run done (stages $FROM-$TO)"
