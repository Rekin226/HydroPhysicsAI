#!/usr/bin/env bash
# Run a command inside one of N global CPU slots (default 4) shared by every data-prep agent,
# with single-threaded BLAS and nice 10, so the GPU calibration batch keeps its CPU cores.
# Usage: dataprep_staging/tools/cpu_slot.sh [--slots K] python script.py args...
#   --slots K  hold K slots at once (for a job that genuinely needs K threads; K <= 2)
N=${DATAPREP_SLOTS:-4}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCKDIR="$REPO/results/dataprep/.cpu_slots"
mkdir -p "$LOCKDIR"
K=1
if [ "${1:-}" = "--slots" ]; then K=$2; shift 2; fi
export OMP_NUM_THREADS=$K MKL_NUM_THREADS=$K OPENBLAS_NUM_THREADS=$K NUMEXPR_NUM_THREADS=$K
held=()
while [ ${#held[@]} -lt "$K" ]; do
  for i in $(seq 1 "$N"); do
    [ ${#held[@]} -ge "$K" ] && break
    fd=$((20 + i))
    eval "exec $fd>\"$LOCKDIR/slot$i\""
    if flock -n "$fd"; then held+=("$fd"); else eval "exec $fd>&-"; fi
  done
  [ ${#held[@]} -lt "$K" ] && sleep 5
done
nice -n 10 "$@"
rc=$?
for fd in "${held[@]}"; do eval "exec $fd>&-"; done
exit $rc
