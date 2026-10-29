#!/usr/bin/env bash
# I1.X5SC.gate0 (coast E2, synthetic identifiability gate 0; model only, spends no look).
# 12 calibrate_coupled column fits = truths {A, N, V} x slow forms {aq, vt} x rolling origins
# {2016-04, 2019-04}, each on its truth's synthetic leveling panel, with the surveys after the
# origin withheld, on calibrate_coupled's own nudged hindcast (gain 1, every month,
# innovation, apex calibrated). Three workers in parallel under MPS. No forward run: the
# evaluator drives the fitted columns with the truth heads on CPU.
#
# Prerequisites (see the declaration's run instructions): the frozen worktree and
# results/twin_runs/par32/py.sh, dump_fit_heads.py, make_synthetic.py, and the registered
# declaration (prereg_x5sc_gate0.py --register). Start it inside tmux, with the MPS daemon up:
#
#   tmux new -s x5sc
#   export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log
#   nvidia-smi                                  # GPU visible, no other compute process
#   nvidia-cuda-mps-control -d                  # only if not already running
#   bash dataprep_staging/x5_slow/run_e2.sh
#
#   DRY=1 bash run_e2.sh     print the exact command lines, run nothing, check nothing
#   NO_MPS=1 bash run_e2.sh  one worker, no MPS (about 3x the wall time)
#
# Guards: the runner holds results/twin_runs/par32/runs/.runner.lock (one batch at a time;
# the evaluator refuses while it is held); jobmeta.py preflight refuses unless the code is
# the frozen worktree and every hash matches the registered declaration; jobmeta.py start
# refuses an arm whose --data or HYDROMIND_GW_DATA is not its synthetic dir, or whose panel
# is not the generator's, and never reruns a completed arm (attempts.jsonl, append-only).
# Code freeze: never edit the worktree while this runs (job.json records the code tree
# before and after each arm). The logs have calibrate_coupled's real-ring numbers redacted.
#
# After the batch (CPU, about a minute, scored once):
#   results/twin_runs/par32/py.sh dataprep_staging/x5_slow/evaluate_e2.py
set -u
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO" || exit 1
unset HYDROMIND_GW_DATA                 # set per arm to its synthetic dir, never globally
PAR=$REPO/results/twin_runs/par32
PY=$PAR/py.sh
SYN=$PAR/synth
RUNS=$PAR/runs
LOG=$RUNS/run.log
WORKERS=3
[ -n "${NO_MPS:-}" ] && WORKERS=1

if [ -n "${DRY:-}" ]; then
    PYD=$PY
    [ -x "$PYD" ] || PYD="$HOME/miniforge3/envs/hydro/bin/python"
    mapfile -t JOBS < <("$PYD" "$X/e2flags.py" jobs "$SYN")
    echo "# python launcher: $PY"
    echo "# preflight: $PY $X/jobmeta.py preflight $RUNS $SYN"
    for j in "${JOBS[@]}"; do
        IFS='|' read -r a t f o d <<< "$j"
        mapfile -t ARGV < <("$PYD" "$X/e2flags.py" argv "$a" "$SYN" "$RUNS")
        echo "HYDROMIND_GW_DATA=$d $PY ${ARGV[*]}"
    done
    exit 0
fi

die() { echo "run_e2.sh: $*" >&2; exit 1; }
[ -n "${TMUX:-}" ] || die "run this inside tmux (CLAUDE.md: long jobs never in a bare terminal)"
[ -x "$PY" ] || die "$PY missing or not executable (frozen-code launcher)"
command -v flock > /dev/null || die "flock not found"
if [ -z "${NO_MPS:-}" ]; then
    export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-$HOME/.mps/pipe}
    export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-$HOME/.mps/log}
    pgrep -u "$(id -u)" -f nvidia-cuda-mps-control > /dev/null \
        || die "MPS daemon not running: start it with 'nvidia-cuda-mps-control -d' (only when no other GPU process runs), or set NO_MPS=1"
fi
nvidia-smi > /dev/null 2>&1 || die "nvidia-smi fails: check the GPU before a batch"
mkdir -p "$RUNS"
exec 9> "$RUNS/.runner.lock"
flock -n 9 || die "another run_e2.sh holds $RUNS/.runner.lock"
note() { echo "[$(date '+%F %T')] [x5sc-gate0] $*" >> "$LOG"; }

"$PY" "$X/jobmeta.py" preflight "$RUNS" "$SYN" >> "$LOG" 2>&1 \
    || { note "preflight refused (see above)"; die "preflight refused; see $LOG"; }
mapfile -t JOBS < <("$PY" "$X/e2flags.py" jobs "$SYN")
[ "${#JOBS[@]}" -eq 12 ] || die "expected 12 arms, got ${#JOBS[@]}"

arm() {  # arm "NAME|TRUTH|FORM|ORIGIN|DATA_DIR"
    local a t f o d rc
    IFS='|' read -r a t f o d <<< "$1"
    local -a ARGV
    mapfile -t ARGV < <("$PY" "$X/e2flags.py" argv "$a" "$SYN" "$RUNS")
    HYDROMIND_GW_DATA="$d" "$PY" "$X/jobmeta.py" start "$RUNS" "$a" "$t" "$f" "$o" "$d" \
        -- "${ARGV[@]}" >> "$LOG" 2>&1
    rc=$?
    case $rc in
        0) ;;
        2) note "$a completed earlier; never rerun"; return ;;
        *) note "$a refused (jobmeta start rc=$rc)"; return ;;
    esac
    note "$a start"
    echo "===== $(date '+%F %T') attempt start" >> "$RUNS/$a.log"
    HYDROMIND_GW_DATA="$d" "$PY" "${ARGV[@]}" 2>&1 \
        | "$PY" "$X/jobmeta.py" redact >> "$RUNS/$a.log"
    rc=${PIPESTATUS[0]}
    "$PY" "$X/jobmeta.py" end "$RUNS" "$a" "$rc" >> "$LOG" 2>&1
    note "$a rc=$rc"
}
worker() {  # worker K N: jobs K, K+N, ...
    local k=$1 n=$2 i
    for ((i = k; i < ${#JOBS[@]}; i += n)); do
        arm "${JOBS[$i]}"
    done
}
note "##### start: ${#JOBS[@]} arms, $WORKERS worker(s), synthetic dir $SYN"
for ((w = 0; w < WORKERS; w++)); do
    worker "$w" "$WORKERS" &
    [ $((w + 1)) -lt "$WORKERS" ] && sleep 20
done
wait
note "##### finished; next (once): $PY $X/evaluate_e2.py"
