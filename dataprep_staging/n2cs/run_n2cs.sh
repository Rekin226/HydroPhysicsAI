#!/usr/bin/env bash
# N2CS (round-8 rank #6): files, the CPU forwards of R1c, the pre-gate, the OSSE and the
# 2028 scoring forward. PREPARED ONLY: the lead runs each step, long ones inside tmux.
#
#   bash dataprep_staging/n2cs/run_n2cs.sh setup        write results/twin_runs/n2cs/py.sh
#        pinned to the clean frozen worktree $HOME/HydroPhysicsAI-chx at 1b08fb6 (N2C's code)
#   bash dataprep_staging/n2cs/run_n2cs.sh declare SEED validate N2CS.osse_design.v1 (the lead
#        registers it BEFORE files; nothing here writes a registration)
#   bash dataprep_staging/n2cs/run_n2cs.sh files        refuses before that declaration: the
#        two unit files (pre-gate) and the seven Changhua-only + seven Yunlin-only pre-origin
#        replays (the builder logs its scoped v3 design look; max filled fraction 0.2)
#   bash dataprep_staging/n2cs/run_n2cs.sh forward-a    CPU forward A: baseline, UCH, UYL,
#        RCH2012..RCH2018 (10 scenarios)
#   bash dataprep_staging/n2cs/run_n2cs.sh forward-b    CPU forward B: baseline,
#        RYL2012..RYL2018 (8 scenarios); A and B may run side by side (CPU only)
#   bash dataprep_staging/n2cs/run_n2cs.sh extract      pre-gate (1e-6 m), per-cell arm check,
#        contrast panel; the panel sha256 goes to attempts.jsonl
#   bash dataprep_staging/n2cs/run_n2cs.sh osse         the OSSE (once)
#   bash dataprep_staging/n2cs/run_n2cs.sh score-files X1_PARQUET SCORING_DIR    2028 only
#   bash dataprep_staging/n2cs/run_n2cs.sh forward-score SCORING_DIR             2028 only:
#        CPU forward S: baseline, NCSCH, NCSYL (3 scenarios)
#   DRY=1 bash ... forward-a|forward-b|forward-score    print the command line, run nothing
#
# The forward is N2C's registered primary command (CPU, R1c main set, --ic-members 1,
# --save-members none) with its PR arm replaced by N2CS's arms. Its stdout goes to a sealed
# log that is never opened; this script prints exit codes and sha256s only. No GPU is used.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
HPY="$HOME/miniforge3/envs/hydro/bin/python"
WT="$HOME/HydroPhysicsAI-chx"
WT_HEAD=1b08fb6
LOG=results/twin_runs/n2cs
PY=$LOG/py.sh
SEAL=$LOG/sealed
ATT=$LOG/attempts.jsonl
FILES=$LOG/r_files
RPL=$FILES/replays
R=results/twin_runs
FWD_A=results/twin_forward/n2cs_osse_a_r1c
FWD_B=results/twin_forward/n2cs_osse_b_r1c
FLAGS="--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation --restart-taper-km 5 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --device cpu --save-members none"
YEARS=(2012 2013 2014 2015 2016 2017 2018)
NOT_BEFORE=2028-01-15
export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"
# the OSSE steps run single-threaded numpy (BLAS threading stalls on a loaded host); the
# forwards keep the default threading
ONE="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

log_attempt() {
    mkdir -p "$LOG"
    "$HPY" -I -c '
import json, sys
rec = {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    rec[k] = int(v) if k == "rc" else v
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")' "$ATT" "$@"
}

frozen_ok() {
    [ -x "$PY" ] || die "$PY missing (run setup)"
    grep -qx "export PYTHONPATH=$WT" "$PY" || die "$PY does not pin $WT"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$WT_HEAD" ] || die "$WT is not $WT_HEAD"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    [ -d "$HYDROMIND_GW_DATA" ] || die "HYDROMIND_GW_DATA $HYDROMIND_GW_DATA missing"
}

fwd_cmd() {   # $1 out, then NAME=PATH pairs
    local out=$1; shift
    printf "%s -m hydrophysics.twin.forward --theta %s --vep-json %s %s" "$PY" \
        "$R/stage3_datum_gate_ccz/stage3_theta.json" \
        "$R/stage3_datum_gate_ccz/coupled_leveling/vep_zonal_leveling.json" "$FLAGS"
    for arm in "$@"; do printf " --recorded-pumping '%s@2023-01'" "$arm"; done
    printf " --out %s\n" "$out"
}

arms_a() {
    echo "UCH=$FILES/unit_changhua.npz"
    echo "UYL=$FILES/unit_yunlin.npz"
    for y in "${YEARS[@]}"; do echo "RCH$y=$RPL/rch$y.npz"; done
}
arms_b() { for y in "${YEARS[@]}"; do echo "RYL$y=$RPL/ryl$y.npz"; done; }

run_forward() {   # $1 tag, $2 out, rest arms
    local tag=$1 out=$2; shift 2
    if [ "${DRY:-0}" = 1 ]; then fwd_cmd "$out" "$@"; return 0; fi
    frozen_ok
    for a in "$@"; do [ -f "${a#*=}" ] || die "${a#*=} missing (run files)"; done
    for s in npz summary.csv; do [ ! -e "$out.$s" ] || die "$out.$s exists"; done
    mkdir -p "$SEAL"; chmod 700 "$SEAL"
    local sealed=$SEAL/$tag.stdout.SEALED err=$LOG/$tag.stderr.log t0 rc=0
    t0=$(utc)
    eval "$(fwd_cmd "$out" "$@")" > "$sealed" 2> "$err" || rc=$?
    chmod 400 "$sealed"
    local h=missing; [ -e "$out.npz" ] && h=$(sha "$out.npz")
    log_attempt event="forward_$tag" rc="$rc" start="$t0" time="$(utc)" \
        worktree_head="$(git -C "$WT" rev-parse HEAD)" py_sh_sha256="$(sha "$PY")" \
        npz_sha256="$h" stdout_sealed_sha256="$(sha "$sealed")"
    echo "forward $tag rc=$rc npz_sha256=$h"
    [ "$rc" = 0 ] || exit "$rc"
}

after_not_before() {
    [ "$(date -u +%Y-%m-%d)" \> "$NOT_BEFORE" ] || [ "$(date -u +%Y-%m-%d)" = "$NOT_BEFORE" ] \
        || die "scoring-time step before $NOT_BEFORE"
}

case "${1:-}" in
    setup)
        [ ! -e "$PY" ] || die "$PY exists"
        mkdir -p "$LOG"
        printf '#!/usr/bin/env bash\nexport PYTHONPATH=%s\nexec %s -P "$@"\n' "$WT" "$HPY" > "$PY"
        chmod 755 "$PY"; frozen_ok; echo "py.sh $(sha "$PY")" ;;
    files)
        [ ! -e results/dataprep/preregistration/N2CS.changhua_size_class.prospective_2026-12.v1.json ] \
            || die "N2CS is registered; its design files are frozen"
        [ -e results/dataprep/preregistration/N2CS.osse_design.v1.json ] \
            || die "register N2CS.osse_design.v1 first (the OSSE design precedes the files)"
        "$HPY" -I "$X/build_r_county.py" unit --scope changhua --out "$FILES/unit_changhua.npz" | tail -n 3
        "$HPY" -I "$X/build_r_county.py" unit --scope yunlin --out "$FILES/unit_yunlin.npz" | tail -n 3
        # build_r_county.py replays logs its own scoped v3 design look before reading the census
        "$HPY" -I "$X/build_r_county.py" replays --out "$RPL" | tail -n 2
        log_attempt event=files rc=0 replays_json_sha256="$(sha "$RPL/replays.json")" ;;
    forward-a) mapfile -t A < <(arms_a); run_forward a "$FWD_A" "${A[@]}" ;;
    forward-b) mapfile -t B < <(arms_b); run_forward b "$FWD_B" "${B[@]}" ;;
    extract) $ONE "$HPY" "$X/osse_n2cs.py" extract ;;
    declare) [ -n "${2:-}" ] || die "declare needs the world seed"
        $ONE "$HPY" -I "$X/osse_n2cs.py" declare --seed "$2" ;;
    osse) $ONE "$HPY" "$X/osse_n2cs.py" run ;;
    score-files)
        after_not_before
        [ -n "${2:-}" ] && [ -n "${3:-}" ] || die "score-files X1_PARQUET SCORING_DIR"
        for s in changhua yunlin; do
            "$HPY" -I "$X/build_r_county.py" x1 --x1 "$2" --first 2023-01 --scope "$s" \
                --out "$3/r_$s.npz" | tail -n 2
        done ;;
    forward-score)
        [ "${DRY:-0}" = 1 ] || after_not_before
        [ -n "${2:-}" ] || die "forward-score SCORING_DIR"
        run_forward s "$2/forward_ncs_r1c" "NCSCH=$2/r_changhua.npz" "NCSYL=$2/r_yunlin.npz" ;;
    *) sed -n '2,24p' "$0"; exit 2 ;;
esac
