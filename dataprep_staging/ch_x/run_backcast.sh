#!/usr/bin/env bash
# I1.CH_backcast.v1 (round 7 rank 18; user ruling R7.l1_pre2012), in the declared order:
#
#   bash dataprep_staging/ch_x/run_backcast.sh look-validate   print the availability look
#   bash dataprep_staging/ch_x/run_backcast.sh look            LOG the availability look
#   bash dataprep_staging/ch_x/run_backcast.sh avail           timestamps-only counts + stop rule
#   bash dataprep_staging/ch_x/run_backcast.sh s0              model only (frozen worktree)
#   bash dataprep_staging/ch_x/run_backcast.sh validate        prereg_bc.py --validate --quiet
#   (the lead registers: prereg_bc.py --write --quiet --expect-self-sha <sha>)
#   bash dataprep_staging/ch_x/run_backcast.sh build           log the product_build look (once;
#        a rerun skips it), then build heads/L1_pre2012 v1 (local only, never committed)
#   bash dataprep_staging/ch_x/run_backcast.sh hindcast        model only: arms A/B 2007-11
#   bash dataprep_staging/ch_x/run_backcast.sh score           once
#
# The worktree and its launcher are run_chx.sh's ($HOME/HydroPhysicsAI-chx at 1b08fb6). No
# step prints a head value; stdout carries counts, check flags, verdict words and sha256s.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
ID=I1.CH_backcast.v1
WT_HEAD=1b08fb6
WT="$HOME/HydroPhysicsAI-chx"
BC=results/twin_runs/chx/backcast
PY=results/twin_runs/chx/py.sh
HPY="$HOME/miniforge3/envs/hydro/bin/python"
DECL=results/dataprep/preregistration/$ID.json
L1PRE=results/dataprep/heads/L1_pre2012/v1
ATT=$BC/attempts.jsonl
export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

die() { echo "REFUSED: $*" >&2; exit 1; }
attempt() { mkdir -p "$BC"; printf '{"time":"%s","step":"%s","rc":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >> "$ATT"; }
worktree_ok() {
    [ -d "$WT" ] || die "$WT missing (bash dataprep_staging/ch_x/run_chx.sh worktree)"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$WT_HEAD" ] || die "$WT is not $WT_HEAD"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    grep -qx "export PYTHONPATH=$WT" "$PY" || die "$PY does not pin $WT"
}
run() { local step=$1; shift; set +e; "$@"; local rc=$?; set -e; attempt "$step" "$rc"; \
    [ "$rc" -eq 0 ] || die "$step failed (rc $rc)"; }

case "${1:-}" in
    look-validate) "$HPY" dataprep_staging/ch_x/avail_l1pre.py look --validate ;;
    look) "$HPY" dataprep_staging/ch_x/avail_l1pre.py look --write ;;
    avail) run avail "$HPY" dataprep_staging/ch_x/avail_l1pre.py count --out "$BC/avail" ;;
    s0)
        worktree_ok
        run s0 "$PY" dataprep_staging/ch_x/hindcast.py s0 --out "$BC/s0" \
            --device "${CHX_DEVICE:-cpu}"
        ;;
    validate) "$HPY" -I dataprep_staging/ch_x/prereg_bc.py --validate --quiet ;;
    build)
        [ -f "$DECL" ] || die "register $ID first"
        "$HPY" dataprep_staging/ch_x/score_bc.py --verify-registration
        [ ! -d "$L1PRE" ] || die "$L1PRE exists (built once)"
        "$HPY" dataprep_staging/ch_x/build_l1pre.py --log-look --skip-if-logged
        run build "$HPY" dataprep_staging/ch_x/build_l1pre.py --build
        ;;
    hindcast)
        [ -f "$DECL" ] || die "register $ID first"
        worktree_ok
        [ -f "$L1PRE/station_meta.parquet" ] || die "build $L1PRE first"
        run hindcast "$PY" dataprep_staging/ch_x/hindcast.py backcast \
            --stations "$L1PRE/station_meta.parquet" --out "$BC/hindcast" \
            --device "${CHX_DEVICE:-cpu}"
        ;;
    score)
        [ -f "$DECL" ] || die "register $ID first"
        run score "$HPY" dataprep_staging/ch_x/score_bc.py score --panel "$BC/hindcast/bc_panel.npz"
        ;;
    *) sed -n '2,16p' "$0"; exit 2 ;;
esac
