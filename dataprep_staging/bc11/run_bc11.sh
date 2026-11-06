#!/usr/bin/env bash
# I1.backcast_2010_11.v1 (round 8 plan #12; user rulings R8.6a-c): the lead's exact order.
# The code must be on main (PR, green CI) before step 1; long steps run inside tmux.
#
#   (0. done 2026-11-06: availability look logged and counted, PASS, origin 2010-12 only)
#   1. bash dataprep_staging/bc11/run_bc11.sh build-look   log the product_build look (own unit)
#      bash dataprep_staging/bc11/run_bc11.sh build        build heads/L1_pre2012 v2 (local)
#   2. bash dataprep_staging/bc11/run_bc11.sh worktree     clean detached worktree at WT_HEAD
#      bash dataprep_staging/bc11/run_bc11.sh e0           GPU: E0 against cf2c7154 (0.01 m)
#   3. bash dataprep_staging/bc11/run_bc11.sh swap         GPU: the Yunlin+Changhua swap gate
#      bash dataprep_staging/bc11/run_bc11.sh hindcast     GPU: F1, F0, B_yl from 2007-01
#   4. bash dataprep_staging/bc11/run_bc11.sh baselines    CPU: 2012-22 CL / rho / sd / datum
#   5. bash dataprep_staging/bc11/run_bc11.sh g0 g1|g2|g3|yr   CPU: synthetic G0 studies
#   6. bash dataprep_staging/bc11/run_bc11.sh validate     the final draft (prints roles)
#   7. (the lead registers: python -I dataprep_staging/bc11/prereg_bc11.py --write --quiet
#       --expect-self-sha <sha>)
#   8. bash dataprep_staging/bc11/run_bc11.sh score        once
#
# GPU steps: BC11_DEVICE=cuda, nvidia-smi checked first, at most 3 jobs under MPS, no edit of
# hydrophysics/twin/* while they run. No step prints a head value; stdout carries counts,
# check flags, gate words and sha256s. Every step's exit code goes to attempts.jsonl.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
ID=I1.backcast_2010_11.v1
WT_HEAD=4c2723e
WT="$HOME/HydroPhysicsAI-bc11run-4c2723e"
RUN=results/twin_runs/chx/backcast_2010_11
PY=$RUN/py.sh
HPY="$HOME/miniforge3/envs/hydro/bin/python"
DECL=results/dataprep/preregistration/$ID.json
PRODUCT=results/dataprep/heads/L1_pre2012/v2
ATT=$RUN/attempts.jsonl
B=dataprep_staging/bc11
export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
DEV=${BC11_DEVICE:-cpu}

die() { echo "REFUSED: $*" >&2; exit 1; }
attempt() { mkdir -p "$RUN"; printf '{"time":"%s","step":"%s","rc":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" >> "$ATT"; }
run() { local step=$1; shift; set +e; "$@"; local rc=$?; set -e; attempt "$step" "$rc"; \
    [ "$rc" -eq 0 ] || die "$step failed (rc $rc)"; }
worktree_ok() {
    [ -d "$WT" ] || die "$WT missing (bash $0 worktree)"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$WT_HEAD" ] || die "$WT is not $WT_HEAD"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    [ -x "$PY" ] || die "$PY missing (bash $0 worktree)"
    grep -qx "export PYTHONPATH=$WT" "$PY" || die "$PY does not pin $WT"
}
gpu_ok() {
    if [ "$DEV" = cuda ]; then nvidia-smi > /dev/null || die "nvidia-smi failed"; fi
}
not_registered() { [ ! -f "$DECL" ] || die "$ID is registered; this step precedes it"; }

case "${1:-}" in
    build-look)
        not_registered
        "$HPY" $B/build_l1pre_v2.py --log-look --skip-if-logged
        ;;
    build)
        not_registered
        [ ! -d "$PRODUCT" ] || die "$PRODUCT exists (built once)"
        run build "$HPY" $B/build_l1pre_v2.py --build
        ;;
    worktree)
        if [ ! -d "$WT" ]; then git worktree add --detach "$WT" "$WT_HEAD"; fi
        mkdir -p "$RUN"
        printf '#!/usr/bin/env bash\nexport PYTHONPATH=%s\nexec %s -P "$@"\n' "$WT" "$HPY" > "$PY"
        chmod 755 "$PY"
        worktree_ok
        echo "worktree $WT at $WT_HEAD; launcher $PY"
        ;;
    e0)
        not_registered; worktree_ok; gpu_ok
        run e0 "$PY" $B/page_run.py e0 --out "$RUN/e0" --device "$DEV" --compile-matvec
        ;;
    swap)
        not_registered; worktree_ok; gpu_ok
        "$HPY" -c "import json,sys; sys.exit(0 if json.load(open('$RUN/e0/e0.json'))['passed'] else 1)" \
            || die "E0 has not passed"
        run swap "$PY" $B/page_run.py swap --out "$RUN/swap" --device "$DEV" --compile-matvec
        ;;
    hindcast)
        not_registered; worktree_ok; gpu_ok
        "$HPY" -c "import json,sys; sys.exit(0 if json.load(open('$RUN/e0/e0.json'))['passed'] else 1)" \
            || die "E0 has not passed"
        [ -f "$PRODUCT/station_meta.parquet" ] || die "build $PRODUCT first"
        run hindcast "$PY" $B/page_run.py backcast --stations "$PRODUCT/station_meta.parquet" \
            --out "$RUN/hindcast" --device "$DEV" --compile-matvec
        ;;
    baselines)
        not_registered
        [ -f "$RUN/hindcast/bc11_panel.npz" ] || die "no panel (bash $0 hindcast)"
        run baselines "$HPY" $B/baselines_bc11.py --panel "$RUN/hindcast/bc11_panel.npz" \
            --out "$RUN/baselines"
        ;;
    g0)
        not_registered
        name=${2:-}
        case "$name" in
            g1) run g0_g1 "$HPY" $B/g0_bc11.py g1 --out "$RUN/g0" ;;
            g2|g3|yr)
                [ -f "$RUN/baselines/bc11_baselines.npz" ] || die "no baselines"
                run "g0_$name" "$HPY" $B/g0_bc11.py "$name" --out "$RUN/g0" \
                    --panel "$RUN/hindcast/bc11_panel.npz" \
                    --baselines "$RUN/baselines/bc11_baselines.npz" ;;
            *) die "g0 needs one of g1 g2 g3 yr" ;;
        esac
        ;;
    validate) "$HPY" -I $B/prereg_bc11.py --validate --quiet ;;
    score)
        [ -f "$DECL" ] || die "register $ID first"
        "$HPY" $B/score_bc11.py --verify-registration
        "$HPY" $B/score_bc11.py score
        ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
