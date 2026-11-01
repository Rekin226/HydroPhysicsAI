#!/usr/bin/env bash
# I1.CHX.v1 (round 7 rank 13): worktree, hindcast panel, G0, CH-J surface-water merge, score.
#
#   bash dataprep_staging/ch_x/run_chx.sh worktree   create the frozen clean detached worktree
#        $HOME/HydroPhysicsAI-chx at 1b08fb6 and its launcher (before registration; no data)
#   bash dataprep_staging/ch_x/run_chx.sh validate   prereg_chx.py --validate --quiet
#   (the lead registers: prereg_chx.py --write --quiet --expect-self-sha <sha>)
#   bash dataprep_staging/ch_x/run_chx.sh hindcast   AFTER registration: R1c arms A/B/C, heads
#        at the externals -> $HC/chx_panel.npz (model only; CPU by default, CHX_DEVICE=cuda
#        under MPS in tmux; never while twin modules are being edited)
#   bash dataprep_staging/ch_x/run_chx.sh g0         AFTER hindcast, CPU: the synthetic gate
#   bash dataprep_staging/ch_x/run_chx.sh build-sw   CH-J's merged surface-water npz (forcing)
#   bash dataprep_staging/ch_x/run_chx.sh score      once; 'not_informative' or 'not decidable'
#        are written without reading a head; CH-J gates only with a verified engineering record
#   bash dataprep_staging/ch_x/run_chx.sh archive "<reason>"   move a failed attempt aside
#
# Nothing here prints a model or observed value: stdout carries shapes, check flags, verdict
# words and sha256s. Every attempt is appended to $LOG/attempts.jsonl.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
ID=I1.CHX.v1
WT_HEAD=1b08fb6
WT="$HOME/HydroPhysicsAI-chx"
LOG=results/twin_runs/chx
HC=results/twin_runs/chx/hindcast
G0D=results/twin_runs/chx/g0
SWD=results/twin_runs/chx/chj_sw
PY=$LOG/py.sh
HPY="$HOME/miniforge3/envs/hydro/bin/python"
DECL=results/dataprep/preregistration/$ID.json
ATT=$LOG/attempts.jsonl
export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

die() { echo "REFUSED: $*" >&2; exit 1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
attempt() { mkdir -p "$LOG"; printf '{"time":"%s","step":"%s","rc":%s,"note":"%s"}\n' \
    "$(utc)" "$1" "$2" "${3:-}" >> "$ATT"; }

worktree_ok() {
    [ -d "$WT" ] || die "$WT missing (bash $0 worktree)"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$WT_HEAD" ] || die "$WT is not $WT_HEAD"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    [ -x "$PY" ] || die "$PY missing (bash $0 worktree)"
    grep -qx "export PYTHONPATH=$WT" "$PY" || die "$PY does not pin $WT"
}

registered() { [ -f "$DECL" ] || die "register $ID first (prereg_chx.py --write)"; \
    "$HPY" dataprep_staging/ch_x/score_chx.py --verify-registration; }

case "${1:-}" in
    worktree)
        if [ ! -d "$WT" ]; then git worktree add --detach "$WT" "$WT_HEAD"; fi
        mkdir -p "$LOG"
        printf '#!/usr/bin/env bash\nexport PYTHONPATH=%s\nexec %s -P "$@"\n' "$WT" "$HPY" > "$PY"
        chmod 755 "$PY"
        worktree_ok
        echo "worktree $WT at $WT_HEAD; launcher $PY"
        ;;
    validate)
        "$HPY" -I dataprep_staging/ch_x/prereg_chx.py --validate --quiet
        ;;
    hindcast)
        registered
        worktree_ok
        [ ! -f "$HC/chx_panel.npz" ] || die "$HC/chx_panel.npz exists (archive it first)"
        set +e
        "$PY" dataprep_staging/ch_x/hindcast.py chx --out "$HC" --check-twinmodel \
            --device "${CHX_DEVICE:-cpu}"
        rc=$?
        set -e
        attempt hindcast "$rc"
        [ "$rc" -eq 0 ] || die "hindcast failed (rc $rc); stderr may be read to diagnose"
        ;;
    g0)
        registered
        [ -f "$HC/chx_panel.npz" ] || die "no panel (bash $0 hindcast)"
        set +e
        "$HPY" dataprep_staging/ch_x/g0_chx.py --panel "$HC/chx_panel.npz" --out "$G0D"
        rc=$?
        set -e
        attempt g0 "$rc"
        [ "$rc" -eq 0 ] || die "g0 failed (rc $rc)"
        ;;
    build-sw)
        "$HPY" dataprep_staging/ch_x/chj_forcing.py build-sw --out "$SWD"
        ;;
    score)
        registered
        set +e
        if [ -f "$LOG/chj/chj_panel.npz" ]; then
            "$HPY" dataprep_staging/ch_x/score_chx.py score --panel "$HC/chx_panel.npz" \
                --g0 "$G0D/g0.json" --chj-panel "$LOG/chj/chj_panel.npz" \
                --chj-metrics "$LOG/chj/chj_metrics.json" \
                --chj-engineering "$LOG/chj/chj_engineering.json"
        else
            "$HPY" dataprep_staging/ch_x/score_chx.py score --panel "$HC/chx_panel.npz" \
                --g0 "$G0D/g0.json"
        fi
        rc=$?
        set -e
        attempt score "$rc"
        [ "$rc" -eq 0 ] || die "score refused or failed (rc $rc)"
        ;;
    archive)
        [ -n "${2:-}" ] || die "archive needs a reason"
        d="$LOG/archive/$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$d"
        for p in "$HC" "$G0D"; do [ -e "$p" ] && mv "$p" "$d/"; done
        attempt archive 0 "$2"
        echo "archived to $d (unopened)"
        ;;
    *)
        sed -n '2,20p' "$0"
        exit 2
        ;;
esac
