#!/usr/bin/env bash
# X5.lever_class_share.v1 (model-only declaration): the two forward runs and the one scoring.
#
#   bash dataprep_staging/x5_class_share/run_class_share.sh check     preconditions only (CPU)
#   bash dataprep_staging/x5_class_share/run_class_share.sh forward   both forwards (GPU, tmux)
#   bash dataprep_staging/x5_class_share/run_class_share.sh score     score once (CPU)
#   bash dataprep_staging/x5_class_share/run_class_share.sh archive "<reason>"
#                                    engineering retry: move a failed attempt's outputs aside,
#                                    unopened (refused once the report exists)
#   DRY=1 bash ... forward           print the forward command lines, run nothing
#
# Code: the frozen worktree $HOME/HydroPhysicsAI-x5run at 5f3af18 (main, clean, detached; the
# code batch 27 wrote results/twin_forward/datum_gate_{q0c,x5r1}.npz with), launched through
# results/twin_runs/par28/py.sh from the repository root. Recipe: batch 27's _forward
# (results/twin_runs/par27/jobs.sh) verbatim, plus the all30 scenario and new --out stems.
# Nothing reads post-2022 observations: the record ends at the origin (2022-12) and the
# projection is climatology. About 4-5 min per forward on the T4, sequential.
#
# Binding to the registered design: every subcommand refuses unless the declaration
# results/dataprep/preregistration/X5.lever_class_share.v1.json exists, is intact
# (INDEX.jsonl and ledger, via the scorer's step 0), and records in ``frozen_code`` the
# sha256 of score_class_share.py, this script and wells_frozen.json as they are now.
#
# Sealed outputs (attempt policy): forward.py computes the all30 scenario and WRITES its
# values before the scorer runs: stdout carries the '=== forward summary ===' block (fan-mean
# layer-2 head change and subsidence per scenario, all30 included) and the rheology-axis
# block, and the run writes results/twin_forward/class_share_<arm>.{npz,summary.csv,
# members.csv}. This script sends the forward's stdout to
# results/twin_runs/class_share/sealed/<arm>_forward.stdout.SEALED, which nobody opens, tails
# or greps before report.json exists, and records its sha256 (and those of the npz and both
# CSVs) in results/twin_runs/class_share/attempts.jsonl. The forward's stderr (tracebacks and
# Python warnings; forward.py prints no value there) goes to <arm>_forward.stderr.log, which
# may be read to diagnose a crash. This script itself prints only exit codes and sha256s.
#
# Engineering check described here, run by the scorer before any all30 value is read
# (step 2): the new baseline and cut30 heads at the 158 frozen screens must reproduce the
# batch-27 files datum_gate_{q0c,x5r1}.npz (max |diff| <= 0.01 m), and the cut30 lever ratio
# x5r1/q0c must reproduce the disclosed 1.3893 / 1.3859 (Changhua) and 1.2730 / 1.2717
# (Yunlin) within 0.01. A failure there is an engineering failure: archive, fix, rerun.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
SELF=dataprep_staging/x5_class_share/run_class_share.sh
SCORER=dataprep_staging/x5_class_share/score_class_share.py
WELLS=dataprep_staging/x5_class_share/wells_frozen.json
ID=X5.lever_class_share.v1
DECL=results/dataprep/preregistration/$ID.json
WT="$HOME/HydroPhysicsAI-x5run"
WT_HEAD=5f3af18
PY=results/twin_runs/par28/py.sh
HPY="$HOME/miniforge3/envs/hydro/bin/python"
R=results/twin_runs
LOG=$R/class_share
SEAL=$LOG/sealed
ATT=$LOG/attempts.jsonl
REPORT=results/dataprep/reports/$ID/report.json
CUT30='cut30:irrigation=0.7@2026-01'
ALL30='all30:irrigation=0.7,aquaculture=0.7,livestock=0.7,domestic=0.7,industry=0.7,other=0.7@2026-01'
FLAGS="--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation --restart-taper-km 5 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --device cuda --compile-matvec --save-members none"
ARMS=(q0c x5r1)

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# append one JSON record (key=value pairs; values are strings or integers) to attempts.jsonl
log_attempt() {
    mkdir -p "$LOG"
    "$HPY" -I -c '
import json, sys
rec = {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    rec[k] = int(v) if v.lstrip("-").isdigit() and k in ("rc", "attempt") else v
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")' "$ATT" "$@"
}

next_attempt() {
    if [ -f "$ATT" ]; then
        "$HPY" -I -c '
import json, sys
n = [json.loads(s).get("attempt", 0) for s in open(sys.argv[1]) if s.strip()]
print(max(n, default=0) + 1)' "$ATT"
    else
        echo 1
    fi
}

preconditions() {
    [ -f "$DECL" ] || die "declare $ID first (prereg_class_share.py --register)"
    for f in "$SCORER" "$SELF" "$WELLS"; do
        want=$("$HPY" -I -c 'import json, sys
print(json.load(open(sys.argv[1])).get("frozen_code", {}).get(sys.argv[2], "none"))' "$DECL" "$f")
        [ "$want" = "$(sha "$f")" ] || die "$f sha256 differs from the registered declaration"
    done
    "$HPY" "$SCORER" --verify-registration > /dev/null \
        || die "the registered declaration does not verify (INDEX/ledger/frozen code)"
    [ -x "$PY" ] || die "$PY missing"
    grep -qx "export PYTHONPATH=$WT" "$PY" || die "$PY does not pin $WT"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$WT_HEAD" ] || die "$WT is not $WT_HEAD"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; no rerun"
}

forward_cmd() {   # the exact command line for arm $1 (printed by DRY=1)
    echo "$PY -m hydrophysics.twin.forward --theta $R/stage3_datum_gate_$1/stage3_theta.json" \
         "--scenario '$CUT30' --scenario '$ALL30' $FLAGS" \
         "--vep-json $R/stage3_datum_gate_$1/coupled_leveling/vep_zonal_leveling.json" \
         "--out results/twin_forward/class_share_$1"
}

do_forward() {
    if [ "${DRY:-0}" = 1 ]; then
        for a in "${ARMS[@]}"; do forward_cmd "$a"; done
        return 0
    fi
    preconditions
    for a in "${ARMS[@]}"; do
        for s in npz summary.csv members.csv; do
            [ ! -e "results/twin_forward/class_share_$a.$s" ] \
                || die "results/twin_forward/class_share_$a.$s exists (archive it first)"
        done
    done
    nvidia-smi > /dev/null || die "nvidia-smi failed"
    mkdir -p "$SEAL"
    chmod 700 "$SEAL"
    n=$(next_attempt)
    decl_sha=$(sha "$DECL")
    for a in "${ARMS[@]}"; do
        out=results/twin_forward/class_share_$a
        sealed=$SEAL/${a}_forward.stdout.SEALED
        err=$LOG/${a}_forward.stderr.log
        t0=$(utc)
        rc=0
        # shellcheck disable=SC2086  # FLAGS is a list of plain tokens
        "$PY" -m hydrophysics.twin.forward --theta "$R/stage3_datum_gate_$a/stage3_theta.json" \
            --scenario "$CUT30" --scenario "$ALL30" $FLAGS \
            --vep-json "$R/stage3_datum_gate_$a/coupled_leveling/vep_zonal_leveling.json" \
            --out "$out" > "$sealed" 2> "$err" || rc=$?
        chmod 400 "$sealed"
        hs=(); for s in npz summary.csv members.csv; do
            if [ -e "$out.$s" ]; then hs+=("$(sha "$out.$s")"); else hs+=(missing); fi
        done
        log_attempt event=forward attempt="$n" arm="$a" rc="$rc" start="$t0" time="$(utc)" \
            worktree_head="$(git -C "$WT" rev-parse HEAD)" py_sh_sha256="$(sha "$PY")" \
            declaration_sha256="$decl_sha" npz_sha256="${hs[0]}" summary_csv_sha256="${hs[1]}" \
            members_csv_sha256="${hs[2]}" stdout_sealed_sha256="$(sha "$sealed")" \
            stdout_sealed="$sealed" stderr_log="$err"
        echo "forward $a: exit $rc (attempt $n; stdout sealed, sha256 $(sha "$sealed"))"
        [ "$rc" = 0 ] || die "forward $a failed: read $err only, then archive and rerun"
    done
    echo "both forwards done; do not open the sealed logs or the class_share CSVs; next: score"
}

do_score() {
    preconditions
    n=$(next_attempt)
    rc=0
    "$HPY" "$SCORER" --q0c results/twin_forward/class_share_q0c.npz \
        --x5r1 results/twin_forward/class_share_x5r1.npz --attempts "$ATT" \
        --out "$REPORT" || rc=$?
    rs=missing; [ -e "$REPORT" ] && rs=$(sha "$REPORT")
    log_attempt event=score attempt="$n" rc="$rc" time="$(utc)" report_sha256="$rs" \
        declaration_sha256="$(sha "$DECL")"
    case $rc in
        0) echo "scored once: $REPORT (sha256 $rs); the sealed logs and CSVs may now be read" ;;
        1|2|3) echo "engineering failure (scorer exit $rc): no all30 value was read;" \
                    "fix the cause, then: run_class_share.sh archive '<reason>' and forward" ;;
        *) echo "scorer exit $rc" ;;
    esac
    return "$rc"
}

do_archive() {
    reason=${1:-}
    [ -n "$reason" ] || die "archive needs a reason"
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; nothing may be rerun"
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    dst=$LOG/archive/$stamp
    mkdir -p "$dst"
    moved=()
    for a in "${ARMS[@]}"; do
        for f in "results/twin_forward/class_share_$a.npz" \
                 "results/twin_forward/class_share_$a.summary.csv" \
                 "results/twin_forward/class_share_$a.members.csv" \
                 "$SEAL/${a}_forward.stdout.SEALED" "$LOG/${a}_forward.stderr.log"; do
            if [ -e "$f" ]; then
                mv "$f" "$dst/"
                moved+=("$(basename "$f")")
            fi
        done
    done
    log_attempt event=archive time="$(utc)" reason="$reason" dir="$dst" \
        moved="${moved[*]:-none}"
    echo "archived ${#moved[@]} files to $dst (unopened); rerun: run_class_share.sh forward"
}

case "${1:-}" in
    check) preconditions; echo "preconditions ok" ;;
    forward) do_forward ;;
    score) do_score ;;
    archive) shift; do_archive "${1:-}" ;;
    *) echo "usage: $0 check|forward|score|archive '<reason>'" >&2; exit 2 ;;
esac
