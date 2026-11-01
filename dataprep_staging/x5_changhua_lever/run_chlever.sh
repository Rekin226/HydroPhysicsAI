#!/usr/bin/env bash
# X5.changhua_lever_model.v1 (model-only declaration, round 5 rank 8): inputs, pre-builds,
# the scored forwards and the one scoring.
#
#   bash dataprep_staging/x5_changhua_lever/run_chlever.sh replays    BEFORE registration, CPU:
#        the seven replayed pre-origin anomaly fields and the two county cuts (replay.py)
#   bash dataprep_staging/x5_changhua_lever/run_chlever.sh prebuild   BEFORE registration, GPU:
#        one baseline-only forward per arm (plus ccz_m1618 with a second column); no lever
#   bash dataprep_staging/x5_changhua_lever/run_chlever.sh precheck   BEFORE registration, CPU:
#        pre-builds against the archived forwards -> precheck.json (frozen in the declaration)
#   bash dataprep_staging/x5_changhua_lever/run_chlever.sh check      preconditions (registered)
#   bash dataprep_staging/x5_changhua_lever/run_chlever.sh forward    AFTER registration, GPU,
#        tmux, under MPS: the eleven scored forwards (sealed)
#   bash dataprep_staging/x5_changhua_lever/run_chlever.sh score      score once (CPU)
#   bash dataprep_staging/x5_changhua_lever/run_chlever.sh archive "<reason>"
#        engineering retry: move a failed attempt's outputs aside, unopened
#   DRY=1 bash ... prebuild|forward    print the forward command lines, run nothing
#
# Code: the frozen clean detached worktree $HOME/HydroPhysicsAI-r6ch at 1b08fb6 (N2C's
# registered code commit), launched through results/twin_runs/r6ch/py.sh from the repository
# root, which must pin PYTHONPATH to the worktree. HYDROMIND_GW_DATA is exported here (the
# class-share runs crashed without it). Forwards: origin 2022-12, climatology after it, one
# parameter set per arm (the main set; N2C's contrast uses the main set), --ic-members 1 as
# N2C, members saved yearly (the 18-member archives are reproduced on the unperturbed main
# member). About 2-5 min per forward on the T4; ccz carries 11 scenarios.
#
# Sealed outputs (attempt policy, copied from X5.lever_class_share.v1): forward.py prints a
# '=== forward summary ===' block with every scenario's fan means and writes the npz, the
# members sidecar and two CSVs before the scorer runs. The scored forwards' stdout goes to
# results/twin_runs/x5_chlever/sealed/<arm>_forward.stdout.SEALED (mode 400), which nobody
# opens, tails or greps before report.json exists; results/twin_forward/chlever_<arm>.* stay
# unopened (the scorer reads only their baseline slice before its step 3). This script prints
# only exit codes and sha256s; stderr (tracebacks, warnings) may be read to diagnose a crash.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
SELF=dataprep_staging/x5_changhua_lever/run_chlever.sh
SCORER=dataprep_staging/x5_changhua_lever/score_chlever.py
REPLAY=dataprep_staging/x5_changhua_lever/replay.py
ID=X5.changhua_lever_model.v1
DECL=results/dataprep/preregistration/$ID.json
WT="$HOME/HydroPhysicsAI-r6ch"
WT_HEAD=1b08fb6
PY=results/twin_runs/r6ch/py.sh
HPY="$HOME/miniforge3/envs/hydro/bin/python"
R=results/twin_runs
LOG=$R/x5_chlever
PRE=$LOG/pre
RPL=$LOG/replays
SEAL=$LOG/sealed
ATT=$LOG/attempts.jsonl
REPORT=results/dataprep/reports/$ID/report.json
ALL30='all30:irrigation=0.7,aquaculture=0.7,livestock=0.7,domestic=0.7,industry=0.7,other=0.7@2026-01'
FLAGS="--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation --restart-taper-km 5 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --device cuda --compile-matvec --save-members yearly"
ARMS=(ccz ccz_m1618 chc chs cdm csw ccs rs x5r1t cpw b1)
YEARS=(2012 2013 2014 2015 2016 2017 2018)
export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

column() {   # the arm's column JSON (ccz_m1618 has only its z3 masked-years columns)
    case $1 in
        ccz_m1618) echo "$R/stage3_datum_gate_ccz_m1618/coupled_leveling_z3_w1618/vep_zonal_leveling.json" ;;
        ccz_m1618_colB) echo "$R/stage3_datum_gate_ccz_m1618/coupled_leveling_z3_w1618_tau3650/vep_zonal_leveling.json" ;;
        *) echo "$R/stage3_datum_gate_$1/coupled_leveling/vep_zonal_leveling.json" ;;
    esac
}

theta() { echo "$R/stage3_datum_gate_${1%_colB}/stage3_theta.json"; }

extra() {   # the scored forward's scenarios beyond the baseline
    printf -- "--scenario '%s'" "$ALL30"
    if [ "$1" = ccz ]; then
        printf -- " --recorded-pumping 'CH30=%s/ch30.csv@2026-01#changhua'" "$RPL"
        printf -- " --recorded-pumping 'YL30=%s/yl30.csv@2026-01#yunlin'" "$RPL"
        for y in "${YEARS[@]}"; do
            printf -- " --recorded-pumping 'RP%s=%s/rp%s.npz@2023-01'" "$y" "$RPL" "$y"
        done
    fi
}

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

worktree_ok() {
    [ -x "$PY" ] || die "$PY missing (create it: see the declaration's recipe.launcher)"
    grep -qx "export PYTHONPATH=$WT" "$PY" || die "$PY does not pin $WT"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$WT_HEAD" ] || die "$WT is not $WT_HEAD"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    [ -d "$HYDROMIND_GW_DATA" ] || die "HYDROMIND_GW_DATA $HYDROMIND_GW_DATA missing"
}

preconditions() {
    [ -f "$DECL" ] || die "declare $ID first (prereg_chlever.py --write)"
    "$HPY" "$SCORER" score --verify-registration > /dev/null \
        || die "the registered declaration does not verify (INDEX/ledger/frozen code)"
    worktree_ok
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; no rerun"
}

pre_cmd() {   # baseline-only pre-build of arm $1 (no --scenario: the baseline only)
    echo "$PY -m hydrophysics.twin.forward --theta $(theta "$1") $FLAGS" \
         "--vep-json $(column "$1") --out $PRE/$1"
}

fwd_cmd() {   # the scored forward of arm $1
    echo "$PY -m hydrophysics.twin.forward --theta $(theta "$1") $(extra "$1") $FLAGS" \
         "--vep-json $(column "$1") --out results/twin_forward/chlever_$1"
}

do_replays() {
    [ ! -e "$DECL" ] || die "$ID is registered; the replays are frozen in it"
    "$HPY" -I "$REPLAY" build --out "$RPL" | tail -n 1
}

do_prebuild() {
    if [ "${DRY:-0}" = 1 ]; then
        for a in "${ARMS[@]}" ccz_m1618_colB; do pre_cmd "$a"; done
        return 0
    fi
    [ ! -e "$DECL" ] || die "$ID is registered; the pre-builds are frozen in it"
    worktree_ok
    nvidia-smi > /dev/null || die "nvidia-smi failed"
    mkdir -p "$PRE"
    for a in "${ARMS[@]}" ccz_m1618_colB; do
        [ ! -e "$PRE/$a.npz" ] || { echo "pre-build $a exists, kept"; continue; }
        rc=0
        eval "$(pre_cmd "$a")" > "$PRE/$a.stdout.log" 2> "$PRE/$a.stderr.log" || rc=$?
        echo "pre-build $a: exit $rc$( [ "$rc" = 0 ] && echo ", sha256 $(sha "$PRE/$a.npz")")"
        [ "$rc" = 0 ] || die "pre-build $a failed: read $PRE/$a.stderr.log, fix, rerun"
    done
}

do_precheck() {
    [ ! -e "$DECL" ] || die "$ID is registered; the precheck is frozen in it"
    "$HPY" "$SCORER" precheck
}

do_forward() {
    if [ "${DRY:-0}" = 1 ]; then
        for a in "${ARMS[@]}"; do fwd_cmd "$a"; done
        return 0
    fi
    preconditions
    for a in "${ARMS[@]}"; do
        for s in npz members.npz summary.csv members.csv; do
            [ ! -e "results/twin_forward/chlever_$a.$s" ] \
                || die "results/twin_forward/chlever_$a.$s exists (archive it first)"
        done
    done
    nvidia-smi > /dev/null || die "nvidia-smi failed"
    mkdir -p "$SEAL"
    chmod 700 "$SEAL"
    n=$(next_attempt)
    decl_sha=$(sha "$DECL")
    for a in "${ARMS[@]}"; do
        out=results/twin_forward/chlever_$a
        sealed=$SEAL/${a}_forward.stdout.SEALED
        err=$LOG/${a}_forward.stderr.log
        t0=$(utc)
        rc=0
        eval "$(fwd_cmd "$a")" > "$sealed" 2> "$err" || rc=$?
        chmod 400 "$sealed"
        hs=(); for s in npz members.npz summary.csv members.csv; do
            if [ -e "$out.$s" ]; then hs+=("$(sha "$out.$s")"); else hs+=(missing); fi
        done
        log_attempt event=forward attempt="$n" arm="$a" rc="$rc" start="$t0" time="$(utc)" \
            worktree_head="$(git -C "$WT" rev-parse HEAD)" py_sh_sha256="$(sha "$PY")" \
            declaration_sha256="$decl_sha" npz_sha256="${hs[0]}" members_npz_sha256="${hs[1]}" \
            summary_csv_sha256="${hs[2]}" members_csv_sha256="${hs[3]}" \
            stdout_sealed_sha256="$(sha "$sealed")" stdout_sealed="$sealed" stderr_log="$err"
        echo "forward $a: exit $rc (attempt $n; stdout sealed, sha256 $(sha "$sealed"))"
        [ "$rc" = 0 ] || die "forward $a failed: read $err only, then archive and rerun"
    done
    echo "all forwards done; do not open the sealed logs or the chlever outputs; next: score"
}

do_score() {
    preconditions
    n=$(next_attempt)
    rc=0
    "$HPY" "$SCORER" score --attempts "$ATT" --out "$REPORT" || rc=$?
    rs=missing; [ -e "$REPORT" ] && rs=$(sha "$REPORT")
    log_attempt event=score attempt="$n" rc="$rc" time="$(utc)" report_sha256="$rs" \
        declaration_sha256="$(sha "$DECL")"
    case $rc in
        0) echo "scored once: $REPORT (sha256 $rs); the sealed logs and outputs may now be read" ;;
        1|2|3) echo "engineering failure (scorer exit $rc): no lever was read; fix the cause," \
                    "then: run_chlever.sh archive '<reason>' and forward" ;;
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
        for f in "results/twin_forward/chlever_$a.npz" \
                 "results/twin_forward/chlever_$a.members.npz" \
                 "results/twin_forward/chlever_$a.summary.csv" \
                 "results/twin_forward/chlever_$a.members.csv" \
                 "$SEAL/${a}_forward.stdout.SEALED" "$LOG/${a}_forward.stderr.log"; do
            if [ -e "$f" ]; then
                mv "$f" "$dst/"
                moved+=("$(basename "$f")")
            fi
        done
    done
    log_attempt event=archive time="$(utc)" reason="$reason" dir="$dst" \
        moved="${moved[*]:-none}"
    echo "archived ${#moved[@]} files to $dst (unopened); rerun: run_chlever.sh forward"
}

case "${1:-}" in
    replays) do_replays ;;
    prebuild) do_prebuild ;;
    precheck) do_precheck ;;
    check) preconditions; echo "preconditions ok" ;;
    forward) do_forward ;;
    score) do_score ;;
    archive) shift; do_archive "${1:-}" ;;
    *) echo "usage: $0 replays|prebuild|precheck|check|forward|score|archive '<reason>'" >&2
       exit 2 ;;
esac
