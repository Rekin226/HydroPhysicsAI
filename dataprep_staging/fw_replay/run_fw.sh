#!/usr/bin/env bash
# FW.weather_replay.v1 (model-only declaration): the forcing gate, the forward runs (replay
# engine hydrophysics/twin/replay.py around the unchanged forward), the reproduction of the
# climatology arm C against the page run, and the one scoring.
#
#   bash dataprep_staging/fw_replay/run_fw.sh pregate <code> design-time dry run of the forcing
#                                   gate BEFORE registration (CPU; booleans only; logged to
#                                   results/twin_runs/fw_replay/design_time/pregate.jsonl)
#   bash dataprep_staging/fw_replay/run_fw.sh check          preconditions only (CPU)
#   bash dataprep_staging/fw_replay/run_fw.sh gate           forcing identity gate (CPU)
#   bash dataprep_staging/fw_replay/run_fw.sh forward C      climatology arm (GPU, tmux)
#   bash dataprep_staging/fw_replay/run_fw.sh repro          C against the page run (CPU)
#   bash dataprep_staging/fw_replay/run_fw.sh forward <arm|primary|reduced|weather|all> [k/n]
#                                   arms in order; 'k/n' runs every n-th arm from the k-th
#                                   (one slice per tmux session under MPS)
#   bash dataprep_staging/fw_replay/run_fw.sh launch <primary|reduced|weather|all> [n]
#                                   n tmux sessions fw1..fwn, one slice each (default 3)
#   bash dataprep_staging/fw_replay/run_fw.sh score          score once (CPU)
#   bash dataprep_staging/fw_replay/run_fw.sh archive "<reason>" [arm ...]
#                                   engineering retry: move outputs aside, unopened
#   DRY=1 bash ... forward <arm>    print the command line, run nothing
#
# Arms: C (climatology, page ensemble; reference and reproduction); W2012..W2022 (PRIMARY:
# the cyclic order of 2012..2022 starting at that year, detrended pumping, page ensemble);
# RD2012..RD2022 and RR2012..RR2022 (report-only: the page theta alone with --ic-members 0,
# detrended and raw pumping); RW2012..RW2022 (report-only, group 'weather': weather alone,
# pumping at climatology; run only when the declaration replays weather). Every arm passes
# both columns (page d290c592 and tau30); the capture keeps them apart. The replayed fields
# (energy+weather or energy) are read from the registered declaration (recipe.replay_fields),
# fixed at registration by the rain manifest.
#
# A failed reproduction (repro exit 3) ends the declaration: once one is logged, gate,
# forward (C included), repro and launch refuse, and the scorer refuses to score.
#
# Code: the forward runs from a clean detached worktree $HOME/HydroPhysicsAI-fwrun-<code>
# through PYTHONPATH (python -P, launched from the repository root), <code> the declaration's
# recipe.run_code; the engine files there must hash to recipe.engine_code_sha256. No twin
# module may be edited in that worktree while jobs run (code freeze).
#
# Sealed outputs: forward.py prints a '=== forward summary ===' block and per-member hindcast
# R2 lines to stdout; this script sends stdout to results/twin_runs/fw_replay/sealed/
# <arm>_forward.stdout.SEALED, which nobody opens before report.json exists, and records its
# sha256 and those of results/twin_forward/fw_replay/<arm>.{npz,summary.csv,members.csv,
# fw.npz} in results/twin_runs/fw_replay/attempts.jsonl. Nobody loads those files before the
# scorer has written the report, except C's reproduction quantities through `repro`. stderr
# (tracebacks, warnings) goes to <arm>_forward.stderr.log and may be read after a crash. This
# script prints only exit codes, sha256s and free GPU memory.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
SELF=dataprep_staging/fw_replay/run_fw.sh
SCORER=dataprep_staging/fw_replay/score_fw.py
ID=FW.weather_replay.v1
DECL=results/dataprep/preregistration/$ID.json
HPY="$HOME/miniforge3/envs/hydro/bin/python"
R=results/twin_runs
G=$R/stage3_datum_gate
LOG=$R/fw_replay
SEAL=$LOG/sealed
ATT=$LOG/attempts.jsonl
GATE=$LOG/gate_forcing.json
OUTD=results/twin_forward/fw_replay
REPORT=results/dataprep/reports/$ID/report.json
PREGATE_LOG=$LOG/design_time/pregate.jsonl
MIN_FREE_MIB=${FW_MIN_FREE_MIB:-5000}
PAGE_THETAS="--theta $G/stage3_theta.json --theta $G/stage3_fold_thetas.json --theta $G/stage3_posterior_tn.json"
ONE_THETA="--theta $G/stage3_theta.json"
VEP="$G/coupled_leveling/vep_zonal_leveling.json,$G/coupled_leveling_tau30/vep_zonal_leveling.json"
SCENARIOS=('cut30:irrigation=0.7@2026-01' 'retire_aqua:aquaculture=0@2026-01' 'cut30_lift:irrigation=0.7@2026-01..2029-12')
RF=chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv
COMMON="--horizon 120 --apex-hold calibrated --column-heads free --column-hpc0-fast-days 365 --rf-timeseries $RF --device cuda --compile-matvec"
F_PAGE="--gain 1.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method innovation --restart-taper-km 5 --save-members none"
F_RED="--gain 1.0 --ic-members 0 --update-method innovation --restart-taper-km 5 --save-members none"
YEARS=(2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022)
PRIMARY=(); RED=(); WEA=()
for y in "${YEARS[@]}"; do PRIMARY+=("W$y"); done
for y in "${YEARS[@]}"; do RED+=("RD$y"); done
for y in "${YEARS[@]}"; do RED+=("RR$y"); done
for y in "${YEARS[@]}"; do WEA+=("RW$y"); done
ARMS=(C "${PRIMARY[@]}" "${RED[@]}" "${WEA[@]}")

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# append one JSON record (key=value pairs) to attempts.jsonl (or to $ATT as overridden)
log_attempt() {
    mkdir -p "$(dirname "$ATT")"
    "$HPY" -I -c '
import fcntl, json, sys
rec = {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    rec[k] = int(v) if v.lstrip("-").isdigit() and k in ("rc", "attempt") else v
with open(sys.argv[1], "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(json.dumps(rec, sort_keys=True) + "\n")' "$ATT" "$@"
}

count_records() {
    if [ ! -f "$ATT" ]; then echo 0; return; fi
    "$HPY" -I -c '
import json, sys
want = dict(kv.split("=", 1) for kv in sys.argv[2:])
n = 0
for s in open(sys.argv[1]):
    if s.strip():
        r = json.loads(s)
        n += all(str(r.get(k)) == v for k, v in want.items())
print(n)' "$ATT" "$@"
}

decl_get() {   # a recipe key of the registered declaration
    "$HPY" -I -c 'import json, sys
print(json.load(open(sys.argv[1]))["recipe"][sys.argv[2]])' "$DECL" "$1"
}

# the cyclic order of 2012..2022 starting at $1, folded as replay.format_years does
seq_of() {
    local y=$1 a b
    if [ "$y" = 2022 ]; then a=2022; else a="$y-2022"; fi
    if [ "$y" = 2012 ]; then echo "$a"; return; fi
    if [ "$y" = 2013 ]; then b=2012; else b="2012-$((y - 1))"; fi
    echo "$a,$b"
}

arm_flags() {   # replay flags, then '--', then the forward's theta and arm flags
    local a=$1
    case $a in
        C) echo "--projection-forcing climatology -- $PAGE_THETAS $F_PAGE" ;;
        W20[12][0-9]) echo "--projection-forcing replay:$(seq_of "${a:1}") --replay-pumping detrended --replay-fields $FIELDS -- $PAGE_THETAS $F_PAGE" ;;
        RD20[12][0-9]) echo "--projection-forcing replay:$(seq_of "${a:2}") --replay-pumping detrended --replay-fields $FIELDS -- $ONE_THETA $F_RED" ;;
        RR20[12][0-9]) echo "--projection-forcing replay:$(seq_of "${a:2}") --replay-pumping raw --replay-fields $FIELDS -- $ONE_THETA $F_RED" ;;
        RW20[12][0-9]) echo "--projection-forcing replay:$(seq_of "${a:2}") --replay-pumping climatology --replay-fields energy+weather -- $ONE_THETA $F_RED" ;;
        *) die "unknown arm $a" ;;
    esac
}

is_arm() { local x; for x in "${ARMS[@]}"; do [ "$x" = "$1" ] && return 0; done; return 1; }

outputs() {
    local o=$OUTD/$1
    echo "$o.npz $o.summary.csv $o.members.csv $o.fw.npz"
}

preconditions() {
    [ -f "$DECL" ] || die "declare $ID first (prereg_fw.py --write)"
    for f in "$SCORER" "$SELF"; do
        want=$("$HPY" -I -c 'import json, sys
print(json.load(open(sys.argv[1])).get("frozen_code", {}).get(sys.argv[2], "none"))' "$DECL" "$f")
        [ "$want" = "$(sha "$f")" ] || die "$f sha256 differs from the registered declaration"
    done
    "$HPY" "$SCORER" --verify-registration > /dev/null \
        || die "the registered declaration does not verify (INDEX/ledger/frozen code/inputs)"
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; no rerun"
    CODE=$(decl_get run_code)
    FIELDS=$(decl_get replay_fields)
    case $FIELDS in energy+weather|energy) ;; *) die "replay_fields $FIELDS" ;; esac
    WT="$HOME/HydroPhysicsAI-fwrun-$CODE"
    check_worktree
    for f in replay forward scenario compaction; do
        want=$("$HPY" -I -c 'import json, sys
print(json.load(open(sys.argv[1]))["recipe"]["engine_code_sha256"][sys.argv[2]])' "$DECL" "$f.py")
        [ "$want" = "$(sha "$WT/hydrophysics/twin/$f.py")" ] \
            || die "$WT/hydrophysics/twin/$f.py differs from the declared engine code"
    done
}

no_failed_repro() {
    [ "$(count_records event=repro rc=3)" = 0 ] \
        || die "a failed reproduction (repro exit 3) is logged: nothing more runs under $ID"
}

check_worktree() {   # $WT at $CODE, clean, and the engine imports from it
    [ -d "$WT" ] || die "worktree $WT missing (git worktree add --detach $WT $CODE)"
    [ "$(git -C "$WT" rev-parse --short=7 HEAD)" = "$CODE" ] || die "$WT is not at $CODE"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    mod=$(PYTHONPATH="$WT" "$HPY" -P -c \
        'import hydrophysics.twin.replay as m; print(m.__file__)' 2> /dev/null) \
        || die "hydrophysics.twin.replay does not import from $WT"
    case $mod in "$WT"/*) ;; *) die "replay imports from $mod, not $WT" ;; esac
}

gpu_check() {
    nvidia-smi > /dev/null || die "nvidia-smi failed"
    read -r total used < <(nvidia-smi --query-gpu=memory.total,memory.used \
        --format=csv,noheader,nounits | head -1 | tr -d ',')
    free=$((total - used))
    echo "GPU memory free: $free MiB (need $MIN_FREE_MIB; other jobs may hold the rest)"
    [ "$free" -ge "$MIN_FREE_MIB" ] || die "not enough free GPU memory ($free MiB)"
}

forward_cmd() {
    local s="" x
    for x in "${SCENARIOS[@]}"; do s="$s --scenario '$x'"; done
    echo "PYTHONPATH=$2 \$HOME/miniforge3/envs/hydro/bin/python -P -m" \
         "hydrophysics.twin.replay $(arm_flags "$1")$s $COMMON --vep-json $VEP" \
         "--out $OUTD/$1"
}

need_gate_and_repro() {
    [ -e "$GATE" ] || die "run the forcing gate first"
    [ "$(count_records event=gate rc=0 code="$CODE" gate_sha256="$(sha "$GATE")")" -ge 1 ] \
        || die "no passing forcing gate is logged at $CODE"
    c=$OUTD/C.npz
    [ -e "$c" ] || die "run forward C and repro first"
    [ "$(count_records event=repro rc=0 code="$CODE" c_npz_sha256="$(sha "$c")")" -ge 1 ] \
        || die "no passing reproduction is logged for this C file at $CODE (run repro)"
}

do_forward_one() {
    local a=$1 f
    for f in $(outputs "$a"); do
        [ ! -e "$f" ] || die "$f exists (archive it first)"
    done
    no_failed_repro
    case $a in RW*) [ "$FIELDS" = energy+weather ] \
        || die "$a: the declaration replays pumping only (no weather-only arm)" ;; esac
    [ "$a" = C ] || need_gate_and_repro
    mkdir -p "$SEAL" "$OUTD"
    chmod 700 "$SEAL"
    n=$(( $(count_records event=forward arm="$a") + 1 ))
    out=$OUTD/$a
    sealed=$SEAL/${a}_forward.stdout.SEALED
    err=$LOG/${a}_forward.stderr.log
    t0=$(utc)
    rc=0
    # shellcheck disable=SC2046  # arm_flags and COMMON are lists of plain tokens
    PYTHONPATH="$WT" "$HPY" -P -m hydrophysics.twin.replay $(arm_flags "$a") \
        --scenario "${SCENARIOS[0]}" --scenario "${SCENARIOS[1]}" \
        --scenario "${SCENARIOS[2]}" $COMMON \
        --vep-json "$VEP" --out "$out" > "$sealed" 2> "$err" || rc=$?
    chmod 400 "$sealed"
    hs=(); for s in npz summary.csv members.csv fw.npz; do
        if [ -e "$out.$s" ]; then hs+=("$(sha "$out.$s")"); else hs+=(missing); fi
    done
    log_attempt event=forward attempt="$n" arm="$a" rc="$rc" start="$t0" time="$(utc)" \
        code="$CODE" worktree_head="$(git -C "$WT" rev-parse HEAD)" \
        declaration_sha256="$(sha "$DECL")" npz_sha256="${hs[0]}" \
        summary_csv_sha256="${hs[1]}" members_csv_sha256="${hs[2]}" \
        capture_sha256="${hs[3]}" stdout_sealed_sha256="$(sha "$sealed")" \
        stdout_sealed="$sealed" stderr_log="$err"
    echo "forward $a: exit $rc (attempt $n, code $CODE; stdout sealed, sha256 $(sha "$sealed"))"
    [ "$rc" = 0 ] || die "forward $a failed: read $err only, then archive and rerun"
}

is_group() { case $1 in primary|reduced|weather|all) return 0 ;; *) return 1 ;; esac; }

expand() {   # a group or arm name -> arms ('all' holds the weather arms unless FIELDS=energy)
    case $1 in
        primary) echo "${PRIMARY[*]}" ;;
        reduced) echo "${RED[*]}" ;;
        weather) echo "${WEA[*]}" ;;
        all) if [ "${FIELDS:-}" = energy ]; then echo "${PRIMARY[*]} ${RED[*]}"
             else echo "${PRIMARY[*]} ${RED[*]} ${WEA[*]}"; fi ;;
        *) echo "$1" ;;
    esac
}

do_forward() {
    local which=${1:-} slice=${2:-1/1}
    [ -n "$which" ] || die "forward needs an arm or a group (primary, reduced, weather, all)"
    is_group "$which" || is_arm "$which" || die "unknown arm or group $which"
    if [ "${DRY:-0}" = 1 ]; then FIELDS='<replay_fields>'; else preconditions; fi
    read -ra all <<< "$(expand "$which")"
    k=${slice%/*}; nn=${slice#*/}
    [[ $k =~ ^[0-9]+$ && $nn =~ ^[0-9]+$ && $k -ge 1 && $k -le $nn ]] || die "slice $slice"
    list=(); for i in "${!all[@]}"; do
        [ $(( i % nn )) = $(( k - 1 )) ] && list+=("${all[$i]}")
    done
    if [ "${DRY:-0}" = 1 ]; then
        for a in "${list[@]}"; do forward_cmd "$a" "\$HOME/HydroPhysicsAI-fwrun-<code>"; done
        return 0
    fi
    no_failed_repro
    gpu_check
    for a in "${list[@]}"; do
        if [ -e "$OUTD/$a.npz" ] && [ "$(count_records event=forward arm="$a" rc=0 \
                npz_sha256="$(sha "$OUTD/$a.npz")")" -ge 1 ]; then
            echo "forward $a: done already (logged), skipped"
            continue
        fi
        do_forward_one "$a"
    done
    echo "done (${list[*]}); do not open the sealed logs or the fw_replay files"
}

do_launch() {
    local which=${1:-} n=${2:-3} i
    [ -n "$which" ] || die "launch needs a group (primary, reduced, weather, all)"
    is_group "$which" || die "unknown group $which"
    preconditions
    no_failed_repro
    [ "$which" != weather ] || [ "$FIELDS" = energy+weather ] \
        || die "the declaration replays pumping only: no weather arms"
    need_gate_and_repro
    pgrep -x nvidia-cuda-mps-control > /dev/null \
        || echo "note: the MPS daemon is not running (nvidia-cuda-mps-control -d); the" \
                "sessions will time-share the GPU"
    for i in $(seq 1 "$n"); do
        tmux has-session -t "fw$i" 2> /dev/null && die "tmux session fw$i exists"
    done
    for i in $(seq 1 "$n"); do
        tmux new-session -d -s "fw$i" "bash $SELF forward $which $i/$n; exec bash"
        echo "tmux session fw$i: forward $which $i/$n"
    done
}

do_pregate() {   # design-time dry run of the forcing gate, before registration
    CODE=${1:-}
    [[ $CODE =~ ^[0-9a-f]{7}$ ]] || die "pregate needs the 7-char run code"
    [ ! -e "$DECL" ] || die "$ID is registered: run 'gate', not 'pregate'"
    WT="$HOME/HydroPhysicsAI-fwrun-$CODE"
    check_worktree
    out=$LOG/design_time/pregate_forcing_$CODE.json
    [ ! -e "$out" ] || die "$out exists (one pregate per run code)"
    mkdir -p "$LOG/design_time"
    rc=0
    PYTHONPATH="$WT" "$HPY" -P -m hydrophysics.twin.replay gate \
        --theta "$G/stage3_theta.json" --rf-timeseries "$RF" --out "$out" || rc=$?
    gs=missing; [ -e "$out" ] && gs=$(sha "$out")
    ATT=$PREGATE_LOG log_attempt event=pregate rc="$rc" time="$(utc)" code="$CODE" \
        gate_sha256="$gs" record="$out"
    echo "design-time forcing pregate: exit $rc (sha256 $gs; booleans only)"
    return "$rc"
}

do_gate() {
    preconditions
    no_failed_repro
    [ ! -e "$GATE" ] || die "$GATE exists (archive it to rerun)"
    rc=0
    PYTHONPATH="$WT" "$HPY" -P -m hydrophysics.twin.replay gate \
        --theta "$G/stage3_theta.json" --rf-timeseries "$RF" --out "$GATE" || rc=$?
    gs=missing; [ -e "$GATE" ] && gs=$(sha "$GATE")
    log_attempt event=gate rc="$rc" time="$(utc)" code="$CODE" gate_sha256="$gs" \
        declaration_sha256="$(sha "$DECL")"
    echo "forcing gate: exit $rc (sha256 $gs)"
    return "$rc"
}

do_repro() {
    preconditions
    no_failed_repro
    c=$OUTD/C.npz
    [ -e "$c" ] || die "run forward C first"
    csha=$(sha "$c")
    for r in 0 3; do
        [ "$(count_records event=repro rc=$r c_npz_sha256="$csha")" = 0 ] \
            || die "a repro (exit $r) is already logged for this C file"
    done
    rc=0
    "$HPY" "$SCORER" --repro --attempts "$ATT" || rc=$?
    log_attempt event=repro rc="$rc" time="$(utc)" code="$CODE" c_npz_sha256="$csha" \
        declaration_sha256="$(sha "$DECL")"
    case $rc in
        0) echo "C reproduces the page run at $CODE; next: launch primary, launch reduced" ;;
        3) echo "C does not reproduce the page run at $CODE (logged): nothing more runs" \
                "under $ID; a v2 must disclose it" ;;
        *) echo "engineering failure (scorer exit $rc): fix the cause and rerun repro" ;;
    esac
    return "$rc"
}

do_score() {
    preconditions
    n=$(( $(count_records event=score) + 1 ))
    rc=0
    "$HPY" "$SCORER" --attempts "$ATT" --out "$REPORT" || rc=$?
    rs=missing; [ -e "$REPORT" ] && rs=$(sha "$REPORT")
    log_attempt event=score attempt="$n" rc="$rc" time="$(utc)" code="$CODE" \
        report_sha256="$rs" declaration_sha256="$(sha "$DECL")"
    case $rc in
        0) echo "scored once: $REPORT (sha256 $rs); the sealed logs and outputs may now be read" ;;
        1|2|3) echo "engineering failure (scorer exit $rc): no gated value was read; fix the" \
                    "cause, then archive and rerun the affected arms" ;;
        *) echo "scorer exit $rc" ;;
    esac
    return "$rc"
}

do_archive() {
    reason=${1:-}
    [ -n "$reason" ] || die "archive needs a reason"
    shift
    local which=("${ARMS[@]}")
    [ "$#" = 0 ] || which=("$@")
    for a in "${which[@]}"; do [ "$a" = gate ] || is_arm "$a" || die "unknown arm $a"; done
    [ ! -e "$REPORT" ] || die "$REPORT exists: $ID is scored; nothing may be rerun"
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    dst=$LOG/archive/$stamp
    mkdir -p "$dst"
    moved=()
    for a in "${which[@]}"; do
        if [ "$a" = gate ]; then fl="$GATE"; else
            fl="$(outputs "$a") $SEAL/${a}_forward.stdout.SEALED $LOG/${a}_forward.stderr.log"
        fi
        for f in $fl; do
            if [ -e "$f" ]; then mv "$f" "$dst/"; moved+=("$(basename "$f")"); fi
        done
    done
    log_attempt event=archive time="$(utc)" reason="$reason" dir="$dst" \
        arms="${which[*]}" moved="${moved[*]:-none}"
    echo "archived ${#moved[@]} files to $dst (unopened)"
}

case "${1:-}" in
    pregate) shift; do_pregate "${1:-}" ;;
    check) preconditions; echo "preconditions ok (code $CODE, fields $FIELDS, worktree $WT)" ;;
    gate) do_gate ;;
    forward) shift; do_forward "${1:-}" "${2:-1/1}" ;;
    launch) shift; do_launch "${1:-}" "${2:-3}" ;;
    repro) do_repro ;;
    score) do_score ;;
    archive) shift; do_archive "$@" ;;
    *) echo "usage: $0 pregate <code>|check|gate|forward <arm|group> [k/n]|launch <group> [n]|" \
            "repro|score|archive '<reason>' [arm ...|gate]" >&2
       exit 2 ;;
esac
