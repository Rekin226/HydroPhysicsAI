#!/usr/bin/env bash
# I1.CHX.v1 secondary, CH-J (round 8 rank 2): the GPU fits of seeds 2027 and 2029 with the
# calibrate_flow CH-J hook (--freeze-from, --pump-layer-shares-county), the columns refitted on
# CH-J's own heads (= the out-of-fold leveling), the forward runs, the counterfactual hindcasts
# and the panel producer (chj_produce.py). PREPARED, NOT RUN: every step refuses until the hook
# is merged and results/twin_runs/chx/chj/chj_engineering.json (score_chx.ENGINEERING_KEYS,
# producer_sha256 = sha256 of dataprep_staging/ch_x/chj_produce.py; also sw_sha256 = sha256 of
# chj_sw.npz, which the jobs and the producer check) is frozen by the lead, before
# 2026-11-22T14:14:55Z.
#
#   bash dataprep_staging/ch_x/run_chj.sh worktree        clean detached worktree at the
#        engineering record's hook_commit ($HOME/HydroPhysicsAI-chj) and its launcher
#   bash dataprep_staging/ch_x/run_chj.sh q0folds SEED    Q0's own recipe on partition SEED
#        (fit + 5 folds + merge; no hook flag) -> stage3_datum_gate_q0_f<SEED>, rerun fresh
#        at the hook commit for BOTH seeds (lead decision: no archived baseline): the fold
#        parents (ruling R8.1e) and Q0's k-fold baseline
#   bash dataprep_staging/ch_x/run_chj.sh q0base SEED     that run's theta, its three-zone
#        column (Q0's recipe, uncapped) on SEED leveling folds and its forward: Q0's
#        leveling / forward baseline
#   bash dataprep_staging/ch_x/run_chj.sh parents SEED    the fold parents: Q0's fold-K theta
#        of the same partition, as theta files whose meta names the fold (chj_fold_parent)
#   bash dataprep_staging/ch_x/run_chj.sh fit SEED | fold SEED K | merge SEED | column SEED |
#        forward SEED | hindcast SEED                     one job (GPU; tmux, MPS)
#   bash dataprep_staging/ch_x/run_chj.sh all SEED        fit, folds 0-4, merge, column,
#        forward, hindcast
#   bash dataprep_staging/ch_x/run_chj.sh hindcast-q0     Q0's free run at the CHX screens (once)
#   bash dataprep_staging/ch_x/run_chj.sh produce         chj_panel.npz / chj_metrics.json /
#        chj_panel.json (CPU, after every job above, both seeds)
#   DRY=1 bash dataprep_staging/ch_x/run_chj.sh all 2027  print the commands only
#
# Order: q0folds S -> parents S and q0base S; a fold job needs its parent; column needs
# merge; forward and hindcast need column; hindcast-q0 needs q0base 2027; produce needs all.
# Under MPS, three jobs at a time (memory: twin-gpu-mps-parallel), in tmux, never while a twin
# module is being edited (code freeze):
#   nvidia-smi && nvidia-cuda-mps-control -d
#   for s in 2027 2029; do tmux new -d -s chj$s "bash dataprep_staging/ch_x/run_chj.sh \
#       q0folds $s && bash dataprep_staging/ch_x/run_chj.sh parents $s && \
#       bash dataprep_staging/ch_x/run_chj.sh all $s && \
#       bash dataprep_staging/ch_x/run_chj.sh q0base $s"; done
#   then: bash dataprep_staging/ch_x/run_chj.sh hindcast-q0 && ... produce
#   echo quit | nvidia-cuda-mps-control                  when every job is done
#
# Census: Q0's default (ruling R8.1d, user_rulings_2026-11-03): no pump-census argument is
# passed, and the script REFUSES a non-empty CHJ_FORCING_ARGS, so Yunlin is exactly Q0 and
# epoch 0 = Q0. Posterior tools are unsupported on a CH-J theta (meta.chj_hook): job()
# refuses any command naming one.
#
# Sealed outputs: every job's stdout (fitted values, scores) goes to
# $LOG/sealed/<job>.stdout.SEALED (mode 400), opened by nobody before chj_report.json exists;
# stderr goes to $LOG/logs/<job>.stderr.log and may be read to diagnose a crash. This script
# prints only exit codes, paths and sha256s. Every attempt is appended to $LOG/attempts.jsonl.
# No --data argument (as run_chx.sh): HYDROMIND_GW_DATA points at the repository's data.
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
ID=I1.CHX.v1
DECL=results/dataprep/preregistration/$ID.json
R=results/twin_runs
FWD=results/twin_forward
LOG=$R/chx/chj
ENG=$LOG/chj_engineering.json
SEAL=$LOG/sealed
ERRS=$LOG/logs
PAR=$LOG/parents
ATT=$LOG/attempts.jsonl
WT="$HOME/HydroPhysicsAI-chj"
PY=$LOG/py.sh
HPY="$HOME/miniforge3/envs/hydro/bin/python"
Q0=$R/stage3_datum_gate_q0
Q0_THETA=$Q0/stage3_theta.json
SW=$R/chx/chj_sw/chj_sw.npz               # run_chx.sh build-sw (chj_forcing.build_sw)
HOOK=hydrophysics/twin/calibrate_flow.py
PRODUCER=dataprep_staging/ch_x/chj_produce.py
CH=results/dataprep/twin_inputs/clean_heads/v1
export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# Q0's recipe (results/twin_runs/par20/jobs.sh, I1.R4Q0.v1) without its --seed
Q0_RECIPE=(--param-mode zonal --epochs 500 --n-folds 5 --fix-eta 0.5 --fix-head-extra 40
           --return-flow --l-min 1e-4 --learn-spread --spread-max-km 10
           --ic-merged-proximal --ground-elev dem --zone-boundaries 205,182,208
           --log-t-min-proximal 58 --well-datum fit --well-datum-sd 5
           --device cuda --compile-matvec --log-every 25 --fold-site-tol-m 200
           --pump-split --wells-dir $CH/wells --stations $CH/fan_stations.parquet
           --day-weighted-months)
# CH-J (chj_forcing.py docstring, CALIBRATE_FLOW_PATCH; I1.CHX.v1 chj.physics)
CHJ_FLAGS=(--zone-county-split --county-copy-sd 0.5
           --pump-layer-shares-county 0.84,0.11,0.025,0.025
           --sw-recharge $SW --sw-components both --sw-layer 0 --sw-county-split)
COLUMN_CAP=0.1                  # chx_const.CHJ_COLUMN_CAP; global (ruling R8.1f, q0_csk01)
# the forward of every Q0 / rung run (results/twin_runs/par20/jobs.sh job_forward)
FWD_FLAGS=(--scenario 'cut30:irrigation=0.7@2026-01' --horizon 120 --gain 1.0 --ic-members 1
           --update-method innovation --restart-taper-km 5 --apex-hold calibrated
           --column-heads free --column-hpc0-fast-days 365 --device cuda --compile-matvec
           --save-members none)
# tools that sample or perturb a theta (a CH-J theta's fixed keys are not parameters)
POSTERIOR_TOOLS=(hydrophysics.twin.uncertainty twin/uncertainty.py n1_gaps/identifiability
                 identifiability.py stage3_posterior)

die() { echo "REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
attempt() {   # key=value ...
    mkdir -p "$LOG"
    "$HPY" -I -c '
import json, sys
rec = dict(kv.split("=", 1) for kv in sys.argv[2:])
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")' "$ATT" "$@"
}

# ruling R8.1d: CH-J runs on Q0's default census; no forcing argument may be passed
[ -z "${CHJ_FORCING_ARGS:-}" ] || die "CHJ_FORCING_ARGS='$CHJ_FORCING_ARGS': ruling R8.1d \
(results/dataprep/rulings/user_rulings_2026-11-03.json) fixes CH-J's pumping census to Q0's \
default (CHJ_FORCING_ARGS empty, so Yunlin is exactly Q0 and epoch 0 = Q0)"

no_posterior() {   # refuse a command that names a posterior tool (meta.chj_hook)
    local a t
    for a in "$@"; do
        for t in "${POSTERIOR_TOOLS[@]}"; do
            case "$a" in *"$t"*) die "$a: posterior tools are unsupported on a CH-J theta";;
            esac
        done
    done
}

seed_ok() { case "${1:-}" in 2027|2029) ;; *) die "seed must be 2027 or 2029 (registered)";; esac; }
run_dir() { echo "$R/stage3_datum_gate_chj_s$1"; }      # score_g2's stage3_datum_gate_<suffix>
q0_dir() { echo "$R/stage3_datum_gate_q0_f$1"; }          # fresh at the hook commit

engineering_ok() {
    [ -f "$ENG" ] || die "$ENG missing: the lead writes it after the hook's PR merges"
    "$HPY" -I - "$ENG" "$DECL" "$REPO/$PRODUCER" <<'PY' || die "engineering record check failed"
import datetime as dt, hashlib, json, sys
eng, doc = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
keys = ("hook_commit", "hook_sha256", "forcing_version", "forcing_sha256", "pregate",
        "frozen_at", "producer_sha256")
miss = [k for k in keys if k not in eng]
assert not miss, f"record lacks {miss}"
utc = lambda s: dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
assert utc(eng["frozen_at"]) <= utc(doc["created"]) + dt.timedelta(days=21), "frozen late"
pg = eng["pregate"] or {}
assert pg.get("passed") and float(pg.get("max_abs_diff_m", 1)) <= 1e-6, "pre-gate not passed"
prod = hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()
assert prod == eng["producer_sha256"], "chj_produce.py differs from the record's producer"
PY
}

worktree_ok() {
    local c
    c=$("$HPY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["hook_commit"])' "$ENG")
    [ -d "$WT" ] || die "$WT missing (bash $0 worktree)"
    [ "$(git -C "$WT" rev-parse HEAD)" = "$(git -C "$WT" rev-parse "$c^{commit}")" ] \
        || die "$WT is not at the hook commit $c"
    [ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || die "$WT is dirty"
    [ "$(sha "$WT/$HOOK")" = "$("$HPY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["hook_sha256"])' "$ENG")" ] \
        || die "$WT/$HOOK sha256 differs from the engineering record"
    [ -x "$PY" ] && grep -qx "export PYTHONPATH=$WT" "$PY" || die "$PY does not pin $WT"
}

preconditions() {
    if [ "${DRY:-0}" = 1 ]; then echo "(DRY: preconditions not checked)"; return 0; fi
    [ -f "$DECL" ] || die "$DECL missing"
    "$HPY" dataprep_staging/ch_x/score_chx.py --verify-registration > /dev/null \
        || die "I1.CHX.v1 frozen code differs from its declaration"
    engineering_ok
    worktree_ok
    [ -f "$SW" ] && [ -f "${SW%.npz}.json" ] || die "$SW missing (run_chx.sh build-sw)"
    [ "$(sha "$SW")" = "$("$HPY" -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("sw_sha256"))' "$ENG")" ] \
        || die "$SW sha256 differs from the engineering record's sw_sha256 (or none)"
    nvidia-smi > /dev/null || die "nvidia-smi failed"
}

job() {   # $1 = job name, rest = the command; stdout sealed, stderr kept
    local name=$1 rc=0 t0
    shift
    no_posterior "$@"
    if [ "${DRY:-0}" = 1 ]; then          # DRY writes nothing
        if [ "$1" = chj ]; then echo "[$name] $(DRYRUN=1 "$@")"; else echo "[$name] $*"; fi
        return 0
    fi
    mkdir -p "$SEAL" "$ERRS"
    [ ! -e "$SEAL/$name.stdout.SEALED" ] || die "$SEAL/$name.stdout.SEALED exists (archive)"
    t0=$(utc)
    "$@" > "$SEAL/$name.stdout.SEALED" 2> "$ERRS/$name.stderr.log" || rc=$?
    chmod 400 "$SEAL/$name.stdout.SEALED"
    attempt event=job job="$name" rc="$rc" start="$t0" time="$(utc)" \
        worktree_head="$(git -C "$WT" rev-parse HEAD)" hook_sha256="$(sha "$WT/$HOOK")" \
        producer_sha256="$(sha "$PRODUCER")" py_sh_sha256="$(sha "$PY")" \
        forcing_args="" census="Q0 default (R8.1d)" \
        sw_sha256="$(sha "$SW")" stdout_sealed_sha256="$(sha "$SEAL/$name.stdout.SEALED")"
    echo "$name: exit $rc"
    [ "$rc" = 0 ] || die "$name failed: read $ERRS/$name.stderr.log only, archive, rerun"
}

hook_applied() {   # $1 = run dir, $2 = seed [, $3 = folds required]: the hook ran, every
    # CH-J fold K ran from Q0's fresh fold-K theta and held out
    # exactly Q0's fold-K entries and sids (chj_produce.check_hook_applied: theta meta, fold
    # files and fold thetas only; nothing is printed but OK / REFUSED)
    [ "${DRY:-0}" = 1 ] && return 0
    "$HPY" -I "$PRODUCER" check-hook --run "$1" --seed "$2" --q0-theta "$Q0_THETA" \
        --q0-fold-dir "$(q0_dir "$2")" ${3:+--require-folds "$3"} \
        || die "the CH-J hook was not applied in $1"
}

parent_of() {   # $1 = seed, $2 = fold (or 'full') -> the --freeze-from file
    if [ "$2" = full ]; then echo "$Q0_THETA"; else echo "$PAR/q0_s$1_fold$2_theta.json"; fi
}

chj() {   # $1 = seed, $2 = fold or 'full', rest = extra flags
    local s=$1 k=$2
    shift 2
    ${DRYRUN:+echo} "$PY" -m hydrophysics.twin.calibrate_flow "${Q0_RECIPE[@]}" --seed "$s" \
        "${CHJ_FLAGS[@]}" --freeze-from "$(parent_of "$s" "$k")" "$@"
}

case "${1:-}" in
    worktree)
        engineering_ok
        c=$("$HPY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["hook_commit"])' "$ENG")
        [ -d "$WT" ] || git worktree add --detach "$WT" "$c"
        mkdir -p "$LOG"
        printf '#!/usr/bin/env bash\nexport PYTHONPATH=%s\nexec %s -P "$@"\n' "$WT" "$HPY" > "$PY"
        chmod 755 "$PY"
        worktree_ok
        echo "worktree $WT at $c; launcher $PY"
        ;;
    q0folds)
        seed_ok "${2:-}"
        preconditions
        out=$(q0_dir "$2")
        job "q0_s$2_fit" "$PY" -m hydrophysics.twin.calibrate_flow "${Q0_RECIPE[@]}" \
            --seed "$2" --fit-only --out "$out"
        for k in 0 1 2 3 4; do
            job "q0_s$2_fold$k" "$PY" -m hydrophysics.twin.calibrate_flow "${Q0_RECIPE[@]}" \
                --seed "$2" --only-fold "$k" --dump-predictions --out "$out"
        done
        job "q0_s$2_merge" "$PY" -m hydrophysics.twin.merge_folds "$out"
        ;;
    q0base)
        # the fresh Q0 run's own (merged) theta, its three-zone column (Q0's recipe, no
        # cap) on SEED leveling folds, and its forward, at the hook commit, in that run's
        # directory (score_g2.metrics reads one directory)
        seed_ok "${2:-}"
        preconditions
        out=$(q0_dir "$2")
        [ "${DRY:-0}" = 1 ] || [ -f "$out/stage3_flow.csv" ] || die "$out not merged (q0folds $2)"
        job "q0_s$2_column" "$PY" -m hydrophysics.twin.calibrate_coupled \
            --theta "$out/stage3_theta.json" \
            --target leveling --configs zonal --hpc0-guard-days 365 --seed "$2" \
            --device cuda --out "$out/coupled_leveling"
        job "q0_s$2_forward" "$PY" -m hydrophysics.twin.forward --theta "$out/stage3_theta.json" \
            "${FWD_FLAGS[@]}" --vep-json "$out/coupled_leveling/vep_zonal_leveling.json" \
            --out "$FWD/datum_gate_q0_f$2"
        ;;
    parents)
        seed_ok "${2:-}"
        src=$(q0_dir "$2")
        mkdir -p "$PAR"
        for k in 0 1 2 3 4; do
            f="$src/stage3_fold$k.json"
            [ -f "$f" ] || die "$f missing (bash $0 q0folds $2 first)"
            dst=$(parent_of "$2" "$k")
            [ ! -e "$dst" ] || die "$dst exists"
            "$HPY" -I - "$f" "$Q0_THETA" "$dst" "$2" "$k" <<'PY'
import hashlib, json, sys
src, q0, dst, seed, k = sys.argv[1:6]
fold, meta = json.load(open(src)), json.load(open(q0))["meta"]
assert int(fold["seed"]) == int(seed) and int(fold["fold"]) == int(k), "fold/seed mismatch"
assert "chj_parent" not in fold, "a CH-J fold, not Q0's"
rec = {"theta": fold["record"]["theta"],
       "meta": {"pump_layer": meta["pump_layer"],
                "chj_fold_parent": {
                    "fold": int(k), "seed": int(seed), "n_folds": int(fold["n_folds"]),
                    "source": src,
                    "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
                    "rule": "CH-J fold parent: Q0's own fold-K fit of the same partition "
                            "(ruling R8.1e)"}}}
with open(dst, "x") as f:
    json.dump(rec, f, indent=1)
PY
            echo "parent $dst sha256 $(sha "$dst")"
        done
        ;;
    hindcast-q0)
        preconditions
        job "q0_hindcast" "$PY" "$PRODUCER" hindcast-q0 --device cuda
        ;;
    produce)
        preconditions
        for s in 2027 2029; do hook_applied "$(run_dir "$s")" "$s" 5; done
        job "chj_produce" "$PY" "$PRODUCER" assemble
        ;;
    fit|fold|merge|column|forward|hindcast|all)
        step=$1
        seed_ok "${2:-}"
        s=$2
        out=$(run_dir "$s")
        preconditions
        do_fit() { job "chj_s${s}_fit" chj "$s" full --fit-only --out "$out"
                   hook_applied "$out" "$s"; }
        do_fold() {
            [ "${DRY:-0}" = 1 ] || [ -f "$(parent_of "$s" "$1")" ] || die "no fold parent (bash $0 parents $s)"
            job "chj_s${s}_fold$1" chj "$s" "$1" --only-fold "$1" --dump-predictions --out "$out"
        }
        do_merge() { job "chj_s${s}_merge" "$PY" -m hydrophysics.twin.merge_folds "$out"
                     hook_applied "$out" "$s" 5; }
        do_column() {
            # both county columns (the six-zone county-split column follows the theta) on
            # CH-J's own heads, Ske/Skv <= 0.1 in every zone (I1.CHX.v1 chj.physics.columns;
            # ruling R8.1f: the cap is global, the Yunlin zones are capped too); its
            # r2_outoffold is the out-of-fold leveling the no-harm gates read
            job "chj_s${s}_column" "$PY" -m hydrophysics.twin.calibrate_coupled \
                --theta "$out/stage3_theta.json" --target leveling --configs zonal \
                --hpc0-guard-days 365 --seed "$s" --device cuda --ske-skv-max "$COLUMN_CAP" \
                --out "$out/coupled_leveling"
        }
        do_forward() {
            job "chj_s${s}_forward" "$PY" -m hydrophysics.twin.forward \
                --theta "$out/stage3_theta.json" "${FWD_FLAGS[@]}" \
                --vep-json "$out/coupled_leveling/vep_zonal_leveling.json" \
                --out "$FWD/datum_gate_chj_s$s"
        }
        do_hindcast() {
            # arms A and B (Changhua at its 2012 pattern) at the CHX screens (model only)
            job "chj_s${s}_hindcast" "$PY" "$PRODUCER" hindcast --seed "$s" --device cuda
        }
        case "$step" in
            fit) do_fit ;;
            fold) [ -n "${3:-}" ] || die "fold needs K"; do_fold "$3" ;;
            merge) do_merge ;;
            column) do_column ;;
            forward) do_forward ;;
            hindcast) do_hindcast ;;
            all)
                do_fit
                for k in 0 1 2 3 4; do do_fold "$k"; done
                do_merge
                do_column
                do_forward
                do_hindcast
                ;;
        esac
        ;;
    *)
        sed -n '2,52p' "$0"
        exit 2
        ;;
esac
