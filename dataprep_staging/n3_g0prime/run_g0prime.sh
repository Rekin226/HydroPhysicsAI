#!/usr/bin/env bash
# N3.G0prime.v1 (model-only declaration) and N3Gprime.piles_conversion.v1: the lead's runs.
# CPU only: no GPU, no forward run, no observation.
#
#   bash dataprep_staging/n3_g0prime/run_g0prime.sh check         hashes and slots only
#   bash dataprep_staging/n3_g0prime/run_g0prime.sh declare-dry [--gate all|identifiable]
#                                    prereg_g0prime.py --validate
#   bash dataprep_staging/n3_g0prime/run_g0prime.sh g0prime       G0' once (~20 min CPU; tmux),
#                                    refuses until N3.G0prime.v1 is registered and intact
#   bash dataprep_staging/n3_g0prime/run_g0prime.sh rings --forward F --sites S
#                                    the ring attenuation N note (CPU, minutes)
#   bash dataprep_staging/n3_g0prime/run_g0prime.sh n3g-dry       prereg_n3g.py --validate
#   bash dataprep_staging/n3_g0prime/run_g0prime.sh archive "<reason>"
#                                    engineering retry: move g0prime/ aside, unopened, logged
#   bash dataprep_staging/n3_g0prime/run_g0prime.sh archive-rings "<reason>"
#                                    ring-note retry (e.g. the wrong forward archive): move
#                                    ring_attenuation/ aside, logged; never touches g0prime/
#
# Registration itself is the lead's (never from this script):
#   python -I dataprep_staging/n3_g0prime/prereg_g0prime.py --write --gate <all|identifiable> \
#       [--gate-basis '<judge/lead decision>'] --expect-self-sha <sha>
#   (after G0' passes, before 2026-11-30T16:00Z)
#   python -I dataprep_staging/n3_g0prime/prereg_n3g.py --write --expect-self-sha <sha>
#
# Where: from the MAIN repository root (results/ lives only there, as run_n3.sh). To run the
# code of a frozen worktree against the main results/ tree instead, set WT to that worktree:
#   WT=<frozen worktree> bash <main root>/dataprep_staging/n3_g0prime/run_g0prime.sh g0prime
# (the scripts are launched from $WT with DATAPREP_REPO=<main root>; g0prime.py refuses unless
# the code it runs hashes as the declaration's frozen_code). Long run in tmux:
#   tmux new -s g0prime 'bash dataprep_staging/n3_g0prime/run_g0prime.sh g0prime; bash'
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
die() { echo "REFUSED: $*" >&2; exit 1; }
[ "$(git rev-parse --git-dir)" = "$(git rev-parse --git-common-dir)" ] \
  || die "run from the main repository root, not a linked worktree (results/ is not there)"
HPY="$HOME/miniforge3/envs/hydro/bin/python"
CODE=${WT:-$REPO}
C=$CODE/dataprep_staging/n3_g0prime
export DATAPREP_REPO=$REPO
OUT=results/dataprep/reports/N3.G0prime.v1
MS=results/dataprep/reports/N3/mapping/model_side.json
DECL=results/dataprep/preregistration/N3.G0prime.v1.json
ATT=$OUT/attempts.jsonl
sha() { sha256sum "$1" | cut -d' ' -f1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log_attempt() {
    mkdir -p "$OUT"
    "$HPY" -I -c '
import json, sys
rec = dict(kv.split("=", 1) for kv in sys.argv[2:])
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")' "$ATT" "$@"
}

cmd=${1:-check}
case "$cmd" in
  check)
    echo "code root $CODE"
    for f in "$C"/*.py "$C"/run_g0prime.sh; do echo "code     $(sha "$f")  ${f#"$CODE"/}"; done
    [ -f "$MS" ] && echo "model_side $(sha "$MS")" || echo "model_side missing"
    [ -f "$DECL" ] && echo "declared: $DECL $(sha "$DECL")" || echo "not declared"
    [ -f "$OUT/g0prime/result.json" ] && echo "g0prime  $(sha "$OUT/g0prime/result.json")" \
      || echo "g0prime  not run"
    ;;
  declare-dry)
    shift
    "$HPY" -I "$C/prereg_g0prime.py" --validate --quiet "$@"
    ;;
  g0prime)
    [ -f "$DECL" ] || die "declare N3.G0prime.v1 first (prereg_g0prime.py --write)"
    [ -f "$OUT/g0prime/result.json" ] && die "$OUT/g0prime/result.json exists: G0' runs once"
    t0=$(utc)
    rc=0
    "$HPY" -I "$C/g0prime.py" run --model-side "$MS" --out "$OUT/g0prime" || rc=$?
    rs=missing
    [ -f "$OUT/g0prime/result.json" ] && rs=$(sha "$OUT/g0prime/result.json")
    log_attempt event=g0prime start="$t0" time="$(utc)" rc="$rc" result_sha256="$rs" \
      declaration_sha256="$(sha "$DECL")" code_root="${WT:+frozen worktree}" \
      g0prime_py_sha256="$(sha "$C/g0prime.py")" g0p_lib_sha256="$(sha "$C/g0p_lib.py")"
    [ "$rc" = 0 ] || die "G0' exit $rc: fix the engineering cause, 'archive' with a reason, rerun"
    ;;
  rings)
    [ -f "$DECL" ] || die "declare N3.G0prime.v1 first (prereg_g0prime.py --write)"
    shift
    rc=0
    "$HPY" -I "$C/ring_attenuation.py" "$@" --out "$OUT/ring_attenuation" || rc=$?
    rs=missing
    [ -f "$OUT/ring_attenuation/result.json" ] && rs=$(sha "$OUT/ring_attenuation/result.json")
    log_attempt event=rings time="$(utc)" rc="$rc" result_sha256="$rs" \
      declaration_sha256="$(sha "$DECL")" args="$*"
    [ "$rc" = 0 ] || die "ring note exit $rc"
    ;;
  n3g-dry)
    "$HPY" -I "$C/prereg_n3g.py" --validate --quiet
    ;;
  archive)
    reason=${2:?give a reason}
    [ -f results/dataprep/preregistration/N3Gprime.piles_conversion.v1.json ] \
      && die "N3G is registered; G0' is frozen"
    if [ -f "$OUT/g0prime/result.json" ] && [ "${FORCE_COMPLETED:-0}" != 1 ]; then
      die "a completed G0' result is final; archive it only for a proven engineering bug" \
          "(FORCE_COMPLETED=1, reason logged and listed by prereg_n3g.py)"
    fi
    d="$OUT/archive/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$d"
    moved=none
    [ -d "$OUT/g0prime" ] && mv "$OUT/g0prime" "$d/" && moved=g0prime
    echo "$reason" > "$d/REASON.txt"
    log_attempt event=archive time="$(utc)" reason="$reason" dir="$d" moved="$moved"
    echo "archived to $d (unopened)"
    ;;
  archive-rings)
    reason=${2:?give a reason}
    [ -d "$OUT/ring_attenuation" ] || die "no ring_attenuation/ to archive"
    d="$OUT/archive_rings/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$d"
    rs=missing
    [ -f "$OUT/ring_attenuation/result.json" ] && rs=$(sha "$OUT/ring_attenuation/result.json")
    mv "$OUT/ring_attenuation" "$d/"
    echo "$reason" > "$d/REASON.txt"
    log_attempt event=archive_rings time="$(utc)" reason="$reason" dir="$d" \
      moved=ring_attenuation result_sha256="$rs"
    echo "archived to $d (g0prime/ untouched)"
    ;;
  *)
    die "unknown subcommand $cmd"
    ;;
esac
