#!/usr/bin/env bash
# N3.piles_prospective.v1: the pre-registration runs (CPU only; no GPU, no forward run).
#
#   bash dataprep_staging/n3_piles/run_n3.sh check        preconditions only (hashes, slots)
#   bash dataprep_staging/n3_piles/run_n3.sh mapping      mapping.json + model_side.json
#                                                          (metadata + frozen model output)
#   bash dataprep_staging/n3_piles/run_n3.sh g0           G0 synthetic gate (~2 min CPU),
#                                                          model-only, BEFORE registration
#   bash dataprep_staging/n3_piles/run_n3.sh competitor   leveling-cell rate product
#                                                          (pre-origin leveling, 'baseline')
#   bash dataprep_staging/n3_piles/run_n3.sh validate     dry run of the registration
#   bash dataprep_staging/n3_piles/run_n3.sh archive "<reason>"
#                                    engineering retry of G0: move result.json aside (never
#                                    after N3 is registered)
#
# Run from the MAIN repository root (as x5_class_share/run_class_share.sh), never from a
# linked worktree: results/ is gitignored and lives only in the main checkout (the forward
# archives, the outputs under $OUT, the leveling product). The code is bound instead by the
# registration's frozen_code sha256 check (prereg_n3.py refuses a changed file), so keep the
# n3_piles files untouched between 'mapping' and '--write'. CPU only, minutes; tmux optional.
# Nothing here reads a pile month, a window head or anything after 2022-12 except the frozen
# model's own projection (model output); 'competitor' reads pre-origin leveling (baseline use)
# and logs that read as a look. Registration itself is the lead's:
#   python -I dataprep_staging/n3_piles/prereg_n3.py --write ... --expect-self-sha <sha>
#   (G0 failed: also --g0-fail-mode level-only --ruling <user ruling record> --ruling-id <id>)
set -euo pipefail
X=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$X/../.." && pwd)
cd "$REPO"
die() { echo "REFUSED: $*" >&2; exit 1; }
[ "$(git rev-parse --git-dir)" = "$(git rev-parse --git-common-dir)" ] \
  || die "run from the main repository root, not a linked worktree (results/ is not there)"
[ -f results/twin_forward/datum_gate_innov.npz ] \
  || die "results/twin_forward/datum_gate_innov.npz missing: run from the main repository root"
HPY="$HOME/miniforge3/envs/hydro/bin/python"
OUT=${OUT:-results/dataprep/reports/N3}
MAP=$OUT/mapping
G0=$OUT/g0
ID=N3.piles_prospective.v1
REG=results/dataprep/preregistration/$ID.json

sha() { sha256sum "$1" | cut -d' ' -f1; }
latest_competitor() { ls -d results/dataprep/n3_leveling_cell_rate/v* 2>/dev/null | sort -V | tail -1; }

cmd=${1:-check}
case "$cmd" in
  check)
    echo "forward  $(sha results/twin_forward/datum_gate_innov.npz)"
    echo "column   $(sha results/twin_runs/stage3_datum_gate/coupled_leveling/vep_zonal_leveling.json)"
    for f in "$X"/*.py; do echo "code     $(sha "$f")  ${f#"$REPO"/}"; done
    [ -f "$MAP/model_side.json" ] && echo "mapping  $(sha "$MAP/model_side.json")" || echo "mapping  missing"
    [ -f "$G0/result.json" ] && echo "g0       $(sha "$G0/result.json")" || echo "g0       missing"
    echo "competitor $(latest_competitor || true)"
    [ -f "$REG" ] && echo "registered: $REG" || echo "not registered"
    ;;
  mapping)
    [ -f "$REG" ] && die "$ID is registered; the mapping is frozen"
    [ -f "$MAP/model_side.json" ] && die "$MAP exists; archive it first"
    "$HPY" -I "$X/build_mapping.py" --out "$MAP"
    ;;
  g0)
    [ -f "$REG" ] && die "$ID is registered; G0 is frozen"
    [ -f "$MAP/model_side.json" ] || die "run 'mapping' first"
    "$HPY" -I "$X/g0_n3.py" run --model-side "$MAP/model_side.json" --out "$G0"
    ;;
  competitor)
    [ -f "$REG" ] && die "$ID is registered; the competitor is frozen"
    [ -f "$MAP/mapping.json" ] || die "run 'mapping' first"
    "$HPY" -I "$X/export_leveling_competitor.py" --mapping "$MAP/mapping.json"
    ;;
  validate)
    C=$(latest_competitor || true)
    "$HPY" -I "$X/prereg_n3.py" --validate --quiet --mapping-dir "$MAP" \
        --g0-result "$G0/result.json" ${C:+--leveling-product "$C"} "${@:2}"
    ;;
  archive)
    [ -f "$REG" ] && die "$ID is registered; nothing is archived after registration"
    reason=${2:?give a reason}
    d="$OUT/archive/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$d"
    [ -d "$G0" ] && mv "$G0" "$d/"
    echo "$reason" > "$d/REASON.txt"
    echo "archived to $d"
    ;;
  *)
    die "unknown subcommand $cmd"
    ;;
esac
