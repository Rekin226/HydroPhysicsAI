#!/usr/bin/env bash
# N3S-L (rank 13) registration steps for the lead: CPU only, no GPU, no forward run.
#
#   bash dataprep_staging/n3_township/run_n3s_l.sh check          tests + dry-run validate
#   bash dataprep_staging/n3_township/run_n3s_l.sh sl-panel       SL source product (reads the
#                                                                 pre-origin leveling panel)
#   bash dataprep_staging/n3_township/run_n3s_l.sh freeze <sl>    ZR/SL/Z predictions + verify
#   bash dataprep_staging/n3_township/run_n3s_l.sh availability   WRA listing titles only
#   bash dataprep_staging/n3_township/run_n3s_l.sh validate <sl-role>   dry run with slots
#   bash dataprep_staging/n3_township/run_n3s_l.sh write <sl-role> <sha>   REGISTER (the lead;
#        <sha> = the prereg_n3s_l.py sha256 the lead reviewed, pasted, never computed here)
#
# Run from the repository root (a frozen worktree is not needed: nothing here is a long job).
# Order: check -> sl-panel -> freeze <the new leveling_statlayer_history dir> -> availability
# -> validate <role> -> write <role> <sha>, all before 2026-11-30T16:00Z. <sl-role> is the
# user's choice: report_only or eligible_ds. Every step refuses to overwrite its output.
# What is read: freeze reads model output (the page forward npz), the township atlas, the
# pre-origin leveling panel through the new product (statistical_layer_history + baseline)
# and the pre-origin WRA summaries' county areas (design_time_evidence). Nothing after 2022, no
# post-origin township summary, no sealed log. availability fetches listing pages only.
set -euo pipefail
PY="${PY:-$HOME/miniforge3/envs/hydro/bin/python}"
D=dataprep_staging/n3_township
OUT=results/dataprep/reports/N3S-L
FREEZE="$OUT/freeze"
AVAIL="$OUT/availability_$(date -u +%Y%m%d).json"

cmd="${1:-}"
case "$cmd" in
  check)
    "$PY" -m pytest -q -p no:cacheprovider "$D/tests"
    "$PY" -I "$D/prereg_n3s_l.py" --validate --quiet ;;
  sl-panel)
    "$PY" -I "$D/freeze_n3s_l.py" sl-panel ;;
  freeze)
    sl="${2:?give the leveling_statlayer_history/preorigin/v<N> directory}"
    "$PY" -I "$D/freeze_n3s_l.py" freeze --sl-product "$sl" --out "$FREEZE"
    "$PY" -I "$D/freeze_n3s_l.py" verify --sl-product "$sl" --out "$FREEZE" ;;
  availability)
    "$PY" -I "$D/prereg_n3s_l.py" availability --out "$AVAIL" ;;
  validate|write)
    role="${2:?give the SL page role: report_only or eligible_ds}"
    av="$(ls -1 "$OUT"/availability_*.json | tail -n 1)"
    args=(--freeze-dir "$FREEZE" --availability "$av" --sl-role "$role" --quiet)
    if [ "$cmd" = write ]; then
      sha="${3:?paste the reviewed prereg_n3s_l.py sha256 (validate prints it)}"
      "$PY" -I "$D/prereg_n3s_l.py" --write "${args[@]}" --expect-self-sha "$sha"
    else
      "$PY" -I "$D/prereg_n3s_l.py" --validate "${args[@]}"
    fi ;;
  *)
    sed -n '2,22p' "$0"; exit 2 ;;
esac
