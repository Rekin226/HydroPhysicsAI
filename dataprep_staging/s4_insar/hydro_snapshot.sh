#!/usr/bin/env bash
# WP S4a acceptance S4a-1: snapshot of the `hydro` conda env, to show the LiCSBAS env setup
# never touched it. Same format as the before/after snapshots of the setup step:
#   <sha256 of the conda-meta file listing>  -
#   <number of conda-meta entries>
#   <mtime of conda-meta/>
#   <mtime of the newest conda-meta entry>
# plus `pip list --format=freeze | LC_ALL=C sort` of the hydro interpreter (read-only).
# Never activates or modifies `hydro`.
# Usage: dataprep_staging/s4_insar/hydro_snapshot.sh <tag>   (writes env/hydro_*_<tag>.txt)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/results/dataprep/insar_tools/s4a_run/env"
H="${HYDRO_ENV:-$HOME/miniforge3/envs/hydro}"
TAG="${1:?tag}"
mkdir -p "$OUT"
{
  ls "$H/conda-meta" | sha256sum
  ls "$H/conda-meta" | wc -l
  stat -c %Y "$H/conda-meta"
  stat -c %Y "$H/conda-meta/$(ls -t "$H/conda-meta" | head -1)"
} > "$OUT/hydro_conda_meta_$TAG.txt"
"$H/bin/python" -I -m pip list --format=freeze --disable-pip-version-check 2>/dev/null \
  | LC_ALL=C sort > "$OUT/hydro_pip_freeze_$TAG.txt"
echo "$OUT/hydro_conda_meta_$TAG.txt $OUT/hydro_pip_freeze_$TAG.txt"
