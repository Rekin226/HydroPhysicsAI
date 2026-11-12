#!/usr/bin/env bash
# Copy the complete project (code + .git + every gitignored input) from the OLD host.
#
#   bash deploy/host/transfer.sh --to USER@NEWHOST:/srv/HydroPhysicsAI   # rsync over SSH
#   bash deploy/host/transfer.sh --out /mnt/usb/hydro_migration          # tar archive + sha256
#
# Options:
#   --lite            skip the large regenerable run outputs (LITE_SKIP below, about 24 GB)
#   --with-scratch    also copy ~/hydro_scratch* (long-run outputs kept outside the repo)
#   --dry-run         rsync only: list what would be sent
#
# Always copied: the whole main checkout including .git (all branches, stashes, history),
# untracked files, .env, *.local.md, data/, chou-shui-data/, AMP_V2/data/, data_fetch_api/,
# results/. Never copied: Python/ruff/pytest caches, *.egg-info, nested .claude/worktrees.
# Other worktrees (~/HydroPhysicsAI-*) are not copied: push their branches first (preflight.sh
# lists them). rsync can be re-run; it resumes and sends only what changed.
set -euo pipefail

REPO="$(git -C "$(dirname "$0")" worktree list --porcelain | awk '/^worktree /{print $2; exit}')"
LITE_SKIP=(results/twin_forward results/surrogate results/twin_runs)
EXCLUDES=(--exclude=__pycache__/ --exclude=.pytest_cache/ --exclude=.ruff_cache/
          --exclude=.mypy_cache/ --exclude='*.egg-info/' --exclude=/.claude/worktrees/)
to="" out="" lite=0 scratch=0 dry=""
while [ $# -gt 0 ]; do
    case "$1" in
        --to) to="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --lite) lite=1; shift ;;
        --with-scratch) scratch=1; shift ;;
        --dry-run) dry="--dry-run"; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
[ -n "$to" ] || [ -n "$out" ] || { echo "need --to or --out (see --help)" >&2; exit 2; }
if [ "$lite" = 1 ]; then for p in "${LITE_SKIP[@]}"; do EXCLUDES+=("--exclude=/$p/"); done; fi
name="$(basename "$REPO")"

if [ -n "$to" ]; then
    # trailing slash: copy the contents of the checkout into the target directory
    rsync -aH --partial --info=progress2 $dry "${EXCLUDES[@]}" "$REPO/" "$to/"
    if [ "$scratch" = 1 ]; then
        dest_host="${to%%:*}"; dest_dir="$(dirname "${to#*:}")"
        for d in "$HOME"/hydro_scratch*; do
            [ -d "$d" ] && rsync -aH --partial --info=progress2 $dry "$d" "$dest_host:$dest_dir/"
        done
    fi
    echo "done. On the new host: cd <target> && git worktree prune && bash deploy/host/verify_host.sh"
    exit 0
fi

mkdir -p "$out"
stamp="$(date +%Y%m%d)"
archive="$out/${name}_${stamp}.tar"
tar_excl=(--exclude=__pycache__ --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.mypy_cache
          --exclude='*.egg-info' --exclude="$name/.claude/worktrees")
if [ "$lite" = 1 ]; then for p in "${LITE_SKIP[@]}"; do tar_excl+=("--exclude=$name/$p"); done; fi
if command -v zstd >/dev/null; then comp=(-I "zstd -T0 -3"); archive="$archive.zst"
else comp=(-z); archive="$archive.gz"; fi
echo "writing $archive (this takes a while for the full results/ tree)"
tar -C "$(dirname "$REPO")" "${tar_excl[@]}" "${comp[@]}" -cf "$archive" "$name"
if [ "$scratch" = 1 ]; then
    for d in "$HOME"/hydro_scratch*; do
        [ -d "$d" ] && tar -C "$HOME" "${comp[@]}" -cf "$out/$(basename "$d")_${stamp}.tar.${archive##*.}" "$(basename "$d")"
    done
fi
(cd "$out" && sha256sum ./*_"${stamp}".tar.* > "SHA256SUMS_${stamp}")
echo "done: $(du -sh "$out" | cut -f1) in $out; verify on the new host with: sha256sum -c SHA256SUMS_${stamp}"
