#!/usr/bin/env bash
# Copy the complete project (code + .git + every gitignored input) from the OLD host.
#
#   bash deploy/host/transfer.sh --to USER@NEWHOST:/srv/HydroPhysicsAI   # rsync over SSH
#   bash deploy/host/transfer.sh --out /mnt/usb/hydro_migration          # tar archive + sha256
#
# Options:
#   --lite             skip the large regenerable run outputs (LITE_SKIP below, about 24 GB)
#   --with-worktrees   also copy the other worktrees (~/HydroPhysicsAI-* etc.) next to the target,
#                      with their gitignored files (results, *.local.md, local data tables)
#   --with-scratch     also copy ~/hydro_scratch* (long-run outputs kept outside the repo)
#   --dry-run          rsync only: list what would be sent
#
# Always copied: the whole main checkout including .git (all branches, stashes, history),
# untracked files, .env, *.local.md, data/, chou-shui-data/, AMP_V2/data/, data_fetch_api/,
# results/. Never copied: Python/ruff/pytest caches, *.egg-info, nested .claude/worktrees.
# Without --with-worktrees the other worktrees stay behind: their commits travel in .git, their
# gitignored files do not (preflight.sh shows how much each one holds).
# rsync can be re-run; it resumes and sends only what changed.
set -euo pipefail

REPO="$(git -C "$(dirname "$0")" worktree list --porcelain | awk '/^worktree /{print $2; exit}')"
LITE_SKIP=(results/twin_forward results/surrogate results/twin_runs)
EXCLUDES=(--exclude=__pycache__/ --exclude=.pytest_cache/ --exclude=.ruff_cache/
          --exclude=.mypy_cache/ --exclude='*.egg-info/' --exclude=/.claude/worktrees/)
to="" out="" lite=0 scratch=0 worktrees=0 dry=""
while [ $# -gt 0 ]; do
    case "$1" in
        --to) to="${2%/}"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --lite) lite=1; shift ;;
        --with-worktrees) worktrees=1; shift ;;
        --with-scratch) scratch=1; shift ;;
        --dry-run) dry="--dry-run"; shift ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
[ -n "$to" ] || [ -n "$out" ] || { echo "need --to or --out (see --help)" >&2; exit 2; }
if [ "$lite" = 1 ]; then for p in "${LITE_SKIP[@]}"; do EXCLUDES+=("--exclude=/$p/"); done; fi
name="$(basename "$REPO")"

# the other worktrees of this repository (the main checkout and nested .claude/worktrees excluded)
other_worktrees() {
    git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2}' | while read -r wt; do
        [ "$wt" = "$REPO" ] && continue
        case "$wt" in "$REPO"/*) continue ;; esac
        [ -d "$wt" ] && echo "$wt"
    done
}

if [ -n "$to" ]; then
    # USER@HOST:/path (remote) or /path (local disk); siblings land next to the target directory
    if [[ "$to" =~ ^[^/]+: ]]; then prefix="${to%%:*}:"; dest="${to#*:}"; else prefix=""; dest="$to"; fi
    parent="$(dirname "$dest")"
    # rsync creates only the last path component: make the parent first (remote: via rsync-path)
    if [ -n "$prefix" ]; then mk=(--rsync-path="mkdir -p $(printf '%q' "$parent") && rsync")
    else mk=(); [ -n "$dry" ] || mkdir -p "$parent"; fi
    send() { rsync -aH --partial --info=progress2 $dry "${mk[@]}" "${EXCLUDES[@]}" "$@"; }
    # trailing slash: copy the contents of the checkout into the target directory
    send "$REPO/" "$to/"
    if [ "$worktrees" = 1 ]; then
        while read -r wt; do
            echo "worktree $(basename "$wt")"
            send "$wt/" "$prefix$parent/$(basename "$wt")/"
        done < <(other_worktrees)
    fi
    if [ "$scratch" = 1 ]; then
        for d in "$HOME"/hydro_scratch*; do [ -d "$d" ] && send "$d/" "$prefix$parent/$(basename "$d")/"; done
    fi
    echo "done. On the new host, in $dest:"
    if [ "$worktrees" = 1 ]; then
        echo "  git worktree repair$(other_worktrees | while read -r wt; do printf ' %s' "$parent/$(basename "$wt")"; done)"
    fi
    echo "  git worktree prune && bash deploy/host/verify_host.sh"
    exit 0
fi

mkdir -p "$out"
stamp="$(date +%Y%m%d)"
archive="$out/${name}_${stamp}.tar"
tar_excl=(--exclude=__pycache__ --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.mypy_cache
          --exclude='*.egg-info' --exclude="$name/.claude/worktrees")
if [ "$lite" = 1 ]; then for p in "${LITE_SKIP[@]}"; do tar_excl+=("--exclude=$name/$p"); done; fi
if command -v zstd >/dev/null; then comp=(-I "zstd -T0 -3"); ext=zst; else comp=(-z); ext=gz; fi
archive="$archive.$ext"
echo "writing $archive (this takes a while for the full results/ tree)"
tar -C "$(dirname "$REPO")" "${tar_excl[@]}" "${comp[@]}" -cf "$archive" "$name"
if [ "$worktrees" = 1 ]; then
    members=()
    while read -r wt; do members+=(-C "$(dirname "$wt")" "$(basename "$wt")"); done < <(other_worktrees)
    if [ "${#members[@]}" -gt 0 ]; then
        echo "writing $out/worktrees_${stamp}.tar.$ext ($(( ${#members[@]} / 3 )) worktrees)"
        tar "${tar_excl[@]}" "${comp[@]}" -cf "$out/worktrees_${stamp}.tar.$ext" "${members[@]}"
    fi
fi
if [ "$scratch" = 1 ]; then
    for d in "$HOME"/hydro_scratch*; do
        [ -d "$d" ] && tar -C "$HOME" "${comp[@]}" -cf "$out/$(basename "$d")_${stamp}.tar.$ext" "$(basename "$d")"
    done
fi
(cd "$out" && sha256sum ./*_"${stamp}".tar.* > "SHA256SUMS_${stamp}")
echo "done: $(du -sh "$out" | cut -f1) in $out; verify on the new host with: sha256sum -c SHA256SUMS_${stamp}"
if [ "$worktrees" = 1 ]; then
    echo "extract worktrees_${stamp}.tar.$ext next to $name, then in $name:"
    echo "  git worktree repair$(other_worktrees | while read -r wt; do printf ' ../%s' "$(basename "$wt")"; done)"
fi
