#!/usr/bin/env bash
# Run on the OLD host before migrating. Reports what git alone would lose: uncommitted or
# unpushed work in the main checkout and every worktree, stashes, the gitignored data that
# only exists on this disk, and jobs still running. Changes nothing.
#
#   bash deploy/host/preflight.sh [REPO_DIR]      # default: the checkout this script is in
set -uo pipefail

REPO="${1:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
REPO="$(git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2; exit}')"
cd "$REPO" || exit 1
warn=0
say()  { printf '%s\n' "$*"; }
flag() { printf '  [!] %s\n' "$*"; warn=$((warn + 1)); }

say "== main checkout: $REPO"
git fetch -q --all 2>/dev/null || flag "git fetch failed (offline?): unpushed check uses stale remotes"

say "== worktrees (uncommitted changes / commits on no remote)"
while read -r wt; do
    [ -d "$wt" ] || { flag "$wt: listed but missing (git worktree prune)"; continue; }
    dirty=$(git -C "$wt" status --porcelain --untracked-files=normal 2>/dev/null | wc -l)
    head=$(git -C "$wt" rev-parse --short HEAD 2>/dev/null)
    branch=$(git -C "$wt" symbolic-ref -q --short HEAD || echo "detached")
    local_only=$(git -C "$wt" rev-list --count HEAD --not --remotes 2>/dev/null || echo "?")
    line="$wt  [$branch @ $head]  changed/untracked files: $dirty  commits on no remote: $local_only"
    if [ "$dirty" != "0" ] || [ "$local_only" != "0" ]; then flag "$line"; else say "  ok  $line"; fi
done < <(git worktree list --porcelain | awk '/^worktree /{print $2}')

say "== local branches with commits on no remote"
while read -r b; do
    n=$(git rev-list --count "$b" --not --remotes 2>/dev/null)
    if [ "${n:-0}" != "0" ]; then flag "branch $b: $n commit(s) not on any remote"; fi
done < <(git for-each-ref --format='%(refname:short)' refs/heads)

n_stash=$(git stash list | wc -l)
[ "$n_stash" = "0" ] || flag "$n_stash stash entr(y/ies): git stash list (stashes do not travel with a push)"

say "== gitignored inputs that exist only on this disk (sizes)"
for p in .env data chou-shui-data AMP_V2/data data_fetch_api results pi_meeting; do
    [ -e "$p" ] && say "  $(du -sh "$p" 2>/dev/null | cut -f1)	$p"
done
find . -path ./.claude -prune -o -name '*.local.md' -print -o -name '*.local.html' -print \
    -o -name '*.local.json' -print 2>/dev/null | sed 's/^/  local-only: /'
for d in "$HOME"/hydro_scratch*; do [ -d "$d" ] && say "  $(du -sh "$d" | cut -f1)	$d (scratch, outside the repo)"; done

say "== running jobs (finish or stop them before copying results/)"
command -v tmux >/dev/null && tmux ls 2>/dev/null | sed 's/^/  tmux: /'
command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null | sed 's/^/  gpu: /'
command -v docker >/dev/null && docker ps --format '  docker: {{.Names}} ({{.Image}}) {{.Status}}' 2>/dev/null

say "== disk"
df -h "$REPO" | tail -1 | sed 's/^/  /'
say ""
if [ "$warn" -gt 0 ]; then
    say "$warn item(s) flagged [!]: commit and push them, or rely on transfer.sh (it copies the"
    say "whole checkout including .git, untracked files and stashes, but not the other worktrees)."
else
    say "nothing flagged."
fi
