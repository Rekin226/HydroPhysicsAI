#!/usr/bin/env bash
# Run on the NEW host after the copy. Checks the tools, the GPU path and the inputs the system
# reads; prints no secret values. Exit status = number of failed checks.
#
#   bash deploy/host/verify_host.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1
fail=0
ok()   { printf '  ok    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=$((fail + 1)); }
note() { printf '  --    %s\n' "$*"; }

echo "== tools"
command -v docker >/dev/null && ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)" || bad "docker not installed"
docker compose version >/dev/null 2>&1 && ok "$(docker compose version)" || bad "docker compose plugin missing"
docker info >/dev/null 2>&1 && ok "docker daemon reachable as $(id -un)" || bad "cannot reach the docker daemon (add the user to group docker, re-login)"
command -v git >/dev/null && ok "git $(git --version | cut -d' ' -f3)" || bad "git missing"
docker buildx version >/dev/null 2>&1 && ok "buildx (per-Dockerfile .dockerignore)" || bad "docker buildx missing (needed for the compute image)"

echo "== GPU (only for the compute profile)"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    ok "$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | head -1)"
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    [ "${drv:-0}" -ge 525 ] && ok "driver >= 525 (torch cu128 wheels)" || bad "driver $drv < 525: torch cu128 will not load"
    if docker info 2>/dev/null | grep -qi nvidia; then ok "NVIDIA Container Toolkit runtime registered"
    else bad "NVIDIA Container Toolkit not registered (nvidia-ctk runtime configure --runtime=docker)"; fi
else
    note "no working nvidia-smi: the web service runs, the gpu profile does not"
fi

echo "== repository"
git rev-parse --git-dir >/dev/null 2>&1 && ok "git checkout at $(git rev-parse --short HEAD) [$(git symbolic-ref -q --short HEAD || echo detached)]" || bad "not a git checkout"
stale=$(git worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2}' | while read -r w; do [ -d "$w" ] || echo "$w"; done | wc -l)
[ "$stale" = 0 ] || note "$stale worktree record(s) point to old-host paths: git worktree repair <copied worktree dirs>, then git worktree prune"

echo "== inputs (gitignored, copied by transfer.sh)"
for p in chou-shui-data AMP_V2/data data_fetch_api results/dataprep results/twin results/releases; do
    if [ -e "$p" ]; then ok "$p ($(du -sh "$p" 2>/dev/null | cut -f1))"; else bad "$p missing"; fi
done
[ -f results/twin/twin_app.html ] && ok "promoted page results/twin/twin_app.html" || bad "results/twin/twin_app.html missing"
for f in .env deploy/host/.env; do
    if [ -f "$f" ]; then
        if grep -q '^ANTHROPIC_API_KEY=..*' "$f"; then ok "$f has ANTHROPIC_API_KEY (value not shown)"
        else note "$f has no ANTHROPIC_API_KEY value (analyst runs examples-and-form only)"; fi
        perms=$(stat -c %a "$f"); [ "$perms" = 600 ] || note "$f mode $perms: chmod 600 $f"
    else note "$f absent"; fi
done
n_local=$(find docs -name '*.local.md' 2>/dev/null | wc -l)
[ "$n_local" -gt 0 ] && ok "$n_local local-only note(s) in docs/" || note "no docs/*.local.md (connection notes for the new host go there)"

echo "== ports"
for port in 80 443 7860; do
    if ss -ltnH "sport = :$port" 2>/dev/null | grep -q .; then note "port $port already in use"
    else ok "port $port free"; fi
done

echo "== disk"
df -h "$ROOT" | tail -1 | sed 's/^/  /'
echo
[ "$fail" = 0 ] && echo "all checks passed." || echo "$fail check(s) failed."
exit "$fail"
