#!/usr/bin/env bash
# Apply the two approved hydrophysics/ changes for the analyst (spec D5, ruling 2026-11-01):
#   01  template.html: URL-hash keys sel (township), tab, mi (Model internals) — deep links
#   02  release.py:    check_payload(allow_private=True) for the private bundle path only
# Both change code_sha256 (release.provenance hashes all of hydrophysics/), so run this only when
# NO twin job is running (project rule: never edit twin modules while GPU jobs run).
# After applying: update ledger page.template_sha256 (and page.html when you rebuild), rebuild
# the page, run browser_check, and run pytest tests/test_viewer_app.py tests/test_twin_claims.py.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if pgrep -af "hydrophysics\.twin\." >/dev/null; then
  echo "refusing: a hydrophysics.twin job is running:"; pgrep -af "hydrophysics\.twin\." | cut -c1-120
  exit 1
fi
here=docs/analyst/d5
git apply --check "$here/01-template-hash-keys.patch" "$here/02-release-allow-private.patch"
git apply "$here/01-template-hash-keys.patch" "$here/02-release-allow-private.patch"
new=$(sha256sum hydrophysics/twin/app/template.html | cut -d' ' -f1)
echo "applied. New template sha256: $new"
echo "next: set claims_v1.json page.template_sha256 to it (your ruling), rebuild the page,"
echo "      python -m hydrophysics.twin.browser_check --page <page> --out results/twin/browser_acceptance"
echo "      pytest -q tests/test_viewer_app.py tests/test_twin_claims.py tests/test_analyst_*.py"
