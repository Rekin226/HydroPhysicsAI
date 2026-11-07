"""Private deployment bundles: one directory per page, switched atomically when every gate passes.

    python -m hydrophysics_analyst bundle --page results/twin/twin_app_20261107c.local.html \\
        --root /srv/twin/bundles --parity results/analyst/parity/parity.json \\
        --eval results/analyst/eval/<date>/summary.json [--run-claims-tests]

Layout (spec §9)::

    <root>/<page_sha>/page.local.html   the (private) build, byte for byte
    <root>/<page_sha>/claims_v1.json    the ledger it binds to
    <root>/<page_sha>/bundle.json       shas, gate results, wording_valid_until, built_utc
    <root>/current.json                 {"page_sha256": ..., "bundle": "<page_sha>"}

Gates, all required: deploy-mode binding (forward sha, template sha, strict ledger, no reads of
individual observations); a payload that is offline and a research scenario; a parity report for
THIS page with zero mismatches; an eval summary for THIS page meeting the selection rule
(accuracy >= 97 %, zero unsafe); optionally the repo's claims tests. A failed gate leaves
``current.json`` untouched (the release.py rule: failed checks never replace the current pointer).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__
from . import binding as BD
from .page import LEDGER, BindingError, PageBundle


class GateFailed(RuntimeError):
    pass


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def wording_valid_until(b: PageBundle) -> str | None:
    """The earliest scoring date of a claim still pending (D): after it the wording may be stale."""
    from hydrophysics.twin import claims as C
    st = {c["id"]: c["status_current"] for c in b.ledger["claims"]}
    dates = [s["not_before"] for cid, s in BD.scoring().items() if "D" in C.codes(st.get(cid, ""))]
    return min(dates) if dates else None


def gates(b: PageBundle, parity: Path | None, evalsum: Path | None, *, run_claims_tests: bool,
          require_parity: bool = True, require_eval: bool = True) -> dict:
    res: dict = {"binding": "pass (deploy mode)"}
    import inspect

    from hydrophysics.twin import release as R
    if "allow_private" in inspect.signature(R.check_payload).parameters:   # after spec D5 patch 02
        errs = R.check_payload(b.payload, allow_private=True)
        if errs:
            raise GateFailed("release.check_payload: " + "; ".join(errs))
        res["release_checks"] = "pass (allow_private)"
    rel = b.payload.get("release") or {}
    if not rel.get("offline", True) or rel.get("status", "research-scenario") != "research-scenario":
        raise GateFailed("payload is not an offline research scenario")
    res["payload"] = "pass"
    if require_parity:
        if not parity or not parity.exists():
            raise GateFailed("no parity report")
        p = json.loads(parity.read_text(encoding="utf-8"))
        if p.get("page_sha256") != b.page_sha256:
            raise GateFailed("the parity report is for another page")
        if p.get("n_mismatch", 1) != 0 or p.get("n_cases", 0) < 100:
            raise GateFailed(f"parity: {p.get('n_mismatch')} mismatches in {p.get('n_cases')} cases")
        res["parity"] = {"cases": p["n_cases"], "sha": p.get("sha")}
    if require_eval:
        if not evalsum or not evalsum.exists():
            raise GateFailed("no eval summary")
        e = json.loads(evalsum.read_text(encoding="utf-8"))
        if e.get("page_sha256") != b.page_sha256:
            raise GateFailed("the eval summary is for another page")
        arm = e.get("decision")
        s = (e.get("summaries") or {}).get(arm) or {}
        if not (s.get("accuracy", 0) >= 97 and s.get("unsafe", 1) == 0):
            raise GateFailed(f"eval: arm {arm} accuracy {s.get('accuracy')} unsafe {s.get('unsafe')}")
        res["eval"] = {"arm": arm, "accuracy": s["accuracy"], "unsafe": s["unsafe"]}
    if run_claims_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_twin_claims.py"],
                           capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
        if r.returncode != 0:
            raise GateFailed("claims tests failed:\n" + r.stdout[-800:])
        res["claims_tests"] = "pass"
    return res


def build_bundle(page: Path, root: Path, *, ledger: Path = LEDGER, parity: Path | None = None,
                 evalsum: Path | None = None, run_claims_tests: bool = False,
                 require_parity: bool = True, require_eval: bool = True) -> dict:
    try:
        b = PageBundle.load(page, ledger, mode="deploy")
    except BindingError as e:
        raise GateFailed(f"binding: {e}") from e
    g = gates(b, parity, evalsum, run_claims_tests=run_claims_tests,
              require_parity=require_parity, require_eval=require_eval)
    d = root / b.page_sha256
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(page, d / "page.local.html")
    shutil.copyfile(ledger, d / "claims_v1.json")
    meta = {"schema": "twin-analyst-bundle/1", "page_sha256": b.page_sha256,
            "ledger_sha256": b.ledger_sha256, "strings_sha256": b.strings_sha256,
            "release": b.payload.get("release"), "public": bool((b.payload.get("release") or {}).get("public")),
            "gates": g, "wording_valid_until": wording_valid_until(b), "code_version": __version__,
            "built_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic_json(d / "bundle.json", meta)
    atomic_json(root / "current.json", {"page_sha256": b.page_sha256, "bundle": b.page_sha256})
    return meta


def current(root: Path) -> Path:
    cur = json.loads((root / "current.json").read_text(encoding="utf-8"))
    return root / cur["bundle"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--parity", type=Path)
    ap.add_argument("--eval", type=Path)
    ap.add_argument("--run-claims-tests", action="store_true")
    a = ap.parse_args(argv)
    try:
        meta = build_bundle(a.page, a.root, ledger=a.ledger, parity=a.parity, evalsum=a.eval,
                            run_claims_tests=a.run_claims_tests)
    except GateFailed as e:
        print(f"bundle NOT published (current.json unchanged): {e}", file=sys.stderr)
        return 1
    print(f"bundle {meta['page_sha256'][:12]} published; wording valid until {meta['wording_valid_until']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
