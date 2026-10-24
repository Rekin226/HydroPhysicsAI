"""Shared helpers for the N1 gap analyses of 2026-10-24 (model outputs only).

Every analysis in this package reads saved model outputs (``results/twin_forward/*.npz``,
the decision page's own payload, run-directory JSON) and never an observation value.
Nothing here scores, registers or records a look at a preregistered test, and every
writer refuses to overwrite an existing file.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, REPORTS_DIR, rel_to_repo  # noqa: E402

OUT_ROOT = REPORTS_DIR / "N1_gaps_2026-10-24"
FWD = REPO / "results" / "twin_forward"
RUNS = REPO / "results" / "twin_runs"
PAGE = REPO / "results" / "twin" / "twin_app.html"
CLASSES = ("irrigation", "aquaculture", "livestock", "domestic", "industry", "other")


def log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source(path: str | os.PathLike) -> dict:
    """``{path, sha256, bytes}`` of an input file, for the provenance block."""
    p = Path(path)
    return {"path": rel_to_repo(p), "sha256": sha256(p), "bytes": p.stat().st_size}


def write_new(path: Path, obj) -> None:
    """Write JSON (dict) or text (str) to a NEW file; refuse to overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists; results are never recomputed or overwritten")
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=1, ensure_ascii=False) + "\n"
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(text)
    log(f"wrote {rel_to_repo(path)}")


def git_head() -> str:
    try:
        head = (REPO / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            return (REPO / ".git" / head[5:]).read_text().strip()[:12]
        return head[:12]
    except OSError:
        return "unknown"


# ----------------------------------------------------------------------------- the page
def load_payload(path: Path = PAGE) -> dict:
    """The decision page's embedded JSON payload (aggregates only; the public build)."""
    s = path.read_text(encoding="utf-8")
    m = re.search(r'<script[^>]*id="payload"[^>]*>(.*?)</script>', s, re.S)
    if m is None:
        raise SystemExit(f"no payload in {path}")
    return json.loads(m.group(1))


def payload_array(D: dict, name: str) -> np.ndarray:
    from hydrophysics.twin.app.prep import unpack

    return unpack(D["arrays"][name])


def page_basis(D: dict) -> np.ndarray:
    """The page's (rescaled) basis rows as the page reads them: ``(R, A, Y)`` cm, with
    the page's reference year subtracted (``classResponse``: dSubs[y] - dSubs[YR])."""
    M = D["meta"]
    R, A, Y = 2 * len(D["classes"]), M["nA"], len(M["years"])
    d = payload_array(D, "dSubs").reshape(R, A, Y)
    yr = M.get("yRef", M["yObs"])
    return d - d[..., yr:yr + 1]


def townships(D: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    """``(town index per cell, english names, eligible-for-headline mask per town)``.
    Eligible = the page's ``topTowns`` filter: not the pooled Changhua row and not a
    low-confidence township (``townSkill.low``)."""
    town = payload_array(D, "town").astype(int).ravel()
    names = [t["en"] for t in D["towns"]]
    skill = D.get("townSkill") or [{"low": True}] * len(names)
    elig = np.array([(not t["pooled"]) and (not s["low"])
                     for t, s in zip(D["towns"], skill, strict=True)])
    return town, names, elig


def town_means(field: np.ndarray, town: np.ndarray, n_towns: int) -> np.ndarray:
    """Mean of ``field`` (..., A) over the cells of each township -> (..., n_towns)."""
    cnt = np.bincount(town, minlength=n_towns).astype(float)
    flat = field.reshape(-1, field.shape[-1])
    out = np.stack([np.bincount(town, weights=row, minlength=n_towns) for row in flat])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / cnt[None, :]
    return out.reshape(field.shape[:-1] + (n_towns,))


def top_towns(avoided_by_town: np.ndarray, elig: np.ndarray, n: int = 3) -> list[int]:
    """The page's headline townships: eligible, finite, positive, largest first."""
    v = np.where(elig & np.isfinite(avoided_by_town) & (avoided_by_town > 0),
                 avoided_by_town, -np.inf)
    order = [int(i) for i in np.argsort(-v) if np.isfinite(v[i])]
    return order[:n]


# ----------------------------------------------------------------------------- the grid
def grid_groups() -> dict:
    """A2's perturbation groups on the 1 km grid: fan zone (0-2) + 3 x Changhua flag."""
    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.inputs import resolve_paths
    from hydrophysics.twin.zone_county import DEFAULT_MAP, county_flags_for_grid
    from hydrophysics.twin.zones import fan_zones

    grid = build_grid(resolve_paths()["polygon"], dx=1000.0)
    ch = np.asarray(county_flags_for_grid(grid, DEFAULT_MAP)[0], bool)
    zone = fan_zones(grid.centroids())
    return {"grid": grid, "changhua": ch, "zone": zone, "grp": zone + 3 * ch.astype(int)}


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def top_decile(v: np.ndarray) -> np.ndarray:
    return v >= np.quantile(v, 0.9)
