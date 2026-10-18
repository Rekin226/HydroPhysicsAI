"""Data-role ledger (plan 4.1, rule U6).

Each WP appends its rows to its own fragment ``results/dataprep/ledger/rows/<WP>.csv``; the
combined ``ledger/data_roles.csv`` is regenerated from the fragments by :func:`assemble`
(``python -m common.ledger assemble``). Columns: the four of plan 4.1
(``number_id, source, role, product_version``) plus ``wp`` and a free-text ``note``.
A source plays one role per number: :func:`append_rows` and :func:`assemble` refuse a second
role for the same (number_id, source).

Repair round (2026-10-12): sources must be explicit - one file, one directory, one product
version or one named source id per row; glob or brace patterns (``* ? [ ] { }``) are refused,
because a pattern never collides with the explicit path of the same file and would let one
source carry two roles unnoticed. Path-like sources are normalised (absolute paths inside the
repository become repository-relative, ``./`` and duplicate or trailing slashes are dropped),
and two sources conflict when one is the other or a directory containing it.
:func:`unledgered` lists reported numbers that have no row (U6).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import io
import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import LEDGER_FILE, LEDGER_ROWS, REPO  # noqa: E402
from common.uses import normalise_role  # noqa: E402

COLUMNS = ("number_id", "source", "role", "product_version", "wp", "note")
_WP = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,31}$")
_GLOB = re.compile(r"[*?\[\]{}]")


class LedgerConflict(ValueError):
    """A source would play two roles for one number (U6)."""


def fragment_path(wp: str, rows_dir: str | os.PathLike = LEDGER_ROWS) -> Path:
    if not _WP.match(wp):
        raise ValueError(f"bad WP id {wp!r}")
    return Path(rows_dir) / f"{wp}.csv"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalise_source(source: str) -> str:
    """Canonical form of a ledger source; raises on glob or brace patterns."""
    s = str(source or "").strip().replace("\\", "/")
    if not s:
        raise ValueError("empty ledger source")
    if _GLOB.search(s):
        raise ValueError(f"ledger source {s!r} is a pattern; list each file, directory or "
                         f"product explicitly (one role per source per number, U6)")
    if "/" in s:
        if s.startswith("/"):
            with contextlib.suppress(ValueError):
                s = Path(s).resolve().relative_to(REPO).as_posix()
        s = re.sub(r"/{2,}", "/", s)
        while s.startswith("./"):
            s = s[2:]
        s = s.rstrip("/") or "/"
    return s


def _clean(row: dict, wp: str) -> dict:
    r = {k: str(row.get(k, "") or "").strip() for k in COLUMNS}
    r["wp"] = r["wp"] or wp
    if r["wp"] != wp:
        raise ValueError(f"row wp {r['wp']!r} != fragment {wp!r}")
    for k in ("number_id", "source", "role", "product_version"):
        if not r[k]:
            raise ValueError(f"ledger row needs {k}: {row}")
    r["role"] = normalise_role(r["role"])
    r["source"] = normalise_source(r["source"])
    return r


def _related(a: str, b: str) -> bool:
    """Same source, or one is a directory (path prefix) containing the other."""
    return a == b or b.startswith(a + "/") or a.startswith(b + "/")


def conflicts(rows: list[dict]) -> list[tuple[str, str, list[str]]]:
    """[(number_id, source, roles)] for every number where one source (or a directory and a
    file inside it) carries more than one role. Sources are normalised first."""
    by_num: dict[str, dict[str, set]] = {}
    for r in rows:
        src = normalise_source(r["source"])
        by_num.setdefault(r["number_id"], {}).setdefault(src, set()).add(r["role"])
    out = []
    for n, srcs in sorted(by_num.items()):
        items = sorted(srcs.items())
        for a, ra in items:
            roles = set(ra)
            for b, rb in items:
                if b != a and _related(a, b):
                    roles |= rb
            if len(roles) > 1:
                out.append((n, a, sorted(roles)))
    return out


def unledgered(number_ids, rows: list[dict] | None = None,
               rows_dir: str | os.PathLike = LEDGER_ROWS) -> list[str]:
    """Reported numbers (ids) that have no ledger row (U6: every reported number has one)."""
    have = {r["number_id"] for r in (read_all(rows_dir) if rows is None else rows)}
    return sorted(set(number_ids) - have)


def append_rows(wp: str, rows: list[dict], rows_dir: str | os.PathLike = LEDGER_ROWS) -> Path:
    """Append rows to the WP's fragment (exact duplicates are skipped). Raises
    :class:`LedgerConflict` before writing if a row gives a source a second role."""
    path = fragment_path(wp, rows_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = [_clean(r, wp) for r in rows]
    with open(path, "a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        content = f.read()
        old = list(csv.DictReader(io.StringIO(content))) if content else []
        bad = conflicts(old + new)
        if bad:
            raise LedgerConflict(f"one role per number violated: {bad}")
        have = {tuple(r.get(k, "") for k in COLUMNS) for r in old}
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            w.writeheader()
        for r in new:
            key = tuple(r[k] for k in COLUMNS)
            if key not in have:
                w.writerow(r)
                have.add(key)
        fcntl.flock(f, fcntl.LOCK_UN)
    return path


def read_all(rows_dir: str | os.PathLike = LEDGER_ROWS) -> list[dict]:
    rows = []
    for p in sorted(Path(rows_dir).glob("*.csv")):
        rows.extend(_read(p))
    return rows


def assemble(rows_dir: str | os.PathLike = LEDGER_ROWS,
             out: str | os.PathLike = LEDGER_FILE) -> tuple[Path, int]:
    """Rebuild ``data_roles.csv`` from all fragments; raises on cross-fragment conflicts."""
    rows = read_all(rows_dir)
    bad = conflicts(rows)
    if bad:
        raise LedgerConflict(f"one role per number violated across fragments: {bad}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    os.replace(tmp, out)
    return out, len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("assemble", help="rebuild data_roles.csv from rows/*.csv")
    sub.add_parser("check", help="report one-role-per-number conflicts")
    a = ap.parse_args(argv)
    if a.cmd == "assemble":
        out, n = assemble()
        print(f"wrote {out} ({n} rows)")
        return 0
    bad = conflicts(read_all())
    for b in bad:
        print("CONFLICT", *b)
    print(f"{len(bad)} conflicts")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
