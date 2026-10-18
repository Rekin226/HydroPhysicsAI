"""Loader guard (rules U1, U2, U3, U5, U7 of plan 2.5).

Every consumer opens a product through :func:`open_product`, declaring the use it makes of
it. The guard raises if

* the use is not in the manifest's ``allowed_uses`` (or is in ``forbidden_uses``);
* the use is a pre-origin use (calibrate, fit a datum, fit a baseline, start, prior, design,
  select, frozen forcing, design-time evidence; :data:`common.uses.PRE_ORIGIN_USES`) and the
  product's ``fit_window_end`` is later than the origin (2022-12), unless

  - an exemption covers that use (U3/U5: ``non_exempt_fit_window_end <= 2022-12`` and every
    post-origin exempt input lists the use), or
  - the consumer declares ``window_end <= 2022-12`` on a product that declares
    ``time_slicing`` whose own fitted parameters end by the origin; only rows dated up to
    ``window_end`` are then readable (:meth:`Product.read_table`; raw paths are refused), or
  - some outputs carry their own ``fit_window_end <= 2022-12`` (pilots that mix windows);
    only those outputs are then readable;

* the product's validation gate has not passed (except for design-time evidence and
  after-the-fact analysis);
* an output file's sha256 differs from the manifest (when ``verify`` is on).

Overrides can only tighten: ``preorigin=True`` marks a use that is not inherently pre-origin
as feeding a model scored on 2023-26 (``preorigin=False`` on a pre-origin use raises), and
``require_gate=True`` demands a passed gate for a gate-exempt use (``require_gate=False`` on
a gate-checked use raises). ``verify=False`` skips only the output re-hash and is recorded in
``Product.notes``.

Row-level rules (U1: flags 2-6 never enter scoring targets, baselines or datum fits; U2:
calibration targets use flags 0-2 only) are applied by :func:`filter_flags`;
:meth:`Product.read_table` always reads the flag column when the table has one, even if the
caller's column list omits it.

Time slicing is type-safe (WP A0fix, 2026-10-12). The manifest's ``time_slicing`` names the
column and its encoding, ``{"column", "params_fit_window_end", "encoding"}``, one of
:data:`TIME_ENCODINGS`:

* ``"iso"`` (the default when ``encoding`` is absent): ISO-8601 calendar strings
  (``YYYY-MM``, ``YYYY-MM-DD``, ``YYYY-MM-DD[T ]hh:mm[:ss[.f]][Z|+hh:mm]``) or a pandas
  datetime64 column, compared at month granularity on the calendar written in the value;
* ``"yyyymm"``: integer months ``YYYYMM`` in 190001-209912 with month 01-12;
* ``"year"``: integer calendar years 1900-2099, compared at year granularity (a row of year
  ``Y`` is readable only when ``Y-12 <= window_end``, the whole year inside the window).

Anything else raises :class:`TimeEncodingError` before a row is returned: integer columns
that are not declared ``yyyymm`` or ``year`` (so an epoch time can never pass as a month),
values outside the declared range, floats, booleans, periods, missing values, bare year
strings and object columns that mix types. Every row of the table is checked, not only the
rows the use keeps. Builders validate the declaration and their output columns before writing
the manifest with :func:`time_slicing_errors`.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from common.manifest import (
    HashMismatch,
    ManifestError,
    load_manifest,
    preorigin_routes,
    validate_manifest,
    verify_hashes,
)
from common.paths import ORIGIN
from common.uses import GATE_EXEMPT_USES, PRE_ORIGIN_USES, USES, flags_for_use

_YM = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

#: Declared encodings of a ``time_slicing`` column (``time_slicing.encoding``).
TIME_ENCODINGS = {
    "iso": "ISO-8601 strings YYYY-MM, YYYY-MM-DD or YYYY-MM-DD[T ]hh:mm[:ss[.f]][Z|+hh:mm], "
           "or a pandas datetime64 column (naive or tz-aware); month granularity",
    "yyyymm": "integer YYYYMM months, 190001-209912 with month 01-12; month granularity",
    "year": "integer calendar years 1900-2099; year granularity: a row of year Y is readable "
            "only when Y-12 <= window_end",
}
#: Encoding assumed when a manifest's time_slicing declares none (A0 r2 manifests).
DEFAULT_TIME_ENCODING = "iso"
_YEAR_MIN, _YEAR_MAX = 1900, 2099
_ISO = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d{1,9})?)?"
                  r"(?:Z|[+-]\d{2}(?::?\d{2})?)?)?)?", re.ASCII)


class TimeEncodingError(ValueError):
    """A time column does not match its declared encoding, so it cannot be sliced."""


class UseNotAllowed(PermissionError):
    """The declared use is not allowed by the product's manifest."""


class LeakageError(PermissionError):
    """A pre-origin use met a product fitted on post-origin data with no covering exemption."""


class GateNotPassed(PermissionError):
    """The product's validation gate has not passed."""


@dataclass
class Access:
    """Result of :func:`check_use`: notes plus any restriction the guard imposes."""
    preorigin: bool
    notes: list[str] = field(default_factory=list)
    window_end: str | None = None          # rows readable only up to this month
    outputs: frozenset | None = None       # only these outputs readable (None = all)
    time_encoding: str | None = None       # declared encoding of the slice column


@dataclass
class Product:
    """A product opened for one declared use."""
    manifest: dict
    directory: Path
    use: str
    preorigin: bool
    notes: list[str] = field(default_factory=list)
    window_end: str | None = None
    allowed_outputs: frozenset | None = None
    time_encoding: str | None = None

    @property
    def outputs(self) -> dict[str, dict]:
        return {r["path"]: r for r in self.manifest["outputs"]}

    def _resolve(self, name: str) -> Path:
        if name not in self.outputs:
            raise KeyError(f"{name!r} is not a listed output of {self.manifest['product']} "
                           f"{self.manifest['version']}")
        if self.allowed_outputs is not None and name not in self.allowed_outputs:
            raise LeakageError(f"{name!r} has fit_window_end > {ORIGIN}; a pre-origin use may "
                               f"read only {sorted(self.allowed_outputs)}")
        return self.directory / name

    def path(self, name: str) -> Path:
        """Path of a listed output (refuses files the manifest does not list, outputs a
        pre-origin use may not read, and any raw access to a time-sliced product)."""
        if self.window_end is not None:
            raise UseNotAllowed(f"{self.manifest['product']}: opened with window_end "
                                f"{self.window_end}; raw file access would bypass the slice, "
                                f"use read_table")
        return self._resolve(name)

    def read_table(self, name: str, columns: list[str] | None = None, strict: bool = False):
        """Read a listed parquet/csv output, apply :func:`filter_flags` for the use and, when
        the product was opened with ``window_end``, keep rows dated up to that month (the
        time column must match the manifest's declared encoding on every row, else
        :class:`TimeEncodingError`)."""
        import pandas as pd
        p = self._resolve(name)
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            have = list(pq.read_schema(p).names)
        elif p.suffix == ".csv":
            have = list(pd.read_csv(p, nrows=0).columns)
        else:
            raise ValueError(f"read_table supports parquet and csv, not {p.suffix}")
        tcol = None
        if self.window_end is not None:
            tcol = self.manifest["time_slicing"]["column"]
            if tcol not in have:
                raise UseNotAllowed(f"{name!r} has no time column {tcol!r}; cannot slice it")
        want = None
        if columns is not None:
            want = list(columns)
            extra = [c for c in ("flag", tcol) if c and c in have and c not in want]
            cols = want + extra
        else:
            cols = None
        df = pd.read_parquet(p, columns=cols) if p.suffix == ".parquet" else \
            pd.read_csv(p, usecols=cols)
        if tcol is not None:          # every row is checked, before the flag filter
            enc = self.time_encoding or time_encoding(self.manifest)
            df = df[slice_mask(df[tcol], enc, self.window_end,
                               what=f"{name!r} column {tcol!r}")]
        if "flag" in df.columns:
            df = filter_flags(df, self.use, strict=strict)
        if want is not None:
            df = df[want]
        return df


# ----------------------------------------------------------------------------- time encodings
def time_encoding(m: dict) -> str:
    """Declared encoding of a manifest's ``time_slicing`` (``"iso"`` when absent); raises
    :class:`TimeEncodingError` for an encoding outside :data:`TIME_ENCODINGS`."""
    ts = m.get("time_slicing") or {}
    enc = ts.get("encoding", DEFAULT_TIME_ENCODING)
    if not (isinstance(enc, str) and enc in TIME_ENCODINGS):
        raise TimeEncodingError(f"{m.get('product')} {m.get('version')}: time_slicing.encoding "
                                f"{enc!r} is not one of {sorted(TIME_ENCODINGS)}")
    return enc


def _ym_key(ym: str) -> int:
    if not (isinstance(ym, str) and _YM.match(ym)):
        raise ValueError(f"{ym!r} must be YYYY-MM")
    return int(ym[:4]) * 100 + int(ym[5:7])


def _iso_key(v) -> int | None:
    """YYYYMM key of one ISO-8601 calendar string, or None when it is not one."""
    if not isinstance(v, str):
        return None
    mt = _ISO.fullmatch(v)
    if mt is None:
        return None
    y, mo = int(mt[1]), int(mt[2])
    if not (_YEAR_MIN <= y <= _YEAR_MAX and 1 <= mo <= 12):
        return None
    if mt[3] is not None:
        try:
            _dt.date(y, mo, int(mt[3]))
        except ValueError:
            return None
        if mt[4] is not None and not (int(mt[4]) <= 23 and int(mt[5]) <= 59
                                      and (mt[6] is None or int(mt[6]) <= 59)):
            return None
    return y * 100 + mo


def _examples(vals, k: int = 3) -> str:
    return ", ".join(repr(v.item() if hasattr(v, "item") else v) for v in list(vals)[:k])


def time_keys(s, encoding: str, what: str = "time column"):
    """int64 array of ``YYYY*100 + MM`` keys, the last calendar month each value covers.

    Only the declared ``encoding`` is accepted (:data:`TIME_ENCODINGS`); any value that does
    not match it, and any missing value, raises :class:`TimeEncodingError`. For ``"year"``
    the key is December of that year, so a year is readable only when it ends in the window.
    """
    import numpy as np
    import pandas as pd
    if encoding not in TIME_ENCODINGS:
        raise TimeEncodingError(f"{what}: encoding {encoding!r} is not one of "
                                f"{sorted(TIME_ENCODINGS)}")
    s = s if isinstance(s, pd.Series) else pd.Series(s)
    dtype = s.dtype
    n_missing = int(s.isna().sum())
    if n_missing:
        raise TimeEncodingError(f"{what}: {n_missing} rows with a missing time value; cannot "
                                f"slice")
    if isinstance(dtype, pd.CategoricalDtype):    # validate the categories, map the codes
        cats = time_keys(pd.Series(dtype.categories), encoding, what=f"{what} (categories)")
        return cats[s.cat.codes.to_numpy()]
    if isinstance(dtype, pd.PeriodDtype) or pd.api.types.is_bool_dtype(dtype):
        raise TimeEncodingError(f"{what}: dtype {dtype} is not a declared time encoding "
                                f"({sorted(TIME_ENCODINGS)}); store ISO strings or datetimes")
    if encoding in ("yyyymm", "year"):
        if not pd.api.types.is_integer_dtype(dtype):
            hint = (" (an integer column with missing values reads as float: drop or fill "
                    "those rows)" if pd.api.types.is_float_dtype(dtype) else "")
            raise TimeEncodingError(f"{what}: encoding {encoding!r} needs an integer column, "
                                    f"got dtype {dtype}{hint}")
        v = s.to_numpy(dtype="int64")
        if encoding == "yyyymm":
            y, mo = v // 100, v % 100
            bad = (y < _YEAR_MIN) | (y > _YEAR_MAX) | (mo < 1) | (mo > 12)
            if bad.any():
                raise TimeEncodingError(
                    f"{what}: {int(bad.sum())} values are not YYYYMM months in 190001-209912 "
                    f"with month 01-12 (e.g. {_examples(np.unique(v[bad]))}); epoch times, "
                    f"YYYYMMDD dates and bare years are refused")
            return v
        bad = (v < _YEAR_MIN) | (v > _YEAR_MAX)
        if bad.any():
            raise TimeEncodingError(
                f"{what}: {int(bad.sum())} values are not calendar years 1900-2099 "
                f"(e.g. {_examples(np.unique(v[bad]))}); epoch times and YYYYMM are refused")
        return v * 100 + 12
    # encoding == "iso"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        y = s.dt.year.to_numpy(dtype="int64")
        mo = s.dt.month.to_numpy(dtype="int64")
        bad = (y < _YEAR_MIN) | (y > _YEAR_MAX)
        if bad.any():
            raise TimeEncodingError(f"{what}: {int(bad.sum())} datetimes outside 1900-2099 "
                                    f"(e.g. {_examples(s[bad].unique())})")
        return y * 100 + mo
    if not (pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)):
        raise TimeEncodingError(
            f"{what}: a {dtype} column is not an ISO date; declare time_slicing.encoding "
            f"'yyyymm' or 'year' for integer months or years (epoch times are never accepted)")
    codes, uniq = pd.factorize(s)
    keys = np.empty(len(uniq), dtype=np.int64)
    bad = []
    for j, u in enumerate(uniq):
        k = _iso_key(u)
        if k is None:
            bad.append(u)
        else:
            keys[j] = k
    if bad:
        raise TimeEncodingError(
            f"{what}: {len(bad)} distinct values are not ISO-8601 dates YYYY-MM[-DD[ hh:mm"
            f"[:ss]]] with year 1900-2099 (e.g. {_examples(bad)}); non-string values, mixed "
            f"types and bare years are refused")
    return keys[codes]


def slice_mask(s, encoding: str, window_end: str, what: str = "time column"):
    """Boolean numpy mask of the rows whose whole period ends by ``window_end`` (YYYY-MM)."""
    return time_keys(s, encoding, what=what) <= _ym_key(window_end)


def time_slicing_errors(m: dict, product_dir: str | os.PathLike | None = None,
                        require_explicit: bool = True) -> list[str]:
    """Problems with a manifest's ``time_slicing`` (empty = sliceable). Builders call it
    before :func:`common.manifest.write_manifest`.

    Checks the declared encoding (``require_explicit``: an ``encoding`` key must be present)
    and, with ``product_dir``, every parquet/csv output that has the slice column: every row
    must match the encoding. At least one output must carry the column."""
    import pandas as pd
    ts = m.get("time_slicing")
    if not ts:
        return []
    errs = []
    if require_explicit and "encoding" not in ts:
        errs.append(f"time_slicing declares no encoding; add one of {sorted(TIME_ENCODINGS)}")
    try:
        enc = time_encoding(m)
    except TimeEncodingError as exc:
        return errs + [str(exc)]
    if product_dir is None:
        return errs
    d = Path(product_dir)
    col = ts.get("column")
    n_with = 0
    for r in m.get("outputs") or []:
        p = d / r["path"]
        if p.suffix not in (".parquet", ".csv") or not p.is_file():
            continue
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            have = pq.read_schema(p).names
        else:
            have = list(pd.read_csv(p, nrows=0).columns)
        if col not in have:
            continue
        n_with += 1
        s = pd.read_parquet(p, columns=[col])[col] if p.suffix == ".parquet" else \
            pd.read_csv(p, usecols=[col])[col]
        try:
            time_keys(s, enc, what=f"{r['path']!r} column {col!r}")
        except TimeEncodingError as exc:
            errs.append(str(exc))
    if n_with == 0:
        errs.append(f"no parquet/csv output has the time_slicing column {col!r}")
    return errs


def check_use(m: dict, use: str, preorigin: bool | None = None,
              require_gate: bool | None = None, window_end: str | None = None) -> Access:
    """Raise if the manifest does not permit ``use``; returns the access granted."""
    if use not in USES:
        raise ValueError(f"unknown use {use!r}; expected one of {sorted(USES)}")
    name = f"{m.get('product')} {m.get('version')}"
    if use in m.get("forbidden_uses", []):
        raise UseNotAllowed(f"{name}: use {use!r} is forbidden by the manifest")
    if use not in m.get("allowed_uses", []):
        raise UseNotAllowed(f"{name}: use {use!r} is not in allowed_uses "
                            f"{m.get('allowed_uses')}")
    inherent = use in PRE_ORIGIN_USES
    if preorigin is False and inherent:
        raise ValueError(f"preorigin=False cannot relax the pre-origin use {use!r} (U3/U5); "
                         f"overrides may only tighten")
    pre = inherent or bool(preorigin)
    acc = Access(preorigin=pre)
    fwe = m["fit_window_end"]
    eff = fwe
    if window_end is not None:
        if not (isinstance(window_end, str) and _YM.match(window_end)):
            raise ValueError(f"window_end {window_end!r} must be YYYY-MM")
        ts = m.get("time_slicing")
        if not ts:
            raise UseNotAllowed(f"{name}: the manifest declares no time_slicing; a window "
                                f"cannot be read from it")
        acc.time_encoding = time_encoding(m)          # raises on an undeclared encoding
        eff = min(fwe, max(window_end, ts["params_fit_window_end"]))
        acc.window_end = window_end
        how = "declared" if "encoding" in ts else "default; time_slicing declares none"
        acc.notes.append(f"time slice: rows up to {window_end} (product parameters fitted to "
                         f"{ts['params_fit_window_end']}; effective window end {eff}; column "
                         f"{ts['column']!r}, encoding {acc.time_encoding!r} ({how}))")
    if pre and eff > ORIGIN:
        routes = preorigin_routes(m, use)
        if window_end is None and routes["whole"]:
            for x in routes["exemptions"]:
                acc.notes.append(f"exemption {x.get('input')!r} (ledger "
                                 f"{x.get('ledger_number_id')}) covers data to "
                                 f"{x.get('data_end')} for use {use!r}")
        elif window_end is None and routes["outputs"]:
            acc.outputs = frozenset(routes["outputs"])
            acc.notes.append(f"only the {len(routes['outputs'])} outputs with fit_window_end "
                             f"<= {ORIGIN} are readable for use {use!r}")
        else:
            hint = (f" (declare window_end <= {ORIGIN} to read a time slice)"
                    if routes["slice"] and window_end is None else "")
            raise LeakageError(f"{name}: pre-origin use {use!r} but effective fit window end "
                               f"{eff} > {ORIGIN} and {routes['reason']} (U3/U5){hint}")
    gate_exempt = use in GATE_EXEMPT_USES
    if require_gate is False and not gate_exempt:
        raise ValueError(f"require_gate=False cannot skip the gate for use {use!r}; overrides "
                         f"may only tighten")
    gate_needed = (not gate_exempt) or bool(require_gate)
    if gate_needed and not m.get("gate", {}).get("passed", False):
        raise GateNotPassed(f"{name}: validation gate not passed ({m.get('gate')})")
    return acc


def open_product(product_dir: str | os.PathLike, use: str, *, preorigin: bool | None = None,
                 require_gate: bool | None = None, window_end: str | None = None,
                 verify: bool = True) -> Product:
    """Open a product version for a declared use, enforcing U5 (see module docstring).

    ``preorigin=True`` declares that a use which is not inherently pre-origin (validator,
    statistical-layer history, ...) feeds something scored on 2023-26. ``window_end`` reads
    only rows up to that month of a product that declares ``time_slicing``, through
    :meth:`Product.read_table`, using the declared ``time_slicing.encoding``.
    """
    m, d = load_manifest(product_dir)
    errs = validate_manifest(m, d)
    if errs:
        raise ManifestError(f"{d}: " + "; ".join(errs))
    acc = check_use(m, use, preorigin=preorigin, require_gate=require_gate,
                    window_end=window_end)
    if verify:
        probs = verify_hashes(m, d, ("outputs",))
        if probs:
            raise HashMismatch(f"{d}: {probs}")
    else:
        acc.notes.append("verify=False: output hashes were not re-checked at open")
    return Product(manifest=m, directory=d, use=use, preorigin=acc.preorigin,
                   notes=acc.notes, window_end=acc.window_end, allowed_outputs=acc.outputs,
                   time_encoding=acc.time_encoding)


def filter_flags(df, use: str, strict: bool = False, flag_col: str = "flag"):
    """Drop rows whose flag is not allowed for ``use`` (U1, U2). ``strict=True`` raises
    instead of dropping."""
    allowed = flags_for_use(use)
    ok = df[flag_col].isin(sorted(allowed))
    if strict and not bool(ok.all()):
        bad = sorted(set(int(x) for x in df.loc[~ok, flag_col].unique()))
        raise UseNotAllowed(f"flags {bad} are not allowed for use {use!r} "
                            f"(allowed {sorted(allowed)})")
    return df[ok]


def assert_releasable(product_dir: str | os.PathLike) -> dict:
    """Raise unless a product may enter an L3 release (plan 2.3, 2.8): valid manifest, layer
    L1 or L2, gate passed, every input/code/output hash reproduced. Returns the manifest."""
    m, d = load_manifest(product_dir)
    errs = validate_manifest(m, d)
    if errs:
        raise ManifestError(f"{d}: " + "; ".join(errs))
    if m["layer"] not in ("L1", "L2"):
        raise GateNotPassed(f"{d}: layer {m['layer']!r} cannot enter a release")
    if not m["gate"]["passed"]:
        raise GateNotPassed(f"{d}: validation gate not passed ({m['gate']})")
    probs = verify_hashes(m, d)
    if probs:
        raise HashMismatch(f"{d}: {probs}")
    return m
