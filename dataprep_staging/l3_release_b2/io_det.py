"""Deterministic writers and a pickle-free npz reader (plan 2.8 item 5: identical rebuilds).

``numpy.savez`` stamps each zip member with the wall-clock time, so two builds of the same
arrays differ byte for byte. :func:`write_npz` writes the same ``.npy`` members with a fixed
timestamp and fixed attributes. Parquet is written by pyarrow without a timestamp; JSON with
a fixed layout and no NaN. Every writer refuses to replace an existing file.
"""
from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np

ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class PickledArray(ValueError):
    """An npz member is an object (pickled) array; it is never unpickled here."""


def _exclusive(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; release files are never replaced")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_npz(path, arrays: dict) -> Path:
    """Write ``arrays`` (name -> array, insertion order kept) as a deflated npz with a fixed
    member timestamp. Object arrays are refused (no pickles)."""
    path = _exclusive(path)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, arr in arrays.items():
            a = np.asanyarray(arr)
            if a.dtype.hasobject:
                raise PickledArray(f"{key}: object arrays are not written")
            zi = zipfile.ZipInfo(f"{key}.npy", date_time=ZIP_EPOCH)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            with zf.open(zi, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, a, allow_pickle=False)
    return path


def read_npz(path) -> tuple[dict, list[str]]:
    """(arrays, skipped): every member that loads with ``allow_pickle=False``; members that
    are pickled object arrays are listed in ``skipped`` and never unpickled."""
    out, skipped = {}, []
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            try:
                out[k] = z[k]
            except ValueError:
                skipped.append(k)
    return out, skipped


def write_parquet(path, df) -> Path:
    """Parquet via pyarrow (no index, zstd); deterministic for identical frames."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = _exclusive(path)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")
    return path


def jsonable(x):
    """Recursively convert numpy scalars/arrays to Python and non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def dumps(obj) -> str:
    return json.dumps(jsonable(obj), ensure_ascii=False, indent=1, allow_nan=False) + "\n"


def write_json(path, obj) -> Path:
    path = _exclusive(path)
    path.write_text(dumps(obj), encoding="utf-8")
    return path


def write_text(path, text: str) -> Path:
    path = _exclusive(path)
    path.write_text(text, encoding="utf-8")
    return path


def copy_file(src, dst) -> Path:
    """Byte copy (content only; the copy's mtime is irrelevant to its hash)."""
    dst = _exclusive(dst)
    shutil.copyfile(src, dst)
    return dst
