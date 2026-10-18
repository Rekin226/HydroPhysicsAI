"""WP P3 re-validation of the rebuild (design-time checks, Yunlin and grid only; no Changhua
census value is read here).

1. SOURCE.json of the two P3 MOA downloads re-hashed (common.source) and parsed record counts.
2. Overlap years against the pilot's downloads (A0 L0 copies moa_crop_038 / moa_rice_J51):
   record-by-record equality of the Changhua and Yunlin rows.
3. LUIMAP classification, features.npz and township.npz rebuilt from L0 (features.py) against
   the landuse_crop_priors pilot copies (byte / array equality).

Usage: python -I verify_rebuild.py      (writes P3_evidence/verify_rebuild.json)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
from common import hashing, source  # noqa: E402

KEY038 = ["年度", "縣市", "鄉鎮", "作物代碼", "作物", "期作"]


def check_sources() -> dict:
    out = {}
    for d in (lib.MOA038_DIR, lib.MOAJ51_DIR):
        with open(d / "SOURCE.json", encoding="utf-8") as fh:
            rec = json.load(fh)
        errs = source.validate_source(rec, d)
        bad = []
        for f in rec["files"]:
            p = d / f["file"]
            if not p.exists():
                bad.append(f"{f['file']} missing")
            elif "sha256" in f and hashing.sha256_file(p) != f["sha256"]:
                bad.append(f"{f['file']} sha256 differs")
        out[d.name] = {"files": len(rec["files"]), "rehash_problems": bad,
                       "validator_errors": errs}
    return out


def _rows(pattern_files) -> pd.DataFrame:
    rows = []
    for f in pattern_files:
        with open(f, encoding="utf-8") as fh:
            rows += json.load(fh)
    return pd.DataFrame(rows)


def overlap_038() -> dict:
    res = {}
    for roc in ("103", "104", "105", "110"):
        for city in ("彰化縣", "雲林縣"):
            new = _rows([lib.MOA038_DIR / f"crop_{city}_{roc}_p0.json"])
            old = _rows([lib.L0 / "moa_crop_038" / f"crop_{city}_{roc}.json"])
            cols = [c for c in new.columns if c in old.columns]
            a = new[cols].astype(str).sort_values(cols).reset_index(drop=True)
            b = old[cols].astype(str).sort_values(cols).reset_index(drop=True)
            same = a.shape == b.shape and bool((a.values == b.values).all())
            ia = set(map(tuple, a.values))
            ib = set(map(tuple, b.values))
            res[f"{city}_{roc}"] = {"new_rows": len(new), "pilot_rows": len(old),
                                    "identical_after_sort": same,
                                    "rows_only_new": len(ia - ib), "rows_only_pilot": len(ib - ia),
                                    "new_cols_not_in_pilot": sorted(set(new.columns) -
                                                                    set(old.columns))}
            if not same:
                # are the pilot rows the annual subset of the new page?
                an = new[new["期作"] == "全年"][cols].astype(str)
                ao = old[old["期作"] == "全年"][cols].astype(str)
                res[f"{city}_{roc}"]["annual_rows_identical"] = (
                    set(map(tuple, an.values)) == set(map(tuple, ao.values)))
    return res


def overlap_j51() -> dict:
    res = {}
    for roc in ("103", "104", "105", "110"):
        new = _rows([lib.MOAJ51_DIR / f"rice_{roc}_p0.json"])
        new = new[new["縣市"].isin(["彰化縣", "雲林縣"])]
        old = _rows([lib.L0 / "moa_rice_J51" / f"rice_{roc}.json"])
        cols = [c for c in new.columns if c in old.columns]
        ia = set(map(tuple, new[cols].astype(str).values))
        ib = set(map(tuple, old[cols].astype(str).values))
        res[roc] = {"new_rows_ch_yl": len(new), "pilot_rows": len(old), "identical": ia == ib,
                    "rows_only_new": len(ia - ib), "rows_only_pilot": len(ib - ia)}
    return res


def crop_table_equality() -> dict:
    """crop_table built from the P3 pages vs from the pilot files, overlap years."""
    t_new = lib.crop_table()
    t_old = lib.crop_table(lib.L0 / "moa_crop_038", lib.L0 / "moa_rice_J51", pilot_layout=True)
    out = {}
    for roc in ("103", "104", "105", "110"):
        a = t_new.loc[roc]
        b = t_old.loc[roc]
        idx = a.index.intersection(b.index)
        cols = [c for c in lib.CROP_FEATS if c in a.columns and c in b.columns]
        d = (a.loc[idx, cols] - b.loc[idx, cols]).abs().to_numpy()
        out[roc] = {"towns_new": int(len(a)), "towns_pilot": int(len(b)),
                    "towns_common": int(len(idx)), "max_abs_diff_ha": float(d.max()),
                    "sum_new_ha": float(a[cols].to_numpy().sum()),
                    "sum_pilot_ha": float(b[cols].to_numpy().sum())}
    return out


def features_equality() -> dict:
    out = {}
    pc = lib.PILOT_LU / "classified_w7.npy"
    out["classified_w7_sha256_equal"] = (hashing.sha256_file(pc) ==
                                         hashing.sha256_file(lib.WORK / "classified_w7.npy"))
    for name in ("features.npz", "township.npz"):
        a = np.load(lib.WORK / name, allow_pickle=True)
        b = np.load(lib.PILOT_LU / name, allow_pickle=True)
        rec = {}
        for k in b.files:
            x, y = a[k], b[k]
            if x.dtype.kind in "fc":
                rec[k] = {"shape_equal": x.shape == y.shape,
                          "max_abs_diff": float(np.nanmax(np.abs(x - y))) if x.shape == y.shape
                          else None}
            else:
                rec[k] = {"equal": bool(x.shape == y.shape and (x == y).all())}
        rec["new_keys"] = sorted(set(a.files) - set(b.files))
        out[name] = rec
    return out


def main() -> None:
    res = {"sources": check_sources(), "overlap_038": overlap_038(), "overlap_j51": overlap_j51(),
           "crop_table_overlap": crop_table_equality(), "features": features_equality()}
    lib.save_json(lib.EVID / "verify_rebuild.json", res)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=lib._jdefault)[:6000])


if __name__ == "__main__":
    main()
