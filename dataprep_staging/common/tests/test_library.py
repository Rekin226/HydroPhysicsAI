"""CPU unit tests for the dataprep staging library (synthetic files only, no real data)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from common import check, flags, guard, hashing, ledger, manifest, source, uses


# ----------------------------------------------------------------------------- helpers
def _product(tmp_path, *, product="heads/L2_filled_preorigin", fwe="2022-12", passed=True,
             allowed=("initial_conditions", "idw_field", "validator"), exemptions=(),
             layer="L2", flag_values=(0, 1, 2, 3, 4), dates=None, **kw):
    d = manifest.new_version_dir(product, root=tmp_path / "dataprep")
    n = len(flag_values)
    df = pd.DataFrame({"value": np.arange(n, dtype=float), "sd": [np.nan, np.nan] + [0.5] * (n - 2),
                       "flag": list(flag_values), "method": ["obs", "obs"] + ["m"] * (n - 2),
                       "product_version": d.name, "source_ids": "s1"})
    if dates is not None:
        df["date"] = list(dates)
    df.to_csv(d / "fill.csv", index=False)
    (d / "validation.json").write_text("{}")
    inp = tmp_path / "input.txt"
    inp.write_text("raw")
    code = tmp_path / "build.py"
    code.write_text("print('x')\n")
    m = manifest.build_manifest(
        d, product=product, layer=layer, inputs=[inp], code=[code],
        parameters={"seed": 1000}, fit_window_end=fwe, validation_report="validation.json",
        gate={"passed": passed, "criteria": "test"}, allowed_uses=list(allowed),
        forbidden_uses=[u for u in ("scoring_target", "baseline", "datum_fit") if u not in allowed],
        exemptions=list(exemptions), **kw)
    manifest.write_manifest(d, m)
    return d, m


_LUIMAP = {"input": "LUIMAP 2024 snapshot", "data_end": "2024-06", "reason": "no earlier map",
           "ledger_number_id": "P3.prior", "sensitivity": "crop-only prior run",
           "uses": ["initial_conditions"]}


# ----------------------------------------------------------------------------- hashing
def test_sha256_file_and_tree(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello" * 1000)
    assert hashing.sha256_file(p) == hashlib.sha256(b"hello" * 1000).hexdigest()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    t1 = hashing.sha256_tree(tmp_path)
    assert t1 == hashing.sha256_tree(tmp_path)
    (tmp_path / "sub" / "b.txt").write_text("c")
    assert hashing.sha256_tree(tmp_path) != t1


def test_mask_and_grid_hash():
    m = np.zeros((3, 4), bool)
    m[1, 2] = True
    assert hashing.mask_hash(m) == hashing.mask_hash(m.astype(int))
    g = hashing.grid_hash(4, 3, 1000.0, 0.0, 0.0, m)
    assert g != hashing.grid_hash(4, 3, 1001.0, 0.0, 0.0, m)
    assert hashing.is_sha256(g) and not hashing.is_sha256("xyz")


# ----------------------------------------------------------------------------- flags
def test_flag_table_matches_plan():
    assert [int(f) for f in flags.Flag] == [0, 1, 2, 3, 4, 5, 6, 8, 9]
    assert set(flags.FLAG_MEANING) == set(flags.Flag)


def test_long_table_validation():
    ok = pd.DataFrame({"value": [1.0, 2.0, np.nan], "sd": [np.nan, 0.3, np.nan],
                       "flag": [0, 3, 8], "method": ["obs", "neigh", ""],
                       "product_version": "v1", "source_ids": "a"})
    assert flags.validate_long_table(ok) == []
    bad = ok.copy()
    bad.loc[1, "sd"] = np.nan
    bad.loc[0, "flag"] = 7
    errs = flags.validate_long_table(bad)
    assert any("without sd" in e for e in errs) and any("invalid flag" in e for e in errs)


def test_grid_validation():
    v = np.zeros((2, 3, 4))
    sd = np.full_like(v, np.nan)
    f = np.zeros(v.shape, np.uint8)
    assert flags.validate_grid(v, sd, f, dates=range(4)) == []
    f[0, 0, 0] = 5
    assert any("finite sd" in e for e in flags.validate_grid(v, sd, f))
    assert any("dtype" in e for e in flags.validate_grid(v, sd, f.astype(np.int16)))


# ----------------------------------------------------------------------------- manifest
def test_manifest_roundtrip_and_no_overwrite(tmp_path):
    d, m = _product(tmp_path)
    m2, d2 = manifest.load_manifest(d)
    assert m2 == m and d2 == d and d.name == "v1"
    assert manifest.validate_manifest(m2, d) == []
    assert manifest.verify_hashes(m2, d) == []
    assert {o["path"] for o in m2["outputs"]} == {"fill.csv", "validation.json"}
    assert [o.get("rows") for o in m2["outputs"] if o["path"] == "fill.csv"] == [5]
    with pytest.raises(FileExistsError):
        manifest.write_manifest(d, m2)
    assert manifest.new_version_dir("heads/L2_filled_preorigin",
                                    root=tmp_path / "dataprep").name == "v2"


def test_manifest_schema_errors(tmp_path):
    d, m = _product(tmp_path)
    bad = dict(m, fit_window_end="2022-13", allowed_uses=["validator", "nonsense"])
    errs = manifest.validate_manifest(bad)
    assert any("fit_window_end" in e for e in errs) and any("unknown uses" in e for e in errs)
    late = dict(m, fit_window_end="2024-06")
    assert any("pre-origin product" in e for e in manifest.validate_manifest(late))
    ex = dict(late, exemptions=[{"input": "LUIMAP 2024", "data_end": "2024-06"}])
    assert any("missing" in e for e in manifest.validate_manifest(ex))
    both = dict(m, forbidden_uses=["validator"])
    assert any("both allowed and forbidden" in e for e in manifest.validate_manifest(both))
    assert any("version" in e for e in manifest.validate_manifest(dict(m, version="v9"), d))


def test_verify_hashes_detects_change(tmp_path):
    d, m = _product(tmp_path)
    (d / "fill.csv").write_text("tampered\n")
    probs = manifest.verify_hashes(m, d)
    assert [p["path"] for p in probs] == ["fill.csv"]
    (tmp_path / "input.txt").write_text("changed")
    with pytest.raises(manifest.HashMismatch):
        manifest.assert_inputs_match(m, d)


# ----------------------------------------------------------------------------- guard (U5)
def test_guard_allowed_and_forbidden(tmp_path):
    d, _ = _product(tmp_path)
    p = guard.open_product(d, "initial_conditions")
    assert p.preorigin and p.path("fill.csv").exists()
    with pytest.raises(KeyError):
        p.path("MANIFEST.json")
    with pytest.raises(guard.UseNotAllowed):
        guard.open_product(d, "scoring_target")
    with pytest.raises(guard.UseNotAllowed):
        guard.open_product(d, "prior")           # not listed as allowed
    with pytest.raises(ValueError):
        guard.open_product(d, "made_up_use")


def test_guard_leakage_and_exemption(tmp_path):
    d, m = _product(tmp_path / "a", product="heads/L2_filled_full", fwe="2026-09",
                    allowed=("validator",))
    assert guard.open_product(d, "validator").preorigin is False
    with pytest.raises(guard.LeakageError):     # caller declares the use feeds a scored model
        guard.open_product(d, "validator", preorigin=True)
    for use in ("initial_conditions", "datum_fit", "baseline", "calibration_target"):
        with pytest.raises(guard.LeakageError):   # in-memory manifest bypassing the writer
            guard.check_use(dict(m, allowed_uses=[use], forbidden_uses=[]), use)
    with pytest.raises(manifest.ManifestError):  # the writer refuses an unreachable use
        _product(tmp_path / "a2", product="heads/L2_filled_full", fwe="2026-09",
                 allowed=("initial_conditions",))
    d2, m2 = _product(tmp_path / "b", product="changhua_prior/preorigin", fwe="2024-06",
                      allowed=("initial_conditions", "validator"), exemptions=[_LUIMAP],
                      non_exempt_fit_window_end="2022-12")
    p = guard.open_product(d2, "initial_conditions")
    assert p.notes and "LUIMAP" in p.notes[0]
    with pytest.raises(guard.UseNotAllowed):
        guard.open_product(d2, "idw_field")
    with pytest.raises(guard.LeakageError):     # exemption does not cover this use
        guard.check_use(dict(m2, allowed_uses=["idw_field"]), "idw_field")


def test_datum_fit_and_baseline_are_preorigin(tmp_path):
    """Verifier probe: fit_window_end 2025-06 must not open for datum_fit or baseline."""
    assert {"datum_fit", "baseline", "design_time_evidence"} <= uses.PRE_ORIGIN_USES
    for use in ("datum_fit", "baseline"):
        with pytest.raises(manifest.ManifestError):
            _product(tmp_path / use, product="heads/L1_clean", fwe="2025-06",
                     allowed=(use, "scoring_target"), flag_values=(0, 1))
    d, m = _product(tmp_path / "x", product="heads/L1_clean", fwe="2025-06",
                    allowed=("scoring_target",), flag_values=(0, 1))
    for use in ("datum_fit", "baseline"):
        with pytest.raises(guard.LeakageError):
            guard.check_use(dict(m, allowed_uses=[use, "scoring_target"],
                                 forbidden_uses=[]), use)


def test_time_sliced_reads(tmp_path):
    dates = ["2021-11", "2022-06", "2022-12", "2023-01", "2025-06", "2026-09"]
    d, _ = _product(tmp_path, product="heads/L1_clean", fwe="2026-09",
                    allowed=("scoring_target", "datum_fit", "baseline", "calibration_target"),
                    flag_values=(0, 1, 2, 0, 1, 0), dates=dates,
                    time_slicing={"column": "date", "params_fit_window_end": "2022-12"})
    with pytest.raises(guard.LeakageError, match="window_end"):
        guard.open_product(d, "datum_fit")
    p = guard.open_product(d, "datum_fit", window_end="2022-12")
    t = p.read_table("fill.csv", columns=["value"])
    assert list(t.columns) == ["value"] and t.value.tolist() == [0.0, 1.0]   # flag 2 dropped
    with pytest.raises(guard.UseNotAllowed):
        p.path("fill.csv")                       # raw access would bypass the slice
    with pytest.raises(guard.LeakageError):
        guard.open_product(d, "baseline", window_end="2023-06")
    c = guard.open_product(d, "calibration_target", window_end="2022-06").read_table("fill.csv")
    assert c.date.tolist() == ["2021-11", "2022-06"]
    s = guard.open_product(d, "scoring_target").read_table("fill.csv")
    assert len(s) == 5 and 2 not in s.flag.tolist()
    late = dict(manifest.load_manifest(d)[0],
                time_slicing={"column": "date", "params_fit_window_end": "2024-01"})
    with pytest.raises(guard.LeakageError):     # QC parameters fitted on 2023+ data
        guard.check_use(late, "datum_fit", window_end="2022-12")
    d2, _ = _product(tmp_path / "n", product="x/plain", allowed=("validator",))
    with pytest.raises(guard.UseNotAllowed):
        guard.open_product(d2, "validator", window_end="2022-12")


# ----------------------------------------------------------------------------- time encodings
def _sliced(tmp_path, values, encoding=None, fmt="csv", dtype=None, name="x/annual"):
    """Product with a time column 't' (declared encoding or none) and 'end', the true last
    month each row covers, opened later for datum_fit with a pre-origin window."""
    d = manifest.new_version_dir(name, root=tmp_path / "dataprep")
    n = len(values)
    t = pd.Series(values, dtype=dtype) if dtype else pd.Series(values)
    df = pd.DataFrame({"value": np.arange(n, dtype=float), "sd": [np.nan] * n, "flag": [0] * n,
                       "method": "obs", "product_version": d.name, "source_ids": "s1", "t": t})
    out = d / f"t.{fmt}"
    df.to_csv(out, index=False) if fmt == "csv" else df.to_parquet(out, index=False)
    (d / "validation.json").write_text("{}")
    code = tmp_path / "build.py"
    code.write_text("pass\n")
    ts = {"column": "t", "params_fit_window_end": "2022-12"}
    if encoding is not None:
        ts["encoding"] = encoding
    m = manifest.build_manifest(
        d, product=name, layer="L1", inputs=[], code=[code], parameters={},
        fit_window_end="2026-09", validation_report="validation.json",
        gate={"passed": True, "criteria": "test"}, time_slicing=ts,
        allowed_uses=["datum_fit", "scoring_target"], forbidden_uses=[])
    manifest.write_manifest(d, m)
    return d, out.name


def _slice(d, fname, window_end="2022-12"):
    return guard.open_product(d, "datum_fit", window_end=window_end).read_table(fname)


def test_time_slice_integer_year_column(tmp_path):
    """Verifier defect: an integer year column was read as YYYYMM (2023 -> '0020-23')."""
    years = [2020, 2021, 2022, 2023, 2025]
    d, f = _sliced(tmp_path / "a", years)                     # no encoding declared
    with pytest.raises(guard.TimeEncodingError, match="declare time_slicing.encoding"):
        _slice(d, f)
    d, f = _sliced(tmp_path / "b", years, fmt="parquet")
    with pytest.raises(guard.TimeEncodingError):
        _slice(d, f)
    d, f = _sliced(tmp_path / "c", years, encoding="year")
    assert _slice(d, f).t.tolist() == [2020, 2021, 2022]
    assert _slice(d, f, "2022-06").t.tolist() == [2020, 2021]  # 2022 ends after 2022-06
    assert _slice(d, f, "2021-12").t.tolist() == [2020, 2021]
    p = guard.open_product(d, "datum_fit", window_end="2022-12")
    assert p.time_encoding == "year" and "encoding 'year' (declared)" in p.notes[0]
    d, f = _sliced(tmp_path / "d", years, encoding="yyyymm")  # wrong declaration
    with pytest.raises(guard.TimeEncodingError, match="190001-209912"):
        _slice(d, f)


def test_time_slice_epoch_seconds(tmp_path):
    days = [(2021, 6, 1), (2022, 12, 15), (2023, 1, 15), (2025, 6, 1)]
    sec = [int(pd.Timestamp(*x, tz="UTC").timestamp()) for x in days]
    for i, (vals, enc) in enumerate([(sec, None), (sec, "yyyymm"), (sec, "year"),
                                     ([s * 1000 for s in sec], None),
                                     ([s // 86400 for s in sec], None),
                                     ([s // 86400 for s in sec], "year"),
                                     ([s // 86400 for s in sec], "yyyymm")]):
        d, f = _sliced(tmp_path / f"e{i}", vals, encoding=enc)
        with pytest.raises(guard.TimeEncodingError):
            _slice(d, f)
    # the same instants as a datetime64 column (naive or tz-aware) slice correctly
    d, f = _sliced(tmp_path / "dt", [pd.Timestamp(*x) for x in days], fmt="parquet")
    assert [str(x)[:7] for x in _slice(d, f).t] == ["2021-06", "2022-12"]
    tz = [pd.Timestamp(*x, tz="Asia/Taipei") for x in days]
    d, f = _sliced(tmp_path / "tz", tz, encoding="iso", fmt="parquet")
    assert len(_slice(d, f)) == 2
    d, f = _sliced(tmp_path / "dy", [pd.Timestamp(*x) for x in days], encoding="yyyymm",
                   fmt="parquet")
    with pytest.raises(guard.TimeEncodingError, match="integer column"):
        _slice(d, f)


def test_time_slice_yyyymm_range(tmp_path):
    ok = [202111, 202212, 202301, 202506]
    d, f = _sliced(tmp_path / "ok", ok, encoding="yyyymm")
    assert _slice(d, f).t.tolist() == [202111, 202212]
    assert _slice(d, f, "2022-11").t.tolist() == [202111]
    d, f = _sliced(tmp_path / "undeclared", ok)                # YYYYMM must be declared
    with pytest.raises(guard.TimeEncodingError):
        _slice(d, f)
    for i, bad in enumerate([202213, 202200, 189912, 210001, 20230115, 2022, 0, -202212]):
        d, f = _sliced(tmp_path / f"b{i}", [202111, bad], encoding="yyyymm")
        with pytest.raises(guard.TimeEncodingError, match="190001-209912"):
            _slice(d, f)
    for bad in (1899, 2100, 202212):
        with pytest.raises(guard.TimeEncodingError, match="1900-2099"):
            guard.time_keys(pd.Series([2020, bad]), "year")
    assert guard.time_keys(pd.Series([190001, 209912]), "yyyymm").tolist() == [190001, 209912]
    assert guard.time_keys(pd.Series([1900, 2099]), "year").tolist() == [190012, 209912]


def test_time_slice_mixed_and_invalid_types(tmp_path):
    mixed = pd.Series(["2022-06", 202301, "2023-02"], dtype=object)
    for enc in ("iso", "yyyymm", "year"):
        with pytest.raises(guard.TimeEncodingError):
            guard.time_keys(mixed, enc)
    d, f = _sliced(tmp_path / "m", ["2022-06", "202301", "2023"])   # csv reads them as str
    with pytest.raises(guard.TimeEncodingError, match="ISO-8601"):
        _slice(d, f)
    d, f = _sliced(tmp_path / "y", ["2021", "2022", "2023"], fmt="parquet")  # bare years
    with pytest.raises(guard.TimeEncodingError):
        _slice(d, f)
    d, f = _sliced(tmp_path / "f", [2021.0, 2022.0, 2023.0], encoding="year", fmt="parquet")
    with pytest.raises(guard.TimeEncodingError, match="integer column"):
        _slice(d, f)
    d, f = _sliced(tmp_path / "na", [2021, None, 2023], encoding="year", fmt="parquet",
                   dtype="Int64")
    with pytest.raises(guard.TimeEncodingError, match="missing"):
        _slice(d, f)
    bad_iso = ["2022-13", "2022/06", " 2022-06", "2022-6", "0020-23", "2022-02-30",
               "2022-06-01T25:00", "２０２２-０６", "Jun 2022", "22-06"]
    for v in bad_iso:
        with pytest.raises(guard.TimeEncodingError):
            guard.time_keys(pd.Series(["2021-01", v]), "iso")
    for s in (pd.Series([True, False]), pd.Series(pd.period_range("2022-11", periods=2,
                                                                  freq="M")),
              pd.Series([1.5e9, 1.6e9]), pd.Series(["2022-06", None])):
        with pytest.raises(guard.TimeEncodingError):
            guard.time_keys(s, "iso")
    good = ["2022-06", "2022-12-31", "2022-12-31 23:59:59", "2023-01-01T00:00:00+08:00",
            "2022-12-01T00:00:00.5Z"]
    assert guard.time_keys(pd.Series(good), "iso").tolist() == [202206, 202212, 202212,
                                                                202301, 202212]
    cat = pd.Series(["2022-12", "2023-01", "2022-12"], dtype="category")
    assert guard.slice_mask(cat, "iso", "2022-12").tolist() == [True, False, True]
    with pytest.raises(guard.TimeEncodingError):
        guard.time_keys(pd.Series(["2022-12", "x"], dtype="category"), "iso")
    assert guard.slice_mask(pd.Series(["2022-12"], dtype="string"), "iso", "2022-12").all()


def test_time_slicing_declaration(tmp_path):
    years = [2021, 2022, 2023]
    d, f = _sliced(tmp_path / "u", years, encoding="epoch")    # unknown encoding
    guard.open_product(d, "scoring_target")                    # no slice: unaffected
    with pytest.raises(guard.TimeEncodingError, match="not one of"):
        guard.open_product(d, "datum_fit", window_end="2022-12")
    m = manifest.load_manifest(d)[0]
    assert any("not one of" in e for e in guard.time_slicing_errors(m, d))
    d, f = _sliced(tmp_path / "ok", years, encoding="year")
    m = manifest.load_manifest(d)[0]
    assert guard.time_slicing_errors(m, d) == []
    bad = dict(m, time_slicing=dict(m["time_slicing"], encoding="yyyymm"))
    assert any("190001-209912" in e for e in guard.time_slicing_errors(bad, d))
    none = dict(m, time_slicing={k: v for k, v in m["time_slicing"].items() if k != "encoding"})
    errs = guard.time_slicing_errors(none, d)
    assert any("declares no encoding" in e for e in errs) and any("ISO" in e for e in errs)
    assert guard.time_slicing_errors(none, require_explicit=False) == []
    gone = dict(m, time_slicing=dict(m["time_slicing"], column="month"))
    assert any("no parquet/csv output" in e for e in guard.time_slicing_errors(gone, d))
    assert guard.time_slicing_errors(dict(m, time_slicing=None), d) == []
    d, f = _sliced(tmp_path / "iso", ["2022-11", "2023-01"])   # A0 r2 style, no encoding
    p = guard.open_product(d, "datum_fit", window_end="2022-12")
    assert p.time_encoding == "iso" and "default" in p.notes[0]
    assert p.read_table(f).t.tolist() == ["2022-11"]


def test_time_slice_checks_rows_outside_the_use_flags(tmp_path):
    """A malformed time value is refused even on a row the use's flag filter would drop."""
    d = manifest.new_version_dir("x/obs", root=tmp_path / "dataprep")
    pd.DataFrame({"value": [1.0, 2.0], "sd": [np.nan, 0.5], "flag": [0, 4],
                  "method": ["obs", "m"], "product_version": "v1", "source_ids": "s",
                  "t": ["2022-06", "2023-13"]}).to_csv(d / "t.csv", index=False)
    (d / "validation.json").write_text("{}")
    code = tmp_path / "build.py"
    code.write_text("pass\n")
    manifest.write_manifest(d, manifest.build_manifest(
        d, product="x/obs", layer="L1", inputs=[], code=[code], parameters={},
        fit_window_end="2026-09", validation_report="validation.json",
        gate={"passed": True, "criteria": "test"}, allowed_uses=["datum_fit"],
        forbidden_uses=[], time_slicing={"column": "t", "params_fit_window_end": "2022-12",
                                         "encoding": "iso"}))
    with pytest.raises(guard.TimeEncodingError):
        _slice(d, "t.csv")


def test_overrides_only_tighten(tmp_path):
    d, _ = _product(tmp_path, allowed=("initial_conditions", "calibration_target", "validator"))
    with pytest.raises(ValueError, match="only tighten"):
        guard.open_product(d, "calibration_target", preorigin=False)
    with pytest.raises(ValueError, match="only tighten"):
        guard.open_product(d, "initial_conditions", require_gate=False)
    assert guard.open_product(d, "validator", preorigin=True).preorigin is True
    p = guard.open_product(d, "validator", verify=False)
    assert any("verify=False" in n for n in p.notes)
    d2, _ = _product(tmp_path / "g", passed=False,
                     allowed=("design_time_evidence", "after_the_fact_analysis"))
    guard.open_product(d2, "after_the_fact_analysis")
    with pytest.raises(guard.GateNotPassed):
        guard.open_product(d2, "after_the_fact_analysis", require_gate=True)


def test_exemption_rules(tmp_path):
    d, m = _product(tmp_path, product="changhua_prior/preorigin", fwe="2024-06",
                    allowed=("initial_conditions",), exemptions=[_LUIMAP],
                    non_exempt_fit_window_end="2022-12")
    errs = manifest.validate_manifest
    star = dict(m, exemptions=[dict(_LUIMAP, uses=["*"])])
    assert any("'*'" in e for e in errs(star))
    no_ne = {k: v for k, v in m.items() if k != "non_exempt_fit_window_end"}
    assert any("non_exempt_fit_window_end is missing" in e for e in errs(no_ne))
    assert any("!= max" in e for e in errs(dict(m, fit_window_end="2025-01")))
    never = dict(m, exemptions=[dict(_LUIMAP, uses=["initial_conditions", "selection"])])
    assert any("never be exempted" in e for e in errs(never))
    late_ne = dict(m, non_exempt_fit_window_end="2023-05")
    assert any("non_exempt_fit_window_end 2023-05" in e for e in errs(late_ne))
    with pytest.raises(guard.LeakageError):
        guard.check_use(late_ne, "initial_conditions")
    second = dict(_LUIMAP, input="crop survey 2023", data_end="2023-12", uses=["prior"])
    two = dict(m, exemptions=[_LUIMAP, second])
    with pytest.raises(guard.LeakageError, match="crop survey"):
        guard.check_use(two, "initial_conditions")
    for name in ("x/pre_origin", "x/Pre-Origin", "y_preorigin"):
        bad = dict(m, product=name, exemptions=[], allowed_uses=["validator"])
        bad.pop("non_exempt_fit_window_end")
        assert any("pre-origin product" in e for e in errs(bad)), name


def test_per_output_window(tmp_path):
    d = tmp_path / "pilot"
    d.mkdir()
    (d / "pre.csv").write_text("flag,value\n0,1\n")
    (d / "post.csv").write_text("flag,value\n0,2\n")
    m = manifest.build_manifest(
        d, product="_pilots/x", layer="pilot", version="2026-10-12.r2", inputs=[], code=[],
        parameters={}, fit_window_end="2026-09", validation_report=None,
        gate={"passed": False, "criteria": "pilot"},
        allowed_uses=["design_time_evidence", "after_the_fact_analysis"], forbidden_uses=[],
        outputs=[{"path": "pre.csv", "fit_window_end": "2022-12"}, {"path": "post.csv"}])
    manifest.write_manifest(d, m)
    p = guard.open_product(d, "design_time_evidence")
    assert p.read_table("pre.csv").value.tolist() == [1]
    with pytest.raises(guard.LeakageError):
        p.path("post.csv")
    a = guard.open_product(d, "after_the_fact_analysis")
    assert a.path("post.csv").exists()
    bad = dict(m, outputs=[dict(o, fit_window_end="2027-01") for o in m["outputs"]])
    assert any("> product fit_window_end" in e for e in manifest.validate_manifest(bad))
    only_post = dict(m, outputs=[o for o in m["outputs"] if o["path"] == "post.csv"])
    assert any("can never pass" in e for e in manifest.validate_manifest(only_post))


def test_read_table_keeps_flag_filter_with_column_subset(tmp_path):
    d, _ = _product(tmp_path, product="x/obs", allowed=("validator", "calibration_target"),
                    flag_values=(0, 1, 2, 3, 5))
    t = guard.open_product(d, "calibration_target").read_table("fill.csv", columns=["value"])
    assert list(t.columns) == ["value"] and t.value.tolist() == [0.0, 1.0, 2.0]


def test_guard_gate_and_hash(tmp_path):
    d, _ = _product(tmp_path, passed=False,
                    allowed=("initial_conditions", "design_time_evidence"))
    with pytest.raises(guard.GateNotPassed):
        guard.open_product(d, "initial_conditions")
    assert guard.open_product(d, "design_time_evidence").use == "design_time_evidence"
    (d / "fill.csv").write_text("x\n")
    with pytest.raises(manifest.HashMismatch):
        guard.open_product(d, "design_time_evidence")
    guard.open_product(d, "design_time_evidence", verify=False)


def test_flag_filtering_u1_u2(tmp_path):
    df = pd.DataFrame({"flag": [0, 1, 2, 3, 4, 5, 6, 8, 9]})
    assert guard.filter_flags(df, "scoring_target").flag.tolist() == [0, 1]
    assert guard.filter_flags(df, "calibration_target").flag.tolist() == [0, 1, 2]
    assert guard.filter_flags(df, "initial_conditions").flag.tolist() == [0, 1, 2, 3, 4]
    assert guard.filter_flags(df, "validator").flag.tolist() == [0, 1, 2, 3, 4, 5, 6]
    with pytest.raises(guard.UseNotAllowed):
        guard.filter_flags(df, "baseline", strict=True)
    d, _ = _product(tmp_path, allowed=("initial_conditions", "calibration_target"))
    t = guard.open_product(d, "calibration_target").read_table("fill.csv")
    assert t.flag.tolist() == [0, 1, 2]


def test_uses_vocabulary():
    assert set(uses.USES) >= uses.PRE_ORIGIN_USES
    assert set(uses.USES) >= uses.OBSERVED_ONLY_USES
    assert uses.normalise_role("Design-time evidence") == "design_time_evidence"
    assert uses.normalise_role("prior-width input") == "prior_width_input"
    with pytest.raises(ValueError):
        uses.normalise_role("target")


# ----------------------------------------------------------------------------- ledger (U6)
def test_ledger_fragment_and_assemble(tmp_path):
    rows_dir = tmp_path / "rows"
    r1 = {"number_id": "P1.mu_elec", "source": "tpc_d007012", "role": "prior",
          "product_version": "county_electricity/preorigin/v1", "note": "Changhua total"}
    ledger.append_rows("P1", [r1, r1], rows_dir=rows_dir)
    rows = list(pd.read_csv(rows_dir / "P1.csv").itertuples())
    assert len(rows) == 1 and rows[0].wp == "P1"
    with pytest.raises(ledger.LedgerConflict):
        ledger.append_rows("P1", [dict(r1, role="validator")], rows_dir=rows_dir)
    ledger.append_rows("A0", [dict(r1, number_id="A0.x", role="input")], rows_dir=rows_dir)
    out, n = ledger.assemble(rows_dir=rows_dir, out=tmp_path / "data_roles.csv")
    assert n == 2 and list(pd.read_csv(out).columns) == list(ledger.COLUMNS)
    (rows_dir / "X.csv").write_text("number_id,source,role,product_version,wp,note\n"
                                    "P1.mu_elec,tpc_d007012,validator,v1,X,\n")
    with pytest.raises(ledger.LedgerConflict):
        ledger.assemble(rows_dir=rows_dir, out=tmp_path / "data_roles.csv")
    with pytest.raises(ValueError):
        ledger.append_rows("P1", [dict(r1, role="target")], rows_dir=rows_dir)


def test_ledger_sources_explicit_and_prefix_conflicts(tmp_path):
    rows_dir = tmp_path / "rows"
    base = {"number_id": "A0.n", "role": "input", "product_version": "v1"}
    for bad in ("results/dataprep/_pilots/*/MANIFEST.json", "a/{b,c}.npz", "x/f?.csv"):
        with pytest.raises(ValueError, match="pattern"):
            ledger.append_rows("A0", [dict(base, source=bad)], rows_dir=rows_dir)
    assert ledger.normalise_source("./a//b/") == "a/b"
    assert ledger.normalise_source(str(ledger.REPO / "results" / "x.csv")) == "results/x.csv"
    ledger.append_rows("A0", [dict(base, source="results/dataprep/_pilots/tpc/")],
                       rows_dir=rows_dir)
    with pytest.raises(ledger.LedgerConflict):   # file inside a directory with another role
        ledger.append_rows("A0", [dict(base, source="results/dataprep/_pilots/tpc/c.csv",
                                       role="validator")], rows_dir=rows_dir)
    ledger.append_rows("A0", [dict(base, source="results/dataprep/_pilots/tpc2/c.csv",
                                   role="validator")], rows_dir=rows_dir)   # not a prefix
    assert ledger.unledgered(["A0.n", "A0.missing"], rows_dir=rows_dir) == ["A0.missing"]


# ----------------------------------------------------------------------------- source (L0)
def test_source_record_and_copy(tmp_path):
    raw = tmp_path / "pilot" / "f.csv"
    raw.parent.mkdir()
    raw.write_text("a,b\n1,2\n")
    l0 = tmp_path / "l0" / "src"
    source.copy_into_l0(raw, l0 / "f.csv")
    source.copy_into_l0(raw, l0 / "f.csv")              # identical: no-op
    other = tmp_path / "pilot" / "g.csv"
    other.write_text("different")
    with pytest.raises(source.SourceError):
        source.copy_into_l0(other, l0 / "f.csv")
    rec = source.build_source(l0, source_id="src", title="t", publisher="p", licence="OGDL-1.0",
                              fetched="2026-10-12",
                              files=[{"file": "f.csv", "url": "https://example.org/f.csv"}])
    source.write_source_json(l0, rec)
    source.write_source_json(l0, rec)                   # identical: no-op
    assert source.check_source_file(l0 / "SOURCE.json")["status"] == "ok"
    with pytest.raises(source.SourceError):
        source.write_source_json(l0, dict(rec, title="changed"))
    bad = source.build_source(l0, source_id="s", title="t", publisher="p", licence="l",
                              fetched="2026-10-12", files=[{"file": "f.csv", "url": None}])
    assert any("no url" in e for e in source.validate_source(bad, l0))
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "x.pdf").write_bytes(b"%PDF")
    (legacy / "SOURCE.json").write_text(json.dumps(
        {"fetched": "2026-10-11", "files": [{"file": "x.pdf", "url": "u", "bytes": 4}]}))
    assert source.check_source_file(legacy / "SOURCE.json")["status"] == "legacy"


# ----------------------------------------------------------------------------- check CLI
def test_check_cli(tmp_path, capsys):
    d, _ = _product(tmp_path)
    root = tmp_path / "dataprep"
    assert check.main([str(root), "--no-sources", "--json", str(tmp_path / "r.json")]) == 0
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["summary"]["manifests"] == 1 and rep["summary"]["manifests_failed"] == 0
    (d / "validation.json").write_text('{"x": 1}')
    assert check.main([str(root), "--no-sources"]) == 1
    assert "sha256" in capsys.readouterr().out


def test_assert_releasable(tmp_path):
    d, _ = _product(tmp_path / "a")
    assert guard.assert_releasable(d)["gate"]["passed"] is True
    d2, _ = _product(tmp_path / "b", passed=False)
    with pytest.raises(guard.GateNotPassed):
        guard.assert_releasable(d2)
    (tmp_path / "a" / "input.txt").write_text("changed")      # an input moved under it
    with pytest.raises(manifest.HashMismatch):
        guard.assert_releasable(d)


def test_check_orphans_and_registry(tmp_path, capsys):
    l0 = tmp_path / "l0"
    (l0 / "src").mkdir(parents=True)
    (l0 / "src" / "a.csv").write_text("a\n")
    rec = source.build_source(l0 / "src", source_id="src", title="t", publisher="p",
                              licence="l", fetched="2026-10-12",
                              files=[{"file": "a.csv", "url": "https://example.org/a"}])
    source.write_source_json(l0 / "src", rec)
    (l0 / "old").mkdir()
    (l0 / "old" / "b.parquet").write_bytes(b"pq")
    reg_root = tmp_path / "dataprep" / "l0_registry"
    args = ["--sources-root", str(l0), "--registry-root", str(reg_root),
            "--json", str(tmp_path / "r.json"), str(tmp_path / "dataprep")]
    assert check.main(args) == 0
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["summary"]["orphans"] == 1 and rep["orphans"][0].endswith("old/b.parquet")
    assert check.main(args + ["--fail-on-orphans"]) == 1
    d = manifest.new_version_dir("l0_registry", root=tmp_path / "dataprep")
    (d / "registry.json").write_text("{}")
    code = tmp_path / "build_registry.py"
    code.write_text("pass\n")
    m = manifest.build_manifest(
        d, product="l0_registry", layer="registry", inputs=[l0 / "old" / "b.parquet"],
        code=[code], parameters={}, fit_window_end="2026-10", validation_report=None,
        gate={"passed": False, "criteria": "registry"}, allowed_uses=[],
        forbidden_uses=sorted(uses.USES))
    manifest.write_manifest(d, m)
    assert check.main(args + ["--fail-on-orphans"]) == 0
    (l0 / "old" / "b.parquet").write_bytes(b"changed")      # a registered L0 file changes
    assert check.main(args) == 1
    capsys.readouterr()
