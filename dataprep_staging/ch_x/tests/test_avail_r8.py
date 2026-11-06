"""CPU tests of the round-8 availability looks (avail_bc11.py, plan #12; avail_chb0809.py, plan
#18) on SYNTHETIC files only: no ls_cache record, no head and no registry is touched; stations,
ls_cache files, ring sites, looks.jsonl and the pinned outputs live in pytest's tmp dirs
(``chx_io.REPO`` is pointed at the tmp dir). The value column of every synthetic ls_cache file
is physically corrupted, so any read of it would raise.

    cd <repository root>
    PYTHONPATH=dataprep_staging ~/miniforge3/envs/hydro/bin/python -m pytest -q \
        -p no:cacheprovider dataprep_staging/ch_x/tests/test_avail_r8.py
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pads = pytest.importorskip("pyarrow.dataset")

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for p in (str(X), str(REPO / "dataprep_staging"), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)
import avail_bc11 as A  # noqa: E402
import avail_chb0809 as B  # noqa: E402
import avail_lib as AL  # noqa: E402
import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402

ZH = {"yunlin": "雲林縣", "changhua": "彰化縣", "other": "嘉義縣"}
_PARQUET_FILE = pq.ParquetFile                  # the real class, for the synthetic writer only
LS = "ls_cache"


# ----------------------------------------------------------------------------- synthetic
def ts_for(months, days: int = 20, hours: int = 12) -> np.ndarray:
    """Timestamps giving exactly ``days`` days of ``hours`` distinct hours in each month."""
    out = []
    for m in months:
        for d in range(1, days + 1):
            day = pd.Timestamp(f"{m}-{d:02d}")
            out.extend(day + pd.Timedelta(hours=h) for h in range(hours))
    return pd.DatetimeIndex(out).values


def write_raw(path: Path, ts) -> None:
    """A raw file with a datetime column and a value column whose bytes are corrupted."""
    n = len(ts)
    tab = pa.table({"datetime": pa.array(np.asarray(ts, "datetime64[ns]")),
                    "value": pa.array(np.full(n, -999.0))})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tab, path, compression="NONE", use_dictionary=False)
    md = _PARQUET_FILE(path).metadata
    b = bytearray(path.read_bytes())
    for rg in range(md.num_row_groups):
        for ci in range(md.num_columns):
            c = md.row_group(rg).column(ci)
            if c.path_in_schema == "value":
                off = c.dictionary_page_offset or c.data_page_offset
                b[off:off + c.total_compressed_size] = b"\xff" * c.total_compressed_size
    path.write_bytes(bytes(b))


def months(first: str, last: str) -> list[str]:
    return [str(p) for p in pd.period_range(first, last, freq="M")]


def setup_repo(tmp_path, monkeypatch, wells: dict, nest: tuple = (), extra_no_file=()):
    """wells: {sid: (county, [months])}; ``nest`` sids sit on a ring site. Returns the
    prereg dir."""
    monkeypatch.setattr(IO, "REPO", tmp_path)
    sids = list(wells) + list(extra_no_file) + ["09999999"]
    county = [wells[s][0] if s in wells else "yunlin" for s in sids[:-1]] + ["yunlin"]
    xy = []
    for i, s in enumerate(sids):
        x, y = (200000.0, 2600000.0) if s in nest else (150000.0 + 1500 * i, 2650000.0)
        xy.append(f"{x:.0f} {y:.0f}")
    zone = [50] * (len(sids) - 1) + [40]                 # the last one is outside the fan
    st = pd.DataFrame({"sid": sids, "CountyName": [ZH[c] for c in county],
                       "LocationByTWD97": xy, "GroundwaterLayerCode": ["1"] * len(sids),
                       "GroundwaterZoneIdentifier": zone,
                       "WellElevation": [10.0] * len(sids), "WellDepth": [90.0] * len(sids)})
    (tmp_path / K.STATIONS).parent.mkdir(parents=True, exist_ok=True)
    st.to_parquet(tmp_path / K.STATIONS)
    lc = tmp_path / K.LS_CACHE
    lc.mkdir(parents=True, exist_ok=True)
    for s, (_, ms) in wells.items():
        write_raw(lc / K.LS_RAW.format(sid=s), ts_for(ms))
    write_raw(lc / K.LS_RAW.format(sid="09999999"), ts_for(["2011-01"]))   # not zone 50
    (tmp_path / K.MLCW_STATIONS).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sub_id": ["r"], "lon": [0], "lat": [0], "x": [200100.0],
                  "y": [2600000.0]}).to_csv(tmp_path / K.MLCW_STATIONS, index=False)
    return tmp_path / "prereg"


def log(mod, pdir, tmp_path):
    """Log the look as ``look --write`` would (bypassing the probe requirement)."""
    from harness import prereg3

    rec = mod.look_record(mod.candidates(), prereg_dir=pdir, registries=[])
    prereg3.record_look(rec["units"], rec["data_scope"], rec["wp"], rec["purpose"],
                        members=rec["members"], note=rec["note"], prereg_dir=pdir,
                        rows_dir=tmp_path / "rows", registries=[])
    return rec


def seed_required_looks(pdir, tmp_path, ids):
    """Synthetic earlier looks on the X5EV and I1CHB units the bc11 note discloses."""
    from harness import prereg3

    for u in A.REQUIRED_PROBE_UNITS:
        prereg3.record_look([u], "heads:2010-01..2011-12", "X5", "reported_after_score",
                            members={u: {"ids": ids[:2], "window": "2010-01..2011-12"}},
                            prereg_dir=pdir, rows_dir=tmp_path / "rows", registries=[])


@pytest.fixture
def read_spy(monkeypatch):
    """Only read_schema and read_table(columns=['datetime']) may touch an ls_cache file;
    read_metadata, ParquetFile, read_pandas, pyarrow.dataset and pandas readers refuse."""
    calls = []
    depth = [0]
    real_rt, real_rs = pq.read_table, pq.read_schema

    def is_ls(src) -> bool:
        return LS in str(src)

    def rt(source, *a, columns=None, **kw):
        calls.append(("read_table", str(source), None if columns is None else list(columns)))
        if is_ls(source) and list(columns or []) != ["datetime"]:
            raise AssertionError(f"ls_cache read beyond datetime: {source} {columns}")
        depth[0] += 1
        try:
            return real_rt(source, *a, columns=columns, **kw)
        finally:
            depth[0] -= 1

    def rs(source, *a, **kw):
        calls.append(("read_schema", str(source), None))
        depth[0] += 1
        try:
            sch = real_rs(source, *a, **kw)
        finally:
            depth[0] -= 1
        assert not any(b"statistic" in k.lower() for k in (sch.metadata or {}))
        return sch

    def refuse(name, real):
        def f(source=None, *a, **kw):
            if depth[0] == 0 and is_ls(source):
                raise AssertionError(f"{name} on an ls_cache file: {source}")
            return real(source, *a, **kw)
        return f

    monkeypatch.setattr(pq, "read_table", rt)
    monkeypatch.setattr(pq, "read_schema", rs)
    for name in ("read_metadata", "ParquetFile", "read_pandas"):
        monkeypatch.setattr(pq, name, refuse(f"pq.{name}", getattr(pq, name)))
    monkeypatch.setattr(pads, "dataset", refuse("pyarrow.dataset", pads.dataset))
    for name in ("read_parquet", "read_csv", "read_feather", "read_table"):
        monkeypatch.setattr(pd, name, refuse(f"pd.{name}", getattr(pd, name)))
    return calls


Y11 = months("2011-01", "2011-12")


def good_bc11() -> list[str]:
    return ["2010-12"] + Y11


# ----------------------------------------------------------------------------- A: constants
def test_bc11_stop_rule_constants_are_the_declared_ones():
    assert A.UNIT == "I1BC11:ls_cache_dates_2009_2011"
    assert A.WINDOW == ("2009-10", "2011-12") and A.SCOPE == "heads:2009-10..2011-12"
    assert (A.MIN_POOLED_WELLS, A.MIN_YEAR_MONTHS, A.COUNTY_LINE_MIN_WELLS) == (25, 9, 10)
    assert A.SCORED_YEAR == ("2011-01", "2011-12") and A.ORIGIN == "2010-12"
    assert A.ORIGIN_WINDOW == ("2010-10", "2010-12") and A.MIN_ORIGIN_MONTHS == 1
    assert A.ORIGIN2 == "2009-12" and A.ORIGIN2_YEAR == ("2010-01", "2010-12")
    assert A.ORIGIN2_WINDOW == ("2009-10", "2009-12")
    assert A.EXCLUDE_NEST and A.NEST_RADIUS_M == 1000.0
    assert A.COUNTIES == ("yunlin", "changhua")
    assert A.OUT == "results/twin_runs/chx/backcast_2010_11/avail/avail.json"
    for s in (">= 25 pooled", ">= 9 available months", "2010-10..2010-12", ">= 10 qualifying",
              "second origin 2009-12", "1000 m", "excluded from the gating count"):
        assert s in A.RULE_TEXT
    for s in ("X5EV", "07010211", "E26", "time-reversed", "mechanism", "I1CHB",
              "excluded only from the gating count", *A.X5EV_UNITS, *A.I1CHB_UNITS):
        assert s in A.DISCLOSURE_TEXT
    c = A.constants()
    assert c["window"] == ["2009-10", "2011-12"] and c["unit"] == A.UNIT
    assert c["nest_radius_m"] == 1000.0 and c["min_pooled_wells"] == 25
    assert len(AL.constants_sha(c)) == 64


# ----------------------------------------------------------------------------- A: rule
def _pool(n_y: int, n_c: int, ms=None):
    ms = good_bc11() if ms is None else ms
    m = {f"09{i:06d}": list(ms) for i in range(n_y)}
    m.update({f"07{i:06d}": list(ms) for i in range(n_c)})
    county = {s: ("yunlin" if s.startswith("09") else "changhua") for s in m}
    return m, county


@pytest.mark.parametrize("n,passed", [(24, False), (25, True)])
def test_bc11_pooled_boundary(n, passed):
    m, c = _pool(n - 10, 10)
    f = A.stop_rule(m, c, set())
    assert f["n_eligible"] == n and f["passed"] is passed


def test_bc11_months_of_2011_boundary():
    eight = ["2010-12"] + Y11[:8]
    nine = ["2010-12"] + Y11[:9]
    assert not A.qualifies(eight) and A.qualifies(nine)
    m, c = _pool(25, 0, eight)
    assert A.stop_rule(m, c, set())["n_eligible"] == 0
    m, c = _pool(25, 0, nine)
    assert A.stop_rule(m, c, set())["passed"]


@pytest.mark.parametrize("origin,ok", [("2010-09", False), ("2010-10", True),
                                       ("2010-11", True), ("2010-12", True)])
def test_bc11_origin_window_edges(origin, ok):
    assert A.qualifies([origin] + Y11) is ok


def test_bc11_january_2011_is_not_an_origin_month():
    assert not A.qualifies(Y11)                         # no month in 2010-10..2010-12


def test_bc11_county_line_needs_ten():
    m, c = _pool(16, 9)
    f = A.stop_rule(m, c, set())
    assert f["passed"] and f["county_line"] == {"yunlin": True, "changhua": False}
    m, c = _pool(15, 10)
    f = A.stop_rule(m, c, set())
    assert f["county_line"] == {"yunlin": True, "changhua": True}
    assert f["n_eligible_by_county"] == {"yunlin": 15, "changhua": 10}


def test_bc11_nest_wells_gate_excluded_and_reported_included():
    m, c = _pool(15, 10)
    f = A.stop_rule(m, c, {"07000000"})
    assert f["gating"] == "nest_excluded"
    assert f["n_eligible"] == 24 and not f["passed"] and f["n_nest"] == 1
    assert f["county_line"]["changhua"] is False
    assert f["nest_included"]["n"] == 25 and f["nest_excluded"]["n"] == 24
    assert f["report_only_other_count"] == {"which": "nest_included", "n": 25,
                                            "by_county": {"yunlin": 15, "changhua": 10}}


def test_bc11_both_counts_switch_with_the_constant(monkeypatch):
    m, c = _pool(15, 10)
    monkeypatch.setattr(A, "EXCLUDE_NEST", False)
    f = A.stop_rule(m, c, {"07000000"})
    assert f["gating"] == "nest_included" and f["passed"] and f["n_eligible"] == 25
    assert f["report_only_other_count"]["n"] == 24


def test_bc11_second_origin_is_report_only():
    o2 = ["2009-11"] + months("2010-01", "2010-09")       # 9 months of 2010, Nov 2009
    m, c = _pool(25, 0, o2 + good_bc11())
    f = A.stop_rule(m, c, set())
    assert f["passed"] and f["origin2_report_only"]["n_eligible"] == 25
    m, c = _pool(25, 0, o2)                              # origin 2009-12 only: STOP
    f = A.stop_rule(m, c, set())
    assert not f["passed"] and f["origin2_report_only"]["n_eligible"] == 25
    m, c = _pool(25, 0, ["2009-09"] + months("2010-01", "2010-09") + good_bc11())
    assert A.stop_rule(m, c, set())["origin2_report_only"]["n_eligible"] == 0


def test_bc11_other_counties_are_not_pooled():
    m, c = _pool(25, 0)
    c = {s: "other" for s in c}
    f = A.stop_rule(m, c, set())
    assert f["n_candidates"] == 0 and not f["passed"]


# ----------------------------------------------------------------------------- A: files
def _bc11_wells(n_y=16, n_c=10):
    w = {f"09{i:06d}": ("yunlin", good_bc11()) for i in range(n_y)}
    w.update({f"07{i:06d}": ("changhua", good_bc11()) for i in range(n_c)})
    return w


def test_bc11_look_record_is_scoped(tmp_path, monkeypatch):
    from harness import prereg3

    pdir = setup_repo(tmp_path, monkeypatch, _bc11_wells(3, 2),
                      extra_no_file=("09888888",))
    ids = A.candidates()
    assert ids == sorted(_bc11_wells(3, 2))            # no file / not zone 50 left out
    rec = A.look_record(ids, prereg_dir=pdir, registries=[])
    assert rec["units"] == [A.UNIT] and set(rec["members"]) == {A.UNIT}
    m = rec["members"][A.UNIT]
    assert "universal" not in m and m["ids"] == ids
    assert m["window"] == "2009-10..2011-12" and not prereg3.member_errors(m)
    assert prereg3.scope_windows(rec["data_scope"]) is not None
    assert rec["purpose"] == "availability_check" and A.RULE_TEXT in rec["note"]
    assert A.DISCLOSURE_TEXT in rec["note"] and AL.sha_tag(A.constants()) in rec["note"]


def test_bc11_count_refuses_then_counts_once_without_reading_values(tmp_path, monkeypatch,
                                                                     read_spy, capsys):
    wells = _bc11_wells()
    wells["09000001"] = ("yunlin", ["2010-12"] + Y11[:8])            # 8 months: out
    wells["09000002"] = ("yunlin", ["2010-09"] + Y11)                # origin too early: out
    pdir = setup_repo(tmp_path, monkeypatch, wells, nest=("07000000",))
    with pytest.raises(SystemExit, match="log the availability look"):
        A.count(prereg_dir=pdir)
    assert not (tmp_path / A.OUT).exists()
    with pytest.raises(OSError):                       # the value column cannot be read
        _PARQUET_FILE(AL.AV.raw_path("09000003")).read(columns=["value"])
    log(A, pdir, tmp_path)
    capsys.readouterr()
    rep = A.count(prereg_dir=pdir)
    out = capsys.readouterr().out
    f = rep["feasibility"]
    # 16 + 10 good wells, two Yunlin wells fail the rule, one Changhua nest well excluded
    assert f["n_eligible"] == 23 and not f["passed"] and f["nest_included"]["n"] == 24
    assert f["n_eligible_by_county"] == {"yunlin": 14, "changhua": 9}
    assert rep["nest_wells"] == ["07000000"]
    ls = [c for c in read_spy if LS in c[1]]
    assert {c[0] for c in ls} == {"read_table", "read_schema"}
    assert [c[2] for c in ls if c[0] == "read_table"] == [["datetime"]] * len(wells)
    assert "STOP" in out and "23" in out and "24" in out
    assert not any(s in out for s in wells) and "-999" not in out
    assert json.loads((tmp_path / A.OUT).read_text())["unit"] == A.UNIT
    with pytest.raises(SystemExit, match="runs once"):
        A.count(prereg_dir=pdir)


def test_bc11_count_has_no_output_choice():
    assert list(inspect.signature(A.count).parameters) == ["prereg_dir"]
    assert list(inspect.signature(B.count).parameters) == ["prereg_dir"]
    for mod in (A, B):
        with pytest.raises(SystemExit):
            mod.main(["count", "--out", "elsewhere"])


@pytest.mark.parametrize("name,value", [("MIN_POOLED_WELLS", 24), ("ORIGIN_WINDOW",
                                         ("2010-09", "2010-12")), ("EXCLUDE_NEST", False)])
def test_bc11_count_refuses_a_rule_changed_after_the_look(tmp_path, monkeypatch, name, value):
    pdir = setup_repo(tmp_path, monkeypatch, _bc11_wells(3, 2))
    log(A, pdir, tmp_path)
    monkeypatch.setattr(A, name, value)
    with pytest.raises(SystemExit, match="constants hash and rule text"):
        A.count(prereg_dir=pdir)
    assert not (tmp_path / A.OUT).exists()


def test_bc11_count_refuses_a_changed_rule_text(tmp_path, monkeypatch):
    pdir = setup_repo(tmp_path, monkeypatch, _bc11_wells(3, 2))
    log(A, pdir, tmp_path)
    monkeypatch.setattr(A, "RULE_TEXT", A.RULE_TEXT.replace(">= 25", ">= 20"))
    with pytest.raises(SystemExit, match="constants hash and rule text"):
        A.count(prereg_dir=pdir)


def test_bc11_count_refuses_when_candidates_changed(tmp_path, monkeypatch):
    pdir = setup_repo(tmp_path, monkeypatch, _bc11_wells(3, 2))
    log(A, pdir, tmp_path)
    new = "07555555"
    st = pd.read_parquet(tmp_path / K.STATIONS)
    st.loc[len(st)] = [new, ZH["changhua"], "160000 2650000", "1", 50, 10.0, 90.0]
    st.to_parquet(tmp_path / K.STATIONS)
    write_raw(tmp_path / K.LS_CACHE / K.LS_RAW.format(sid=new), ts_for(Y11))
    with pytest.raises(SystemExit, match="differs from the logged members"):
        A.count(prereg_dir=pdir)


def test_bc11_twenty_five_pass_through_files(tmp_path, monkeypatch):
    wells = _bc11_wells(15, 10)
    wells["07000099"] = ("changhua", ["2010-10"] + Y11[:9])           # edge cases qualify
    pdir = setup_repo(tmp_path, monkeypatch, wells)
    log(A, pdir, tmp_path)
    f = A.count(prereg_dir=pdir)["feasibility"]
    assert f["n_eligible"] == 26 and f["passed"] and all(f["county_line"].values())


def test_bc11_origin2_through_files_and_2009_09_ignored(tmp_path, monkeypatch):
    o2 = ["2009-09", "2009-10", "2009-11", "2009-12"] + months("2010-01", "2010-09")
    wells = {"09000000": ("yunlin", o2), "09000001": ("yunlin", o2[1:2] + o2[4:])}
    wells["09000002"] = ("yunlin", ["2009-09"] + months("2010-01", "2010-09"))  # Sep only
    pdir = setup_repo(tmp_path, monkeypatch, wells)
    log(A, pdir, tmp_path)
    rep = A.count(prereg_dir=pdir)
    assert rep["n_months_by_well"] == {"09000000": 12, "09000001": 10, "09000002": 9}
    o = rep["feasibility"]["origin2_report_only"]
    assert o["nest_excluded"]["wells"] == ["09000000", "09000001"] and o["n_eligible"] == 2
    assert rep["feasibility"]["n_eligible"] == 0 and not rep["feasibility"]["passed"]


def test_bc11_thin_month_is_not_available(tmp_path, monkeypatch):
    pdir = setup_repo(tmp_path, monkeypatch, {"09000000": ("yunlin", [])})
    lc = tmp_path / K.LS_CACHE / K.LS_RAW.format(sid="09000000")
    ts = np.concatenate([ts_for(["2010-12"]), ts_for(Y11[:8]), ts_for(["2011-09"], days=19),
                         ts_for(["2011-10"], hours=11)])
    write_raw(lc, ts)
    log(A, pdir, tmp_path)
    rep = A.count(prereg_dir=pdir)
    assert rep["n_months_by_well"]["09000000"] == 9
    assert rep["feasibility"]["n_eligible"] == 0


def test_bc11_probe_reports_earlier_overlapping_looks(tmp_path, monkeypatch):
    from harness import prereg3

    pdir = setup_repo(tmp_path, monkeypatch, _bc11_wells(3, 2))
    ids = A.candidates()
    prereg3.record_look(["X5EV:confined_ls_cache_2010_2011"], "heads:2010-01..2011-12", "X5",
                        "reported_after_score", members={"X5EV:confined_ls_cache_2010_2011": {
                            "ids": ids[:2], "window": "2010-01..2011-12"}},
                        prereg_dir=pdir, rows_dir=tmp_path / "rows", registries=[])
    prereg3.record_look(["Z:later"], "heads:2013-01..2014-12", "Z", "probe",
                        members={"Z:later": {"ids": ids}}, prereg_dir=pdir,
                        rows_dir=tmp_path / "rows", registries=[])
    rec = A.look_record(ids, prereg_dir=pdir, registries=[])
    assert [(r["unit"], r["n_shared"]) for r in rec["probe"]] == [
        ("X5EV:confined_ls_cache_2010_2011", 2)]
    assert "X5EV:confined_ls_cache_2010_2011 (reported_after_score" in rec["note"]
    assert AL.missing_units(rec["probe"], A.REQUIRED_PROBE_UNITS) == [
        u for u in A.REQUIRED_PROBE_UNITS if u != "X5EV:confined_ls_cache_2010_2011"]


def test_bc11_cli_look_requires_probe_units_and_refuses_a_second_write(tmp_path, monkeypatch,
                                                                       capsys):
    pdir = setup_repo(tmp_path, monkeypatch, _bc11_wells(3, 2))
    ids = A.candidates()
    assert A.main(["look", "--validate"], prereg_dir=pdir, registries=[]) == 1
    with pytest.raises(SystemExit, match="missing from the probe"):
        A.main(["look", "--write"], prereg_dir=pdir, registries=[])
    assert AL.logged_look(A.UNIT, pdir) is None
    seed_required_looks(pdir, tmp_path, ids)
    capsys.readouterr()
    assert A.main(["look", "--validate"], prereg_dir=pdir, registries=[]) == 0
    assert A.main(["look", "--write"], prereg_dir=pdir, registries=[]) == 0
    out = capsys.readouterr().out
    assert not any(s in out for s in ids)              # no member station id on stdout
    assert f"{len(ids)} members" in out and AL.sha_tag(A.constants()) in out
    assert all(u in out for u in A.REQUIRED_PROBE_UNITS)
    lk = AL.logged_look(A.UNIT, pdir)
    assert AL.note_matches(lk, A.constants(), A.RULE_TEXT)
    with pytest.raises(SystemExit, match="already logged"):
        A.main(["look", "--write"], prereg_dir=pdir, registries=[])


def test_logged_look_refuses_duplicates(tmp_path, monkeypatch):
    pdir = setup_repo(tmp_path, monkeypatch, _bc11_wells(2, 1))
    log(A, pdir, tmp_path)
    assert AL.logged_look(A.UNIT, pdir) is not None
    assert AL.logged_look(B.UNIT, pdir) is None        # units are distinct
    log(A, pdir, tmp_path)
    with pytest.raises(SystemExit, match="runs once"):
        AL.logged_look(A.UNIT, pdir)


# ----------------------------------------------------------------------------- B
W0809 = months("2008-01", "2009-12")


def test_chb0809_constants_are_the_declared_ones():
    assert B.UNIT == "I1CHB0:ls_cache_rowcounts_2008_2009"
    assert B.WINDOW == ("2008-01", "2009-12") and B.SCOPE == "heads:2008-01..2009-12"
    assert (B.MIN_WELLS, B.MIN_MONTHS) == (10, 18) and B.COUNTIES == ("changhua",)
    assert B.EXCLUDE_NEST and "not decidable" in B.STOP_LINE
    assert B.OUT == "results/twin_runs/chx/backcast_2008_09/avail/avail.json"
    for s in (">= 10 Changhua wells", ">= 18 available months", "2008-01..2009-12",
              "excluded from the gating count", "report-only", "1000 m"):
        assert s in B.RULE_TEXT
    assert "moea_public_well_tables:2008-01..2024-12" in B.DISCLOSURE_TEXT
    c = B.constants()
    assert (c["min_wells"], c["min_months"], c["exclude_nest"]) == (10, 18, True)
    assert c["unit"] == B.UNIT and c["window"] == ["2008-01", "2009-12"]


def _ch(n: int, ms):
    m = {f"07{i:06d}": list(ms) for i in range(n)}
    return m, {s: "changhua" for s in m}


@pytest.mark.parametrize("k,passed", [(17, False), (18, True)])
def test_chb0809_months_boundary(k, passed):
    m, c = _ch(10, W0809[:k])
    assert B.stop_rule(m, c, set())["passed"] is passed


@pytest.mark.parametrize("n,passed", [(9, False), (10, True)])
def test_chb0809_wells_boundary(n, passed):
    m, c = _ch(n, W0809[:18])
    assert B.stop_rule(m, c, set())["passed"] is passed


def test_chb0809_window_edges_both_counts_and_county(monkeypatch):
    m, c = _ch(10, months("2007-01", "2007-12") + W0809[:17] + months("2010-01", "2010-12"))
    assert not B.stop_rule(m, c, set())["passed"]       # outside months never count
    m, c = _ch(10, W0809[-18:])
    f = B.stop_rule(m, c, {"07000000"})
    assert not f["passed"] and f["n_nest"] == 1 and f["gating"] == "nest_excluded"
    assert f["nest_excluded"]["n"] == 9 and f["nest_included"]["n"] == 10
    assert f["report_only_other_count"] == {"which": "nest_included", "n": 10}
    monkeypatch.setattr(B, "EXCLUDE_NEST", False)
    f = B.stop_rule(m, c, {"07000000"})
    assert f["passed"] and f["report_only_other_count"]["n"] == 9
    m, c = _ch(10, W0809)
    c = {s: "yunlin" for s in c}
    assert B.stop_rule(m, c, set())["n_changhua_candidates"] == 0


def test_chb0809_look_scoped_count_once_without_values(tmp_path, monkeypatch, read_spy,
                                                        capsys):
    from harness import prereg3

    wells = {f"07{i:06d}": ("changhua", W0809[:18]) for i in range(10)}
    wells["07000010"] = ("changhua", W0809[:17])
    wells["09000000"] = ("yunlin", W0809)                              # not a member
    pdir = setup_repo(tmp_path, monkeypatch, wells)
    with pytest.raises(SystemExit, match="log the availability look"):
        B.count(prereg_dir=pdir)
    capsys.readouterr()
    assert B.main(["look", "--write"], prereg_dir=pdir, registries=[]) == 0
    out = capsys.readouterr().out
    assert not any(s in out for s in wells) and "11 members" in out
    lk = AL.logged_look(B.UNIT, pdir)
    m = lk["members"][B.UNIT]
    assert lk["units"] == [B.UNIT] and "universal" not in m
    assert "09000000" not in m["ids"] and len(m["ids"]) == 11
    assert m["window"] == "2008-01..2009-12" and not prereg3.member_errors(m)
    assert AL.note_matches(lk, B.constants(), B.RULE_TEXT)
    with pytest.raises(SystemExit, match="already logged"):
        B.main(["look", "--write"], prereg_dir=pdir, registries=[])
    rep = B.count(prereg_dir=pdir)
    out = capsys.readouterr().out
    assert rep["feasibility"]["n_eligible"] == 10 and rep["feasibility"]["passed"]
    assert "PASS" in out and not any(s in out for s in wells) and "-999" not in out
    ls = [c for c in read_spy if LS in c[1]]
    assert [c[2] for c in ls if c[0] == "read_table"] == [["datetime"]] * 11
    assert (tmp_path / B.OUT).is_file()
    with pytest.raises(SystemExit, match="runs once"):
        B.count(prereg_dir=pdir)


def test_chb0809_count_refuses_a_rule_changed_after_the_look(tmp_path, monkeypatch):
    pdir = setup_repo(tmp_path, monkeypatch, {"07000000": ("changhua", W0809)})
    log(B, pdir, tmp_path)
    monkeypatch.setattr(B, "MIN_MONTHS", 17)
    with pytest.raises(SystemExit, match="constants hash and rule text"):
        B.count(prereg_dir=pdir)


def test_spy_refuses_other_readers(tmp_path, read_spy):
    p = tmp_path / LS / "gw__x__raw.parquet"
    write_raw(p, ts_for(["2011-01"]))
    for f in (lambda: pq.read_metadata(p), lambda: pq.ParquetFile(p),
              lambda: pq.read_pandas(p), lambda: pads.dataset(p),
              lambda: pd.read_parquet(p), lambda: pq.read_table(p),
              lambda: pq.read_table(p, columns=["value"])):
        with pytest.raises(AssertionError):
            f()
    assert pq.read_table(p, columns=["datetime"]).num_rows == 20 * 12


# ----------------------------------------------------------------------------- frozen files
@pytest.mark.parametrize("reg_id", ["I1.CH_backcast.v1", "I1.CHX.v1"])
def test_registered_ch_x_files_untouched(reg_id):
    """The ch_x files these registrations hash-bind are unchanged (skips without results/)."""
    reg = REPO / "results" / "dataprep" / "preregistration" / f"{reg_id}.json"
    if not reg.is_file():
        pytest.skip("registry not present (CI)")
    doc = json.loads(reg.read_text(encoding="utf-8"))
    fz = doc.get("frozen_code") or {}
    bound = {p: h for p, h in fz.items() if p.startswith(f"{K.X}/")}
    assert bound, "no frozen ch_x code found"
    for p, h in bound.items():
        assert IO.sha_or_missing(REPO / p) == h, p
