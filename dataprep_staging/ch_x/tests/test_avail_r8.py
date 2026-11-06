"""CPU tests of the round-8 availability looks (avail_bc11.py, plan #12; avail_chb0809.py, plan
#18) on SYNTHETIC files only: no ls_cache record, no head and no registry is touched; stations,
ls_cache files, ring sites and looks.jsonl live in pytest's tmp dirs. The value column of every
synthetic ls_cache file is physically corrupted, so any read of it would raise.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \
        dataprep_staging/ch_x/tests/test_avail_r8.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

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
    md = pq.ParquetFile(path).metadata
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
    from harness import prereg3

    rec = mod.look_record(mod.candidates(), prereg_dir=pdir, registries=[])
    prereg3.record_look(rec["units"], rec["data_scope"], rec["wp"], rec["purpose"],
                        members=rec["members"], note=rec["note"], prereg_dir=pdir,
                        rows_dir=tmp_path / "rows", registries=[])
    return rec


@pytest.fixture
def read_spy(monkeypatch):
    """Records every pyarrow read_table call; refuses any non-datetime ls_cache read."""
    calls = []
    real = pq.read_table

    def spy(source, *a, columns=None, **kw):
        calls.append((str(source), None if columns is None else list(columns)))
        if K.LS_CACHE.split("/")[-1] in str(source) and list(columns or []) != ["datetime"]:
            raise AssertionError(f"ls_cache read beyond datetime: {source} {columns}")
        return real(source, *a, columns=columns, **kw)

    real_pd = pd.read_parquet

    def pd_spy(path, *a, **kw):
        if K.LS_CACHE.split("/")[-1] in str(path):
            raise AssertionError(f"pandas read of an ls_cache file: {path}")
        return real_pd(path, *a, **kw)

    monkeypatch.setattr(pq, "read_table", spy)
    monkeypatch.setattr(pd, "read_parquet", pd_spy)
    return calls


Y11 = months("2011-01", "2011-12")


def good_bc11() -> list[str]:
    return ["2010-12"] + Y11


# ----------------------------------------------------------------------------- A: constants
def test_bc11_stop_rule_constants_are_the_declared_ones():
    assert A.UNIT == "I1BC11:ls_cache_dates_2009_2011"
    assert A.WINDOW == ("2009-09", "2011-12") and A.SCOPE == "heads:2009-09..2011-12"
    assert (A.MIN_POOLED_WELLS, A.MIN_YEAR_MONTHS, A.COUNTY_LINE_MIN_WELLS) == (25, 9, 10)
    assert A.SCORED_YEAR == ("2011-01", "2011-12") and A.ORIGIN == "2010-12"
    assert A.ORIGIN_WINDOW == ("2010-10", "2010-12") and A.MIN_ORIGIN_MONTHS == 1
    assert A.ORIGIN2 == "2009-12" and A.ORIGIN2_YEAR == ("2010-01", "2010-12")
    assert A.ORIGIN2_WINDOW == ("2009-10", "2009-12")
    assert A.EXCLUDE_NEST and A.NEST_RADIUS_M == 1000.0
    assert A.COUNTIES == ("yunlin", "changhua")
    for s in (">= 25 pooled", ">= 9 available months", "2010-10..2010-12", ">= 10 qualifying",
              "second origin 2009-12", "1000 m"):
        assert s in A.RULE_TEXT
    for s in ("X5EV", "07010211", "E26", "time-reversed", "mechanism.py", "I1CHB"):
        assert s in A.DISCLOSURE_TEXT


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


def test_bc11_nest_wells_excluded_before_counting():
    m, c = _pool(15, 10)
    f = A.stop_rule(m, c, {"07000000"})
    assert f["n_eligible"] == 24 and not f["passed"] and f["n_excluded_nest"] == 1
    assert f["county_line"]["changhua"] is False


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
    assert m["window"] == "2009-09..2011-12" and not prereg3.member_errors(m)
    assert prereg3.scope_windows(rec["data_scope"]) is not None
    assert rec["purpose"] == "availability_check" and A.RULE_TEXT in rec["note"]
    assert A.DISCLOSURE_TEXT in rec["note"]


def test_bc11_count_refuses_then_counts_once_without_reading_values(tmp_path, monkeypatch,
                                                                     read_spy):
    wells = _bc11_wells()
    wells["09000001"] = ("yunlin", ["2010-12"] + Y11[:8])            # 8 months: out
    wells["09000002"] = ("yunlin", ["2010-09"] + Y11)                # origin too early: out
    pdir = setup_repo(tmp_path, monkeypatch, wells, nest=("07000000",))
    out = tmp_path / "avail"
    with pytest.raises(SystemExit, match="log the availability look"):
        A.count(out, prereg_dir=pdir)
    assert not (out / "avail.json").exists()
    with pytest.raises(OSError):                       # the value column cannot be read
        pq.ParquetFile(AL.AV.raw_path("09000003")).read(columns=["value"])
    log(A, pdir, tmp_path)
    rep = A.count(out, prereg_dir=pdir)
    f = rep["feasibility"]
    # 16 + 10 good wells, two Yunlin wells fail the rule, one Changhua nest well excluded
    assert f["n_eligible"] == 23 and not f["passed"]
    assert f["n_eligible_by_county"] == {"yunlin": 14, "changhua": 9}
    assert rep["nest_excluded"] == ["07000000"]
    ls = [c for c in read_spy if "ls_cache" in c[0]]
    assert len(ls) == len(wells) and all(c[1] == ["datetime"] for c in ls)
    assert json.loads((out / "avail.json").read_text())["unit"] == A.UNIT
    with pytest.raises(SystemExit, match="runs once"):
        A.count(out, prereg_dir=pdir)


def test_bc11_count_refuses_when_candidates_changed(tmp_path, monkeypatch):
    pdir = setup_repo(tmp_path, monkeypatch, _bc11_wells(3, 2))
    log(A, pdir, tmp_path)
    new = "07555555"
    st = pd.read_parquet(tmp_path / K.STATIONS)
    st.loc[len(st)] = [new, ZH["changhua"], "160000 2650000", "1", 50, 10.0, 90.0]
    st.to_parquet(tmp_path / K.STATIONS)
    write_raw(tmp_path / K.LS_CACHE / K.LS_RAW.format(sid=new), ts_for(Y11))
    with pytest.raises(SystemExit, match="differs from the logged members"):
        A.count(tmp_path / "avail", prereg_dir=pdir)


def test_bc11_twenty_five_pass_through_files(tmp_path, monkeypatch):
    wells = _bc11_wells(15, 10)
    wells["07000099"] = ("changhua", ["2010-10"] + Y11[:9])           # edge cases qualify
    pdir = setup_repo(tmp_path, monkeypatch, wells)
    log(A, pdir, tmp_path)
    f = A.count(tmp_path / "avail", prereg_dir=pdir)["feasibility"]
    assert f["n_eligible"] == 26 and f["passed"] and all(f["county_line"].values())


def test_bc11_thin_month_is_not_available(tmp_path, monkeypatch):
    pdir = setup_repo(tmp_path, monkeypatch, {"09000000": ("yunlin", [])})
    lc = tmp_path / K.LS_CACHE / K.LS_RAW.format(sid="09000000")
    ts = np.concatenate([ts_for(["2010-12"]), ts_for(Y11[:8]), ts_for(["2011-09"], days=19),
                         ts_for(["2011-10"], hours=11)])
    write_raw(lc, ts)
    log(A, pdir, tmp_path)
    rep = A.count(tmp_path / "avail", prereg_dir=pdir)
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
    assert ">= 10 Changhua wells" in B.RULE_TEXT and ">= 18 available months" in B.RULE_TEXT


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


def test_chb0809_window_edges_and_nest_and_county():
    m, c = _ch(10, months("2007-01", "2007-12") + W0809[:17] + months("2010-01", "2010-12"))
    assert not B.stop_rule(m, c, set())["passed"]       # outside months never count
    m, c = _ch(10, W0809[-18:])
    f = B.stop_rule(m, c, {"07000000"})
    assert not f["passed"] and f["n_excluded_nest"] == 1
    assert f["n_ge_min_months_incl_nest_report_only"] == 10
    m, c = _ch(10, W0809)
    c = {s: "yunlin" for s in c}
    assert B.stop_rule(m, c, set())["n_changhua_candidates"] == 0


def test_chb0809_look_scoped_count_once_without_values(tmp_path, monkeypatch, read_spy):
    from harness import prereg3

    wells = {f"07{i:06d}": ("changhua", W0809[:18]) for i in range(10)}
    wells["07000010"] = ("changhua", W0809[:17])
    wells["09000000"] = ("yunlin", W0809)                              # not a member
    pdir = setup_repo(tmp_path, monkeypatch, wells)
    out = tmp_path / "avail0809"
    with pytest.raises(SystemExit, match="log the availability look"):
        B.count(out, prereg_dir=pdir)
    rec = log(B, pdir, tmp_path)
    m = rec["members"][B.UNIT]
    assert rec["units"] == [B.UNIT] and "universal" not in m
    assert "09000000" not in m["ids"] and len(m["ids"]) == 11
    assert m["window"] == "2008-01..2009-12" and not prereg3.member_errors(m)
    rep = B.count(out, prereg_dir=pdir)
    assert rep["feasibility"]["n_eligible"] == 10 and rep["feasibility"]["passed"]
    ls = [c for c in read_spy if "ls_cache" in c[0]]
    assert len(ls) == 11 and all(c[1] == ["datetime"] for c in ls)
    with pytest.raises(SystemExit, match="runs once"):
        B.count(out, prereg_dir=pdir)


# ----------------------------------------------------------------------------- frozen files
def test_registered_ch_x_files_untouched():
    """The ch_x files I1.CH_backcast.v1 hash-binds are unchanged (skips without results/)."""
    reg = REPO / "results" / "dataprep" / "preregistration" / "I1.CH_backcast.v1.json"
    if not reg.is_file():
        pytest.skip("registry not present (CI)")
    doc = json.loads(reg.read_text(encoding="utf-8"))
    fz = doc.get("frozen_code") or {}
    bound = {p: h for p, h in fz.items() if p.startswith(f"{K.X}/")}
    assert bound, "no frozen ch_x code found"
    for p, h in bound.items():
        assert IO.sha_or_missing(REPO / p) == h, p
