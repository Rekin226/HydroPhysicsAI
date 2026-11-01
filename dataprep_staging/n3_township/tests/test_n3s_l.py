"""CPU tests of N3S-L (rank 13): product builder, freeze, registration script and scorer.

Synthetic data only: no WRA summary, leveling panel or head product is opened, the forward
run is a small synthetic grid, nothing is written to results/ (every product, registration
and look goes to pytest's tmp dirs) and no network call is made.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \
        dataprep_staging/n3_township/tests
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for _p in (str(X), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_prospective_product as B  # noqa: E402
import freeze_n3s_l as F  # noqa: E402
import prereg_n3s_l as P  # noqa: E402
import score_n3s_l as S  # noqa: E402
from common import manifest  # noqa: E402
from harness import prereg3  # noqa: E402

UTC = dt.timezone.utc
BEFORE_DEADLINE = dt.datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
SCORING_DAY = dt.datetime(2028, 9, 1, 12, 0, tzinfo=UTC)
X0, Y0, DX = 160000.0, 2600000.0, 1000.0
INLAND = {11: ("雲林縣", "斗六市"), 12: ("彰化縣", "溪州鄉")}
LISTED = ("麥寮鄉", "臺西鄉", "四湖鄉", "口湖鄉")


@pytest.fixture(autouse=True)
def _no_real_products(monkeypatch):
    """No test may open a real data product or reach the network."""
    from common import guard

    real_open = guard.open_product

    def only_tmp(d, *a, **k):
        if str(REPO / "results") in str(Path(d).resolve()):
            raise AssertionError("tests must not open a real data product")
        return real_open(d, *a, **k)
    monkeypatch.setattr(guard, "open_product", only_tmp)
    monkeypatch.setattr(P, "fetch_titles", lambda s: (_ for _ in ()).throw(
        AssertionError("no network in tests")))


# ----------------------------------------------------------------------------- synthetic world
def tile_keys() -> dict:
    k = {i: (S.COAST_COUNTY[t], t) for i, t in enumerate(S.COAST)}
    k.update(INLAND)
    return k


def polys() -> dict:
    out = {}
    for i, key in tile_keys().items():
        r, c = divmod(i, 4)
        x, y = X0 + c * 3000, Y0 + r * 3000
        out[key] = [np.array([[x, y], [x + 3000, y], [x + 3000, y + 3000], [x, y + 3000]])]
    return out


def tile_rate(key) -> float:
    if key == ("雲林縣", "斗六市"):
        return 5.0
    if key == ("彰化縣", "溪州鄉"):
        return 0.5
    return 0.2 if key else 0.0


def fake_run(scale: float = 1.0) -> F.ForwardRun:
    mask = np.ones((12, 12), dtype=bool)
    keys = tile_keys()
    rates = []
    for i in range(12):
        for j in range(12):
            k = (i // 3) * 4 + (j // 3)
            rates.append(tile_rate(keys.get(k)) * scale)
    months = pd.period_range("2012-01", "2032-12", freq="M")
    t = np.arange(len(months)) / 12.0
    s = np.asarray(rates)[:, None] * t[None, :]
    return F.ForwardRun(s, months, mask, X0, Y0, DX)


def sl_rate_of(key) -> float:
    if key and key[1] in LISTED:
        return 3.5
    if key == ("彰化縣", "溪州鄉"):
        return 3.5
    return 1.0


def fake_panel(noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, key in tile_keys().items():
        r, c = divmod(i, 4)
        x, y = X0 + c * 3000 + 1500, Y0 + r * 3000 + 1500
        rate = sl_rate_of(key)
        for yr in range(2007, 2022):
            z = 10.0 - rate / 100.0 * (yr - 2007) + noise * rng.standard_normal()
            rows.append({"sid": f"B{i:02d}", "datetime": pd.Timestamp(f"{yr}-04-15"),
                         "elev_m": z, "x": x, "y": y, "flag": 0})
        # a survey after 2021-09 that SL must ignore
        rows.append({"sid": f"B{i:02d}", "datetime": pd.Timestamp("2021-11-20"),
                     "elev_m": 0.0, "x": x, "y": y, "flag": 0})
    return pd.DataFrame(rows)


def pre_area() -> pd.DataFrame:
    yl = [100, 90, 95, 80, 70, 72, 60, 64, 50, 55, 40]
    ch = [5, 0, 3, 3, 1, 0, 2, 0, 0, 1, 0]
    rows = [{"county": "雲林", "year_label": str(101 + i), "area_km2": v, "edition_roc": 112}
            for i, v in enumerate(yl)]
    rows += [{"county": "彰化", "year_label": str(101 + i), "area_km2": v, "edition_roc": 112}
             for i, v in enumerate(ch)]
    return pd.DataFrame(rows)


def make_freeze(tmp: Path, run=None) -> Path:
    out = tmp / "freeze"
    F.freeze(out, run or fake_run(), polys(), fake_panel(), pre_area())
    return out


def towns() -> dict:
    d = {"雲林縣": [], "彰化縣": []}
    for (c, t) in tile_keys().values():
        d[c].append(t)
    return {k: sorted(v) for k, v in d.items()}


def _row(key, value, ok=True):
    fv = {k: ok for k in value}
    return {"key": key, "value": value, "fields_verified": fv, "verified": ok,
            "any_verified": ok, "pdf": value, "text": value, "text2": {}, "agree_with": "text"}


def transcriptions(tmp: Path, rocs=(114, 115, 116, 117)) -> Path:
    pages = []
    for roc in rocs:
        pdf = tmp / f"roc{roc}_00.pdf"
        pdf.write_bytes(b"%PDF-synthetic " + str(roc).encode())

        def item(kind, pdf=pdf):
            return {"pdf": str(pdf), "page": 1, "kind": kind, "group": "summary"}
        pages.append({"item": item("statement"), "rows": [
            _row(["雲林", roc], {"townships_over_3cm": list(LISTED), "says_none_over_3cm": False,
                                "max_rate_cm_yr": 4.0, "max_rate_township": "口湖鄉",
                                "significant_area_km2": 10.0}),
            _row(["彰化", roc], {"townships_over_3cm": [], "says_none_over_3cm": True,
                                "max_rate_cm_yr": 2.0, "max_rate_township": "溪州鄉",
                                "significant_area_km2": 0.0})]})
        for kind, cty, area in (("interval_yl", "雲林", 10.0), ("interval_chc", "彰化", 0.0)):
            pages.append({"item": item(kind), "rows": [_row(
                [cty, f"{roc - 1}.5", f"{roc}.5"],
                {"max_rate_cm_yr": 4.0 if cty == "雲林" else 2.0,
                 "max_rate_township": "口湖鄉" if cty == "雲林" else "溪州鄉",
                 "area_gt2_km2": area + 5, "area_gt3_km2": area, "class_areas": {}})]})
        pages.append({"item": item("area_county"), "rows": [
            _row(["雲林", str(roc)], {"area_km2": 10.0}),
            _row(["彰化", str(roc)], {"area_km2": 0.0})]})
    p = tmp / "transcriptions.json"
    p.write_text(json.dumps({"schema": "x4-transcriptions/1", "pages": pages},
                            ensure_ascii=False))
    return p


def fake_forward_file(tmp: Path, monkeypatch) -> Path:
    f = tmp / "datum_gate_fake.npz"
    f.write_bytes(b"synthetic forward stand-in")
    monkeypatch.setattr(S, "FORWARD", str(f))
    monkeypatch.setattr(S, "MODEL_SHA", S.sha256(f))
    return f


def register_doc(tmp: Path, monkeypatch, sl_role="report_only") -> tuple[Path, Path]:
    """The real document (PENDING slots allowed by write_prereg) in a tmp registry."""
    fake_forward_file(tmp, monkeypatch)
    pdir, rows = tmp / "prereg", tmp / "rows"
    d = P.doc(sl_role=sl_role, prereg_dir=pdir)
    prereg3.write_prereg(d, prereg_dir=pdir, rows_dir=rows)
    return pdir, rows


def add_look(pdir: Path, prereg_id: str, test_id: str, wp="N3") -> None:
    now = prereg3._now()
    prereg3._v1._append_chained(pdir / "looks.jsonl", {
        "time": now, "logged": now, "wp": wp, "purpose": "score", "test_id": test_id,
        "prereg_id": prereg_id, "units": ["N3:synthetic"], "data_scope": "x:2027-01",
        "scope_windows": "2027-01..2027-01", "members": {}, "seeded": False,
        "note": "test fixture", "tool": "test"})


def stand_in(pdir: Path, rows: Path, prereg_id: str) -> None:
    na = prereg3.not_applicable
    prereg3.write_prereg({"id": prereg_id, "wp": "N3", "kind": "declaration",
                          "title": "synthetic stand-in", "tests": na("x"), "zones": na("x"),
                          "priors": na("x"), "burn_in": na("x"), "folds": na("x"),
                          "fresh_units": na("x"), "thresholds": na("x"),
                          "gate_scores": na("x"), "primary_model": na("x"),
                          "inputs": na("x"), "interpretation": "test fixture only"},
                         prereg_dir=pdir, rows_dir=rows)


# ----------------------------------------------------------------------------- static
def test_no_home_paths_or_user_names():
    import getpass

    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    user = getpass.getuser()
    for p in sorted(X.rglob("*.py")) + sorted(X.rglob("*.sh")):
        txt = p.read_text()
        assert not pat.search(txt), p
        assert len(user) < 3 or user not in txt, p


def test_module_header_and_constants():
    for p in sorted(X.rglob("*.py")):
        assert "from __future__ import annotations" in p.read_text(), p
    assert dt.datetime(2026, 11, 30, 16, 0, tzinfo=UTC) == S.REG_DEADLINE
    assert S.NOT_BEFORE.date() == dt.date(2028, 1, 15)
    assert S.SURVEYS == (116, 117) and S.EXCLUDED_ROC_MAX == 115
    assert set(P.FROZEN) == {f"{S.E}/{n}" for n in ("score_n3s_l.py", "freeze_n3s_l.py",
                                                    "build_prospective_product.py",
                                                    "prereg_n3s_l.py")}
    for y in S.SURVEYS:
        assert S.UNITS[y].endswith(f"roc{y}") and len(set(S.TESTS[y].values())) == 2
        assert prereg3.scope_windows(S.DATA_SCOPE[y]) is not None
    sys.path.insert(0, str(REPO / "dataprep_staging/x5_nudge"))
    import prereg_x5nh
    assert tuple(prereg_x5nh.COAST) == S.COAST and prereg_x5nh.DISTAL_KM == S.DISTAL_KM
    assert prereg_x5nh.TOL == S.TOL and prereg_x5nh.LISTED_FLOOR == S.LISTED_FLOOR


def test_product_uses_are_prospective_only():
    from common.uses import USES
    assert "prospective_scorer" in B.ALLOWED
    assert not set(B.ALLOWED) & set(B.FORBIDDEN)
    assert {"selection", "model_design", "calibration_target", "baseline",
            "design_time_evidence", "statistical_layer_history"} <= set(B.FORBIDDEN)
    assert set(B.ALLOWED) | set(B.FORBIDDEN) <= set(USES)


# ----------------------------------------------------------------------------- SL
def test_sl_mean3_eb_recovers_and_ignores_late_surveys():
    sl = F.sl_mean3_eb(fake_panel())
    assert len(sl) == len(tile_keys())
    want = {f"B{i:02d}": sl_rate_of(k) for i, k in tile_keys().items()}
    got = dict(zip(sl["sid"], sl["sl_rate"], strict=True))
    for sid, r in want.items():
        assert got[sid] == pytest.approx(r, rel=2e-3)    # noise-free up to leap days
    assert (sl["n"] == F.SL_N).all()
    iv = F.sl_intervals(fake_panel())
    assert iv["end"].max() <= pd.Timestamp(F.SL_LAST_END)


def test_sl_shrinkage_with_noise():
    sl = F.sl_mean3_eb(fake_panel(noise=0.01, seed=3))
    assert ((sl["shrink_B"] >= 0) & (sl["shrink_B"] <= 1)).all()
    assert (sl["shrink_B"] > 0).any()
    lo = np.minimum(sl["mean_rate"], sl["prior_mean"]) - 1e-12
    hi = np.maximum(sl["mean_rate"], sl["prior_mean"]) + 1e-12
    assert ((sl["sl_rate"] >= lo) & (sl["sl_rate"] <= hi)).all()
    # an isolated benchmark keeps its own mean
    p = fake_panel(noise=0.01, seed=3)
    far = p[p["sid"] == "B00"].assign(sid="FAR", x=X0 + 90000.0)
    sl2 = F.sl_mean3_eb(pd.concat([p, far]))
    r = sl2[sl2["sid"] == "FAR"].iloc[0]
    assert r["shrink_B"] == 0 and r["sl_rate"] == r["mean_rate"]


def test_area_tolerance_median_of_consecutive_changes():
    tol = F.area_tolerance(pre_area())
    yl = np.abs(np.diff([100, 90, 95, 80, 70, 72, 60, 64, 50, 55, 40]))
    assert tol["雲林縣"] == pytest.approx(float(np.median(yl)))
    assert tol["彰化縣"] == pytest.approx(float(np.median(np.abs(np.diff(
        [5, 0, 3, 3, 1, 0, 2, 0, 0, 1, 0])))))


# ----------------------------------------------------------------------------- decision rules
def _bounds(listed=LISTED, upper_listed=4.0, upper_other=3.0):
    rows = []
    for t in S.COAST:
        ls = t in listed
        rows.append({"county": "雲林" if S.COAST_COUNTY[t] == "雲林縣" else "彰化",
                     "township": t, "survey_year": 2027, "listed_by_all": ls,
                     "listed_by_any": ls, "ambiguous": False,
                     "peak_rate_upper_cm_yr": upper_listed if ls else upper_other})
    return pd.DataFrame(rows)


def test_l1_year_rules():
    rows = S.coast_rows(_bounds(), 116)
    peaks = {"ZR": {t: 0.2 for t in S.COAST},
             "SL": {t: (3.5 if t in LISTED else 1.0) for t in S.COAST},
             "Z": {t: 0.0 for t in S.COAST}}
    y = S.l1_year(rows, peaks, True)
    assert y["status"] == "informative" and y["n_listed"] == 4
    assert y["arms"]["SL"]["verdict"] == "supported"
    assert y["arms"]["ZR"]["verdict"] == "not_supported"
    assert y["arms"]["Z"]["verdict"] == "not_supported"
    # net Z clause: up to 2 broken upper bounds are allowed (the literal reading would fail)
    peaks["SL"]["北港鎮"] = 4.0
    a = S.l1_year(rows, peaks, True)["arms"]["SL"]
    assert a["upper_broken"] == 1 and a["z_clause"] and a["verdict"] == "supported"
    assert not S.z_clause(a, {"lower_met": 0, "upper_broken": 0}, "literal")
    peaks["SL"]["東勢鄉"] = 4.0
    assert S.l1_year(rows, peaks, True)["arms"]["SL"]["verdict"] == "supported"
    peaks["SL"]["水林鄉"] = 4.0                                        # 3 broken
    assert S.l1_year(rows, peaks, True)["arms"]["SL"]["verdict"] == "not_supported"
    # the Z clause still binds against a competitor that does better on the net count
    assert not S.z_clause({"lower_met": 2, "upper_broken": 2}, {"lower_met": 2,
                                                                 "upper_broken": 1})
    # no prediction: lower missed, upper kept
    peaks["SL"] = {t: (3.5 if t in LISTED else 1.0) for t in S.COAST}
    peaks["SL"]["麥寮鄉"] = np.nan
    a = S.l1_year(rows, peaks, True)["arms"]["SL"]
    assert a["lower_met"] == 3 and a["upper_broken"] == 0 and a["verdict"] == "supported"
    # fewer than 3 listed: not informative; Yunlin list missing: not evaluable
    few = S.coast_rows(_bounds(listed=("麥寮鄉", "臺西鄉")), 116)
    assert S.l1_year(few, peaks, True)["status"] == "not_informative"
    assert S.l1_year(rows, peaks, False)["status"] == "not_evaluable"
    # ambiguous listing carries no bound
    amb = _bounds()
    amb.loc[amb["township"] == "麥寮鄉", "ambiguous"] = True
    a = S.l1_year(S.coast_rows(amb, 116), peaks, True)
    assert a["n_listed"] == 3


def test_l1_passability_net_vs_literal():
    assert S.L1_Z_CLAUSE == "net"
    assert S.l1_passability(4, 0.0, "net") == 1.0 == S.l1_passability(4, 0.0, "literal")
    net, lit = S.l1_passability(4, 1.0, "net", 500), S.l1_passability(4, 1.0, "literal", 500)
    assert net > 0.8 and lit < net
    t = P.passability_table()
    assert {r["n_listed"] for r in t["rows"]} == set(P.PASSABILITY_N)
    assert all(r["net"] >= r["literal"] for r in t["rows"])


def test_l1_combine():
    s, n = {"arms": {"SL": {"verdict": "supported"}}}, {"arms": {"SL": {"verdict":
                                                                       "not_supported"}}}
    ni = {"status": "not_informative", "arms": {"SL": {"verdict": "not_informative"}}}
    assert S.l1_combine({116: s, 117: s})["SL"]["verdict"] == "supported"
    assert S.l1_combine({116: s, 117: n})["SL"]["verdict"] == "not_supported"
    assert S.l1_combine({116: s, 117: ni})["SL"]["verdict"] == "supported"
    assert S.l1_combine({116: ni, 117: ni})["SL"]["verdict"] == "undecided"


def test_l2_rules():
    r = S.l2_county_year(10.0, 9.0, 36.0, 5.0)
    assert r["branch"] == "ratio" and r["pass"]
    assert not S.l2_county_year(10.0, 16.0, 36.0, 5.0)["pass"]          # outside x1.5
    assert not S.l2_county_year(10.0, 9.0, 10.5, 5.0)["pass"]           # SL closer
    assert S.l2_county_year(10.0, 9.0, 0.0, 5.0)["pass"]                # SL 0: infinitely far
    assert not S.l2_county_year(10.0, 0.0, 0.0, 5.0)["pass"]
    z = S.l2_county_year(0.0, 2.0, 9.0, 3.0)
    assert z["branch"] == "zero" and z["pass"]
    assert not S.l2_county_year(0.0, 0.0, 0.0, 3.0)["pass"]             # tie fails
    assert not S.l2_county_year(0.0, 4.0, 9.0, 3.0)["pass"]             # beyond tolerance
    assert not S.l2_county_year(0.0, 1.0, 9.0, None)["evaluable"]
    assert not S.l2_county_year(None, 1.0, 9.0, 3.0)["evaluable"]
    ok, bad, ne = {"evaluable": True, "pass": True}, {"evaluable": True, "pass": False}, \
        {"evaluable": False, "pass": None}
    assert S.l2_combine({1: ok, 2: ok, 3: ok, 4: bad})["verdict"] == "supported"
    assert S.l2_combine({1: ok, 2: ok, 3: bad, 4: bad})["verdict"] == "not_supported"
    assert S.l2_combine({1: ok, 2: ok, 3: ne, 4: ne})["verdict"] == "undecided"


def test_printed_area_source_order():
    area = pd.DataFrame([{"county": "雲林", "year_label": "116", "area_km2": 12.0,
                          "edition_roc": 116}])
    iv = pd.DataFrame([{"county": "雲林", "survey_year": 2027, "area_gt3_km2": 11.0}])
    st = pd.DataFrame([{"county": "雲林", "survey_year": 2027, "significant_area_km2": 10.0}])
    r = S.printed_area(area, iv, st, "雲林縣", 116)
    assert r["value"] == 12.0 and r["source"] == "table_1_1" and r["sources_disagree"]
    r = S.printed_area(area.iloc[0:0], iv, st, "雲林縣", 116)
    assert r["value"] == 11.0 and r["source"] == "table_2_x"
    assert S.printed_area(None, None, None, "彰化縣", 116)["value"] is None


def test_final_mapping_roles():
    l1 = {"ZR": {"verdict": "not_supported"}, "SL": {"verdict": "supported"}}
    l2 = {"verdict": "supported"}
    m = S.final_mapping(l1, l2, "report_only")
    assert "failed a registered" in m["C26"] and m["C05"].startswith("returns as a narrowed S")
    assert "may " not in m["C05"]
    assert m["C26b"].startswith("report-only")
    m = S.final_mapping(l1, l2, "eligible_ds")
    assert m["C26b"].startswith("S (prospective, statistical layer")
    assert m["C25"] == "unchanged"


# ----------------------------------------------------------------------------- freeze
def test_freeze_verify_and_zr_reproduction(tmp_path):
    out = make_freeze(tmp_path)
    fz = S.load_freeze(out)
    assert set(fz["tol"]) == set(S.COUNTIES)
    pred = fz["pred"]
    assert set(pred["arm"]) == set(S.ARMS) and set(pred["survey_roc"]) == set(S.SURVEYS)
    za = S.frozen_values(pred, "ZR", 116, "county", "area_gt3_km2")
    assert za == {"雲林縣": 9.0, "彰化縣": 0.0}
    sa = S.frozen_values(pred, "SL", 116, "county", "area_gt3_km2")
    assert sa == {"雲林縣": 36.0, "彰化縣": 9.0}
    S.check_zr_frozen(fake_run(), fz, 116)
    # SL's township peak is the max of its cell field over the township's cells (as ZR)
    slp = S.frozen_values(pred, "SL", 116, "township", "peak_cm_yr")
    assert slp["麥寮鄉"] == pytest.approx(3.5, rel=2e-3)
    assert slp["北港鎮"] == pytest.approx(1.0, rel=2e-3)
    assert set(S.frozen_values(pred, "SL", 116, "township", "benchmark_peak_cm_yr")) == \
        set(S.COAST)
    with pytest.raises(S.EngineeringFailure):
        S.check_zr_frozen(fake_run(scale=1.1), fz, 116)
    rec = F.verify(out, lambda d: F.freeze(d, fake_run(), polys(), fake_panel(), pre_area()))
    assert rec["identical"] and not (tmp_path / "freeze.verify").exists()
    (out / "cells.csv").write_text("tampered\n")
    with pytest.raises(S.EngineeringFailure, match="changed since the freeze"):
        S.load_freeze(out)


# ----------------------------------------------------------------------------- product
def test_builder_refuses_without_registration_and_session(tmp_path, monkeypatch):
    t = transcriptions(tmp_path)
    with pytest.raises(SystemExit, match="not in the registry"):
        B.build(t, [116], root=tmp_path / "dp", prereg_dir=tmp_path / "empty", towns=towns())
    pdir, _ = register_doc(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="open-session"):
        B.build(t, [116], root=tmp_path / "dp", prereg_dir=pdir, towns=towns())
    with pytest.raises(SystemExit, match="only registered"):
        B.build(t, [115], root=tmp_path / "dp", prereg_dir=pdir, towns=towns(),
                require_session=False)


def test_builder_keeps_only_registered_future_years(tmp_path, monkeypatch):
    pdir, _ = register_doc(tmp_path, monkeypatch)
    out = B.build(transcriptions(tmp_path), [116], root=tmp_path / "dp", prereg_dir=pdir,
                  towns=towns(), require_session=False)
    assert out == tmp_path / "dp" / S.PRODUCT / "v1"
    m, d = manifest.load_manifest(out)
    assert manifest.validate_manifest(m, d) == []
    assert "prospective_scorer" in m["allowed_uses"] and m["gate"]["passed"]
    assert m["parameters"]["registration_id"] == S.ID and m["parameters"]["survey_rocs"] == [116]
    for name in ("township_bounds.csv", "township_status.csv", "county_intervals.csv",
                 "township_statements.csv"):
        df = pd.read_csv(out / name)
        assert set(df["survey_year"]) == {2027}, name
    area = pd.read_csv(out / "county_area_by_year.csv")
    assert set(area["year_label"].astype(str)) == {"116"}
    v = json.loads((out / "validation.json").read_text())
    assert v["dropped_rows"]["township_statements.csv"]["le_roc115"] == 4
    assert v["completeness"]["116"]["yunlin_list"]
    ps = S.product_status(out)
    assert ps["ok"] and ps["survey_rocs"] == [116]


# ----------------------------------------------------------------------------- scorer
def _score_args(tmp, product, freeze, roc, out="score", stage1=None, rehearsal=True,
                run=None, sl_role=None):
    return SimpleNamespace(survey_roc=roc, product=str(product), freeze_dir=str(freeze),
                           stage1_report=str(stage1) if stage1 else None, rehearsal=rehearsal,
                           out=str(tmp / out), run=run or fake_run(), sl_role=sl_role)


def test_rehearsal_both_years(tmp_path, monkeypatch):
    pdir, _ = register_doc(tmp_path, monkeypatch)
    fz = make_freeze(tmp_path)
    reps = {}
    for roc in S.SURVEYS:
        td = tmp_path / f"t{roc}"
        td.mkdir()
        prod = B.build(transcriptions(td), [roc], root=tmp_path / "dp", prereg_dir=pdir,
                       towns=towns(), require_session=False)
        (prod / S.SYNTHETIC_MARKER).write_text("synthetic\n")
        a = _score_args(tmp_path, prod, fz, roc, stage1=(tmp_path / "score" / "report_roc116"
                                                        ".json") if roc == 117 else None,
                        sl_role="eligible_ds")
        reps[roc] = S.run_score(a, prereg_dir=pdir)
    y = reps[116]["year"]
    assert y["L1"]["status"] == "informative" and y["L1"]["n_listed"] == 4
    assert y["L1"]["arms"]["SL"]["verdict"] == "supported"
    assert y["L1"]["arms"]["ZR"]["verdict"] == "not_supported"
    assert y["L1"]["intervals"]["雲林縣"] == ["2026-05", "2027-05", "printed"]
    l2 = y["L2_county_years"]
    assert l2["雲林縣"]["branch"] == "ratio" and l2["雲林縣"]["pass"]
    assert l2["彰化縣"]["branch"] == "zero" and l2["彰化縣"]["pass"]
    assert y["L3_report_only"]["status"] == "not_evaluable"
    c = reps[117]["combined"]
    assert c["L1"]["SL"]["verdict"] == "supported" and c["L1"]["ZR"]["verdict"] == \
        "not_supported"
    assert c["L2"]["verdict"] == "supported" and c["L2"]["n_pass"] == 4
    assert c["page_mapping_applied"]["C26b"].startswith("S (prospective")
    with pytest.raises(FileExistsError):
        S.run_score(_score_args(tmp_path, prod, fz, 117, stage1=tmp_path / "score" /
                                "report_roc116.json"), prereg_dir=pdir)
    # nothing was logged by a rehearsal
    looks = pdir / "looks.jsonl"
    assert not looks.exists() or not [lk for lk in prereg3.read_chained(looks)
                                      if lk.get("prereg_id") == S.ID]


def test_rehearsal_needs_marker_and_117_needs_stage1(tmp_path, monkeypatch):
    pdir, _ = register_doc(tmp_path, monkeypatch)
    fz = make_freeze(tmp_path)
    prod = B.build(transcriptions(tmp_path), [117], root=tmp_path / "dp", prereg_dir=pdir,
                   towns=towns(), require_session=False)
    with pytest.raises(S.ScoringRefused, match="SYNTHETIC"):
        S.run_score(_score_args(tmp_path, prod, fz, 117), prereg_dir=pdir)
    (prod / S.SYNTHETIC_MARKER).write_text("synthetic\n")
    with pytest.raises(S.ScoringRefused, match="stage1"):
        S.run_score(_score_args(tmp_path, prod, fz, 117), prereg_dir=pdir)
    with pytest.raises(S.ScoringRefused, match="not registered"):
        S.run_score(_score_args(tmp_path, prod, fz, 115), prereg_dir=pdir)


def avail_file(tmp: Path, name: str, rows116, checked=SCORING_DAY) -> Path:
    """An availability record: ROC116 rows (title, published) in the summary series."""
    listing = {s: [{"title": "114年 x", "published": "114-08"}] for s in S.AVAIL_SERIES}
    listing["007005"] = listing["007005"] + list(rows116)
    f = tmp / name
    f.write_text(json.dumps(S.availability_record(listing, checked=checked.isoformat()),
                            ensure_ascii=False))
    return f


def test_open_session_availability_rules(tmp_path, monkeypatch):
    pdir, rows = register_doc(tmp_path, monkeypatch)
    monkeypatch.setattr(S, "_now", lambda: SCORING_DAY)
    sess = tmp_path / "sess"
    with pytest.raises(S.ScoringRefused, match="availability"):
        S.open_session(116, None, sess, prereg_dir=pdir, rows_dir=rows)
    # only the monthly series (007001) names 116: not the summary
    listing = {s: [{"title": "114年 x", "published": "114-08"}] for s in S.AVAIL_SERIES}
    listing["007001"].append({"title": "116年6月份地層下陷監測月報", "published": "116-07"})
    f = tmp_path / "monthly.json"
    f.write_text(json.dumps(S.availability_record(listing, checked=SCORING_DAY.isoformat()),
                            ensure_ascii=False))
    with pytest.raises(S.ScoringRefused, match="not listed"):
        S.open_session(116, None, sess, availability=f, prereg_dir=pdir, rows_dir=rows)
    ok_rows = [{"title": "116年度地層下陷檢測成果摘要報告", "published": "117-04"}]
    stale = avail_file(tmp_path, "stale.json", ok_rows, SCORING_DAY - dt.timedelta(days=20))
    with pytest.raises(S.ScoringRefused, match="days old"):
        S.open_session(116, None, sess, availability=stale, prereg_dir=pdir, rows_dir=rows)
    # first listed after the deadline: refused; listed before it: the session may open later
    late = avail_file(tmp_path, "late.json", [{"title": "116年度 摘要", "published": "119-01"}])
    with pytest.raises(S.ScoringRefused, match="after its deadline"):
        S.open_session(116, None, sess, availability=late, prereg_dir=pdir, rows_dir=rows)
    after = dt.datetime(2030, 1, 10, tzinfo=UTC)
    monkeypatch.setattr(S, "_now", lambda: after)
    f = avail_file(tmp_path, "ok.json", ok_rows, after)
    rec = S.open_session(116, None, sess, availability=f, prereg_dir=pdir, rows_dir=rows)
    assert rec["availability"]["first_listed"] == "2028-04"
    assert rec["availability"]["basis"] == "first_listed"


def test_real_flow_in_tmp_registry(tmp_path, monkeypatch):
    pdir, rows = register_doc(tmp_path, monkeypatch, sl_role="report_only")
    fz = make_freeze(tmp_path)
    t = transcriptions(tmp_path)
    av = avail_file(tmp_path, "avail.json",
                    [{"title": "116年度地層下陷檢測成果摘要報告", "published": "117-04"}])
    monkeypatch.setattr(S, "_now", lambda: dt.datetime(2027, 12, 1, tzinfo=UTC))
    with pytest.raises(S.ScoringRefused, match="not before"):
        S.open_session(116, None, tmp_path / "sess", prereg_dir=pdir, rows_dir=rows,
                       availability=av)
    monkeypatch.setattr(S, "_now", lambda: SCORING_DAY)
    # an N3 registration exists: refused until its score look and report exist
    stand_in(pdir, rows, "N3.piles_prospective_2026-12.v1")
    with pytest.raises(S.ScoringRefused, match="N3 has not been scored"):
        S.open_session(116, None, tmp_path / "sess", prereg_dir=pdir, rows_dir=rows,
                       availability=av)
    n3 = tmp_path / "n3_report.json"
    n3.write_text(json.dumps({"prereg_id": "N3.piles_prospective_2026-12.v1",
                              "verdict": "synthetic"}))
    with pytest.raises(S.ScoringRefused, match="N3 has not been scored"):
        S.open_session(116, n3, tmp_path / "sess", prereg_dir=pdir, rows_dir=rows,
                       availability=av)
    add_look(pdir, "N3.piles_prospective_2026-12.v1", "N3P.synthetic")
    # the builder refuses before the session (no value read before the score look)
    with pytest.raises(SystemExit, match="open-session"):
        B.build(t, [116], root=tmp_path / "dp", prereg_dir=pdir, towns=towns())
    # scoring refuses before the session too
    with pytest.raises(S.ScoringRefused, match="open-session"):
        S.run_score(_score_args(tmp_path, tmp_path, fz, 116, rehearsal=False),
                    prereg_dir=pdir, rows_dir=rows)
    rec = S.open_session(116, n3, tmp_path / "sess", prereg_dir=pdir, rows_dir=rows,
                         availability=av)
    assert rec["order"]["ok"] and set(rec["assert_can_score"]) == set(S.TESTS[116].values())
    with pytest.raises(FileExistsError):
        S.open_session(116, n3, tmp_path / "sess", prereg_dir=pdir, rows_dir=rows,
                       availability=av)
    prod = B.build(t, [116], root=tmp_path / "dp", prereg_dir=pdir, towns=towns())
    rep = S.run_score(_score_args(tmp_path, prod, fz, 116, rehearsal=False), prereg_dir=pdir,
                      rows_dir=rows)
    assert rep["sl_role"] == "report_only" and not rep["rehearsal"]
    assert rep["year"]["L1"]["arms"]["SL"]["verdict"] == "supported"
    looks = [lk for lk in prereg3.read_chained(pdir / "looks.jsonl")
             if lk["prereg_id"] == S.ID]
    assert {lk["test_id"] for lk in looks} == set(S.TESTS[116].values())
    # a synthetic product is refused in real mode; a rehearsal stage 1 cannot enter real 117
    (prod / S.SYNTHETIC_MARKER).write_text("x")
    with pytest.raises(S.ScoringRefused):
        S.run_score(_score_args(tmp_path, prod, fz, 116, out="o2", rehearsal=False),
                    prereg_dir=pdir, rows_dir=rows)
    fake = tmp_path / "s1.json"
    fake.write_text(json.dumps({"id": S.ID, "survey_roc": 116, "rehearsal": True,
                                "year": rep["year"]}))
    with pytest.raises(S.ScoringRefused, match="rehearsal ROC116"):
        S.combine(rep["year"], fake, "report_only", rehearsal=False)
    # ROC117 never published: the not-published record carries the final verdict of ROC116
    s1 = tmp_path / "score" / "report_roc116.json"
    monkeypatch.setattr(S, "_now", lambda: dt.datetime(2031, 1, 2, tzinfo=UTC))
    with pytest.raises(S.ScoringRefused, match="stage1"):
        S.not_published(117, tmp_path / "score", prereg_dir=pdir, rows_dir=rows)
    fin = S.not_published(117, tmp_path / "score", s1, prereg_dir=pdir, rows_dir=rows)
    c = fin["combined"]
    assert fin["verdict"] == S.NOT_PUBLISHED and fin["sl_role"] == "report_only"
    assert c["years_scored"] == [116] and c["not_published"] == [117]
    assert c["L1"]["SL"]["verdict"] == "supported" and c["L1"]["ZR"]["verdict"] == \
        "not_supported"
    assert c["L2"]["verdict"] == "undecided" and c["L2"]["n_evaluable"] == 2
    m = c["page_mapping_applied"]
    assert "failed a registered" in m["C26"] and m["C05"].startswith("W, 'undecided")
    assert m["C26b"].startswith("report-only")
    on_disk = json.loads((tmp_path / "score" / "report_roc117.json").read_text())
    assert on_disk["combined"]["page_mapping_applied"] == m


def test_not_published_record(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_now", lambda: dt.datetime(2029, 6, 1, tzinfo=UTC))
    with pytest.raises(S.ScoringRefused, match="has not passed"):
        S.not_published(116, tmp_path)
    monkeypatch.setattr(S, "_now", lambda: dt.datetime(2030, 1, 2, tzinfo=UTC))
    rec = S.not_published(116, tmp_path)
    assert rec["verdict"] == S.NOT_PUBLISHED
    y = {"L1": {"status": "not_informative", "arms": {a: {"verdict": "not_informative"}
                                                      for a in S.ARMS}},
         "L2_county_years": {c: {"evaluable": True, "pass": True} for c in S.COUNTIES}}
    c = S.combine(y, tmp_path / "report_roc116.json", "report_only")
    assert c["years_scored"] == [117] and c["L2"]["verdict"] == "undecided"
    # neither year published: every verdict undecided, the mapping still applied
    c = S.combine(None, tmp_path / "report_roc116.json", "eligible_ds")
    assert c["years_scored"] == [] and c["not_published"] == [116, 117]
    assert c["L2"]["verdict"] == "undecided"
    assert c["page_mapping_applied"]["C26"].startswith("N, 'undecided")
    assert c["page_mapping_applied"]["C26b"].startswith("D removed")


# ----------------------------------------------------------------------------- registration
def test_availability_parsing_and_block(tmp_path, monkeypatch):
    st = P.roc_status(["114年地層下陷檢測成果摘要報告 114", "115年彰雲地區地層下陷檢測成果 115"])
    assert st == {"115": "listed", "116": "not_listed", "117": "not_listed"}
    assert P.roc_status(["116年度 成果"])["116"] == "listed"
    assert P.roc_status(["115~116年 成果"])["116"] == "listed"
    # the publication column never marks a year listed; it only dates the first listing
    r = P.availability_record({s: [{"title": "114年 x", "published": "116-03"},
                                   {"title": "117年度 摘要", "published": "2028-11"}]
                               for s in P.SERIES}, checked="2028-12-01T00:00:00+00:00")
    st = r["series"]["007005"]
    assert st["status"]["116"] == "not_listed" and st["status"]["117"] == "listed"
    assert st["first_listed"]["117"] == "2028-11" and st["first_listed"]["116"] is None
    assert S.published_ym("115/10") == "2026-10" and S.published_ym("") is None
    rec = P.availability_record({s: ["114年 x"] for s in P.SERIES},
                                checked="2026-11-18T00:00:00+00:00")
    assert rec["unpublished_116_117"]
    f = tmp_path / "avail.json"
    f.write_text(json.dumps(rec))
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    assert P.availability_block(f)["status"] == "ok"
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE + dt.timedelta(days=30))
    assert "days old" in P.availability_block(f)["status"]
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    rec2 = P.availability_record({s: ["116年 x"] for s in P.SERIES},
                                 checked="2026-11-18T00:00:00+00:00")
    f.write_text(json.dumps(rec2))
    assert "do not register" in P.availability_block(f)["status"]


def test_doc_validates_and_states_rulings(tmp_path, monkeypatch):
    fake_forward_file(tmp_path, monkeypatch)
    d = P.doc(prereg_dir=tmp_path / "prereg")
    assert prereg3.validate_prereg(d, prereg_dir=tmp_path / "prereg") == []
    assert [t["test_id"] for t in d["tests"]] == [S.TESTS[116]["L1"], S.TESTS[116]["L2"],
                                                   S.TESTS[117]["L1"], S.TESTS[117]["L2"]]
    assert d["fresh_units"] == [S.UNITS[116], S.UNITS[117]]
    assert d["sl_page_role"]["role"] == P.PENDING
    assert d["binds"]["frozen_predictions"]["status"] == P.PENDING
    assert d["availability"]["status"] == P.PENDING
    for k in ("looked_2023_25", "physics_near_z", "roc115", "monthly_report_channel",
              "multiplicity", "l4_dropped"):
        assert k in d["disclosures"]
    assert "ONCE" in d["attempt_policy"] and "prospective_scorer" in \
        d["product_definition"]["allowed_uses"]
    assert "no cut number" in d["page_mapping"]["never"]
    txt = json.dumps(d, ensure_ascii=False)
    assert not re.search(r"(/home/|-home-)[A-Za-z0-9_]", txt)
    assert P.check_blocking(d, tmp_path / "prereg") == []


def _synthetic_slots(tmp: Path, monkeypatch) -> tuple[Path, Path]:
    """A freeze dir whose provenance names a synthetic SL product, and an availability record."""
    fwd = fake_forward_file(tmp, monkeypatch)
    sl_dir = tmp / "slprod" / "v1"
    sl_dir.mkdir(parents=True)
    (sl_dir / "MANIFEST.json").write_text(json.dumps({
        "product": F.SL_PRODUCT, "version": "v1", "fit_window_end": "2021-12",
        "allowed_uses": list(F.SL_ALLOWED), "forbidden_uses": list(F.SL_FORBIDDEN),
        "gate": {"passed": True}}))
    towns_f = tmp / "towns.json"
    towns_f.write_text("{}")
    out = tmp / "freeze"
    prov = {"forward": {"path": str(fwd), "sha256": S.sha256(fwd)},
            "towns": {"path": str(towns_f), "sha256": S.sha256(towns_f)},
            "sl_product_manifest": {"path": str(sl_dir / "MANIFEST.json"),
                                    "sha256": S.sha256(sl_dir / "MANIFEST.json")}}

    def build(d):
        F.freeze(d, fake_run(), polys(), fake_panel(), pre_area(), provenance=prov)
    build(out)
    F.verify(out, build)
    av = tmp / "avail.json"
    av.write_text(json.dumps(P.availability_record({s: ["114年 x"] for s in P.SERIES},
                                                   checked="2026-11-18T00:00:00+00:00")))
    return out, av


def test_write_refuses_without_slots(tmp_path, monkeypatch):
    fake_forward_file(tmp_path, monkeypatch)
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    rc = P.main(["--write", "--quiet", "--expect-self-sha", S.sha256(Path(P.__file__))],
                prereg_dir=pdir, rows_dir=rows)
    assert rc == 1 and not (pdir / f"{S.ID}.json").exists()


def test_write_with_slots_in_tmp_registry(tmp_path, monkeypatch):
    fz, av = _synthetic_slots(tmp_path, monkeypatch)
    assert P.freeze_block(fz)["status"] == "ok"
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    sha = S.sha256(Path(P.__file__))
    base = ["--write", "--quiet", "--freeze-dir", str(fz), "--availability", str(av)]
    assert P.main(base + ["--expect-self-sha", sha], prereg_dir=pdir, rows_dir=rows) == 1
    assert P.main(base + ["--sl-role", "report_only", "--expect-self-sha", "0" * 64],
                  prereg_dir=pdir, rows_dir=rows) == 1
    monkeypatch.setattr(S, "_now", lambda: S.REG_DEADLINE)
    assert P.main(base + ["--sl-role", "report_only", "--expect-self-sha", sha],
                  prereg_dir=pdir, rows_dir=rows) == 1
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    assert P.main(base + ["--sl-role", "eligible_ds", "--expect-self-sha", sha],
                  prereg_dir=pdir, rows_dir=rows) == 0
    doc, _ = prereg3.load_prereg(S.ID, prereg_dir=pdir, rows_dir=rows)
    assert doc["sl_page_role"]["role"] == "eligible_ds"
    assert doc["binds"]["frozen_predictions"]["status"] == "ok"
    assert {t["test_id"] for t in doc["tests"]} == {t for y in S.SURVEYS
                                                    for t in S.TESTS[y].values()}
    real = REPO / "results/dataprep/preregistration/INDEX.jsonl"
    assert not real.exists() or S.ID not in real.read_text()


def test_freeze_block_rejects_wrong_forward_or_sl_use(tmp_path, monkeypatch):
    fz, _ = _synthetic_slots(tmp_path, monkeypatch)
    # an SL source without the baseline use (SL is L2's comparator) is refused
    mf = tmp_path / "slprod" / "v1" / "MANIFEST.json"
    m = json.loads(mf.read_text())
    m["allowed_uses"] = ["statistical_layer_history"]
    m["forbidden_uses"] = ["baseline"]
    mf.write_text(json.dumps(m))
    summ = json.loads((fz / "summary.json").read_text())
    assert summ["provenance"]["sl_product_manifest"]["sha256"] != S.sha256(mf)
    prov = summ["provenance"]
    prov["sl_product_manifest"]["sha256"] = S.sha256(mf)
    fz2 = tmp_path / "freeze2"

    def build(d):
        F.freeze(d, fake_run(), polys(), fake_panel(), pre_area(), provenance=prov)
    build(fz2)
    F.verify(fz2, build)
    assert P.freeze_block(fz2)["status"].startswith("SL source does not grant")
    monkeypatch.setattr(S, "MODEL_SHA", "0" * 64)
    assert P.freeze_block(fz)["status"] == "forward is not the page model's"


def test_sl_product_grants_both_uses(tmp_path):
    from common import guard
    src = tmp_path / "panel_stand_in.parquet"
    fake_panel().to_parquet(src)
    d = F.build_sl_panel(root=tmp_path / "dp", panel=fake_panel(), src=src)
    m, _ = manifest.load_manifest(d)
    assert set(F.SL_USES) <= set(m["allowed_uses"])
    assert not set(F.SL_USES) & set(m["forbidden_uses"])
    assert m["fit_window_end"] == "2021-12" and m["gate"]["passed"]
    assert "50 cm/yr" in m["parameters"]["flags"]
    for u in F.SL_USES:
        guard.open_product(d, u)
    for u in ("prospective_scorer", "selection", "calibration_target"):
        with pytest.raises(guard.UseNotAllowed):
            guard.open_product(d, u)
    p = F.read_sl_panel(d)
    assert len(p) == len(fake_panel()) and (p["flag"] == 0).all()
