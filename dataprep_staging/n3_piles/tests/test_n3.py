"""CPU tests of the N3 pile registration, its G0, products, scorer and the rank-12 diagnostic.

Synthetic worlds only: a small synthetic forward archive (no real model run), synthetic
station lists and wells, synthetic transcriptions, synthetic window heads. No real data
product is opened (the guard refuses any path under results/), nothing is written to
results/dataprep/preregistration (every registration and look goes to pytest's tmp dirs).

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \
        dataprep_staging/n3_piles/tests
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

import build_mapping as BM  # noqa: E402
import build_v3_manifest as V3  # noqa: E402
import export_leveling_competitor as LC  # noqa: E402
import g0_n3 as G  # noqa: E402
import n3lib as L  # noqa: E402
import posthoc_pile_conversion as PH  # noqa: E402
import prereg_n3 as P  # noqa: E402
import score_n3 as S  # noqa: E402
import well_columns as WC  # noqa: E402
from harness import prereg3  # noqa: E402

UTC = dt.timezone.utc
BS = L.BS                              # N2's shared scorer, as n3lib imports it
needs_repo = pytest.mark.skipif(not (REPO / L.FORWARD).exists() or not (REPO / L.COLUMN).exists(),
                                reason="gitignored results/ tree not present")
BEFORE_DEADLINE = dt.datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
SCORING_DAY = dt.datetime(2028, 1, 20, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_real_products(monkeypatch):
    """The guard refuses every product under the repository's results/ tree."""
    from common import guard

    real = guard.open_product

    def only_tmp(path, *a, **k):
        p = Path(path).resolve()
        if (REPO / "results").resolve() in p.parents:
            raise AssertionError(f"tests must not open a real product ({p})")
        return real(path, *a, **k)

    monkeypatch.setattr(guard, "open_product", only_tmp)


# ----------------------------------------------------------------------------- a synthetic world
X0, Y0, DX, NX, NY = 100000.0, 2500000.0, 1000.0, 30, 30
SITES = {"BM_TKJS(3)": ("土庫國中", 400, 105500.0, 2505500.0),
         "BM_TKJS(1)": ("土庫國中", 300, 105505.0, 2505505.0),
         "BM_NTUH(1)": ("中科虎尾", 300, 112500.0, 2512500.0),
         "BM_SJES": ("溪州國小", 300, 120500.0, 2520500.0)}
WELLS = [("Y1", 2, 105600.0, 2505600.0), ("Y2", 3, 105700.0, 2505400.0),
         ("H1", 1, 112600.0, 2512500.0), ("H2", 4, 112400.0, 2512600.0),
         ("C1", 1, 121000.0, 2520500.0), ("C2", 2, 121500.0, 2521000.0),
         ("FAR", 2, 128000.0, 2528000.0)]
DATES = pd.date_range("2012-01-01", "2032-12-01", freq="MS")


def grid() -> dict:
    return {"mask": np.ones((NY, NX), bool), "dx": np.float64(DX), "x0": np.float64(X0),
            "y0": np.float64(Y0)}


def head_series(phase: float = 0.0) -> np.ndarray:
    t = np.arange(len(DATES))
    return 10.0 + 1.5 * np.sin(2 * np.pi * (t / 12.0) + phase) + 0.3 * np.sin(t / 7.0)


def write_npz(path: Path, g_cm_per_m: float = 1.0) -> Path:
    """Every cell and layer shares one head series; the column is g x the drawdown plus a
    steady creep, so the model's conversion is exactly g (cm/m)."""
    A, T = NX * NY, len(DATES)
    h = head_series()
    heads = np.broadcast_to(h, (3, 4, A, T)).astype(np.float32).copy()
    dd = -(h - h[0])
    subs_m = (g_cm_per_m * dd + 0.05 * np.arange(T)) / 100.0
    subs = np.broadcast_to(subs_m, (2, 3, A, T)).astype(np.float32).copy()
    np.savez(path, dates=np.array([d.strftime("%Y-%m-%dT00:00:00") for d in DATES]),
             origin=np.int64(131), heads_mean=heads, subs_mean_by_rheology=subs,
             rheology_labels=np.array(["coupled_leveling", "tau30y"]),
             scenario_names=np.array(["baseline", "cut30", "retire_aqua"]), **grid())
    return path


def sites_df() -> pd.DataFrame:
    return pd.DataFrame([{"identifier": k, "pile": v[0], "depth_m": v[1], "x_twd97": v[2],
                          "y_twd97": v[3]} for k, v in SITES.items()])


def wells_df() -> pd.DataFrame:
    return pd.DataFrame([{"sid": s, "layer": lay, "x": x, "y": y} for s, lay, x, y in WELLS])


def make_mapping_dir(tmp: Path) -> tuple[Path, Path]:
    """mapping.json + model_side.json from the synthetic world; the forward sha field is set
    to the registered one (the prereg checks it), as a build on the real archive would."""
    npz = write_npz(tmp / "fw.npz")
    mapping = BM.build_mapping(sites_df(), wells_df(), grid())
    d = tmp / "mapping"
    d.mkdir()
    (d / "mapping.json").write_text(json.dumps({"piles": mapping}, ensure_ascii=False))
    ms = BM.model_side(mapping, npz, npz)
    ms["forward"]["sha256"] = L.FORWARD_SHA
    ms["mapping_sha256"] = L.sha256(d / "mapping.json")
    (d / "model_side.json").write_text(json.dumps(ms, ensure_ascii=False, default=float))
    return d, npz


# ----------------------------------------------------------------------------- hygiene
def test_no_home_paths_or_user_names():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in sorted(X.rglob("*.py")) + sorted(X.glob("*.sh")):
        assert not pat.search(p.read_text()), p


def test_module_header_and_constants():
    for p in sorted(X.rglob("*.py")):
        assert "from __future__ import annotations" in p.read_text(), p
    assert dt.datetime(2026, 11, 30, 16, 0, tzinfo=UTC) == L.DEADLINE
    assert L.NOT_BEFORE.date() == dt.date(2028, 1, 15)
    assert L.WINDOW == ("2026-12", "2027-11") and L.WINDOW24 == ("2026-12", "2028-11")
    assert L.PRIMARY == ("tuku", "huwei") and L.PILES["xizhou"]["role"] == "report_only"
    assert "秀潭國小" in L.EXCLUDED
    assert prereg3.scope_windows(L.SCOPE12) == [(12 * 2026 + 11, 12 * 2027 + 10)]
    assert set(P.FROZEN) == {f"{L.E}/{n}" for n in (
        "n3lib.py", "g0_n3.py", "build_mapping.py", "well_columns.py", "build_v3_manifest.py",
        "export_leveling_competitor.py", "score_n3.py", "prereg_n3.py")} | {L.SHARED_SCORER}
    assert (REPO / L.SHARED_SCORER).is_file()
    for sc in (L.SCOPE_CONV12, L.SCOPE_CONV24):
        assert prereg3.scope_windows(sc)[0][0] == 12 * 2026 + 8              # from 2026-09
    assert prereg3.scope_windows(L.HEADS_WINDOW_YL) == [(12 * 2026 + 8, 12 * 2026 + 9),
                                                        (12 * 2027 + 10, 12 * 2028 + 10)]
    assert f"{L.E}/posthoc_pile_conversion.py" not in P.FROZEN


# ----------------------------------------------------------------------------- n3lib
def test_layer_fill_rules():
    assert L.layer_fill({2, 3}) == {1: [2], 2: [2], 3: [3], 4: [3]}
    assert L.layer_fill({1, 4}) == {1: [1], 2: [1], 3: [4], 4: [4]}
    assert L.layer_fill({1, 2}) == {1: [1], 2: [2], 3: [2], 4: [2]}
    assert L.layer_fill(set()) == {}
    tie = L.layer_fill({1, 3}, depths=(0.0, 50.0, 100.0, 150.0))
    assert tie[2] == [1, 3]


def test_nest_dd_layer_rule_and_gaps():
    ms = ["2027-01", "2027-02", "2027-03"]
    idx = ["2026-12"] + ms
    heads = {"a": pd.Series([10.0, 9.0, 8.0, 8.5], index=idx),
             "b": pd.Series([5.0, 5.0, 4.0, np.nan], index=idx)}
    dd = L.nest_dd(heads, {"a": 2, "b": 3}, ms)
    # layers 1,2 <- a (dD 1, 1, -0.5); layers 3,4 <- b (dD 0, 1, nan)
    assert dd["2027-01"] == pytest.approx(0.5)
    assert dd["2027-02"] == pytest.approx(1.0)
    assert np.isnan(dd["2027-03"])
    ch = {"a": pd.Series({"2027-01": -1.0}), "b": pd.Series({"2027-01": 0.0})}
    assert L.nest_dd_from_changes(ch, {"a": 2, "b": 3}, ["2027-01"])["2027-01"] == \
        pytest.approx(0.5)


def test_fe_slope_and_bootstrap():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    g = np.repeat(["p", "q"], 30)
    y = 0.8 * x + np.where(g == "p", 5.0, -3.0) + rng.normal(0, 0.05, 60)
    assert L.fe_slope(y, x, g) == pytest.approx(0.8, abs=0.02)
    e1, e2 = L.conversion_estimate(y, x, g), L.conversion_estimate(y, x, g)
    assert e1 == e2 and e1["lo"] < 0.8 < e1["hi"]
    idx = L.block_indices(10, 4, 3, np.random.default_rng(1))
    assert idx.shape == (4, 10) and idx.max() < 10
    assert np.all(np.diff(idx[:, :3], axis=1) == 1)


def test_conversion_verdict_categories():
    v = L.conversion_verdict
    assert v(1.0, 0.8, 1.2, 1.0)["verdict"] == "conversion_supported"
    assert v(0.4, 0.3, 0.5, 1.0)["verdict"] == "direction_supported_size_not"
    assert v(2.0, 1.8, 2.2, 1.0)["verdict"] == "direction_supported_size_not"
    assert v(1.0, -0.1, 2.0, 1.0)["verdict"] == "not_supported"
    assert v(-1.0, -1.2, -0.8, 1.0)["verdict"] == "not_supported"
    assert v(np.nan, 0, 1, 1.0)["verdict"] == "not_supported"           # as N2's rule
    assert v(-0.1, 0.2, 0.5, 1.0)["verdict"] == "not_supported"         # negative point
    assert v(1.0, 0.8, 1.2, np.nan)["verdict"] == "not_informative"     # model side only
    assert v(1.0, 0.8, 1.2, 0.0)["verdict"] == "not_informative"
    r = v(0.76, 0.6, 0.9, 0.76)
    assert r["rho"] == pytest.approx(1.0) and r["rho_ci"][0] < 1 < r["rho_ci"][1]


def test_conversion_verdict_equals_shared_rule_on_a_grid():
    """N3's verdict is N2's shared decide() on rho (RMSE clause passed as satisfied)."""
    name = {"response_supported": "conversion_supported"}
    grid = [-2.0, -0.5, 0.0, 0.3, 0.66, 0.67, 1.0, 1.5, 1.51, 3.0, np.nan]
    for g_mod in (0.76, -0.5, 2.0):
        for a in grid:
            for b in grid:
                for g in grid:
                    lo, hi = a * g_mod, b * g_mod
                    got = L.conversion_verdict(g * g_mod, lo, hi, g_mod)["verdict"]
                    rlo, rhi = sorted((lo / g_mod, hi / g_mod)) \
                        if np.isfinite(lo) and np.isfinite(hi) else (np.nan, np.nan)
                    want = BS.decide(True, (g * g_mod) / g_mod, rlo, rhi, 0.0, 1.0)
                    assert got == name.get(want, want), (g_mod, a, b, g)


def test_level_verdict():
    obs, mod = {"tuku": 2.0, "huwei": 1.5}, {"tuku": 1.9, "huwei": 1.6}
    assert L.level_verdict(obs, mod, {"tuku": 5.0, "huwei": 5.0})["verdict"] == \
        "level_supported"
    assert L.level_verdict(obs, mod, {"tuku": 2.0, "huwei": 5.0})["verdict"] == \
        "level_not_supported"
    out = L.level_verdict({"tuku": 2.0}, mod, {"tuku": 5.0, "huwei": 5.0})
    assert out["verdict"] == "not_evaluable" and out["missing"] == ["huwei"]


def test_season_coefficients_rules():
    ms = [f"2027-{m:02d}" for m in range(6, 12)]
    inc = pd.Series([-0.1] * 6, index=ms)
    dd = pd.Series([-0.2] * 6, index=ms)                  # head rises 1.2 m
    r = L.season_coefficients(inc, dd, [2027], threshold_cm=1.0)
    s = r["seasons"][0]
    assert s["rebound_cm"] == pytest.approx(0.6) and s["rise_m"] == pytest.approx(1.2)
    assert r["pooled"]["k"] == pytest.approx(0.5) and r["pooled"]["ci90"] is None
    assert "missing" in L.season_coefficients(inc.drop(ms[2]), dd, [2027],
                                              threshold_cm=1.0)["seasons"][0]["dropped"]
    big = inc.copy()
    big[ms[0]] = -3.0
    assert "discontinuity" in L.season_coefficients(big, dd, [2027],
                                                    threshold_cm=1.0)["seasons"][0]["dropped"]
    dok = pd.Series({m: m != ms[1] for m in ms})
    assert "depth" in L.season_coefficients(inc, dd, [2027], threshold_cm=1.0,
                                            depth_ok=dok)["seasons"][0]["dropped"]
    two = pd.concat([inc, inc.rename(lambda m: m.replace("2027", "2028"))])
    dd2 = pd.concat([dd, (dd * 2).rename(lambda m: m.replace("2027", "2028"))])
    r2 = L.season_coefficients(two, dd2, [2027, 2028], threshold_cm=1.0)
    assert r2["pooled"]["n"] == 2 and r2["pooled"]["ci90"] is not None
    assert L.discontinuity_threshold([0.1, -0.3]) == pytest.approx(1.5)
    assert L.discontinuity_threshold([0.01]) == pytest.approx(1.0)


def test_report_only_lines_keys():
    rng = np.random.default_rng(3)
    n = 24
    dd = rng.normal(size=n)
    g = np.repeat(["a", "b"], 12)
    inc = 0.7 * dd + rng.normal(0, 0.05, n)
    out = L.report_only_lines(inc, dd, np.roll(dd, 1), np.roll(dd, 2), g, inc * 1.1, 0.7)
    assert set(out) == {"lag0_2", "sign_split", "calibration_slope_obs_on_zr"}
    assert out["lag0_2"]["coef"][0] == pytest.approx(0.7, abs=0.1)


# ----------------------------------------------------------------------------- mapping, model side
def test_mapping_and_model_side(tmp_path):
    d, npz = make_mapping_dir(tmp_path)
    mp = json.loads((d / "mapping.json").read_text())["piles"]
    assert [w["sid"] for w in mp["tuku"]["nest"]] == ["Y1", "Y2"]
    assert mp["tuku"]["layer_fill"] == {"1": [2], "2": [2], "3": [3], "4": [3]}
    assert {w["sid"] for w in mp["xizhou"]["nest"]} == {"C1", "C2"}       # 3 km, not FAR
    assert mp["tuku"]["fallback_site"]["depth_m"] == 300
    ms = json.loads((d / "model_side.json").read_text())
    assert ms["g_mod_12"]["g_mod"] == pytest.approx(1.0, abs=1e-3)       # g = 1 cm/m exactly
    assert ms["g_mod_24"]["n"] == 48 and ms["informative_model_side_pass"]
    w = ms["world"]["piles"]["tuku"]
    assert w["months"][0] == L.REF_MONTH and w["months"][-1] == L.WINDOW24[1]
    assert ms["zr_level"]["tuku"]["zr_12m_cm"] == pytest.approx(
        sum(i for m, i in zip(w["months"], w["inc_cm"], strict=True)
            if L.WINDOW[0] <= m <= L.WINDOW[1]), abs=1e-6)
    h = ms["hindcast"]["piles"]["huwei"]["months"]
    assert h[0] == "2016-05" and h[-1] == "2022-12"
    assert "xizhou_r1c" in ms and "rehearsal" in ms["xizhou_r1c"]
    # column pairing: recorded, and the paired hindcast slope is the report-only comparator
    assert ms["column_pairing"]["column_heads"] is None                 # synthetic archive
    assert "DISPLAYED" in ms["g_mod_definition"]
    assert ms["g_mod_hindcast_paired"]["g_mod"] == pytest.approx(1.0, abs=1e-3)
    assert ms["g_mod_hindcast_paired"]["months"] == ["2016-06", "2022-12"]
    assert "k_mod_hindcast_paired" in ms["xizhou_r1c"]


def test_model_side_records_free_column_heads(tmp_path):
    npz = write_npz(tmp_path / "fw.npz")
    with np.load(npz) as z:
        arrs = {k: z[k] for k in z.files}
    np.savez(npz, **arrs, forward_options=np.array(json.dumps({"column_heads": "free"})),
             gain=np.float64(1.0), hindcast_gain=np.float64(0.0))
    mapping = BM.build_mapping(sites_df(), wells_df(), grid())
    ms = BM.model_side(mapping, npz, None)
    cp = ms["column_pairing"]
    assert cp["column_heads"] == "free" and not cp["window_heads_drive_column"]
    assert cp["hindcast_paired"] and "DISPLAYED" in cp["wording"]


def test_mapping_refuses_wrong_site_name():
    s = sites_df()
    s.loc[s.identifier == "BM_SJES", "pile"] = "其他"
    with pytest.raises(SystemExit):
        BM.build_mapping(s, wells_df(), grid())


# ----------------------------------------------------------------------------- G0
def test_ar1_marginal_sd():
    e = G.ar1(200000, 1.5, 0.9, np.random.default_rng(0))
    assert e.std() == pytest.approx(1.5, rel=0.05)


def test_judge_cells_rule():
    def cells(pc1=0.9, pc04=0.9, false=0.0, cov=0.9):
        out = []
        for sd in G.SD_GRID:
            for phi in G.PHI_GRID:
                for rho in G.RHOS:
                    pc = pc1 if rho == 1.0 else pc04
                    out.append({"sd": sd, "phi": phi, "rho": rho, "p_correct": pc,
                                "coverage": cov,
                                "probs": {"conversion_supported": pc if rho == 1.0 else false}})
        return out
    assert G.judge_cells(cells())["pass"]
    assert not G.judge_cells(cells(pc1=0.7))["pass"]
    assert not G.judge_cells(cells(false=0.2))["pass"]
    assert not G.judge_cells(cells(cov=0.8))["pass"]
    assert G.decision({"pass": False}, {"pass": True})["primary_horizon"] == 24
    assert G.decision({"pass": False}, {"pass": False})["conversion_role"] == "report_only"


def test_g0_run_writes_once(tmp_path):
    d, _ = make_mapping_dir(tmp_path)
    res = G.run(d / "model_side.json", tmp_path / "g0", reps=4, B=50)
    assert set(res) >= {"horizon_12", "horizon_24", "decision", "rank12_report_only", "code"}
    assert res["code"][f"{L.E}/n3lib.py"] == L.sha256(X / "n3lib.py")
    assert len(res["horizon_12"]["cells"]) == 12
    with pytest.raises(FileExistsError):
        G.run(d / "model_side.json", tmp_path / "g0", reps=4, B=50)


# ----------------------------------------------------------------------------- well columns
def test_well_columns_acceptance_and_mapping():
    t = {"positive_means": "rise",
         "wells": [{"pile_name": "土庫國中", "well_name": "土庫（２）", "month_change_m": -0.5,
                    "year_change_m": 1.0}]}
    a = WC.rows_from(t, "2027-01", "pdf")
    b = WC.rows_from(t, "2027-01", "text")
    assert a["well"].iloc[0] == "土庫(2)"
    assert bool(WC.accept(a, b)["verified"].iloc[0])
    t2 = json.loads(json.dumps(t))
    t2["wells"][0]["month_change_m"] = -0.4
    c = WC.rows_from(t2, "2027-01", "text")
    assert not bool(WC.accept(a, c)["verified"].iloc[0])
    assert bool(WC.accept(a, c, b)["verified"].iloc[0])                 # third read settles
    st = pd.DataFrame({"sid": ["Y1", "Y9"], "WellName": ["土庫(2)", "土庫(2)"],
                       "x_twd97": [0.0, 9000.0], "y_twd97": [0.0, 0.0]})
    assert WC.map_wells(pd.Series(["土庫（２）"]), st, (100.0, 0.0)) == {"土庫(2)": "Y1"}
    acc = WC.accept(a, b)
    ch, lay = WC.changes_from_printed(acc, "土庫國中", {"土庫(2)": "Y1"}, {"Y1": 2})
    assert lay == {"Y1": 2} and ch["Y1"]["2027-01"] == -0.5
    assert "properties" in WC.SCHEMA_WELLS and "井名" in WC.SYSTEM_WELLS
    assert "positive_means" in WC.SCHEMA_WELLS["required"]


def test_well_columns_sign_legend():
    t = {"positive_means": "fall",
         "wells": [{"pile_name": "土庫國中", "well_name": "Y", "month_change_m": 0.3,
                    "year_change_m": 0.1}]}
    a, b = WC.rows_from(t, "2027-01", "pdf"), WC.rows_from(t, "2027-01", "text")
    ch, _ = WC.changes_from_printed(WC.accept(a, b), "土庫國中", {"Y": "Y1"}, {"Y1": 2})
    assert ch["Y1"]["2027-01"] == pytest.approx(-0.3)                 # printed fall -> h fell
    t2 = {**t, "positive_means": "rise"}
    assert not bool(WC.accept(a, WC.rows_from(t2, "2027-01", "text"))["verified"].iloc[0])
    none = WC.rows_from({"wells": t["wells"]}, "2027-01", "pdf")
    acc = WC.accept(none, none)
    assert acc["positive_means"].iloc[0] == "not_printed"
    assert WC.changes_from_printed(acc, "土庫國中", {"Y": "Y1"}, {"Y1": 2})[0] == {}
    assert np.isnan(WC.head_change(0.3, "not_printed"))
    good = WC.accept(a, b).assign(month="2027-01")
    assert list(V3.well_rows(good)["positive_means"]) == ["fall"]
    with pytest.raises(V3.UndeclaredSign):
        V3.well_rows(acc)
    with pytest.raises(V3.UndeclaredSign):
        V3.well_rows(good.drop(columns="positive_means"))


# ----------------------------------------------------------------------------- products
def synth_transcription(tmp: Path, ms: dict, rho: float = 1.0, start: str = L.WINDOW[0],
                        end: str = L.WINDOW[1]) -> Path:
    rows = []
    for key in ("tuku", "huwei", "xizhou"):
        w = ms["world"]["piles"][key] if key != "xizhou" else \
            ms["xizhou_r1c"]["world"]["piles"]["xizhou"]
        inc = dict(zip(w["months"], w["inc_cm"], strict=True))
        mlist = L.months(start, end)
        vals = {m: round(rho * inc.get(m, 0.0), 1) for m in mlist}
        for m in mlist:
            yv = round(sum(vals[q] for q in L.months(L.prev_month(m, 11), m) if q in vals), 1) \
                if m == end else None
            rows.append({"month": m, "pile": L.PILES[key]["pile"],
                         "depth_m": L.PILES[key]["depth_m"], "month_cm": vals[m],
                         "year_cm": yv, "verified": True, "chain_ok": None,
                         "file": f"integrated_{m}.pdf"})
    p = tmp / "deep_piles.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_v3_refuses_pre_window_and_builds(tmp_path):
    from common import guard
    from common.manifest import load_manifest

    d, _ = make_mapping_dir(tmp_path)
    ms = json.loads((d / "model_side.json").read_text())
    old = synth_transcription(tmp_path, ms, start="2026-10", end="2027-11")
    with pytest.raises(V3.FutureOnly):
        V3.build(old, tmp_path / "root")
    p = synth_transcription(tmp_path, ms)
    out = V3.build(p, tmp_path / "root")
    m, _ = load_manifest(out)
    guard.check_use(m, "prospective_scorer")
    for bad in ("baseline", "scoring_target", "selection", "design_time_evidence"):
        with pytest.raises(guard.UseNotAllowed):
            guard.check_use(m, bad)
    t = guard.open_product(out, "prospective_scorer").read_table(V3.PILE_TABLE)
    assert t["month"].min() == L.WINDOW[0] and set(t["flag"]) == {0}
    with pytest.raises(FileExistsError):
        V3.build(p, tmp_path / "root")


def make_competitor(tmp: Path, mapping_dir: Path, rate: float = 5.0) -> Path:
    mp = json.loads((mapping_dir / "mapping.json").read_text())["piles"]
    rows = [{"sid": f"B{k}", "x": mp[k]["site"]["x"] + 10, "y": mp[k]["site"]["y"],
             "rate_cm_yr": rate} for k in L.PRIMARY]
    rows.append({"sid": "BFAR", "x": X0 + 29000, "y": Y0 + 29000, "rate_cm_yr": 99.0})
    return LC.build(mapping_dir / "mapping.json", tmp, tmp / "root", rates=pd.DataFrame(rows),
                    grid=grid(), prereg_dir=tmp / "prereg")


def test_leveling_competitor_rules(tmp_path):
    from common.manifest import load_manifest

    d, _ = make_mapping_dir(tmp_path)
    mp = json.loads((d / "mapping.json").read_text())
    rates = pd.DataFrame([{"sid": "a", "x": X0 + 5400, "y": Y0 + 5600, "rate_cm_yr": 2.0},
                          {"sid": "b", "x": X0 + 12900, "y": Y0 + 12500, "rate_cm_yr": 3.0},
                          {"sid": "c", "x": X0 + 13900, "y": Y0 + 12500, "rate_cm_yr": 7.0}])
    tab = LC.competitor_table(rates, mp, grid()).set_index("pile_key")
    assert tab.loc["tuku", "rule"] == "pile_cell" and tab.loc["tuku", "rate_cm_yr"] == 2.0
    assert tab.loc["huwei", "rule"] == "pile_cell" and tab.loc["huwei", "rate_cm_yr"] == 3.0
    far = rates.assign(x=rates.x + 1200)
    tab2 = LC.competitor_table(far, mp, grid()).set_index("pile_key")
    assert tab2.loc["tuku", "rule"] == "within_1.5km"
    out = make_competitor(tmp_path, d)
    m, _ = load_manifest(out)
    assert m["allowed_uses"] == LC.ALLOWED and m["fit_window_end"] == "2021-12"
    assert m["gate"]["passed"]
    looks = prereg3.read_chained(tmp_path / "prereg" / "looks.jsonl")      # the read is logged
    assert [(lk["purpose"], lk["data_scope"]) for lk in looks] == [(LC.LOOK_PURPOSE,
                                                                     LC.LOOK_SCOPE)]
    assert looks[0]["members"][LC.LOOK_UNIT]["ids"] == ["BFAR", "Bhuwei", "Btuku"]


# ----------------------------------------------------------------------------- registration
def heads_product(tmp: Path, npz: Path, mapping: dict) -> Path:
    from common.manifest import build_manifest, write_manifest

    with np.load(npz) as z:
        heads = z["heads_mean"][0]
    labels = [d.strftime("%Y-%m") for d in DATES]
    rows = []
    for key in mapping:
        for w in mapping[key]["nest"]:
            for t, m in enumerate(labels):
                if "2026-09" <= m <= "2028-11":
                    rows.append({"sid": w["sid"], "month": m, "flag": 0,
                                 "value": float(heads[w["layer"] - 1, w["cell"], t])})
    d = tmp / "heads_root" / "window_heads" / "v1"
    d.mkdir(parents=True)
    pd.DataFrame(rows).drop_duplicates(["sid", "month"]).to_parquet(
        d / "heads_monthly.parquet", index=False)
    (d / "validation.json").write_text("{}")
    m = build_manifest(d, product="window_heads", layer="L1", inputs=[], code=[f"{L.E}/n3lib.py"],
                       parameters={}, fit_window_end="2028-11",
                       validation_report="validation.json",
                       gate={"passed": True, "criteria": "synthetic"},
                       allowed_uses=["prospective_scorer"], forbidden_uses=["selection"])
    write_manifest(d, m)
    return d


def add_score_look(pdir: Path, prereg_id: str, test_id: str) -> None:
    from harness import prereg as v1

    now = prereg3._now()
    v1._append_chained(pdir / "looks.jsonl", {
        "time": now, "logged": now, "wp": "SYN", "purpose": "score", "test_id": test_id,
        "prereg_id": prereg_id, "units": ["syn"], "data_scope": "heads:2026-11..2027-10",
        "scope_windows": "2026-11..2027-10", "members": {}, "seeded": False, "note": "",
        "tool": "test"})


def full_g0(tmp: Path, mapping_dir: Path, monkeypatch) -> Path:
    monkeypatch.setattr(P, "G0_REPS", 3)
    G.run(mapping_dir / "model_side.json", tmp / "g0", reps=3, B=L.BOOT_B)
    return tmp / "g0" / "result.json"


def write_ruling(root: Path, applies=(L.ID,), rid="N3.level_only") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / "user_rulings_synthetic.json"
    p.write_text(json.dumps({"schema": P.RULINGS_SCHEMA, "date": "2026-11-20",
                             "ruling_by": "user", "rulings": [
                                 {"id": rid, "question": "G0 failed: register level-only?",
                                  "decision": "Approved (synthetic)",
                                  "applies_to": list(applies)}]}, ensure_ascii=False))
    return p


def synth_inventory(declared=L.PRIMARY) -> dict:
    return {"yunlin": {}, "yunlin_declared": list(declared), "n_yunlin": len(declared),
            "xizhou": {"seasons_upper_bound": [2019, 2020, 2021]}, "rule": "synthetic"}


@needs_repo
def test_register_and_score_in_tmp_registry(tmp_path, monkeypatch):
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    d, npz = make_mapping_dir(tmp_path)
    g0 = full_g0(tmp_path, d, monkeypatch)
    comp = make_competitor(tmp_path, d)
    self_sha = L.sha256(X / "prereg_n3.py")
    monkeypatch.setattr(L, "_now", lambda: BEFORE_DEADLINE)
    rdir = tmp_path / "rulings"
    good = write_ruling(rdir)
    kw = {"prereg_dir": pdir, "rows_dir": rows, "rulings_dir": rdir}

    def args(sha=self_sha, mode=None, ruling=None):
        a = ["--write", "--quiet", "--mapping-dir", str(d), "--g0-result", str(g0),
             "--leveling-product", str(comp), "--expect-self-sha", sha]
        a += ["--g0-fail-mode", mode] if mode else []
        return a + (["--ruling", str(ruling), "--ruling-id", "N3.level_only"] if ruling else [])

    dec = json.loads(g0.read_text())["decision"]
    mode, rul = None, None
    if dec["primary_horizon"] is None:          # G0 failed: refused unless a user ruling
        assert P.main(args(), **kw) == 1
        assert P.main(args(mode="level-only"), **kw) == 1                # no ruling
        other = write_ruling(tmp_path / "r2", applies=("X5.other.v1",))
        assert P.main(args(mode="level-only", ruling=other), **kw) == 1  # outside rulings dir
        bad = write_ruling(rdir / "sub", applies=("X5.other.v1",))
        assert P.main(args(mode="level-only", ruling=bad), **kw) == 1    # does not name N3
        mode, rul = "level-only", good
    assert P.main(args("0" * 64, mode, rul), **kw) == 1
    base = args(mode=mode, ruling=rul)
    monkeypatch.setattr(L, "_now", lambda: L.DEADLINE)
    assert P.main(base, **kw) == 1                                      # deadline passed
    monkeypatch.setattr(L, "_now", lambda: BEFORE_DEADLINE)
    assert P.main(base, **kw) == 0
    doc, _ = prereg3.load_prereg(L.ID, prereg_dir=pdir, rows_dir=rows)
    assert doc["roles"][L.TEST_LEVEL] == "gating"
    if rul is not None:
        rb = doc["rulings"]["G0 level-only (N3.level_only)"]
        assert rb["status"] == "ok" and rb["decision"] == "Approved (synthetic)"
        assert "user ruling" in doc["page_mapping"]["before_scoring"]["C31"]
    conv = doc["tests"][0]
    assert conv["fresh_units"] == [L.UNIT, L.UNIT_HEADS_YL]
    assert conv["external_units"][L.UNIT_HEADS_YL]["ids"] == ["H1", "H2", "Y1", "Y2"]
    assert doc["tests"][4]["external_units"][L.UNIT_HEADS_XZ]["ids"] == ["C1", "C2"]
    assert doc["g0"]["attempts"]["scratch"]["result_sha256"].startswith("5deadeba")
    assert "DISPLAYED" in doc["tests"][0]["metric"]["g_mod"]
    assert "column_pairing" in doc["disclosures"]
    assert doc["roles"][L.TEST_XZ] == "report_only"
    assert [t["test_id"] for t in doc["tests"]] == [L.TEST_CONV, L.TEST_LEVEL, L.TEST_CONV24,
                                                    L.TEST_LEVEL24, L.TEST_XZ]
    assert "C31" in doc["page_mapping"]["before_scoring"]
    assert "ONCE" in doc["attempt_policy"]
    real_index = REPO / "results/dataprep/preregistration/INDEX.jsonl"
    assert not real_index.exists() or L.ID not in real_index.read_text()
    # ---- scoring day
    mapping = json.loads((d / "mapping.json").read_text())["piles"]
    ms = json.loads((d / "model_side.json").read_text())
    piles = V3.build(synth_transcription(tmp_path, ms), tmp_path / "root")
    heads = heads_product(tmp_path, npz, mapping)
    n2c = tmp_path / "n2c_report.json"
    a = SimpleNamespace(horizon="12", piles_product=piles, heads_product=heads,
                        heads_table="heads_monthly.parquet", stations=None, n2c_report=n2c,
                        out=tmp_path / "score", part="piles")
    monkeypatch.setattr(L, "_now", lambda: BEFORE_DEADLINE)
    with pytest.raises(S.ScoringRefused, match="not before"):
        S.run_score(a, prereg_dir=pdir, rows_dir=rows)
    monkeypatch.setattr(L, "_now", lambda: SCORING_DAY)
    nope = SimpleNamespace(**{**vars(a), "piles_product": tmp_path / "no_product",
                              "out": tmp_path / "ne"})
    with pytest.raises(S.ScoringRefused, match="--final"):
        S.run_score(nope, prereg_dir=pdir, rows_dir=rows)
    nope.final = True
    assert S.run_score(nope, prereg_dir=pdir, rows_dir=rows)["verdict"] == "not_evaluable"
    with pytest.raises(S.ScoringRefused, match="session order"):
        S.run_score(a, prereg_dir=pdir, rows_dir=rows)
    assert not any(lk.get("prereg_id") == L.ID
                   for lk in prereg3.read_chained(pdir / "looks.jsonl"))
    add_score_look(pdir, L.N2_ID, L.N2_TEST)
    add_score_look(pdir, L.N2C_ID, L.N2C_TEST)
    rep = S.run_score(a, prereg_dir=pdir, rows_dir=rows)
    conv, lev = rep["T_conversion"], rep["T_level"]
    assert conv["verdict"] == "conversion_supported" and conv["rho"] == pytest.approx(1, abs=0.1)
    assert conv["role"] == doc["roles"][L.TEST_CONV]
    assert set(conv["report_only"]) >= {"sign_split", "calibration_slope_obs_on_zr"}
    assert lev["verdict"] == "level_supported"
    assert lev["observed_detail"]["tuku"]["construction"] == "trailing12"
    assert "T_xizhou" not in rep and rep["part"] == "piles"
    looks = {(lk["test_id"], lk["purpose"]) for lk in prereg3.read_chained(pdir / "looks.jsonl")
             if lk.get("prereg_id") == L.ID}
    assert looks == {(L.TEST_CONV, "score"), (L.TEST_LEVEL, "score")}
    with pytest.raises(FileExistsError):
        S.run_score(a, prereg_dir=pdir, rows_dir=rows)
    # ---- the 溪州 arm after N2C's step-(2) report, in a second output directory
    a.out, a.part = tmp_path / "score_xz", "xizhou"
    with pytest.raises(S.ScoringRefused, match="step-\\(2\\)"):
        S.run_score(a, prereg_dir=pdir, rows_dir=rows)
    n2c.write_text(json.dumps({"prereg_id": L.N2C_ID, "verdict": "synthetic"}))
    rep2 = S.run_score(a, prereg_dir=pdir, rows_dir=rows)
    assert rep2["T_xizhou"]["role"] == "report_only" and "seasons" in rep2["T_xizhou"]
    assert "T_conversion" not in rep2
    looks = [(lk["test_id"], lk["purpose"]) for lk in prereg3.read_chained(pdir / "looks.jsonl")
             if lk.get("prereg_id") == L.ID]
    assert sorted(looks) == sorted([(L.TEST_CONV, "score"), (L.TEST_LEVEL, "score"),
                                    (L.TEST_XZ, "score")])          # each scored once
    # ---- rank 12: declared after N3, G0 lines decide; nothing is opened when they fail
    monkeypatch.setattr(PH, "metadata_inventory", lambda *a, **k: synth_inventory())
    pargs = ["declare", "--n3-mapping-dir", str(d), "--g0-result", str(g0)]
    assert PH.main(pargs + ["--validate"], prereg_dir=pdir, rows_dir=rows) == 0
    assert PH.main(pargs + ["--write", "--expect-self-sha", "0" * 64], prereg_dir=pdir,
                   rows_dir=rows) == 1
    ph_sha = L.sha256(X / "posthoc_pile_conversion.py")
    assert PH.main(pargs + ["--write", "--expect-self-sha", ph_sha], prereg_dir=pdir,
                   rows_dir=rows) == 0
    lines = PH.g0_lines(json.loads(g0.read_text()))
    if not any(lines.values()):
        n_before = len(prereg3.read_chained(pdir / "looks.jsonl"))
        r = PH.run(tmp_path / "ph", prereg_dir=pdir, rows_dir=rows)
        assert r["values_read"] is False and r["yunlin"]["line"] == "not_identifiable"
        assert len(prereg3.read_chained(pdir / "looks.jsonl")) == n_before


def test_problems_for_write_slots_pending(monkeypatch):
    monkeypatch.setattr(L, "_now", lambda: BEFORE_DEADLINE)
    d = P.doc()
    assert d["model_side"]["status"] == P.PENDING and d["g0"]["status"] == P.PENDING
    assert d["roles"][L.TEST_CONV] == P.NOT_REGISTERED == d["roles"][L.TEST_LEVEL]
    assert d["fresh_units"] == [L.UNIT, L.UNIT_XZ, L.UNIT_HEADS_YL, L.UNIT_HEADS_XZ]
    assert "default_without_ruling" in d["page_mapping"]
    assert any("not registered" in b for b in P.problems_for_write(d, None, "refuse"))


def test_doc_validates_with_synthetic_slots(tmp_path, monkeypatch):
    d, _ = make_mapping_dir(tmp_path)
    g0 = full_g0(tmp_path, d, monkeypatch)
    comp = make_competitor(tmp_path, d)
    doc = P.doc(d, g0, comp, prereg_dir=tmp_path / "prereg")
    assert doc["model_side"]["status"] == "ok" and doc["g0"]["status"] == "ok"
    assert doc["leveling_competitor"]["status"] == "ok"
    assert prereg3.validate_prereg(doc, prereg_dir=tmp_path / "prereg") == []
    bad = P.g0_block(g0, {"files": {"model_side.json": {"sha256": "x" * 64}}})
    assert "another model_side" in bad["status"]
    monkeypatch.setattr(P, "G0_REPS", 500)
    assert "not a full" in P.g0_block(g0, {})["status"]
    # archived attempts are listed with their reason
    arc = tmp_path / "archive" / "20261101T000000Z"
    (arc / "g0").mkdir(parents=True)
    (arc / "g0" / "result.json").write_text(g0.read_text())
    (arc / "REASON.txt").write_text("engineering: synthetic\n")
    att = P.g0_attempts(g0)
    assert att["n_archived"] == 1 and att["archived"][0]["reason"] == "engineering: synthetic"
    assert att["archived"][0]["results"][0]["sha256"] == L.sha256(g0)
    # a stale model_side without the column-pairing fields is refused
    ms = json.loads((d / "model_side.json").read_text())
    ms.pop("column_pairing")
    (d / "model_side.json").write_text(json.dumps(ms, default=float))
    assert "column-pairing" in P.mapping_block(d)["status"]


def test_ruling_block_rules(tmp_path):
    root = tmp_path / "rulings"
    good = write_ruling(root)
    assert P.ruling_block(good, "N3.level_only", root)["status"] == "ok"
    assert P.ruling_block(None, None, root)["status"] == "absent"
    assert "not found" in P.ruling_block(good, "other", root)["status"]
    assert "not under" in P.ruling_block(good, "N3.level_only", tmp_path / "x")["status"]
    lead = json.loads(good.read_text())
    lead["ruling_by"] = "lead"
    good.write_text(json.dumps(lead))
    assert "not a user ruling" in P.ruling_block(good, "N3.level_only", root)["status"]


def test_posthoc_estimators_on_synthetic_rows(tmp_path, monkeypatch):
    d, npz = make_mapping_dir(tmp_path)
    mapping = json.loads((d / "mapping.json").read_text())
    ms = json.loads((d / "model_side.json").read_text())
    labels = [x.strftime("%Y-%m") for x in DATES]
    with np.load(npz) as z:
        heads = z["heads_mean"][0]
    feed = []
    for key in L.PRIMARY:
        w = ms["hindcast"]["piles"][key]
        for m, i in zip(w["months"], w["inc_cm"], strict=True):
            feed.append({"sensor_id": L.PILES[key]["feed_preorigin"], "month": m,
                         "value": 0.5 * i, "flag": 0})
    hs = {}
    for k in mapping["piles"]:
        for wl in mapping["piles"][k]["nest"]:
            hs[wl["sid"]] = pd.Series(heads[wl["layer"] - 1, wl["cell"]], index=labels)
    monkeypatch.setattr(PH, "_feed_rows", lambda cols: pd.DataFrame(feed))
    monkeypatch.setattr(PH, "_heads_rows", lambda ids: {s: hs[s] for s in ids if s in hs})
    y = PH.yunlin(ms, mapping)
    assert y["line"] == "computed" and y["k_model_over_obs"] == pytest.approx(2.0, rel=0.05)
    assert "C27" in y["wording"] and y["n_piles"] == 2
    y1 = PH.yunlin(ms, mapping, ("huwei",))                               # declared n = 1
    assert y1["line"] == "computed" and y1["piles_used"] == ["huwei"]
    assert y1["k_model_over_obs"] == pytest.approx(2.0, rel=0.05)
    assert PH.yunlin(ms, mapping, ())["line"] == "not_evaluable"
    assert PH.inverse_interval([0.4, 0.5]) == pytest.approx([2.0, 2.5])
    assert PH.inverse_interval([-0.1, 0.5]) is None
    assert PH.inverse_interval(None) is None
    flat = [{**r, "value": 0.0} for r in feed]                         # rho 0 -> no k
    monkeypatch.setattr(PH, "_feed_rows", lambda cols: pd.DataFrame(flat))
    y0 = PH.yunlin(ms, mapping)
    assert y0["k_model_over_obs"] is None and "not defined" in y0["wording"]
    w = ms["xizhou_r1c"]["hindcast"]["piles"]["xizhou"]
    pr = pd.DataFrame([{"station": "溪州國小", "depth_m": 300, "month": m, "value": 10 * i,
                        "flag": 0} for m, i in zip(w["months"], w["inc_cm"], strict=True)])
    monkeypatch.setattr(PH, "_printed_rows", lambda cols: pr)
    x = PH.xizhou_line(ms, mapping)
    assert x["line"] in ("computed", "not_evaluable")
    if x["line"] == "computed":
        assert x["k_model_over_obs"] == pytest.approx(1.0, rel=0.05)


def test_posthoc_metadata_inventory(tmp_path):
    fm = tmp_path / "mapping.csv"
    pd.DataFrame([{"sensor_id": "TKJS(1)", "first_month": "2016-06", "last_month": "2025-02",
                   "value_col_not_read": 1},
                  {"sensor_id": "NTUH(1)", "first_month": "2021-06", "last_month": "2025-01",
                   "value_col_not_read": 1}]).to_csv(fm, index=False)
    pv = tmp_path / "validation.json"
    pv.write_text(json.dumps({"pile_publication": {"dropped_months": ["2022-10", "2021-12"]}}))
    inv = PH.metadata_inventory(fm, pv)
    assert inv["yunlin"]["tuku"]["span_months_upper_bound"] == 79
    assert inv["yunlin"]["huwei"]["span_months_upper_bound"] == 19
    assert inv["yunlin_declared"] == ["tuku"] and inv["n_yunlin"] == 1         # n = 1
    assert inv["xizhou"]["seasons_upper_bound"] == [2019, 2020, 2021]
    assert PH._span_months("2023-01", "2024-01") == 0


def test_xizhou_depth_rule_fires_on_another_depth():
    ms = L.months("2027-06", "2027-11")
    rows = pd.DataFrame([{"pile": "溪州國小", "depth_m": 300, "month": m} for m in ms]
                        + [{"pile": "溪州國小", "depth_m": 250, "month": "2027-08"},
                           {"pile": "土庫國中", "depth_m": 250, "month": "2027-09"}])
    ok = S.xizhou_depth_ok(rows, ms)
    assert not ok["2027-08"] and ok["2027-09"] and ok.sum() == 5
