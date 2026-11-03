"""FE.encompass_2026-12.v1 (round-8 rank 9), on synthetic inputs only.

- the two-contrast within-well estimator equals its definition, removes per-well offsets, is
  linear in y (the identity behind the lambda power points); the persistence-error (PE) arm's
  world and the B1 mechanism (a layer persistence error drives the one-contrast lambda, not
  lambda_P);
- the well x quarter (pigeonhole) bootstrap: a weighted replicate equals the explicitly
  duplicated sample;
- the reading rule (fixed margin 0.25); page wording in both languages, never-words checked on
  each language's own output;
- the B2 physics-world construction: window shifts, 2023-26 shifts, 2012-22 shifts, truth =
  ZR0 + const, and real_truths' scenario mapping;
- seeds have no default; refused seeds raise; injected inputs need test_only, a TEST.* id
  and a non-real registry;
- G0 on FABRICATED records (no test computes a rate from simulated worlds); the B2 switch;
  traceability of decidability.json; the smoke writes no rate;
- the runners refuse before registration, before 2028-01-15, after 2028-01-31, when a report
  exists, a missing window product, a non-canonical E02c report and an E02d record without a
  window; every attempt is appended to attempts.jsonl;
- the scorer's pure part and chain of custody; the draft's key fields.

Checks that need the gitignored results/ tree skip when it is absent (CI).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")         # hydrophysics.twin.coverage imports it

REPO = Path(__file__).resolve().parents[1]
D_DIR = REPO / "dataprep_staging" / "fe_encompass"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_saved = list(sys.path)
L = _load("fe_lib", D_DIR / "fe_lib.py")
D = _load("decide_fe", D_DIR / "decide_fe.py")
SF = _load("score_fe", D_DIR / "score_fe.py")
PR = _load("prereg_fe", D_DIR / "prereg_fe.py")
sys.path[:] = _saved

HAVE_RESULTS = (REPO / "results" / "dataprep" / "preregistration" / "INDEX.jsonl").is_file()
NEED_RESULTS = pytest.mark.skipif(not HAVE_RESULTS, reason="results/ (gitignored) is absent")
SEED = 424242
LATE = dt.datetime(2028, 1, 20, tzinfo=dt.timezone.utc)
EARLY = dt.datetime(2027, 6, 1, tzinfo=dt.timezone.utc)
AFTER = dt.datetime(2028, 2, 1, 0, 0, 1, tzinfo=dt.timezone.utc)
WIN = D._FE_IN_AXIS


@pytest.fixture(autouse=True)
def _staging_on_path():
    saved = list(sys.path)
    D.S._paths()
    for p in (str(D_DIR), str(REPO / "dataprep_staging" / "fv_stat_v2"),
              str(REPO / "dataprep_staging" / "gov_heads_window")):
        if p not in sys.path:
            sys.path.insert(0, p)
    yield
    sys.path[:] = saved


def _ols(xs, y, m):
    """Explicit within-well least squares (reference implementation)."""
    rows = []
    for w in range(y.shape[0]):
        sel = m[w]
        if not sel.any():
            continue
        cols = [x[w, sel] - x[w, sel].mean() for x in xs]
        rows.append((np.column_stack(cols), y[w, sel] - y[w, sel].mean()))
    X = np.vstack([r[0] for r in rows])
    Y = np.concatenate([r[1] for r in rows])
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def _panel(W=40, lp=0.5, lc=0.3, noise=0.05, seed=1):
    r = np.random.default_rng(seed)
    T = len(L.MONTHS)
    xc = r.standard_normal((W, T)) * 0.2 + r.standard_normal((W, 1)) * 2
    xp = 0.5 * xc + r.standard_normal((W, T)) * 0.2 + r.standard_normal((W, 1)) * 3
    y = lp * xp + lc * xc + noise * r.standard_normal((W, T)) + r.standard_normal((W, 1)) * 5
    m = r.random((W, T)) > 0.15
    return xp, xc, y, m


def _frame(W=120, *, truth="phys", seed=3):
    """Synthetic step-(5)-like rows. truth 'phys': obs = PHYS + noise; 'stat': obs = STAT."""
    r = np.random.default_rng(seed)
    rows = []
    for w in range(W):
        sid = f"{'09' if w % 2 else '07'}{w:06d}"
        lvl, a0 = r.standard_normal() * 3, r.standard_normal() * 0.5
        for j, t in enumerate(L.MONTHS):
            clim = lvl + 0.2 * np.sin(2 * np.pi * j / 12)
            stat = clim + a0 * 0.9 ** (j + 1)
            phys = clim + 0.4 * np.sin(2 * np.pi * (j + w % 5) / 12) + 1.0
            base = phys if truth == "phys" else stat
            rows.append((sid, t, base + 0.03 * r.standard_normal(), stat, phys, clim))
    return pd.DataFrame(rows, columns=["station_id", "date", "observed_m", "stat_m", "phys_m",
                                       "clim_m"])


def _fab_h0(inp, seed=5):
    """A ZR0-like series with a non-seasonal post-2022 feature (identifiable beyond CLIM)."""
    r = np.random.default_rng(seed)
    t = np.arange(D.T_AX)
    return (r.standard_normal((len(inp.ids), 1)) * 3 + 0.5 * np.sin(2 * np.pi * t / 12)
            + 0.4 * np.sin(2 * np.pi * t / 7) * (t >= D.O.T_FIT))


# ----------------------------------------------------------------------------- static
def test_conventions():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in [*sorted(D_DIR.glob("*.py")), Path(__file__)]:
        txt = p.read_text(encoding="utf-8")
        assert "from __future__ import annotations" in txt, p
        assert not pat.search(txt), p
        if p.parent == D_DIR:
            assert "write_prereg(" not in txt, p
    assert not list(D_DIR.glob("*.json")) and not list(D_DIR.glob("*.csv"))


def test_identities_streams_and_cells():
    assert L.ID == "FE.encompass_2026-12.v1" and L.UNIT == "FE:heads_2026-12_2027-10_E02xE04"
    assert len(L.MONTHS) == 11 and str(L.MONTHS[0].date()) == "2026-12-01"
    assert list(L.QIDX) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    assert L.DELTA == 0.25 and D.G0["p_inf_ar_max"] == 0.075
    assert D.G0["p_inf_ar_cell_max"] == 0.15 == D.G0["p_inf_pe_cell_max"]
    assert set(D.STREAMS.values()).isdisjoint({D.S.STREAM_E04, D.S.STREAM_NULL,
                                               D.S.STREAM_SELFCHECK, 31, 32, 33})
    assert len(set(D.STREAMS.values())) == len(D.STREAMS)
    assert {"mask_ar", "mask_phys", "subset_ar", "subset_phys"} <= set(D.STREAMS)
    kinds = [c["kind"] for c in D.ar_cells()]
    assert kinds.count("grid") == 9 and {"basin", "mcar", "w100"} <= set(kinds)
    assert [(c["rho_hist"], c["rho_win"]) for c in D.pe_cells()] == [(0.93, 0.80),
                                                                     (0.80, 0.93)]
    assert D.PE_SHAPES == {"none": None, "fast": 0.6, "slow": 0.99}
    assert len(D.phys_cells()) == 12 and D.TRUTHS == ("R0", "Q0", "R1c", "X5R1")
    assert set(D.S.FROZEN) <= set(D.FROZEN) and {L.LIB, L.DECIDE, L.SCORER, D.GOV} <= set(
        D.FROZEN)
    assert L.PREREG not in D.FROZEN and all((REPO / p).is_file() for p in D.FROZEN)
    assert not {"d", "dc", "xc"} & {f.name for f in dataclasses.fields(D.Inputs)}   # U7


# ----------------------------------------------------------------------------- estimator
def test_two_contrast_matches_definition():
    xp, xc, y, m = _panel()
    W, T = y.shape
    got = L.coefs(np.ones((1, T)), np.ones((1, W)), [xp, xc], y, m)[0]
    assert np.allclose(got, _ols([xp, xc], y, m), rtol=1e-10)
    assert got[0] == pytest.approx(0.5, abs=0.05) and got[1] == pytest.approx(0.3, abs=0.05)
    raw = L.coefs(np.ones((1, T)), np.ones((1, W)), [xp, xc], y, m, demean=False)[0]
    assert abs(raw[0] - 0.5) > 0.1


def test_pigeonhole_replicate_equals_duplicated_sample():
    xp, xc, y, m = _panel(W=12, seed=5)
    cq, cw = L.boot_weights(12, 3, np.random.default_rng(9))
    assert cw.sum(axis=1).tolist() == [12] * 3
    got = L.coefs(cq, cw, [xp, xc], y, m)
    for b in range(3):
        rows_x, rows_y = [], []
        for w in range(12):
            for _ in range(int(cw[b, w])):
                idx = [t for t in range(11) if m[w, t] for _ in range(int(cq[b, t]))]
                if not idx:
                    continue
                X = np.column_stack([xp[w, idx], xc[w, idx]])
                rows_x.append(X - X.mean(axis=0))
                rows_y.append(y[w, idx] - y[w, idx].mean())
        want = np.linalg.lstsq(np.vstack(rows_x), np.concatenate(rows_y), rcond=None)[0]
        assert np.allclose(got[b], want, rtol=1e-8)


def test_linearity_shift_identity():
    xp, xc, y, m = _panel(seed=7)
    w = L.boot_weights(xp.shape[0], 200, np.random.default_rng(1))
    a = L.coefs(*w, [xp, xc], y, m)
    b = L.coefs(*w, [xp, xc], y - 0.5 * xp, m)
    assert np.allclose(b[:, 0] - a[:, 0], -0.5, atol=1e-9)
    assert np.allclose(b[:, 1], a[:, 1], atol=1e-9)


def test_b1_persistence_error_moves_one_contrast_not_lambda_p():
    """The reviewer's mechanism, noise-free: the layer persists at 0.93, the truth decays at
    0.80, dC carries no anomaly term; the one-contrast lambda absorbs the persistence error,
    lambda_P of the two-contrast fit stays near 0."""
    r = np.random.default_rng(11)
    W, h = 147, np.arange(3, 14)[None, :]
    a0 = r.standard_normal((W, 1)) * 1.5
    stat = a0 * 0.93 ** h
    y = a0 * 0.80 ** h - stat
    xc = -stat
    dc = 0.3 * r.standard_normal((W, 1)) * np.sin(2 * np.pi * (h + 2) / 12)
    m = np.ones_like(y, bool)
    two = _ols([dc + xc, xc], y, m)
    one = _ols([dc + xc], y, m)
    assert abs(two[0]) < 0.01 < 0.02 < abs(one[0])


def test_pe_world_follows_both_persistences():
    inp = D.fabricated_inputs(30, seed=SEED)
    c = D.pe_cells()[0]
    truth = D.pe_truth(inp, c, 0, seed=SEED)
    hist = D.S.world(inp.layout(), c["rho_hist"], D.PE_SHARE, D.STREAMS["pe"], c["cell"], 0,
                     seed=SEED) * inp.sd[:, None]
    t0 = int(inp.last_fit[0])
    assert np.array_equal(truth[:, :t0 + 1], hist[:, :t0 + 1])     # history: rho_hist
    assert not np.allclose(truth[:, t0 + 1:], hist[:, t0 + 1:])     # window: regenerated
    r = D.pe_record(inp, c, 0, D.Pool(dc=np.zeros((1, 30, 11)), xp=np.zeros((1, 30, 11)),
                                      xc=np.zeros((1, 30, 11))), seed=SEED, B=20)
    assert set(r) == {"none", "fast", "slow", "one_none", "one_fast", "one_slow"}
    a0 = np.ones(3)
    hh = np.array([[1, 2], [1, 2], [1, 2]])
    assert np.allclose(D.pe_shape(a0, hh, 0.6), 0.5 * 0.6 ** hh)
    assert not D.pe_shape(a0, hh, None).any()


def test_degenerate_and_collinear():
    xp, xc, y, m = _panel()
    e0 = L.estimate([np.zeros_like(xp), xc], y, m, rng=np.random.default_rng(2), B=50)
    assert e0["P"]["lower"] is None and e0["bad_share"] == 1.0
    e1 = L.estimate([2 * xc, xc], y, m, rng=np.random.default_rng(2), B=50)
    assert e1["bad_share"] == 1.0


def test_panel_filters():
    f = _frame(W=10)
    f = pd.concat([f, f.assign(date=pd.Timestamp("2026-11-01")).drop_duplicates("station_id")])
    f = f[~((f["station_id"] == f["station_id"].iloc[0])
            & (pd.to_datetime(f["date"]) > pd.Timestamp("2027-02-01")))]
    p = L.panel(f)
    assert len(p["ids"]) == 9 and p["m"].shape == (9, 11) and p["rows"] == 99
    with pytest.raises(L.EngineeringFailure, match="duplicate"):
        L.panel(pd.concat([f, f.head(1)]))


def test_partial_rms():
    xc = np.tile(np.arange(11.0), (3, 1))
    m = np.ones((3, 11), bool)
    assert L.partial_rms(2.0 * xc + 7.0, xc, m) == pytest.approx(0.0, abs=1e-12)
    assert L.contrast_rms(xc + np.array([[0.0], [5.0], [9.0]]), m) == pytest.approx(
        np.sqrt(10.0))


# ----------------------------------------------------------------------------- readings
@pytest.mark.parametrize("lo,hi,want,small", [
    (0.1, 0.9, "physics_informative", False), (0.05, 0.2, "physics_informative", True),
    (-0.2, 0.2, "stat_encompasses", False), (-0.2, 0.3, "undecided", False),
    (0.0, 0.24, "stat_encompasses", False), (-0.3, 0.25, "undecided", False)])
def test_reading_rule(lo, hi, want, small):
    est = {"lambda": (lo + hi) / 2, "lower": lo, "upper": hi}
    assert L.reading(est, informative=True, evaluable=True) == {"reading": want,
                                                                "small": small}


def test_reading_gates_first():
    est = {"lambda": 0.5, "lower": 0.2, "upper": 0.8}
    assert L.reading(est, informative=False, evaluable=True)["reading"] == "not_informative"
    assert L.reading(est, informative=True, evaluable=False)["reading"] == "not_evaluable"
    assert L.reading(est, bad_share=0.02, informative=True,
                     evaluable=True)["reading"] == "not_evaluable"


def test_wording_both_languages_and_never_words():
    est = {"lambda": 0.12, "lower": -0.1, "upper": 0.2}
    for key in ("physics_informative", "stat_encompasses", "undecided", "not_informative"):
        for role in ("gating", "report_only"):
            w = L.wording({"reading": key, "small": key == "physics_informative"}, est,
                          n_wells=140, role=role, why="G0 not passed")
            assert set(w) == {"en", "zh"} and "{" not in w["en"] + w["zh"]
            assert not any(bad in w["en"].lower() for bad in L.WORDING["never"]["en"])
            assert not any(bad in w["zh"] for bad in L.WORDING["never"]["zh"])
    w = L.wording({"reading": "stat_encompasses", "small": False}, est, n_wells=140,
                  role="gating")
    assert "encompassing coefficient" in w["en"] and "涵蓋係數" in w["zh"]
    assert "0.25" in w["en"] and "0.25" in w["zh"]
    w = L.wording({"reading": "undecided", "small": False}, est, n_wells=140,
                  role="report_only", why="not measured before the deadline")
    assert w["en"].startswith("Report-only") and w["zh"].startswith("僅供報告")


# ----------------------------------------------------------------------------- B2 world
def test_b2_physics_world_construction():
    inp = D.fabricated_inputs(30, seed=SEED)
    h0 = _fab_h0(inp)
    h = h0 + 0.3 + 0.1 * np.cos(2 * np.pi * np.arange(D.T_AX) / 5)[None, :]
    base = D.phys_parts(inp, h, h0, 0, 0, seed=SEED)
    hw = h.copy()
    hw[:, WIN] += 0.8                                   # window months only
    q = D.phys_parts(inp, hw, h0, 0, 0, seed=SEED)
    for a, b in (("phys", "stat"), ("clim", "stat")):
        assert np.allclose(q[a] - q[b], base[a] - base[b], atol=1e-12)
    assert np.allclose(q["obs"] - base["obs"], 0.8)
    hp = h.copy()
    hp[:, D.O.T_FIT:int(inp.last_fit.min()) + 1] += 0.5      # 2023-01..origin
    q = D.phys_parts(inp, hp, h0, 0, 0, seed=SEED)
    assert np.allclose(q["phys"], base["phys"], atol=1e-12)
    assert not np.allclose(q["stat"], base["stat"])
    hf = h.copy()
    hf[:, :D.O.T_FIT] += 0.25                                 # 2012-22 (datum months)
    q = D.phys_parts(inp, hf, h0, 0, 0, seed=SEED)
    assert np.allclose(q["phys"] - base["phys"], 0.25, atol=1e-12)


def test_b2_truth_equal_zr0_gives_lambda_p_near_one():
    inp = dataclasses.replace(D.fabricated_inputs(120, seed=SEED), sd=np.full(120, 0.01))
    h0 = _fab_h0(inp)
    y, xp, xc, _ = D.phys_world(inp, h0 + 0.7, h0, 0, 0, seed=SEED)
    est = L.estimate([xp, xc], y, np.ones_like(y, bool), rng=np.random.default_rng(1), B=50)
    assert est["P"]["lambda"] == pytest.approx(1.0, abs=0.05)


def test_real_truths_scenario_mapping():
    inp = D.fabricated_inputs(4, seed=SEED)
    reg = {f"{t}_b{b}": {"path": f"{t}_b{b}", "npz_sha256": "s"} for t in D.TRUTHS
           for b in range(D.V2.PLAN.n_batch)}

    def heads(path, cells, layers):
        if path == "ZR0":
            return {"scenario_names": np.array(["baseline"])}, np.zeros((1, 4, D.T_AX))
        t, b = path.split("_b")
        names = D.V2.PLAN.names(int(b))
        val = (np.arange(len(names)) + 1000 * int(b) + 100000 * D.TRUTHS.index(t))
        return ({"scenario_names": np.array(names)},
                np.broadcast_to(val[:, None, None], (len(names), 4, D.T_AX)).astype(float))

    def verified(rel, sha, repo=None):
        return "ZR0" if rel == L.ZR0_NPZ else rel

    h_zr0, gen, used = D.real_truths(inp, reg, verified=verified, heads=heads)
    got = [(ti, k, float(h[0, 0])) for ti, k, h in gen]
    assert len(got) == 4 * D.K_PHYS and set(used["fv2_batches"]) == set(reg)
    for ti, k, v in got:
        b, j = divmod(k, D.V2.PLAN.batch)
        assert v == 100000 * ti + 1000 * b + 1 + j       # h[1 + j]: scenario 0 is baseline


# ----------------------------------------------------------------------------- seeds
def test_seed_has_no_default_and_refused_seeds():
    for fn in (D.run, D.rng_for, D.ar_record, D.pe_record, D.pe_truth, D.phys_world,
               D.phys_parts, D.phys_records, D.sig_record, D.variant, D.study,
               D.fabricated_inputs, D.fabricated_truths, SF.score_frame, PR.draft_prereg):
        p = inspect.signature(fn).parameters["seed"]
        assert p.default is inspect.Parameter.empty, fn.__name__
    for bad in (D.SMOKE_SEED, *D.S.REFUSED_SEEDS):
        with pytest.raises(L.EngineeringFailure, match="refused"):
            D.check_seed(bad)
        with pytest.raises(L.EngineeringFailure, match="refused"):
            PR.draft_prereg(bad, ids=["09000001"])
    for bad in (True, -1, 1.5, "7"):
        with pytest.raises(L.EngineeringFailure):
            D.check_seed(bad)
    with pytest.raises(SystemExit):
        D.main(["run", "--prereg-id", L.ID])
    with pytest.raises(SystemExit):
        SF.main(["score", "--prereg-id", L.ID])


# ----------------------------------------------------------------------------- study
def test_variants_use_arm_streams_and_drop_wells():
    rows, m = D.variant("mcar", 147, 10, 0, arm="ar", seed=SEED)
    assert m.shape == (len(rows), 11) and (m.sum(axis=1) >= 6).all() and len(rows) <= 147
    rows2, m2 = D.variant("mcar", 147, 10, 0, arm="phys", seed=SEED)
    assert not np.array_equal(m[:50], m2[:50])                 # distinct streams per arm
    rows, m = D.variant("w100", 147, 11, 0, arm="ar", seed=SEED)
    assert len(rows) == 100 == len(set(rows)) and m.all()
    rows, m = D.variant("grid", 147, 0, 0, arm="ar", seed=SEED)
    assert len(rows) == 147 and m.all()


def _pool(W=30, n=3, seed=2):
    r = np.random.default_rng(seed)
    return D.Pool(*(r.standard_normal((n, W, 11)) * 0.2 for _ in range(3)))


def test_arms_shapes_and_determinism_on_fabricated_inputs():
    inp = D.fabricated_inputs(30, seed=SEED)
    pool = _pool()
    for c in (D.ar_cells()[4], D.ar_cells()[9], D.ar_cells()[10]):
        r1 = D.ar_record(inp, c, 0, pool, seed=SEED, B=20)
        assert r1 == D.ar_record(inp, c, 0, pool, seed=SEED, B=20)
        assert set(r1) == {"ar", "one_ar"}
    with pytest.raises(ValueError):            # w100 needs >= 100 wells
        D.ar_record(inp, D.ar_cells()[11], 0, pool, seed=SEED, B=20)
    inp120 = D.fabricated_inputs(120, seed=SEED)
    h_zr0, gen = D.fabricated_truths(inp120, 1, seed=SEED)
    ti, k, h = next(gen)
    recs, (dc, xp, xc) = D.phys_records(inp120, h, h_zr0, ti, k, seed=SEED, B=20)
    assert set(recs) == set(D.VARIANTS) and dc.shape == xp.shape == xc.shape == (120, 11)
    assert set(D.sig_record(inp, D.sig_cells()[0], 0, pool, seed=SEED, B=20)) == {"sig",
                                                                                "sig_c"}
    assert 0 <= pool.pick("ar", 11, D.K_AR, 499) < 3


def test_generator_is_fv_stat_v2s(monkeypatch):
    calls = []
    real = D.S.world

    def spy(*a, **k):
        calls.append(a[1:5])
        return real(*a, **k)

    monkeypatch.setattr(D.S, "world", spy)
    inp = D.fabricated_inputs(20, seed=SEED)
    pool = _pool(W=20)
    c = D.ar_cells()[0]
    D.ar_record(inp, c, 3, pool, seed=SEED, B=10)
    D.sig_record(inp, c, 3, pool, seed=SEED, B=10)
    D.pe_truth(inp, D.pe_cells()[1], 3, seed=SEED)
    assert calls == [(c["rho"], c["share"], 72, c["cell"]), (c["rho"], c["share"], 73,
                                                             c["cell"]),
                     (0.80, D.PE_SHARE, 83, 1)]


def _w(lo, hi):
    return {"lambda": (lo + hi) / 2, "lower": lo, "upper": hi, "bad_share": 0.0}


def _fab(ar=(-0.1, 0.2), pe=(-0.1, 0.2), phys=(0.6, 1.4), k=4):
    a = [[{"ar": _w(*ar), "one_ar": _w(*ar)}] * k for _ in D.ar_cells()]
    p = [[{**{s: _w(*pe) for s in D.PE_SHAPES}, **{f"one_{s}": _w(0.1, 0.4)
                                                   for s in D.PE_SHAPES}}] * k
         for _ in D.pe_cells()]
    ph = {c["cell"]: [_w(*phys)] * k for c in D.phys_cells()}
    sig = [[{"sig": _w(0.8, 1.2), "sig_c": _w(-0.1, 0.1)}] * 2 for _ in D.sig_cells()]
    return a, p, ph, sig


def test_g0_on_fabricated_records():
    s = D.summarise(*_fab(), x_partial_rms=np.full(8, 0.2))
    assert s["g0"]["pass"] and s["role_if_on_time"] == "gating"
    assert s["P"]["one_contrast_report_only"]["pe"]["p"] == 1.0
    ph = s["P"]["phys"]
    assert ph["lambda_1.0"]["informative"]["p"] == 1.0
    assert ph["lambda_0.5"]["informative"]["p"] == 1.0
    assert ph["lambda_0.25"]["informative"]["p"] == 0.0
    assert set(s["median_lambda_p_by_truth_report_only"]) == set(D.TRUTHS)
    assert "bootstrap_calibration" in s and len(s["cells_pe"]) == 6
    assert not D.summarise(*_fab(ar=(0.05, 0.2)), x_partial_rms=np.full(8, 0.2))["g0"][
        "conditions"]["a_false_informative"]
    assert not D.summarise(*_fab(pe=(0.05, 0.3)), x_partial_rms=np.full(8, 0.2))["g0"][
        "conditions"]["a2_persistence_error"]
    s = D.summarise(*_fab(phys=(-0.1, 0.2)), x_partial_rms=np.full(8, 0.2))
    assert not s["g0"]["conditions"]["b_power"]
    assert not s["g0"]["conditions"]["c_false_encompass"]
    s = D.summarise(*_fab(ar=(-0.1, 0.4)), x_partial_rms=np.full(8, 0.01))
    assert not s["g0"]["conditions"]["d_null_encompass"]
    assert not s["g0"]["conditions"]["e_contrast"]


def test_role_cap_keyed_to_registration_time():
    assert D.role_cap("2026-11-20T10:00:00.000000Z")[0] == "gating"
    assert D.role_cap("2026-11-25T23:59:59Z")[0] == "gating"
    assert D.role_cap("2026-11-26T00:00:01.000000Z")[0] == "report_only"
    assert D.role_cap(None)[0] == "report_only"
    assert PR.planned_role(dt.datetime(2026, 11, 27, tzinfo=dt.timezone.utc))[
        "if_registered_now"] == "report_only"


def test_smoke_writes_no_rate(tmp_path):
    rep = D.run_smoke(tmp_path / "smoke", k=1, B=10)
    assert rep["status"] == "ran ok"
    got = json.loads((tmp_path / "smoke" / "smoke.json").read_text())
    assert set(got) == {"id", "smoke", "status", "seed", "k", "B", "inputs", "note"}
    with pytest.raises(L.EngineeringFailure):
        D.run_smoke(REPO / "results" / "fe_smoke_never")


def test_decidability_run_refusals_are_logged(tmp_path):
    out = tmp_path / "rep" / "decidability.json"
    pre = tmp_path / "prereg"
    fab = D.fabricated_inputs(seed=1)
    with pytest.raises(L.Refused, match="register it first"):
        D.run(SEED, L.ID, out=str(out), prereg_dir=pre)
    with pytest.raises(L.Refused, match="serves"):
        D.run(SEED, "FE.other", out=str(out), prereg_dir=pre)
    with pytest.raises(L.EngineeringFailure, match="refused"):
        D.run(D.SMOKE_SEED, L.ID, out=str(out), prereg_dir=pre)
    with pytest.raises(L.Refused, match="test_only"):
        D.run(SEED, "TEST.fe", out=str(out), prereg_dir=pre, inputs=fab)
    with pytest.raises(L.Refused, match="real registration id"):
        D.run(SEED, L.ID, out=str(out), prereg_dir=pre, inputs=fab, test_only=True)
    with pytest.raises(L.Refused, match="real registry"):
        D.run(SEED, "TEST.fe", out=str(out), prereg_dir=None, inputs=fab, test_only=True)
    with pytest.raises(L.Refused, match="register it first"):
        D.run(SEED, "TEST.fe", out=str(out), prereg_dir=pre, inputs=fab, test_only=True)
    out.write_text("{}")
    with pytest.raises(L.Refused, match="run once"):
        D.run(SEED, L.ID, out=str(out), prereg_dir=pre)
    rows = [json.loads(x) for x in (out.parent / L.ATTEMPTS_NAME).read_text().splitlines()]
    assert [r["event"] for r in rows].count("start") == 8
    assert all(r["event"] in ("start", "Refused", "EngineeringFailure") for r in rows)


def test_verify_decidability_traceability(tmp_path):
    reg = {"R0_b0": {"path": "p", "npz_sha256": "a" * 64}}
    doc = {"decidability": {"fv2_batches": reg}}
    dec = {"truth_files_used": {"fv2_batches": reg, "zr0": {"sha256": L.ZR0_SHA}}}
    p = tmp_path / "decidability.json"
    p.write_text(json.dumps(dec))
    with pytest.raises(L.EngineeringFailure, match="success row"):
        D.verify_decidability(dec, p, doc, "d" * 64)
    D.log_attempt(tmp_path / L.ATTEMPTS_NAME, {"event": "success", "stage": "decidability",
                                               "report_sha256": D.O.sha256(p),
                                               "declaration_sha256": "d" * 64})
    D.verify_decidability(dec, p, doc, "d" * 64)
    with pytest.raises(L.EngineeringFailure, match="registered FV v2"):
        D.verify_decidability({"truth_files_used": {"fv2_batches": {}, "zr0": {
            "sha256": L.ZR0_SHA}}}, p, doc, "d" * 64)
    with pytest.raises(L.EngineeringFailure, match="fv_zr0"):
        D.verify_decidability({"truth_files_used": {"fv2_batches": reg, "zr0": {
            "sha256": "0"}}}, p, doc, "d" * 64)
    p.write_text(json.dumps({**dec, "x": 1}))
    with pytest.raises(L.EngineeringFailure, match="success row"):
        D.verify_decidability(dec, p, doc, "d" * 64)


# ----------------------------------------------------------------------------- scorer
def _args(tmp_path, **kw):
    a = dict(seed=SEED, prereg_id=L.ID, window_product=str(tmp_path / "wp"),
             e02_score_dir=None, e04_score_dir=None, e02c_report=None, e02d_report=None,
             decidability=str(tmp_path / "dec.json"), out=str(tmp_path / "r" / "report.json"))
    a.update(kw)
    return argparse.Namespace(**a)


def _registered(monkeypatch):
    monkeypatch.setattr(D, "index_record", lambda pid, d=None: {"id": pid, "created": "x"})
    doc = {"frozen_code": {L.SCORER: D.O.sha256(D_DIR / "score_fe.py")}}
    monkeypatch.setattr(D, "load_registered", lambda *a, **k: (doc, "a" * 64))


def test_scorer_refuses_registration_dates_and_twice(tmp_path, monkeypatch):
    with pytest.raises(L.Refused, match="not registered"):
        SF.run_score(_args(tmp_path), prereg_dir=tmp_path / "pre", now=LATE)
    monkeypatch.setattr(D, "index_record", lambda pid, d=None: {"id": pid, "created": "x"})
    with pytest.raises(L.Refused, match="not before 2028-01-15"):
        SF.run_score(_args(tmp_path), prereg_dir=tmp_path / "pre", now=EARLY)
    with pytest.raises(L.Refused, match="no later than 2028-01-31"):
        SF.run_score(_args(tmp_path), prereg_dir=tmp_path / "pre", now=AFTER)
    out = tmp_path / "r" / "report.json"
    out.write_text("{}")
    with pytest.raises(L.Refused, match="scored once"):
        SF.run_score(_args(tmp_path), prereg_dir=tmp_path / "pre", now=LATE)
    out.unlink()
    with pytest.raises(L.Refused, match="not registered or not intact"):
        SF.run_score(_args(tmp_path), prereg_dir=tmp_path / "pre", now=LATE)
    ev = [json.loads(x)["event"] for x in (tmp_path / "r" / L.ATTEMPTS_NAME)
          .read_text().splitlines()]
    assert [e for e in ev if e != "start"] == ["Refused"] * 5 and not out.exists()


def test_scorer_e02c_canonical_and_window_product(tmp_path, monkeypatch):
    _registered(monkeypatch)
    out = tmp_path / "r" / "report.json"
    rogue = tmp_path / "rogue.json"
    rogue.write_text(json.dumps({"prereg_id": L.E02C_ID, "verdict": "not_scorable"}))
    with pytest.raises(L.Refused, match="not the canonical"):
        SF.run_score(_args(tmp_path, e02c_report=str(rogue)), now=LATE)
    canon = tmp_path / "canon" / "report.json"
    canon.parent.mkdir()
    monkeypatch.setattr(L, "E02C_REPORT", str(canon))
    chain = {"e02_step5": {"evaluation.csv_sha256": "0" * 64}}
    canon.write_text(json.dumps({"prereg_id": L.E02C_ID, **chain}))   # no window_product
    with pytest.raises(L.Refused, match="does not exist"):            # S5: typo'd path
        SF.run_score(_args(tmp_path, e02c_report=str(canon)), now=LATE)
    wp = tmp_path / "wp"
    wp.mkdir()
    with pytest.raises(L.Refused, match="readable MANIFEST"):
        SF.run_score(_args(tmp_path, e02c_report=str(canon)), now=LATE)
    (wp / "MANIFEST.json").write_text(json.dumps({"product": "x"}))
    with pytest.raises(L.Refused, match="names no window_product"):
        SF.run_score(_args(tmp_path, e02c_report=str(canon)), now=LATE)
    canon.write_text(json.dumps({"prereg_id": L.E02C_ID, "window_product": str(tmp_path),
                                 **chain}))
    with pytest.raises(L.Refused, match="not the one E02c"):
        SF.run_score(_args(tmp_path, e02c_report=str(canon)), now=LATE)
    assert not out.exists()
    canon.write_text(json.dumps({"prereg_id": L.E02C_ID, "window_product": str(wp),
                                 **chain}))
    copy = tmp_path / "copy.json"
    copy.write_bytes(canon.read_bytes())                              # byte-identical copy
    rep = SF.run_score(_args(tmp_path, e02c_report=str(copy)), now=LATE)
    assert rep["verdict"] == "not_evaluable" and "prospective_scorer" in rep["why"]


def test_scorer_e02c_not_scorable_is_final(tmp_path, monkeypatch):
    _registered(monkeypatch)
    canon = tmp_path / "canon.json"
    monkeypatch.setattr(L, "E02C_REPORT", str(canon))
    canon.write_text(json.dumps({"prereg_id": L.E02C_ID, "verdict": "not_scorable"}))
    rep = SF.run_score(_args(tmp_path, e02c_report=str(canon),
                             window_product=str(tmp_path / "never")), now=LATE)
    assert rep["verdict"] == "not_evaluable" and "E02c not scorable" in rep["why"]
    assert "E02c not scorable" in rep["wording"]["en"]
    with pytest.raises(L.Refused, match="scored once"):
        SF.run_score(_args(tmp_path, e02c_report=str(canon)), now=LATE)
    assert SF.e02c_not_scorable(tmp_path / "missing.json") is None


def test_score_frame_readings():
    rp = SF.score_frame(_frame(truth="phys"), seed=SEED, role="gating")
    assert rp["wells_scored"] == 120 and rp["reading"]["reading"] == "physics_informative"
    assert rp["primary"]["lambda"] == pytest.approx(1.0, abs=0.05)
    assert {"lambda_c_report_only", "one_contrast_report_only", "raw_level_report_only"} <= \
        set(rp)
    rs = SF.score_frame(_frame(truth="stat"), seed=SEED, role="report_only",
                        why_role="G0 not passed")
    assert rs["reading"]["reading"] == "stat_encompasses"
    assert rs["wording"]["en"].startswith("Report-only")
    rn = SF.score_frame(_frame(W=50), seed=SEED, role="gating")
    assert rn["reading"]["reading"] == "not_evaluable" and rn["primary"] is None


def test_frame_from_step5_chain():
    f = _frame(W=5)
    shared = sorted(f["station_id"].unique())
    e02 = f.rename(columns={"phys_m": "predicted_m"})[["station_id", "date", "observed_m",
                                                       "predicted_m"]]
    e04 = f.rename(columns={"stat_m": "predicted_m", "clim_m": "baseline_m"})[
        ["station_id", "date", "observed_m", "predicted_m", "baseline_m"]]
    j, prov = SF.frame_from_step5(e02, e04, shared)
    assert prov["shared_rows"] == len(f) and {"stat_m", "phys_m", "clim_m"} <= set(j.columns)
    with pytest.raises(L.EngineeringFailure, match="rows differ"):
        SF.frame_from_step5(e02, e04.iloc[:-3], shared)
    with pytest.raises(L.EngineeringFailure, match="observed_m differ"):
        SF.frame_from_step5(e02, e04.assign(observed_m=e04["observed_m"] + 1e-6), shared)


def test_role_from(tmp_path):
    sha, early, late = "a" * 64, "2026-11-20T00:00:00Z", "2026-11-28T00:00:00Z"
    assert SF.role_from(None, sha, early)[0] == "report_only"
    ok = {"declaration_sha256": sha, "on_time": True, "role": "gating", "g0": {"pass": True}}
    assert SF.role_from(ok, sha, early) == ("gating", "G0 passed")
    assert SF.role_from(ok, sha, late)[0] == "report_only"
    assert SF.role_from({**ok, "on_time": False}, sha, early)[0] == "report_only"
    with pytest.raises(L.EngineeringFailure):
        SF.role_from(ok, "b" * 64, early)
    p = tmp_path / "decidability.json"
    p.write_text(json.dumps(ok))
    with pytest.raises(L.EngineeringFailure):          # traceability is verified first
        SF.role_from(ok, sha, early, doc={"decidability": {"fv2_batches": {}}}, dec_path=p)


@NEED_RESULTS
def test_order_status_chain_and_e02d_window(tmp_path):
    man = json.loads((REPO / L.E02_DIR / "manifest.json").read_text())
    man4 = json.loads((REPO / L.E04_DIR / "manifest.json").read_text())
    d2, d4 = tmp_path / "e02", tmp_path / "e04"
    for d, mm in ((d2, man), (d4, man4)):
        d.mkdir()
        (d / "evaluation.csv").write_text("station_id\n")
        (d / "report.json").write_text(json.dumps({"registration_manifest": mm}))
    e02d = tmp_path / "e02d.json"
    e02d.write_text(json.dumps({"prereg_id": L.E02D_ID, "verdict": "not_evaluable"}))
    e02c = tmp_path / "e02c.json"
    e02c.write_text(json.dumps({"prereg_id": L.E02C_ID, "e02_step5": {
        "evaluation.csv_sha256": D.O.sha256(d2 / "evaluation.csv")}}))
    st = SF.order_status(tmp_path / "pre", e02c, e02d, d2, d4)
    assert not st["ok"] and "names no window" in st["e02d"]["why"]
    e02d.write_text(json.dumps({"prereg_id": L.E02D_ID, "verdict": "not_evaluable",
                                "window": "A"}))
    assert SF.order_status(tmp_path / "pre", e02c, e02d, d2, d4)["ok"]
    e02c.write_text(json.dumps({"prereg_id": L.E02C_ID,
                                "e02_step5": {"evaluation.csv_sha256": "0" * 64}}))
    st = SF.order_status(tmp_path / "pre", e02c, e02d, d2, d4)
    assert not st["ok"] and not st["e02_chain"]["ok"]


# ----------------------------------------------------------------------------- draft
def test_draft_key_fields_with_injected_ids():
    ids = [f"09{i:06d}" for i in range(147)]
    d = PR.draft_prereg(SEED, ids=ids)
    t = d["tests"][0]
    assert d["id"] == L.ID and d["kind"] == "test" and d["wp"] == "FE"
    assert t["fresh_units"] == [L.UNIT] == d["fresh_units"]
    assert t["external_units"][L.UNIT]["ids"] == ids and t["data_scope"] == L.DATA_SCOPE
    assert d["design"]["seed"] == SEED and d["decidability"]["streams"] == D.STREAMS
    assert set(d["frozen_code"]) == set(D.FROZEN)
    assert "pe" in d["decidability"]["arms"] and "over_persist" not in d["decidability"]["arms"]
    assert "u7" in d["decidability"] and "traceability" in d["decidability"]
    assert L.E02C_REPORT in d["scoring"]["e02c_report"]
    assert "E02c not scorable" in d["scoring"]["e02c_not_scorable"]
    words = {k: v for k, v in d["page_mapping"]["wording"].items() if k != "never"}
    assert "no-information reference" not in json.dumps(words)
    assert "DRAFT" not in json.dumps(d) and "created" not in d
    assert PR.FE_FIXER_LOOK[0] in d["disclosures"]["builder"]


@NEED_RESULTS
def test_draft_validates_on_the_real_tree():
    from harness import prereg3

    d = PR.draft_prereg(SEED)
    assert prereg3.validate_prereg(d) == []
    assert len(d["tests"][0]["external_units"][L.UNIT]["ids"]) == L.N_SHARED
    fv = d["binds"]["fv_stat_v2"]
    assert fv["declaration"]["sha256"] == L.FV_STAT_V2_DECL_SHA and fv["generator_code_unchanged"]
    assert len(d["decidability"]["fv2_batches"]) == 20
    inp = D.load_inputs()
    assert inp.fitmask.shape == (L.N_SHARED, D.O.T_FIT) and (inp.sd > 0).all()
