"""CPU tests of FM.origin_memory.v1 (round-7 rank 2): no forward run, no GPU, nothing
registered, no look logged, nothing written outside pytest's tmp dirs. Every forward output
here is SYNTHETIC: fake npz files with the FS layout (and the forbidden keys hindcast_r2 and
gate present, to prove they are never loaded) whose start differences decay at known rates.
The tmp-registry round trip hashes the gitignored results/ files and skips when they are
absent, so these tests live outside the repository's testpaths:

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        --basetemp=<scratch dir> dataprep_staging/fm_origin_memory/tests
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
sys.path.insert(0, str(X))
sys.path.insert(0, str(REPO / "dataprep_staging"))
import fm_common as C  # noqa: E402
import prereg_fm as P  # noqa: E402
import score_fm as S  # noqa: E402

N_T = 252
needs_results = pytest.mark.skipif(
    not all((REPO / p).exists() for p in P.INPUT_SHA256),
    reason="results/ inputs (FS forwards, page run, FS registration) not present")


# ----------------------------------------------------------------------------- synthetic
def dates(n=N_T, y0=2012) -> np.ndarray:
    return np.array([f"{y0 + i // 12}-{i % 12 + 1:02d}-01T00:00:00" for i in range(n)])


def base_heads(n_scen: int, n_t: int = N_T) -> np.ndarray:
    t = np.arange(n_t, dtype=np.float32)
    base = (10.0 + 0.002 * np.arange(C.N_ACTIVE, dtype=np.float32)[None, :, None]
            + np.array([0.0, -1.0, -2.0, -3.0], dtype=np.float32)[:, None, None]
            + 0.3 * np.sin(2 * np.pi * t / 12.0)[None, None, :])
    out = np.broadcast_to(base, (n_scen, 4, C.N_ACTIVE, n_t)).copy()
    for s in range(1, n_scen):                       # scenario levers from 2026-01
        out[s, :, :, 168:] -= 0.5 * s
    return out


def decay(amp_by_layer, tau, n_t: int = N_T, i0: int = S.I0) -> np.ndarray:
    """(4, 1, n_t) start offset amp * exp(-(L-1)/tau) at lead L >= 1 (zero before)."""
    t = np.arange(n_t, dtype=np.float64)
    lead = t - i0
    shape = np.where(lead >= 1, np.exp(-(lead - 1) / tau), 0.0)
    return (np.asarray(amp_by_layer, dtype=np.float64)[:, None, None] * shape[None, None, :])


def fake_npz(path: Path, names, hm, *, origin: int = S.I0, extra: bool = True) -> Path:
    mask = np.zeros((40, 60), dtype=bool)
    mask.flat[:C.N_ACTIVE] = True
    kw = dict(dates=dates(hm.shape[-1]), origin=origin, heads_mean=hm.astype(np.float32),
              scenario_names=np.array(names), mask=mask)
    if extra:      # keys a scorer must never load
        kw.update(hindcast_r2=np.array([0.9, 0.8]), gate=json.dumps({"r2_kfold": 0.5}),
                  heads_std=np.zeros(1))
    np.savez(path, **kw)
    return path


def world(t: Path, s1=((0.77, 0.77, 0.77, 0.77), 3.5), s2=((0.77, 0.77, 2.0, 2.0), 5.0),
          s2_const: float | None = None, s2_bump: tuple[int, float] | None = None) -> dict:
    hm0 = base_heads(5)
    f = {"page": fake_npz(t / "page.npz", S.REF_SCEN, hm0[:3]),
         "S0": fake_npz(t / "S0.npz", S.FS_SCEN, hm0)}
    h1 = hm0 + decay(*s1)[None]
    f["S1"] = fake_npz(t / "S1.npz", S.FS_SCEN, h1)
    h2 = hm0 + (decay(*s2)[None] if s2_const is None
                else np.where(np.arange(N_T) > S.I0, s2_const, 0.0)[None, None, None, :])
    if s2_bump is not None:                          # a re-growth at one lead, every screen
        lead, amp = s2_bump
        h2[..., S.I0 + lead] += amp
    f["S2"] = fake_npz(t / "S2.npz", S.FS_SCEN, h2)
    return f


def expected_lstar(amp: float, tau: float, floor: float = C.FLOOR_M) -> int:
    """The first lead L with amp * exp(-(L-1)/tau) < floor."""
    return int(math.floor(tau * math.log(amp / floor))) + 2


@pytest.fixture
def registered(monkeypatch):
    """Pretend the declaration is registered with the current frozen code; accept synthetic
    files in place of the sha-bound FS files."""
    monkeypatch.setattr(C, "load_registered", lambda pid, *a, **k: (
        {"frozen_code": {p: C.sha256(REPO / p) for p in S.FROZEN}}, "f" * 64))
    monkeypatch.setattr(C, "check_sha", lambda path, want, what: C.sha256(path))


def run_main(f: dict, out: Path | None) -> int:
    a = [f"--s0={f['S0']}", f"--s1={f['S1']}", f"--s2={f['S2']}", f"--ref={f['page']}"]
    return S.main(a + ([f"--out={out}"] if out else []))


# ----------------------------------------------------------------------------- static
def test_screens_frozen_and_equal_fs():
    assert C._screens_sha() == C.SCREENS_SHA256
    assert len(C.CELLS) == 158 and "".join(C.COUNTY).count("Y") == 84
    sys.path.insert(0, str(REPO / "dataprep_staging/fs_start"))
    import score_fs as FS

    assert (C.CELLS, C.LAYERS, C.COUNTY, C.SCREENS_SHA256) == (
        FS.CELLS, FS.LAYERS, FS.COUNTY, FS.SCREENS_SHA256)
    assert C.PAGE_MEMBERS == FS.PAGE_MEMBERS
    assert P.scorer_problems() == []


def test_no_home_paths_in_staged_files():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")     # a home directory with a user name
    for p in sorted(X.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sh", ".json", ".md"):
            assert not pat.search(p.read_text()), p


def test_declaration_validates_and_binds():
    from harness import prereg3

    d = P.doc()
    assert prereg3.validate_prereg(d) == []
    assert d["kind"] == "declaration" and d["id"] == "FM.origin_memory.v1"
    assert set(d["frozen_code"]) == set(S.FROZEN)
    declared = {r["path"]: r["sha256"] for r in d["inputs"] if isinstance(r, dict)}
    assert {p: declared[p] for p in S.FROZEN} == d["frozen_code"]
    assert all(declared[p] == h for p, h in S.FILES.values())
    assert declared[S.REF[0]] == S.REF[1]
    th = d["thresholds"]
    assert "0.05 m" in th["lstar"] and "every L' in L..48" in th["lstar"]
    assert "L* <= 47" in th["g_def"] and "0.0 m" in th["g_eng"]
    assert th["wordings"] == S.WORDINGS
    assert "hindcast_r2" in d["recipe"]["key_whitelist"]
    assert "about +0.77 m" in d["disclosed"]["fs_endpoints_seen"]
    assert "moves no claim status" in d["title"]
    assert "page-line ruling" in " ".join(
        d["evidence_and_ledger_mapping"]["within_first_year_or_beyond_first_year"])


def test_the_fs_files_are_the_ones_fs_scored():
    """The sha256 bound here are those of FS's scored report (copied, checked when present)."""
    rep = REPO / P.FS_REPORT[0]
    if not rep.exists():
        pytest.skip("FS report not present")
    files = json.loads(rep.read_text())["files"]
    for arm, (_, h) in S.FILES.items():
        assert files[arm]["sha256"] == h


def test_prereg_cli_validate_is_a_dry_run(capsys):
    reg = REPO / "results/dataprep/preregistration"
    before = sorted(reg.glob("FM.*")) if reg.exists() else []
    P.main(["--validate", "--quiet"])
    out = capsys.readouterr().out
    assert "validate_prereg: valid" in out and "scorer/binding checks: ok" in out
    assert (sorted(reg.glob("FM.*")) if reg.exists() else []) == before


@needs_results
def test_declaration_round_trips_in_tmp_registry(tmp_path):
    from harness import prereg3

    path, h = prereg3.write_prereg(P.doc(), prereg_dir=tmp_path / "prereg",
                                   rows_dir=tmp_path / "rows", registries=[])
    got, sha = C.load_registered(S.PREREG_ID, prereg_dir=tmp_path / "prereg",
                                 rows_dir=tmp_path / "rows")
    assert sha == h and got["id"] == S.PREREG_ID
    assert C.check_frozen(got, S.FROZEN) == got["frozen_code"]
    with pytest.raises(C.EngineeringError, match="frozen code differs"):
        C.check_frozen({**got, "frozen_code": {p: "0" * 64 for p in S.FROZEN}}, S.FROZEN)


# ----------------------------------------------------------------------------- helpers
def test_whitelist_refuses_forbidden_keys(tmp_path):
    p = fake_npz(tmp_path / "x.npz", S.FS_SCEN, base_heads(5))
    with C.WhitelistedNpz(p, C.FM_KEYS) as z:
        assert "hindcast_r2" in z.names and "gate" in z.names     # listing is metadata
        for k in ("hindcast_r2", "gate", "heads_std"):
            with pytest.raises(C.EngineeringError, match="not whitelisted"):
                z[k]
        z["origin"]
        assert z.keys_read == ["origin"]
    with pytest.raises(C.EngineeringError, match="never contain"):
        C.WhitelistedNpz(p, (*C.FM_KEYS, "hindcast_r2"))


def test_stays_below_and_first_exceeds():
    f = C.FLOOR_M
    assert C.stays_below([np.array([1.0, 0.04, 0.06, 0.01, 0.0])]) == 4
    assert C.stays_below([np.array([0.01, 0.02])]) == 1
    assert C.stays_below([np.array([1.0, f])]) is None             # strict
    assert C.stays_below([np.array([1, 0.01, 0.01]), np.array([1, 1, 0.01])]) == 3
    assert C.first_exceeds(np.array([0, 1, 3]), np.array([2, 2, 2])) == 3
    assert C.first_exceeds(np.array([0, 2]), np.array([2, 2])) is None
    assert C.stays_above(np.array([3, 0, 3, 3]), np.array([2, 2, 2, 2])) == 3
    assert C.lead_month("2022-12", 1) == "2023-01" and C.lead_month("2022-12", 48) == "2026-12"
    assert C.lead_month("2019-12", 36) == "2022-12"
    assert C.month_index("2012-01", "2022-12") == 131 == S.I0


def test_efold_of_an_exponential():
    L = np.arange(48)
    r = C.efold(0.77 * np.exp(-L / 3.5))
    assert r["tau_1e_months"] == pytest.approx(3.5, abs=1e-9)
    assert r["tau_fit_months"] == pytest.approx(3.5, rel=1e-9)
    assert C.efold(np.zeros(5))["tau_1e_months"] is None


def test_wording_cases():
    assert S.case_of(11) == "within_first_year" and S.case_of(12) == "within_first_year"
    assert S.case_of(13) == "beyond_first_year" and S.case_of(47) == "beyond_first_year"
    assert S.case_of(48) == S.case_of(None) == "longer_than_record"
    w = S.wording(11, "window_beyond_memory")
    assert "about 11 months (from 2023-11)" in w["C30_heads"]
    assert w["E02_E21_clause"].startswith("in this model the 2026-11..2027-10 window lies "
                                          "beyond the start memory (about 11 months")
    assert "in part" in S.wording(20, "window_partly_within_memory")["E02_E21_clause"]
    w0 = S.wording(None, "window_within_memory")
    assert w0["C30_heads"] is None and "lies within" in w0["E02_E21_clause"]
    for bad in ((None, "window_beyond_memory"), (11, "window_within_memory")):
        with pytest.raises(ValueError):
            S.wording(*bad)
    assert S.window_case(None, None) == "window_within_memory"
    assert S.window_case(20, 20) == "window_beyond_memory"
    assert S.window_case(47, 47) == "window_beyond_memory"
    assert S.window_case(20, 53) == S.window_case(20, None) == "window_partly_within_memory"
    for case in S.WORDINGS.values():
        assert "this model" in case["E23"]
    assert all(c.startswith("in this model") for c in S.E02_E21_CLAUSES.values())
    assert (S.WINDOW_E02_FIRST_LEAD, S.WINDOW_E02_LAST_LEAD) == (47, 58)
    assert C.lead_month("2022-12", 47) == "2026-11" and C.lead_month("2022-12", 58) == "2027-10"


# ----------------------------------------------------------------------------- scoring
def test_synthetic_beyond_first_year(tmp_path, registered, monkeypatch):
    """S1 decays fast everywhere (L 11); S2 decays slower in layers 3-4 (2.0 m, tau 5):
    p90 over screens is set by the slow layers only if they hold > 10 % of the screens."""
    f = world(tmp_path)
    seen = []
    orig = np.lib.npyio.NpzFile.__getitem__

    def spy(self, key):
        seen.append(key)
        return orig(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", spy)
    out = tmp_path / "rep.json"
    assert run_main(f, out) == 0
    assert set(seen) <= set(C.FM_KEYS) and not set(seen) & set(C.NEVER_KEYS)
    r = json.loads(out.read_text())
    lay = np.asarray(C.LAYERS)
    share_deep = float(np.mean(lay >= 2))
    pr = r["primary"]
    # S2's p90 is the deep layers' curve when more than 10 % of the screens are deep
    assert share_deep > 0.10
    want = max(expected_lstar(0.77, 3.5), expected_lstar(2.0, 5.0))
    assert pr["lstar_months"] == want == 20
    assert pr["lstar_per_arm"] == {"S1": expected_lstar(0.77, 3.5), "S2": 20}
    assert pr["wording"]["case"] == "beyond_first_year" and pr["g_def_pass"]
    assert pr["wording"]["E02_E21_clause_case"] == "window_beyond_memory"
    assert pr["window_e02_beyond_memory"] and pr["lstar_through_2027_10"] == 20
    assert pr["lstar_month"] == "2024-08"
    assert pr["n2_unit_efold"]["S1"]["tau_1e_months"] == pytest.approx(3.5, abs=1e-3)
    assert r["g_eng"]["max_abs_diff_m"] == 0.0
    rep = r["reported_not_gating"]
    assert all(v["max_abs_dev_from_baseline_curve_m"] < 1e-5
               for v in rep["scenarios_check"].values())
    assert rep["continuation_to_2032"]["lstar_through_2032"] == 20
    assert rep["lstar_by_layer"]["L1"] == expected_lstar(0.77, 5.0)
    assert set(r["files"]["S1"]["keys_read"]) <= set(C.FM_KEYS)


def test_synthetic_within_first_year(tmp_path, registered):
    f = world(tmp_path, s2=((0.77,) * 4, 3.5))
    out = tmp_path / "rep.json"
    assert run_main(f, out) == 0
    pr = json.loads(out.read_text())["primary"]
    assert pr["lstar_months"] == 11 and pr["wording"]["case"] == "within_first_year"
    assert "about 11 months" in pr["wording"]["E23"]


def test_synthetic_longer_than_record(tmp_path, registered):
    f = world(tmp_path, s2_const=0.1)
    out = tmp_path / "rep.json"
    assert run_main(f, out) == 0
    pr = json.loads(out.read_text())["primary"]
    assert pr["lstar_months"] is None and not pr["g_def_pass"]
    assert pr["wording"]["case"] == "longer_than_record"
    assert pr["wording"]["C30_heads"] is None
    assert pr["wording"]["E02_E21_clause_case"] == "window_within_memory"
    assert not pr["window_e02_beyond_memory"]


@pytest.mark.parametrize("lead, case", [(52, "window_partly_within_memory"),
                                        (58, "window_partly_within_memory"),
                                        (59, "window_beyond_memory")])
def test_window_clause_reads_through_2027_10(tmp_path, registered, lead, case):
    """L* is decided through 2026-12 only; a difference that re-grows inside the 2026-11..
    2027-10 window (leads 47..58) turns the E02/E21 clause to 'partly within' without
    changing L*, G-def or the E23/C30 wordings; one after 2027-10 does not."""
    f = world(tmp_path, s2_bump=(lead, 0.2))
    out = tmp_path / "rep.json"
    assert run_main(f, out) == 0
    pr = json.loads(out.read_text())["primary"]
    assert pr["lstar_months"] == 20 and pr["g_def_pass"]
    assert pr["wording"]["case"] == "beyond_first_year"
    assert pr["wording"]["E02_E21_clause_case"] == case
    assert pr["window_e02_beyond_memory"] == (case == "window_beyond_memory")


def test_g_eng_failure_reads_no_curve(tmp_path, registered, capsys):
    f = world(tmp_path)
    hm = base_heads(3)
    hm[0, C.LAYERS[0], C.CELLS[0], 10] += 1e-3        # a screen, 2012-11
    f["page"] = fake_npz(tmp_path / "page2.npz", S.REF_SCEN, hm)
    out = tmp_path / "rep.json"
    assert run_main(f, out) == 3
    assert not out.exists() and "no curve computed" in capsys.readouterr().err


def test_engineering_checks(tmp_path, registered):
    f = world(tmp_path)
    bad = base_heads(5)
    f2 = {**f, "S1": fake_npz(tmp_path / "S1b.npz", S.FS_SCEN, bad, origin=95)}
    assert run_main(f2, tmp_path / "a.json") == 2
    f3 = {**f, "S2": fake_npz(tmp_path / "S2b.npz", S.FS_SCEN[:4], bad[:4])}
    assert run_main(f3, tmp_path / "b.json") == 2
    nan = base_heads(5)
    nan[0, 1, 2018, 140] = np.nan
    f4 = {**f, "S1": fake_npz(tmp_path / "S1n.npz", S.FS_SCEN, nan)}
    assert run_main(f4, tmp_path / "c.json") == 2


def test_scored_once_and_needs_out(tmp_path, registered):
    f = world(tmp_path)
    out = tmp_path / "rep.json"
    assert run_main(f, out) == 0
    assert run_main(f, out) == 1
    assert run_main(f, None) == 1


def test_unregistered_or_drifted_is_engineering(tmp_path, monkeypatch):
    f = world(tmp_path)
    assert run_main(f, tmp_path / "r.json") == 1            # no registration in this tree
    monkeypatch.setattr(C, "load_registered", lambda pid, *a, **k: (
        {"frozen_code": {p: "0" * 64 for p in S.FROZEN}}, "f" * 64))
    assert run_main(f, tmp_path / "r.json") == 1
    monkeypatch.setattr(C, "load_registered", lambda pid, *a, **k: (
        {"frozen_code": {p: C.sha256(REPO / p) for p in S.FROZEN}}, "f" * 64))
    assert run_main(f, tmp_path / "r.json") == 1            # synthetic files: sha differs
    assert not (tmp_path / "r.json").exists()
