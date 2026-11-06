"""PB.decomposed_band.v1 (round-8 rank 16, PB part; ruling R8.13) on SYNTHETIC inputs only
(made-up wells, members, half-widths and family departures; no real value):

- the identity check of E02d's frozen artefact (1e-9 m): passes on a consistent artefact and
  refuses every perturbation (sigma, half-width, drift, ensemble, centre, sd, lead, a row);
- the zone ribbon: median per zone and month, the 5-well zone minimum, labels by window, no
  station id in the output;
- the shared range (hull of 0, the member part and the families) and the leave-one-family-out
  coverage, checked by hand on a small case;
- the U7 guard: no coverage of the relative recipe in any month (2023-26 included), no
  coverage of any kind before 2026-12, no observation product among the inputs; the guard is
  not in hydrophysics/twin/coverage.py, bands.py or harness/coverage3.py;
- the declaration draft validates with prereg3.validate_prereg, carries the design, the R8.13
  labels, the dropped leave-one-forcing-year-out, and the FB / EXS status; nothing is written;
- the builder refuses unless the declaration is registered, its design and frozen code match,
  bound inputs match, and the output does not exist.

Checks that need the gitignored results/ tree skip when it is absent (CI).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]
PB_DIR = REPO / "dataprep_staging" / "pb_band"
for _p in (str(PB_DIR), str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_pb as B  # noqa: E402
import e02dio as io  # noqa: E402
import e02dlib as L  # noqa: E402
import pb_const as K  # noqa: E402
import prereg_pb as P  # noqa: E402
from harness import prereg, prereg3  # noqa: E402

HAVE_RESULTS = (REPO / "results" / "dataprep" / "preregistration" / "INDEX.jsonl").is_file()
NEED_RESULTS = pytest.mark.skipif(not HAVE_RESULTS, reason="results/ (gitignored) is absent")
ZONES = {"Changhua": (("Changhua:z1", 40), ("Changhua:z2", 34)),
         "Yunlin": (("Yunlin:z1", 44), ("Yunlin:z2", 40))}
DECISIONS = {"Changhua": {"grouping": "zone", "include_drift": True, "tau2_per_year2": 0.02},
             "Yunlin": {"grouping": "zone", "include_drift": False, "tau2_per_year2": 0.01}}


# ----------------------------------------------------------------------------- synthetic
def _synthetic(seed=7):
    """A consistent synthetic world: wells table, 36 members, artefacts A/B/C built with
    E02d's own functions, centres and family departures (all made up)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c, code in (("Changhua", "07"), ("Yunlin", "09")):
        for z, n in ZONES[c]:
            for _ in range(n):
                rows.append({"station_id": f"{code}99{len(rows):04d}", "county": c,
                             "group": z, "sd_fit_rel_m": float(rng.uniform(0.3, 1.2)),
                             "t_centre_year": float(rng.uniform(2015, 2019))})
    wells = pd.DataFrame(rows)
    ids = wells["station_id"].tolist()
    dates = pd.date_range("2012-01-01", "2029-10-01", freq="MS")
    members = {"members": rng.normal(0.0, 0.5, (K.N_MEMBERS, len(ids), len(dates))),
               "station_id": np.array(ids), "dates": np.array([str(d) for d in dates])}
    groups = wells["group"].to_numpy()
    tau2 = np.array([DECISIONS[c]["tau2_per_year2"] if DECISIONS[c]["include_drift"] else 0.0
                     for c in wells["county"]])
    anchor = L.lead_anchor("origin", wells["t_centre_year"].to_numpy(float))
    mkeys = [B.month_key(d) for d in dates]
    arts, centres = {}, {}
    for win, (first, last) in K.WINDOWS.items():
        months = B.month_list(first, last)
        ti = [mkeys.index(m) for m in months]
        ens = L.sigma_ens_rel(members["members"][:, :, ti], groups)
        dt_y = L.year_frac(pd.DatetimeIndex([f"{m}-01" for m in months]))[None, :] - \
            anchor[:, None]
        drift = np.sqrt(tau2[:, None] * dt_y ** 2)
        sig = L.sigma_idio(wells["sd_fit_rel_m"].to_numpy(float), ens, tau2, dt_y)
        pred = rng.normal(0.0, 2.0, sig.shape)
        lead = B.lead_months(months)
        art, cen = [], []
        for i, s in enumerate(ids):
            for j, m in enumerate(months):
                art.append({"station_id": s, "county": wells.at[i, "county"], "layer": 2,
                            "group": groups[i], "date": f"{m}-01", "lead_months": int(lead[j]),
                            "predicted_m": pred[i, j],
                            "sd_fit_rel_m": wells.at[i, "sd_fit_rel_m"],
                            "sigma_ens_rel_m": ens[i, j], "drift_m": drift[i, j],
                            "sigma_idio_m": sig[i, j], "half_width_m": K.Z90 * sig[i, j]})
                cen.append({"station_id": s, "date": f"{m}-01", "predicted_m": pred[i, j]})
        arts[win] = pd.DataFrame(art, columns=list(B.ART_COLS))
        centres[win] = pd.DataFrame(cen)
    mu = {c: {k: rng.normal(0.3 * j, 0.4, ((wells["county"] == c).sum(), 11))
              for j, k in enumerate(K.FAMILIES)} for c in K.COUNTIES}
    return wells, members, arts, centres, mu


@pytest.fixture(scope="module")
def world():
    return _synthetic()


def _ident(world, arts=None, centres=None, wells=None, members=None):
    w, m, a, c, _ = world
    return B.identity_check(arts or a, w if wells is None else wells, DECISIONS, "origin",
                            members or m, centres or c)


# ----------------------------------------------------------------------------- identity
def test_identity_passes_on_a_consistent_artefact(world):
    r = _ident(world)
    assert r["ok"] and r["tol_m"] == 1e-9
    for win in K.WINDOWS:
        d = r["windows"][win]
        assert d["rows"] == K.N_WELLS * K.WINDOW_MONTHS_N[win]
        assert all(v <= 1e-9 for k, v in d.items() if k != "rows")


@pytest.mark.parametrize("col", ["sigma_idio_m", "half_width_m", "drift_m", "sigma_ens_rel_m",
                                 "predicted_m", "sd_fit_rel_m"])
def test_identity_refuses_a_perturbed_column(world, col):
    arts = {k: v.copy() for k, v in world[2].items()}
    arts["B"].loc[5, col] = arts["B"].loc[5, col] + 1e-8
    with pytest.raises(io.EngineeringFailure, match="identity check failed"):
        _ident(world, arts=arts)


def test_identity_refuses_a_shifted_centre_a_bad_lead_and_a_missing_row(world):
    cen = {k: v.copy() for k, v in world[3].items()}
    cen["A"].loc[0, "predicted_m"] += 1e-6
    with pytest.raises(io.EngineeringFailure, match="predicted_m"):
        _ident(world, centres=cen)
    arts = {k: v.copy() for k, v in world[2].items()}
    arts["C"].loc[3, "lead_months"] += 1
    with pytest.raises(io.EngineeringFailure, match="lead_months"):
        _ident(world, arts=arts)
    arts = {k: v.copy() for k, v in world[2].items()}
    arts["A"] = arts["A"].drop(index=7).reset_index(drop=True)
    with pytest.raises(io.EngineeringFailure, match="wells x"):
        _ident(world, arts=arts)


def _consistent(arts, win, row, col, delta):
    """Perturb ``col`` at one row and recompute sigma_idio / half_width consistently, so only
    the recomputation of ``col`` itself can catch it."""
    a = {k: v.copy() for k, v in arts.items()}
    f = a[win]
    f.loc[row, col] += delta
    f.loc[row, "sigma_idio_m"] = float(np.sqrt(f.loc[row, "sd_fit_rel_m"] ** 2
                                               + f.loc[row, "sigma_ens_rel_m"] ** 2
                                               + f.loc[row, "drift_m"] ** 2))
    f.loc[row, "half_width_m"] = K.Z90 * f.loc[row, "sigma_idio_m"]
    return a


@pytest.mark.parametrize("col", ["drift_m", "sd_fit_rel_m", "sigma_ens_rel_m"])
def test_identity_recomputes_each_term(world, col):
    arts = _consistent(world[2], "A", 5, col, 1e-6)
    with pytest.raises(io.EngineeringFailure, match=col):
        _ident(world, arts=arts)


def test_identity_recomputes_drift_where_the_county_has_no_drift(world):
    wells = world[0]
    row = int(np.flatnonzero(world[2]["C"]["county"].to_numpy() == "Yunlin")[0])
    assert not DECISIONS["Yunlin"]["include_drift"] and wells is not None
    arts = _consistent(world[2], "C", row, "drift_m", 0.05)
    with pytest.raises(io.EngineeringFailure, match="drift_m"):
        _ident(world, arts=arts)


def test_identity_refuses_a_group_change_and_a_member_change(world):
    arts = {k: v.copy() for k, v in world[2].items()}
    arts["A"].loc[0, "group"] = "Changhua:z2"
    with pytest.raises(io.EngineeringFailure, match="group"):
        _ident(world, arts=arts)
    m = dict(world[1])
    m["members"] = m["members"].copy()
    m["members"][0, 0, -1] += 0.1
    with pytest.raises(io.EngineeringFailure, match="sigma_ens_rel_m"):
        _ident(world, members=m)


# ----------------------------------------------------------------------------- ribbon
BIG_HW = (1.0, 1.0, 1.0, 1.0, 1.0, 10.0)      # asymmetric: median 1.0, mean 2.5


def _tiny_art(n_small=4, n_big=6):
    rows = []
    for z, n, base in (("Yunlin:big", n_big, None), ("Yunlin:small", n_small, 9.0)):
        for i in range(n):
            for m in B.month_list(*K.WINDOWS["A"]):
                hw = BIG_HW[i] if base is None else base + 0.1 * i
                rows.append({"station_id": f"09{z[-3:]}{i}", "county": "Yunlin", "group": z,
                             "date": f"{m}-01", "half_width_m": hw})
    a = pd.DataFrame(rows)
    return {"A": a, "B": a.iloc[:0].copy(), "C": a.iloc[:0].copy()}


def test_ribbon_median_and_zone_minimum():
    r = B.ribbon(_tiny_art(n_small=4, n_big=6))["A"]
    assert [z["zone"] for z in r["zones"]] == ["Yunlin:big"]
    z = r["zones"][0]
    assert z["n_wells"] == 6 and z["county"] == "Yunlin"
    assert z["half_width_m"] == [1.0] * 11                     # the median, not the mean 2.5
    assert r["not_drawn"] == {"zones": 1, "wells": 4}
    assert r["label"]["en"] == "decided by a registered test, E02d, 2028" and r["drawn"]
    assert r["decided_by"]["registration"] == K.E02D_ID
    r5 = B.ribbon(_tiny_art(n_small=5, n_big=6))["A"]
    assert [z["zone"] for z in r5["zones"]] == ["Yunlin:big", "Yunlin:small"]


def test_ribbon_month_with_fewer_than_5_rows_is_null():
    arts = _tiny_art(n_small=4, n_big=6)
    a = arts["A"]
    first = B.month_list(*K.WINDOWS["A"])[0]
    drop = a[(a["group"] == "Yunlin:big") & (a["date"] == f"{first}-01")].index[:2]
    a.loc[drop, "half_width_m"] = np.nan            # 4 finite rows left in the first month
    z = B.ribbon(arts)["A"]["zones"][0]
    assert z["n_wells"] == 6 and z["half_width_m"][0] is None
    assert z["half_width_m"][1:] == [1.0] * 10


def test_ribbon_draws_window_a_only_and_carries_no_ids(world):
    r = B.ribbon(world[2])
    assert r["A"]["label"] == K.LABELS["ribbon_A"] and r["A"]["drawn"] is True
    assert r["B"]["drawn"] is r["C"]["drawn"] is False
    assert r["B"]["label"] is None and r["C"]["label"] is None
    assert K.DRAWN_RIBBON_WINDOWS == ("A",)
    assert r["B"]["decided_by"]["test_id"] == "E02yP.rel_2027-11"
    assert r["C"]["decided_by"]["not_before"] == "2030-01-15"
    assert all(len(z["half_width_m"]) == K.WINDOW_MONTHS_N[w] for w in r for z in r[w]["zones"])
    B.no_station_ids(r, world[0]["station_id"])
    assert "station_id" not in json.dumps(r)


# ----------------------------------------------------------------------------- shared + LOFO
def test_hull_includes_zero_members_and_families():
    md = np.array([[0.1, -0.2], [0.3, 0.0]])
    fam = {"R0": np.array([1.0, -1.0]), "Q0": np.array([0.5, 0.5]),
           "R1c": np.array([-0.5, 2.0]), "X5R1": np.array([0.2, 0.1])}
    lo, hi = B.hull(md, fam)
    assert lo.tolist() == [-0.5, -1.0] and hi.tolist() == [1.0, 2.0]
    lo, hi = B.hull(md, fam, exclude=("R0", "R1c"))
    assert lo.tolist() == [0.0, -0.2] and hi.tolist() == [0.5, 0.5]
    lo, hi = B.hull(np.full((2, 2), 5.0), {k: np.full(2, 5.0) for k in K.FAMILIES})
    assert lo.tolist() == [0.0, 0.0]


def test_member_part_is_centred_county_median_by_hand():
    # 2 members x 3 wells (2 Yunlin, 1 Changhua) x 1 month; member mean differs per well
    X = np.array([[[1.0], [5.0], [0.0]],
                  [[3.0], [5.0], [2.0]]])
    out = B.member_county_dev(X, ["Yunlin", "Yunlin", "Changhua"], [0])
    # centred: a = X - mean_m X -> wells (-1, 1), (0, 0), (-1, 1)
    assert out["Changhua"][:, 0].tolist() == [-1.0, 1.0]
    assert out["Yunlin"][:, 0].tolist() == [-0.5, 0.5]      # median of 2 wells
    X3 = np.array([[[0.0], [0.0], [9.0], [0.0]], [[2.0], [2.0], [-9.0], [0.0]]])
    o3 = B.member_county_dev(X3, ["Yunlin", "Yunlin", "Yunlin", "Changhua"], [0])
    assert o3["Yunlin"][:, 0].tolist() == [-1.0, 1.0]       # median, not mean (+1.67/-1.67)


def test_family_part_is_county_median_by_hand():
    # 3 wells x 2 months per county; one outlying well: median 0.1 / -0.2, mean 3.4 / 2.6
    w = np.array([[0.0, -0.2], [0.1, -0.3], [10.0, 8.3]])
    mu = {c: {k: w + j for j, k in enumerate(K.FAMILIES)} for c in K.COUNTIES}
    out = B.family_county_dep(mu)
    assert np.allclose(out["Yunlin"]["R0"], [0.1, -0.2])
    assert np.allclose(out["Changhua"]["X5R1"], [3.1, 2.8])


def test_shared_range_by_hand():
    T = 2
    md = {c: np.array([[-0.3, -0.1], [0.2, 0.4]]) for c in K.COUNTIES}
    fam = {c: {"R0": np.array([0.5, -0.6]), "Q0": np.zeros(T), "R1c": np.zeros(T),
               "X5R1": np.zeros(T)} for c in K.COUNTIES}
    r = B.shared_range(md, fam, ["2026-12", "2027-01"])["Yunlin"]
    assert r["lo_m"] == [-0.3, -0.6] and r["hi_m"] == [0.5, 0.4]
    assert r["member_lo_m"] == [-0.3, -0.1] and r["member_hi_m"] == [0.2, 0.4]


def test_lofo_by_hand():
    months = B.month_list(*K.WINDOWS["A"])
    T = len(months)
    md = np.vstack([np.full(T, -0.1), np.full(T, 0.1)])
    fam = {"R0": np.full(T, 0.05), "Q0": np.full(T, 0.08), "R1c": np.full(T, 1.0),
           "X5R1": np.full(T, 0.9)}
    fam["X5R1"][:3] = 1.2                    # X5R1 above R1c in the first 3 months
    r = B.lofo({c: md for c in K.COUNTIES}, {c: fam for c in K.COUNTIES}, months)
    c = r["counties"]["Yunlin"]
    assert c["per_family"]["R0"]["inside"] == T and c["per_family"]["Q0"]["inside"] == T
    assert c["per_family"]["R1c"]["inside"] == 3           # inside only while X5R1 is higher
    assert c["per_family"]["X5R1"]["inside"] == T - 3       # above R1c in the first 3 months
    assert c["pooled"] == {"inside": 3 * T, "n": 4 * T, "coverage": 0.75}
    assert c["min_family"] == round(3 / T, 4)
    g = c["fb_groups_report_only"]
    assert g["R1cX5R1"]["inside"] == 0 and g["R0Q0"]["inside"] == 2 * T
    assert r["months"] == months


def test_assemble_on_synthetic_world(world):
    wells, members, arts, centres, mu = world
    out = B.assemble(arts, wells, DECISIONS, "origin", members, centres, mu)
    assert out["identity"]["ok"]
    assert out["shared"]["label"]["en"] == "cannot be tested before 2030 by design"
    for c in K.COUNTIES:
        s = out["shared"]["counties"][c]
        assert len(s["lo_m"]) == 11 and all(lo <= 0 <= hi for lo, hi in
                                            zip(s["lo_m"], s["hi_m"], strict=True))
        assert set(s["family_m"]) == set(K.FAMILIES)
        assert out["lofo"]["counties"][c]["pooled"]["n"] == 44
    assert out["lofo"]["months"][0] == "2026-12" and out["lofo"]["months"][-1] == "2027-10"
    assert out["u7"]["relative_coverage_computed"] is False
    assert "leave_one_forcing_year_out" in out["dropped"]
    assert out["wells_by_county"] == {"Yunlin": 84, "Changhua": 74}
    text = json.dumps(out)
    assert not any(s in text for s in wells["station_id"])


def test_assemble_refuses_a_station_id_in_the_output(world):
    wells, members, arts, centres, mu = world
    sid = wells.loc[0, "station_id"]
    zone = wells.loc[0, "group"]
    w2 = wells.copy()
    w2.loc[w2["group"] == zone, "group"] = sid               # a zone named after a well
    a2 = {k: v.assign(group=v["group"].where(v["group"] != zone, sid)) for k, v in arts.items()}
    with pytest.raises(io.EngineeringFailure, match="station ids"):
        B.assemble(a2, w2, DECISIONS, "origin", members, centres, mu)


# ----------------------------------------------------------------------------- summary checks
E02D_DOC = {"binds": {"dr": {"decisions": DECISIONS}}}


def test_summary_checks():
    dec, lead = B.summary_checks({"decisions": DECISIONS, "lead_definition": "origin"},
                                 E02D_DOC, {"lead_definition": "origin"})
    assert dec == DECISIONS and lead == "origin"
    bad = json.loads(json.dumps(DECISIONS))
    bad["Yunlin"]["tau2_per_year2"] = 0.011
    with pytest.raises(io.EngineeringFailure, match="decisions"):
        B.summary_checks({"decisions": bad, "lead_definition": "origin"}, E02D_DOC,
                         {"lead_definition": "origin"})
    for summ_lead, dr_lead in (("datum_centre", "origin"), (None, "origin"),
                               ("origin", None), ("origin", "datum_centre")):
        summ = {"decisions": DECISIONS}
        if summ_lead is not None:
            summ["lead_definition"] = summ_lead
        with pytest.raises(io.EngineeringFailure, match="lead_definition"):
            B.summary_checks(summ, E02D_DOC, {} if dr_lead is None else
                             {"lead_definition": dr_lead})


# ----------------------------------------------------------------------------- U7
@pytest.mark.parametrize("months", [["2023-01"], ["2024-06", "2025-01"], ["2026-11"],
                                    B.month_list("2026-12", "2027-10")])
def test_u7_refuses_any_relative_coverage(months):
    with pytest.raises(B.U7Refused):
        B.u7_guard(months, "relative")


def test_u7_relative_message_names_the_sealed_span():
    with pytest.raises(B.U7Refused, match="no 2023-26 coverage of the relative recipe"):
        B.u7_guard(["2025-03-01"], "relative")


@pytest.mark.parametrize("months", [["2026-11"], ["2023-01", "2026-12"], ["2022-12"],
                                    B.month_list("2026-06", "2027-10")])
def test_u7_refuses_shared_coverage_before_2026_12(months):
    with pytest.raises(B.U7Refused, match="before 2026-12"):
        B.u7_guard(months, "shared")


def test_u7_allows_window_months_and_lofo_refuses_early_months():
    ms = B.month_list("2026-12", "2027-10")
    assert B.u7_guard([f"{m}-01" for m in ms], "shared") == ms
    T = 11
    md = {c: np.zeros((2, T)) for c in K.COUNTIES}
    fam = {c: {k: np.zeros(T) for k in K.FAMILIES} for c in K.COUNTIES}
    with pytest.raises(B.U7Refused):
        B.lofo(md, fam, B.month_list("2025-12", "2026-10"))
    with pytest.raises(ValueError):
        B.u7_guard(ms, "absolute")


def test_no_observation_inputs():
    assert B.observation_inputs(B.bound_files()) == []
    with pytest.raises(B.U7Refused):
        B.assert_no_observation_inputs(["AMP_V2/data/wells/x.parquet"])
    with pytest.raises(B.U7Refused):
        B.assert_no_observation_inputs(["results/twin/prospective_2026-11/step5/rows.csv"])
    doc = P.draft_prereg(code_sha={f: "0" * 64 for f in K.FROZEN})
    assert B.observation_inputs([r["path"] for r in doc["inputs"]]) == []


def test_guard_lives_only_in_pb():
    for f in ("hydrophysics/twin/coverage.py", "hydrophysics/twin/bands.py",
              "dataprep_staging/harness/coverage3.py"):
        text = (REPO / f).read_text(encoding="utf-8")
        assert "u7_guard" not in text and K.PB_ID not in text
    src = (PB_DIR / "build_pb.py").read_text(encoding="utf-8")
    for bad in ("AMP_V2", "hydrophysics.twin.heads", "load_heads", "prospective.score"):
        assert bad not in src


def test_no_station_ids():
    B.no_station_ids({"sha": "16ac3c3783cea9f3c94560448d06c64ec8dc8fbb450effa28b465469a931cc32",
                      "x": 1.2345})
    with pytest.raises(io.EngineeringFailure):
        B.no_station_ids({"zones": [{"well": "99999999"}]})
    with pytest.raises(io.EngineeringFailure):
        B.no_station_ids({"note": "x"}, ids=["note"])


# ----------------------------------------------------------------------------- draft
def _validate(doc, tmp_path):
    return prereg3.validate_prereg(doc, units_registry=None, prereg_dir=tmp_path,
                                   registries=[])


def test_draft_validates_and_carries_the_design(tmp_path):
    doc = P.draft_prereg(code_sha={f: "0" * 64 for f in K.FROZEN})
    assert doc["id"] == "TEST." + K.PB_ID and doc["kind"] == "declaration"
    assert _validate(doc, tmp_path) == []
    real = P.draft_prereg()
    assert real["id"] == K.PB_ID and _validate(real, tmp_path) == []
    assert "MISSING" not in real["frozen_code"].values()
    d = real["design"]
    assert d == json.loads(json.dumps(B.design_spec()))
    assert d["ribbon"]["min_zone_wells"] == 5
    assert "leave_one_forcing_year_out" in d["dropped"]
    assert d["lofo"]["months"].startswith("window A only")
    pm = real["page_mapping"]
    assert pm["part_1_ribbon"]["label_A"]["en"] == "decided by a registered test, E02d, 2028"
    assert pm["part_2_shared"]["label"]["en"] == "cannot be tested before 2030 by design"
    assert real["ruling"]["decision"] == K.RULING_DECISION
    assert "E29" in real["status_16"]["FB"] and "E02x" in real["status_16"]["EXS"]
    files = real["binds"]["files"]
    assert set(files) == set(B.bound_files())
    assert all(len(h) == 64 for h in files.values())
    assert not any(str(r["path"]).startswith("/") for r in real["inputs"])
    assert not any(r["path"].startswith(K.TWIN_PREFIX) for r in real["inputs"])
    # lead decisions B1/B2, S1, S2, S4, S5
    p1 = pm["part_1_ribbon"]
    assert p1["rows"] == {"A": "C14a-r"} and p1["windows_drawn"] == ["A"]
    assert "E02y" not in json.dumps(p1) and "C14b" not in json.dumps(pm)
    assert "bn_head_fy" not in pm["replaces"] and "bn_head_fy_pb" in pm["kept"]
    assert "cov_line" in pm["kept"]
    what = pm["part_2_shared"]["what"]["en"]
    assert "spread among model variants only" in what and "forcing-year" in what
    assert "between-zone" in what and "FV v1 baseline forwards of FV v2's four truth" in what
    assert "results/twin_forward/fv_zr0.npz" in files
    assert files["results/twin_forward/fv_zr0.npz"].startswith("a1833d5f")
    assert "FB builder" in real["disclosures"]["zr0"]
    ep = real["disclosures"]["engineering_print"]
    assert "builder's choice" in ep and "61/40/34/13/10" in ep
    assert "A 2.157-4.904 m; B 2.267-4.954 m; C 2.408-5.014 m" in ep
    assert "dataprep_staging/e02c/score_e02c.py" in real["frozen_code"]
    kept = pm["kept"]["bn_head_fy_pb"]
    assert kept["registration"] == {"id": K.FV2_ID, "sha256": K.FV2_SHA,
                                    "wording": "readings.reading_2.wording.c14c"}
    assert kept["clause"]["en"].startswith("the model's own forcing-year range")
    assert "run spread" not in kept["clause"]["en"] and "執行離散" not in kept["clause"]["zh"]
    assert "Changhua:mid+dist" in p1["zone_names"] and "pb_small" in p1["small_zones"]
    assert "強迫" not in json.dumps(real, ensure_ascii=False)


def test_draft_main_writes_nothing(tmp_path, capsys):
    before = sorted(tmp_path.iterdir())
    rc = P.main(["--validate"], prereg_dir=tmp_path)
    out = capsys.readouterr().out
    assert rc == 0 and "nothing written" in out and "validate_prereg: OK" in out
    assert sorted(tmp_path.iterdir()) == before


# ----------------------------------------------------------------------------- builder refusals
def _register_fake(tmp_path, doc):
    path = tmp_path / f"{doc['id']}.json"
    data = (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode()
    path.write_bytes(data)
    prereg._append_chained(tmp_path / "INDEX.jsonl", {
        "id": doc["id"], "wp": doc["wp"], "kind": doc["kind"], "created": "2026-11-06T00:00:00Z",
        "sha256": io.sha256(path), "path": str(path), "tool": "test", "test_ids": []})


def test_build_refuses_unregistered(tmp_path):
    (tmp_path / "INDEX.jsonl").write_text("")
    with pytest.raises(io.Refused, match="not registered"):
        B.run(tmp_path / "twin", tmp_path / "out", prereg_dir=tmp_path)
    assert not (tmp_path / "out").exists()


def _fake_doc(files):
    doc = P.draft_prereg()
    doc["binds"] = {"registrations": {}, "files": files}
    return doc


def test_build_refuses_changed_design_inputs_observations_and_rewrite(tmp_path):
    twin = tmp_path / "twin"
    (twin / "x").mkdir(parents=True)
    (twin / "x" / "a.csv").write_text("a\n1\n")
    good = {"results/twin/x/a.csv": io.sha256(twin / "x" / "a.csv")}
    d1 = tmp_path / "r1"
    d1.mkdir()
    doc = _fake_doc({"results/twin/x/a.csv": "0" * 64})
    _register_fake(d1, doc)
    with pytest.raises(io.Refused, match="bound inputs differ"):
        B.run(twin, tmp_path / "out", prereg_dir=d1)
    d2 = tmp_path / "r2"
    d2.mkdir()
    doc = _fake_doc({**good, "AMP_V2/data/wells/w.parquet": "0" * 64})
    _register_fake(d2, doc)
    with pytest.raises(B.U7Refused):
        B.run(twin, tmp_path / "out", prereg_dir=d2)
    d3 = tmp_path / "r3"
    d3.mkdir()
    doc = _fake_doc(good)
    doc["design"]["ribbon"]["min_zone_wells"] = 3
    _register_fake(d3, doc)
    with pytest.raises(io.Refused, match="design differs"):
        B.run(twin, tmp_path / "out", prereg_dir=d3)
    d4 = tmp_path / "r4"
    d4.mkdir()
    _register_fake(d4, _fake_doc(good))
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / K.OUT_JSON).write_text("{}")
    with pytest.raises(io.Refused, match="written once"):
        B.run(twin, tmp_path / "out", prereg_dir=d4)


def test_build_wires_the_summary_checks(tmp_path):
    """run() refuses through summary_checks: lead_definition and decisions mismatches."""
    twin = tmp_path / "twin"
    (twin / "e02d_relative").mkdir(parents=True)
    summ = twin / "e02d_relative" / "summary.json"
    for case, body, msg in (
            ("lead", {"decisions": DECISIONS, "lead_definition": "datum_centre"},
             "lead_definition"),
            ("nolead", {"decisions": DECISIONS}, "lead_definition"),
            ("dec", {"decisions": {"Yunlin": {}}, "lead_definition": "origin"}, "decisions")):
        summ.write_text(json.dumps(body))
        d = tmp_path / f"reg_{case}"
        d.mkdir()
        _register_fake(d, _fake_doc({K.SUMMARY: io.sha256(summ)}))
        _register_fake(d, {"id": K.E02D_ID, "wp": "E02d", "kind": "test", **E02D_DOC})
        _register_fake(d, {"id": K.DR_ID, "wp": "DR", "kind": "declaration",
                           "lead_definition": "origin"})
        with pytest.raises(io.EngineeringFailure, match=msg):
            B.run(twin, tmp_path / f"out_{case}", prereg_dir=d)
        assert not (tmp_path / f"out_{case}").exists()


def test_build_refuses_changed_frozen_code(tmp_path):
    doc = _fake_doc({})
    doc["frozen_code"] = dict(doc["frozen_code"], **{K.BUILDER: "0" * 64})
    _register_fake(tmp_path, doc)
    with pytest.raises(io.Refused, match="code changed"):
        B.run(tmp_path / "twin", tmp_path / "out", prereg_dir=tmp_path)


# ----------------------------------------------------------------------------- real registry
@NEED_RESULTS
def test_real_draft_is_ready_or_is_what_was_registered():
    doc = P.draft_prereg()
    if io.index_record(K.PB_ID) is None:
        assert P.problems(doc) == []
        return
    # registered: the only readiness problem is that, and the registered document is this
    # draft (every drafted field except the inputs, which write_prereg hashes and sizes)
    assert P.problems(doc) == [f"{K.PB_ID} is already registered"]
    reg, _ = io.registered(K.PB_ID, frozen=K.FROZEN)
    assert {k: v for k, v in doc.items() if k != "inputs"} == \
        {k: reg[k] for k in doc if k != "inputs"}
    assert len(reg["inputs"]) == len(doc["inputs"])        # registered paths may be resolved
    assert all(str(r["path"]).endswith(d["path"]) and r["sha256"] == d.get("sha256", r["sha256"])
               for r, d in zip(reg["inputs"], doc["inputs"], strict=True))


@NEED_RESULTS
@pytest.mark.skipif(not (REPO / "results" / "twin" / "e02d_relative").is_dir(),
                    reason="this checkout's results/twin has no e02d_relative/")
def test_bound_files_hash_in_this_checkout():
    want = B.bound_files()
    assert B.sha_problems(B.file_shas(want, REPO / "results" / "twin"), want) == []


# ----------------------------------------------------------------------------- kept FV v2 clause
#: the page's placeholders for pb_const's registered format fields, and FV v2's {lbl} as the
#: page renders it in bn_head_fy (EN and ZH)
CLAUSE_FIELDS = {"{n2_lo:+.2f}": "{n2lo}", "{n2_hi:+.2f}": "{n2hi}", "{l2_lo:+.2f}": "{l2lo}",
                 "{l2_hi:+.2f}": "{l2hi}", "{forcing}": "{forcing}"}
CLAUSE_LBL = {"en": "post-failure exploratory, FV.decidability.v2",
              "zh": "失敗後探索性研究，FV.decidability.v2"}


def _template_key(lang: str, key: str) -> str:
    import re

    text = (REPO / "hydrophysics" / "twin" / "app" / "template.html").read_text(encoding="utf-8")
    en, zh = text.split("const T = {")[1].split("\nzh: {")
    block = en if lang == "en" else zh.split("\n}};")[0]
    m = re.findall(rf'(?<![\w"]){key}:\s*"([^"]*)"', block)
    assert len(m) == 1, (lang, key, m)
    return m[0]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_template_fy_clause_is_the_registered_clause_verbatim(lang):
    clause = K.FV2_C14C_CLAUSE_EN if lang == "en" else K.FV2_C14C_CLAUSE_ZH
    for f, page in CLAUSE_FIELDS.items():
        assert f in clause, f
        clause = clause.replace(f, page)
    clause = clause.replace("{lbl}", CLAUSE_LBL[lang])
    if lang == "en":
        clause = clause[0].upper() + clause[1:]
    page = _template_key(lang, "bn_head_fy_pb")
    assert page == clause + ("." if lang == "en" else "。")
    # the full line keeps the same {lbl} rendering, so the clause is that line's clause
    assert CLAUSE_LBL[lang] in _template_key(lang, "bn_head_fy")


def _fv2_doc(en: str, zh: str) -> dict:
    return {"id": K.FV2_ID, "wp": "FV", "kind": "declaration",
            "truths": [{"tag": t, "v1_forward": {"sha256": "0" * 64}} for t in K.FAMILIES],
            "readings": {"reading_2": {"wording": {"c14c": {"en": en, "zh": zh}}}}}


def test_readiness_checks_the_kept_clause_verbatim(tmp_path):
    doc = P.draft_prereg(code_sha={f: "0" * 64 for f in K.FROZEN})
    msg = "does not hold the kept clause verbatim"
    good = tmp_path / "good"
    good.mkdir()
    _register_fake(good, _fv2_doc("run spread, a parameter spread, plus " +
                                  K.FV2_C14C_CLAUSE_EN, "前文" + K.FV2_C14C_CLAUSE_ZH))
    assert not [b for b in P.problems(doc, good) if msg in b]
    bad = tmp_path / "bad"
    bad.mkdir()
    _register_fake(bad, _fv2_doc(K.FV2_C14C_CLAUSE_EN.replace("damped", "reduced"),
                                 K.FV2_C14C_CLAUSE_ZH))
    hits = [b for b in P.problems(doc, bad) if msg in b]
    assert hits == [f"FV v2's registered C14c wording (en) {msg}"]
