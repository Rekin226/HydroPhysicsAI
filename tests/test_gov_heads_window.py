"""GOV.heads_window_product.v1 (draft): the staged heads window MANIFEST template against every
registered scorer precondition that reads the window product (round-8 step 0, audit fix F5).

The preconditions are the frozen functions themselves (e02c/score_e02c.py product_grants_scorer,
e02d/score_e02d.py product_covers), run on synthetic product directories in pytest's tmp dir:
each reads the MANIFEST alone, so no head table exists or is opened. N2 and N2C have no
precondition code yet; the declaration's covering rule stands in for them. The template and
the declaration are built in code (``template()``, ``declaration_core()``,
``draft_declaration()``); their JSON copies are local, gitignored CLI outputs, compared only when
present. Checks that need the gitignored registry (results/dataprep/preregistration) skip
cleanly when it is absent (CI).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]
MOD = REPO / "dataprep_staging" / "gov_heads_window" / "heads_window.py"
_spec = importlib.util.spec_from_file_location("gov_heads_window_heads_window", MOD)
HW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(HW)

NEED_REGISTRY = pytest.mark.skipif(not (HW.PREREG / "INDEX.jsonl").exists(),
                                   reason="the pre-registration index (results/dataprep/"
                                          "preregistration) is gitignored, absent here")
KEYS = sorted(HW.PRECONDITIONS)
YEAR1 = [k for k in KEYS if not HW.PRECONDITIONS[k]["needs_window"]]
YEARS23 = [k for k in KEYS if HW.PRECONDITIONS[k]["needs_window"]]
STATION = re.compile(r"\b(?:0[79]|10)[0-9A-Z]{6}\b")


@pytest.fixture(autouse=True)
def _staging_on_path():
    """dataprep_staging on sys.path for each test only (the scorers import `common` lazily);
    restored afterwards so the rest of the suite sees the path it had."""
    saved = list(sys.path)
    HW._paths()
    yield
    sys.path[:] = saved


def _template() -> dict:
    return HW.template()


def _product(root: Path, name: str, *, window="keep", allowed=None, gated=True,
             fit_window_end="2029-10") -> Path:
    """A synthetic window product directory: the instantiated template plus empty files. The
    default product holds data to 2029-10, so it may declare the full window."""
    d = root / name / "v1"
    d.mkdir(parents=True)
    (d / "heads.csv").write_text("station_id,date,head_m,flag\n", encoding="utf-8")
    (d / "validation.json").write_text("{}\n", encoding="utf-8")
    rec = {"path": "heads.csv", "sha256": "0" * 64}
    m = HW.instantiate(_template(), version="v1", created="2029-11-10",
                       inputs=[dict(rec, path="data/synthetic.csv")],
                       code=[dict(rec, path="dataprep_staging/gov_heads_window/heads_window.py")],
                       git_commit="0" * 40, fit_window_end=fit_window_end, outputs=[rec],
                       validation_report="validation.json",
                       gate={"passed": gated, "criteria": "synthetic test product"},
                       window=None if window in ("keep", None) else tuple(window))
    if window is None:
        del m["parameters"][HW.FIELD]
    if allowed is not None:
        m["allowed_uses"] = list(allowed)
        m["forbidden_uses"] = [u for u in HW.all_uses() if u not in allowed]
    (d / "MANIFEST.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return d


# ---- the template ---------------------------------------------------------------------------
def test_template_is_built_in_code_and_holds_no_station_id():
    text = HW.template_text()
    assert json.loads(text) == HW.template()
    assert not STATION.search(text)
    t = _template()
    assert t["parameters"][HW.FIELD] == ["2026-11", "2029-10"]
    for w, (first, last) in HW.WINDOWS.items():                 # windows A, B and C
        assert t["parameters"]["windows"][w] == [first, last]
        assert HW._covers(t["parameters"][HW.FIELD], (first, last)), w
    assert t["allowed_uses"] == ["prospective_scorer", "validator", "after_the_fact_analysis"]
    assert set(t["allowed_uses"]) | set(t["forbidden_uses"]) == set(HW.all_uses())
    assert t["gate"]["passed"] is False                          # an unfilled template never scores
    assert HW.check_governance(t) == []


@pytest.mark.skipif(not HW.TEMPLATE.exists(), reason="local template copy not written")
def test_local_template_copy_if_written_is_the_builder_output():
    assert HW.TEMPLATE.read_text(encoding="utf-8") == HW.template_text()


def test_instantiated_template_is_a_valid_manifest(tmp_path):
    from common.manifest import load_manifest, validate_manifest

    d = _product(tmp_path, "full")
    m, _ = load_manifest(d)
    assert validate_manifest(m, d) == []
    assert HW.check_governance(m) == []


def test_governance_refuses_a_widened_or_extra_use_manifest():
    t = _template()
    wide = copy.deepcopy(t)
    wide["parameters"][HW.FIELD] = ["2026-11", "2030-10"]
    assert any("outside the ruling's scope" in p for p in HW.check_governance(wide))
    extra = copy.deepcopy(t)
    extra["allowed_uses"].append("selection")
    extra["forbidden_uses"].remove("selection")
    assert any("not granted" in p for p in HW.check_governance(extra))
    with pytest.raises(ValueError, match="scope"):
        HW.instantiate(t, version="v1", created="2028-01-10", inputs=[], code=[],
                       git_commit="0", fit_window_end="2027-10", outputs=[],
                       validation_report="v.json", gate={"passed": True, "criteria": "x"},
                       window=("2026-11", "2031-10"))


def test_declared_window_may_not_end_after_the_data():
    """A product whose data end in 2027-10 cannot keep the template's 2026-11..2029-10 window
    (it would grant years 2-3 before any year-2/3 month exists); narrowed to its data it is
    accepted."""
    t = _template()
    kw = dict(version="v1", created="2028-01-10", inputs=[], code=[], git_commit="0",
              fit_window_end="2027-10", outputs=[], validation_report="v.json",
              gate={"passed": True, "criteria": "x"})
    with pytest.raises(ValueError, match="fit_window_end 2027-10"):
        HW.instantiate(t, **kw)
    with pytest.raises(ValueError, match="fit_window_end 2027-10"):
        HW.instantiate(t, **kw, window=("2026-11", "2028-10"))
    m = HW.instantiate(t, **kw, window=HW.WINDOWS["A"])
    assert m["parameters"][HW.FIELD] == ["2026-11", "2027-10"]
    assert HW.check_governance(m) == []
    late = copy.deepcopy(m)
    late["fit_window_end"] = "2027-09"
    assert any("after the product's fit_window_end" in p for p in HW.check_governance(late))
    unfilled = copy.deepcopy(m)
    unfilled["fit_window_end"] = "<YYYY-MM>"
    assert any("fit_window_end must be YYYY-MM" in p for p in HW.check_governance(unfilled))


# ---- every registered scorer precondition -----------------------------------------------------
def test_the_registered_scorers_are_listed():
    regs = {k.split(":")[0] for k in KEYS}
    assert regs == {HW.E02C, HW.N3F, HW.E02D, HW.E02Y, HW.N2, HW.N2C}
    assert sorted(YEARS23) == sorted([f"{HW.E02Y}:B", f"{HW.E02Y}:C"])


@pytest.mark.parametrize("key", KEYS)
def test_every_precondition_accepts_the_template(tmp_path, key):
    ok, why = HW.check_precondition(key, _product(tmp_path, "full"))
    assert ok, (key, why)


@pytest.mark.parametrize("key", KEYS)
def test_every_precondition_rejects_a_product_without_the_scorer_use(tmp_path, key):
    d = _product(tmp_path, "noscorer", allowed=["validator", "after_the_fact_analysis"])
    ok, why = HW.check_precondition(key, d)
    assert not ok and "prospective_scorer" in why, (key, why)


@pytest.mark.parametrize("key", KEYS)
def test_every_precondition_rejects_an_ungated_product(tmp_path, key):
    ok, why = HW.check_precondition(key, _product(tmp_path, "ungated", gated=False))
    assert not ok and "gate" in why, (key, why)


@pytest.mark.parametrize("key", YEARS23)
def test_years_2_3_reject_a_product_without_the_window(tmp_path, key):
    ok, why = HW.check_precondition(key, _product(tmp_path, "nowindow", window=None))
    assert not ok and HW.FIELD in why, (key, why)


@pytest.mark.parametrize("key", YEAR1)
def test_year_1_scorers_accept_a_product_without_the_window(tmp_path, key):
    """Their registered rule: year 1 is the vocabulary's own window, so no declaration is
    needed (E02c/N3F do not read the field at all; E02d accepts its absence)."""
    ok, why = HW.check_precondition(key, _product(tmp_path, "nowindow", window=None))
    assert ok, (key, why)


def test_a_window_that_does_not_cover_is_rejected(tmp_path):
    short = _product(tmp_path, "short", window=("2026-11", "2028-10"))
    assert HW.check_precondition(f"{HW.E02Y}:B", short)[0]
    assert not HW.check_precondition(f"{HW.E02Y}:C", short)[0]
    late = _product(tmp_path, "late", window=("2027-11", "2029-10"))
    for key in (f"{HW.E02D}:A", HW.N2, HW.N2C):
        assert not HW.check_precondition(key, late)[0], key


@pytest.mark.parametrize("key", [HW.E02C, HW.N3F])
def test_year_1_frozen_checks_get_the_covering_rule(tmp_path, key):
    """E02c and N3F's frozen product_grants_scorer reads only the use grant, so a product that
    declares only years 2-3 passes it; check_precondition adds the covering rule and rejects
    that product for their year-1 months (the frozen scorer code is not edited)."""
    s_c, _ = HW._scorers()
    late = _product(tmp_path, "late", window=("2027-11", "2029-10"))
    assert s_c.product_grants_scorer(late)[0]                 # the frozen check alone passes
    ok, why = HW.check_precondition(key, late)
    assert not ok and "does not cover 2026-11..2027-10" in why, why
    year1 = _product(tmp_path, "year1", window=HW.WINDOWS["A"], fit_window_end="2027-10")
    ok, why = HW.check_precondition(key, year1)
    assert ok and "covers" in why, why


# ---- frozen code and the draft declaration ---------------------------------------------------
def test_uses_vocabulary_is_untouched():
    assert HW.sha256_file(REPO / HW.USES_FILE) == HW.USES_SHA256


@NEED_REGISTRY
def test_the_preconditions_run_the_registered_code():
    assert HW.frozen_code_problems() == []


def test_declaration_core_is_built_in_code():
    d = HW.declaration_core()
    text = json.dumps(d, indent=1, ensure_ascii=False)
    assert not STATION.search(text)
    assert d["id"] == HW.DECL_ID and d["status"].startswith("DRAFT: not registered")
    assert d["template"]["sha256"] == hashlib.sha256(HW.template_text().encode()).hexdigest()
    assert set(d["scorer_preconditions"]) == set(KEYS)
    assert d["uses_vocabulary"]["sha256"] == HW.USES_SHA256
    assert "station_lists" not in d


@NEED_REGISTRY
def test_draft_declaration_holds_hashes_only():
    d = HW.draft_declaration()
    text = json.dumps(d, indent=1, ensure_ascii=False)
    assert not STATION.search(text)
    assert {k: v for k, v in d.items() if k != "station_lists"} == HW.declaration_core()
    sl = d["station_lists"]
    assert {k: sl[k]["n"] for k in ("E02_wells158", "N3F_external66", "N2_yunlin84",
                                    "N2C_changhua74")} == {
        "E02_wells158": 158, "N3F_external66": 66, "N2_yunlin84": 84, "N2C_changhua74": 74}
    lists = [v for v in sl.values() if isinstance(v, dict) and "n" in v]
    assert len(lists) == 4 and all(re.fullmatch(r"[0-9a-f]{64}", v["sha256"]) for v in lists)


@NEED_REGISTRY
def test_draft_declaration_is_not_registered_and_the_local_copy_is_current():
    built = HW.draft_declaration()
    if HW.DRAFT.exists():
        assert json.loads(HW.DRAFT.read_text(encoding="utf-8")) == built
    idx = [json.loads(line) for line in (HW.PREREG / "INDEX.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    assert HW.DECL_ID not in {r.get("id") for r in idx}
