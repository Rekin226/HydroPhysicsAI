"""GOV.heads_window_product.v1 (DRAFT): governance of the heads window product that the
registered prospective scorers read (round-8 step 0, audit fix F5).

Why. ``common/uses.py`` describes ``prospective_scorer`` as "the registered prospective
scorer (Nov 2026 - Oct 2027)". User ruling R7.heads_years23 extended the heads scorer to the
year-2/3 windows of the same frozen forecast (2027-11..2029-10), and E02y's registered scorer
(``e02d/score_e02d.py::product_covers``) reads that extension only from the window product's
MANIFEST field ``parameters.prospective_scorer_window = [first, last]``. ``uses.py`` is NOT
edited: E02c binds its hash (d1355821) and restores it at scoring. This module therefore fixes
the MANIFEST side instead:

* a staged MANIFEST template (``template()``, the source of truth; ``MANIFEST_template.json``
  is only a local, gitignored copy the CLI writes) that grants ``prospective_scorer`` (and only
  ``validator`` and ``after_the_fact_analysis`` beside it) and declares the window for A, B
  and C (2026-11..2029-10);
* ``instantiate`` fills the template for a real product directory and ``check_governance``
  validates the governance fields of any window MANIFEST, including that the declared window
  does not end after the product's own ``fit_window_end`` (the data end);
* ``PRECONDITIONS`` lists every registered scorer that reads the window product, with the
  frozen function that checks it (``check_precondition`` runs it on a product directory, from
  the MANIFEST alone: no table is read), and adds this declaration's covering rule wherever
  the frozen function does not read the declared window (E02c, N3F, N2, N2C);
* ``draft_declaration`` builds the draft declaration (the source of truth; the ``.draft.json``
  file is a local, gitignored copy) with the station-list hashes (158 E02 wells, 66 N3F
  external wells, N2's 84 Yunlin and N2C's 74 Changhua wells), read from the registrations'
  own unit lists. Only hashes and counts are written; the ids stay in the registrations under
  results/. ``declaration_core`` is the part that needs no registry.

Nothing here registers anything: the lead registers GOV.heads_window_product.v1 (prereg3)
after the code is merged. No liveness probe runs before the dependent registrations exist.

    python -I dataprep_staging/gov_heads_window/heads_window.py template   # write the local copy
    python -I dataprep_staging/gov_heads_window/heads_window.py draft      # write the local copy
    python -I dataprep_staging/gov_heads_window/heads_window.py check <product dir>
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGING = REPO / "dataprep_staging"
PREREG = REPO / "results" / "dataprep" / "preregistration"

DECL_ID = "GOV.heads_window_product.v1"
#: Local, gitignored outputs of the CLI (dataprep_staging/.gitignore blocks *.json). The code
#: (template() / draft_declaration()) is the source of truth; nothing reads these files.
TEMPLATE = HERE / "MANIFEST_template.json"
DRAFT = HERE / "GOV.heads_window_product.v1.draft.json"
RULING = "user ruling 2026-11-01 (sixth set) (R7.heads_years23)"
RULING_PENDING = "R8.14 (asked 2026-11-02): confirm this template implements R7.heads_years23"
FIELD = "prospective_scorer_window"
USES_FILE = "dataprep_staging/common/uses.py"
USES_SHA256 = "d13558213580cc682f1ba61bf1d16bd328580ef34410888c6bd58c592805a47b"

#: The ruling's scope: the frozen forecast's years 1-3.
SCOPE = ("2026-11", "2029-10")
WINDOWS = {"A": ("2026-11", "2027-10"), "B": ("2027-11", "2028-10"), "C": ("2028-11", "2029-10")}
ALLOWED = ["prospective_scorer", "validator", "after_the_fact_analysis"]
FLAGS = [0, 1]
PRODUCT = "heads/L1_window"
_YM = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_STATION = re.compile(r"\b(?:0[79]|10)[0-9A-Z]{6}\b")

E02C = "E02c.prospective_2026-11.v1"
N3F = "N3F.ext_level.prospective_2026-11.v1"
E02D = "E02d.decomposed_prospective_2026-12.v1"
E02Y = "E02y.years23_2027-11_2029-10.v1"
N2 = "N2.policy_response.prospective_2026-11.v1"
N2C = "N2C.changhua_response.prospective_2026-11.v1"

#: Every registered scorer that reads the heads window product, keyed "<registration>[:<window>]".
#: ``code`` is the frozen file whose function decides; ``needs_window`` says whether the
#: registered rule requires parameters.prospective_scorer_window.
PRECONDITIONS = {
    E02C: {"months": ("2026-11", "2027-10"), "needs_window": False,
           "code": "dataprep_staging/e02c/score_e02c.py", "function": "product_grants_scorer",
           "registered_rule": "the window heads product MANIFEST lists prospective_scorer "
                              "(flags 0-1); otherwise not_evaluable"},
    N3F: {"months": ("2026-11", "2027-10"), "needs_window": False,
          "code": "dataprep_staging/e02c/score_e02c.py", "function": "product_grants_scorer",
          "registered_rule": "the window heads product MANIFEST lists prospective_scorer "
                             "(flags 0-1); otherwise not_informative (run_score_ext)"},
    E02D + ":A": {"months": ("2026-12", "2027-10"), "needs_window": False,
                  "code": "dataprep_staging/e02d/score_e02d.py",
                  "function": "product_grants_scorer and product_covers(A)",
                  "registered_rule": "prospective_scorer and, if the MANIFEST declares "
                                     "prospective_scorer_window, it covers 2026-12..2027-10"},
    E02Y + ":B": {"months": WINDOWS["B"], "needs_window": True,
                  "code": "dataprep_staging/e02d/score_e02d.py",
                  "function": "product_grants_scorer and product_covers(B)",
                  "registered_rule": "prospective_scorer AND a declared prospective_scorer_window "
                                     "covering 2027-11..2028-10"},
    E02Y + ":C": {"months": WINDOWS["C"], "needs_window": True,
                  "code": "dataprep_staging/e02d/score_e02d.py",
                  "function": "product_grants_scorer and product_covers(C)",
                  "registered_rule": "prospective_scorer AND a declared prospective_scorer_window "
                                     "covering 2028-11..2029-10"},
    N2: {"months": ("2026-11", "2027-10"), "needs_window": False, "code": None,
         "function": "governance rule of this declaration (no registered precondition code)",
         "registered_rule": "none in code: the N2 scorer is not written yet; the registration "
                            "scores flags 0-1 window heads in the 2028-01-15 session"},
    N2C: {"months": ("2026-11", "2027-10"), "needs_window": False, "code": None,
          "function": "governance rule of this declaration (no registered precondition code)",
          "registered_rule": "none in code: the N2C scorer is not written yet; the registration "
                             "scores flags 0-1 window heads in the 2028-01-15 session"},
}


# ----------------------------------------------------------------------------- helpers
def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def station_list_sha256(ids) -> str:
    """sha256 of the sorted unique ids, one per line, each line ending in a newline (UTF-8)."""
    return hashlib.sha256("".join(f"{s}\n" for s in sorted(set(map(str, ids))))
                          .encode("utf-8")).hexdigest()


def _covers(window, months) -> bool:
    first, last = (str(x) for x in window)
    return first <= months[0] and months[1] <= last


def all_uses() -> list[str]:
    """The use vocabulary (read from the frozen uses.py, never edited)."""
    _paths()
    from common.uses import USES

    return sorted(USES)


def forbidden_uses() -> list[str]:
    return [u for u in all_uses() if u not in ALLOWED]


def _paths() -> None:
    for p in (STAGING / "e02d", STAGING / "e02c", STAGING):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


# ----------------------------------------------------------------------------- template
def template() -> dict:
    """The staged MANIFEST template. Governance fields are fixed; ``<...>`` fields are filled by
    ``instantiate`` when the real window product is built. The gate is not passed in the
    template, so an unfilled template is never scorable."""
    return {
        "_template": {
            "declaration": DECL_ID,
            "ruling": RULING,
            "status": "template; fill every '<...>' field with instantiate() when the window "
                      "product is built. allowed_uses, forbidden_uses and parameters."
                      f"{FIELD} / flags / governance are fixed by the declaration",
            "fill": ["version", "created", "inputs", "code", "git_commit", "fit_window_end",
                     "outputs", "validation_report", "gate"],
        },
        "schema": "dataprep-manifest/2",
        "product": PRODUCT,
        "version": "<v<N>>",
        "created": "<YYYY-MM-DD>",
        "layer": "L1",
        "inputs": [],
        "code": [],
        "git_commit": "<commit>",
        "parameters": {
            FIELD: list(SCOPE),
            "flags": list(FLAGS),
            "governance": DECL_ID,
            "windows": {k: list(v) for k, v in WINDOWS.items()},
        },
        "fit_window_end": "<YYYY-MM: the product's data end, at most 2029-10; the declared "
                          f"{FIELD} may not end after it>",
        "exemptions": [],
        "outputs": [],
        "validation_report": "<validation.json>",
        "gate": {"passed": False, "criteria": "<the product's own validation gate>"},
        "allowed_uses": list(ALLOWED),
        "forbidden_uses": forbidden_uses(),
        "personal_data": "none (station ids only; no names or contacts)",
    }


def template_text() -> str:
    """The template as the CLI writes it (the bytes the declaration hashes)."""
    return json.dumps(template(), indent=1, ensure_ascii=False) + "\n"


def instantiate(tpl: dict, *, version: str, created: str, inputs: list, code: list,
                git_commit: str, fit_window_end: str, outputs: list, validation_report: str,
                gate: dict, window: tuple[str, str] | None = None) -> dict:
    """A concrete MANIFEST from the template. ``window`` may narrow the declared window (a
    year-specific product) but never widen it beyond the ruling's scope. The declared window
    may not end after ``fit_window_end``: a product built before 2029-10 must narrow it to the
    months its data reach, or ``check_governance`` refuses it here."""
    m = copy.deepcopy(tpl)
    m.pop("_template", None)
    m.update({"version": version, "created": created, "inputs": inputs, "code": code,
              "git_commit": git_commit, "fit_window_end": fit_window_end, "outputs": outputs,
              "validation_report": validation_report, "gate": gate})
    if window is not None:
        m["parameters"][FIELD] = list(window)
    probs = check_governance(m)
    if probs:
        raise ValueError("; ".join(probs))
    return m


def check_governance(m: dict) -> list[str]:
    """Governance problems of a window MANIFEST (empty = conforms). The MANIFEST alone."""
    e = []
    au, fu = m.get("allowed_uses") or [], m.get("forbidden_uses") or []
    if "prospective_scorer" not in au or "prospective_scorer" in fu:
        e.append("prospective_scorer must be allowed and not forbidden")
    extra = sorted(set(au) - set(ALLOWED))
    if extra:
        e.append(f"uses beyond {ALLOWED} are not granted by {DECL_ID}: {extra}")
    missing = sorted(set(forbidden_uses()) - set(fu))
    if missing:
        e.append(f"uses not forbidden: {missing}")
    par = m.get("parameters") or {}
    w = par.get(FIELD)
    if not (isinstance(w, list) and len(w) == 2 and all(isinstance(x, str) and _YM.match(x)
                                                        for x in w)):
        e.append(f"parameters.{FIELD} must be [first, last] (YYYY-MM)")
    else:
        if w[0] > w[1]:
            e.append(f"parameters.{FIELD} first > last")
        if not _covers(SCOPE, w):
            e.append(f"parameters.{FIELD} {w} reaches outside the ruling's scope {SCOPE}")
    if par.get("flags") != FLAGS:
        e.append(f"parameters.flags must be {FLAGS} (prospective_scorer reads flags 0-1)")
    if par.get("governance") != DECL_ID:
        e.append(f"parameters.governance must name {DECL_ID}")
    fwe = m.get("fit_window_end")
    if isinstance(fwe, str) and _YM.match(fwe):
        if fwe > SCOPE[1]:
            e.append(f"fit_window_end {fwe} > {SCOPE[1]}")
        if isinstance(w, list) and len(w) == 2 and isinstance(w[1], str) and w[1] > fwe:
            e.append(f"parameters.{FIELD} ends {w[1]}, after the product's fit_window_end "
                     f"{fwe}: a product cannot grant months its data do not reach")
    elif "_template" not in m:
        e.append("fit_window_end must be YYYY-MM (the product's data end)")
    return e


def covering_rule(m: dict, months) -> tuple[bool, str]:
    """This declaration's rule for every scorer: a product that declares a window must cover
    the scorer's months; a product without one serves only year 1 (the vocabulary's own
    window)."""
    w = (m.get("parameters") or {}).get(FIELD)
    if w is None:
        if _covers(WINDOWS["A"], months):
            return True, "no declared window; year 1 is the vocabulary's own window"
        return False, f"no declared {FIELD}: a year-1 product does not grant {months}"
    try:
        ok = _covers(w, months)
    except (TypeError, ValueError) as exc:
        return False, f"unreadable {FIELD} {w!r} ({exc})"
    return ok, f"{FIELD} {w} {'covers' if ok else 'does not cover'} {months[0]}..{months[1]}"


# ----------------------------------------------------------------------------- preconditions
def _scorers():
    _paths()
    import score_e02c
    import score_e02d

    return score_e02c, score_e02d


def check_precondition(key: str, product_dir) -> tuple[bool, str]:
    """Run the registered precondition of scorer ``key`` (a PRECONDITIONS key) on a product
    directory, plus this declaration's covering rule where the frozen code does not read the
    declared window. The MANIFEST alone is read.

    E02c and N3F's frozen ``product_grants_scorer`` checks only the use grant, so a product
    that declares only years 2-3 would pass it; the covering rule rejects that product for
    their year-1 months. E02d/E02y's frozen ``product_covers`` already reads the window. N2 and
    N2C have no precondition code; the covering rule is theirs."""
    spec = PRECONDITIONS[key]
    s_c, s_d = _scorers()
    ok, why = s_c.product_grants_scorer(product_dir)
    if not ok:
        return ok, why
    if spec["code"] == "dataprep_staging/e02d/score_e02d.py":
        return s_d.product_covers(product_dir, key.rsplit(":", 1)[1])
    from common.manifest import load_manifest

    m, _ = load_manifest(product_dir)
    ok, rule = covering_rule(m, spec["months"])
    if spec["code"] == "dataprep_staging/e02c/score_e02c.py":
        return ok, f"{why}; {DECL_ID} covering rule: {rule}"
    return ok, rule


def frozen_code_problems(prereg_dir: Path = PREREG) -> list[str]:
    """Each precondition's code must be the file its registration froze (sha256)."""
    probs = []
    for key, spec in PRECONDITIONS.items():
        if spec["code"] is None:
            continue
        rid = key.split(":")[0]
        doc = json.loads((Path(prereg_dir) / f"{rid}.json").read_text(encoding="utf-8"))
        want = (doc.get("frozen_code") or {}).get(spec["code"])
        got = sha256_file(REPO / spec["code"])
        if want != got:
            probs.append(f"{key}: {spec['code']} sha256 {got[:8]} != registered {str(want)[:8]}")
    return probs


# ----------------------------------------------------------------------------- declaration
def _units(doc: dict) -> dict[str, list[str]]:
    out = {}
    for t in doc.get("tests") or []:
        for name, u in (t.get("external_units") or {}).items():
            out[name] = list(u["ids"])
    return out


def station_lists(prereg_dir: Path = PREREG) -> dict:
    """The station lists the window product must hold, from the registrations' unit lists."""
    def load(rid):
        p = Path(prereg_dir) / f"{rid}.json"
        return json.loads(p.read_text(encoding="utf-8")), sha256_file(p)

    (c, c_sha), (f, f_sha) = load(E02C), load(N3F)
    (d, d_sha), (y, y_sha) = load(E02D), load(E02Y)
    (n2, n2_sha), (n2c, n2c_sha) = load(N2), load(N2C)
    e02 = set(_units(c)["I1:prospective_heads_2026-11_2027-10_R0wells"])
    same = {E02D: set(next(iter(_units(d).values()))), N2: set(_units(n2)[
        "N2:heads_prospective_2026-11_2027-10"])}
    same.update({f"{E02Y} {k}": set(v) for k, v in _units(y).items()})
    bad = sorted(k for k, v in same.items() if v != e02)
    if bad or len(e02) != 158:
        raise ValueError(f"the 158-well list differs between registrations: {bad}")
    ext = set(_units(f)["N3F:ext_heads_2026-11_2027-10"])
    if ext != set(f["external_predictions"]["kept_ids"]) or len(ext) != 66 or ext & e02:
        raise ValueError("the N3F external list is not 66 kept wells disjoint from the 158")
    yl = {s for s in e02 if s.startswith("09")}
    ch = set(_units(n2c)["N2C:heads_changhua_prospective_2026-11_2027-10"])
    if len(yl) != 84 or len(ch) != 74 or ch != {s for s in e02 if s.startswith("07")}:
        raise ValueError("N2's Yunlin 84 / N2C's Changhua 74 do not split the 158")

    def rec(ids, src):
        return {"n": len(ids), "sha256": station_list_sha256(ids), "source": src}

    return {
        "definition": "sha256 of the sorted unique station ids, one per line, each line "
                      "ending in a newline (UTF-8); ids are never written here",
        "E02_wells158": rec(e02, {
            "registration": E02C, "registration_sha256": c_sha,
            "field": "tests[*].external_units['I1:prospective_heads_2026-11_2027-10_R0wells'].ids",
            "identical_in": [E02D, E02Y + " (all four tests)", N2]}),
        "N3F_external66": rec(ext, {
            "registration": N3F, "registration_sha256": f_sha,
            "field": "external_predictions.kept_ids (= tests[0] unit ids)",
            "disjoint_from": "E02_wells158"}),
        "N2_yunlin84": rec(yl, {
            "registration": N2, "registration_sha256": n2_sha,
            "field": "the county-09 (Yunlin) ids of its 158; N2's primary wells"}),
        "N2C_changhua74": rec(ch, {
            "registration": N2C, "registration_sha256": n2c_sha,
            "field": "tests[0].external_units['N2C:heads_changhua_prospective_2026-11_"
                     "2027-10'].ids (= the county-07 ids of the 158)"}),
        "registrations_also_checked": {E02D: d_sha, E02Y: y_sha},
    }


def declaration_core() -> dict:
    """The draft declaration without its station lists: everything that needs no registry."""
    pre = {}
    for key, spec in PRECONDITIONS.items():
        rec = {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}
        if spec["code"]:
            rec["code_sha256"] = sha256_file(REPO / spec["code"])
        pre[key] = rec
    return {
        "id": DECL_ID,
        "kind": "declaration",
        "status": "DRAFT: not registered. The lead registers it (harness.prereg3) after this code "
                  "is merged; until then it binds nothing. No liveness probe runs before the "
                  "dependent registrations exist.",
        "title": "Governance of the heads window product: the MANIFEST field "
                 f"parameters.{FIELD} carries R7.heads_years23's year-2/3 extension of "
                 "prospective_scorer, without editing common/uses.py",
        "rulings": {"R7.heads_years23": RULING, "pending": RULING_PENDING},
        "why": "E02y's windows B and C can be scored only on a product whose MANIFEST declares "
               f"parameters.{FIELD}; no held product declares it, and the use vocabulary still "
               "reads 'Nov 2026 - Oct 2027'. Without this declaration C14b's D could become "
               "unscorable by accident in 2029.",
        "uses_vocabulary": {
            "file": USES_FILE, "sha256": USES_SHA256,
            "text": "prospective_scorer: the registered prospective scorer (Nov 2026 - Oct 2027)",
            "edit": "none: E02c binds this hash (d1355821) and restores it at scoring; the "
                    "year-2/3 extension lives in the product MANIFEST only"},
        "manifest_field": {
            "path": f"parameters.{FIELD}", "format": "[first, last], YYYY-MM",
            "scope": list(SCOPE), "windows": {k: list(v) for k, v in WINDOWS.items()},
            "rule": "a product that declares a window must cover the months of every scorer that "
                    "reads it; a product without one serves only year 1 (2026-11..2027-10)"},
        "template": {"source": "dataprep_staging/gov_heads_window/heads_window.py template()",
                     "sha256": hashlib.sha256(template_text().encode("utf-8")).hexdigest(),
                     "sha256_of": "template_text(): json.dumps(template(), indent=1, "
                                  "ensure_ascii=False) + newline, UTF-8 (the bytes the CLI "
                                  "'template' command writes to the local, gitignored "
                                  "MANIFEST_template.json)",
                     "allowed_uses": list(ALLOWED), "flags": list(FLAGS),
                     "builder": "dataprep_staging/gov_heads_window/heads_window.py instantiate()"},
        "data_end_rule": f"parameters.{FIELD}'s last month may not be after the product's "
                         "fit_window_end (the data end); instantiate() and check_governance() "
                         "refuse such a product",
        "scorer_preconditions": pre,
        "known_gaps": [
            "E02c and N3F's frozen code checks only the prospective_scorer grant; it does not "
            f"read parameters.{FIELD}, so a product declaring only years 2-3 would pass it. "
            "check_precondition() adds this declaration's covering rule for both and rejects "
            "such a product for their year-1 months; the frozen scorers themselves are not "
            "edited, so the scoring session must run check_precondition() before scoring.",
            "N2 and N2C have no precondition code (their scorers are not written); this "
            "declaration's covering rule applies to them, and their scorers must call it.",
        ],
        "tests": "tests/test_gov_heads_window.py",
    }


def draft_declaration(prereg_dir: Path = PREREG) -> dict:
    """The draft declaration (not registered): ``declaration_core()`` plus the station-list
    hashes read from the registry. Hashes and counts only. Needs results/dataprep/
    preregistration (gitignored)."""
    d = declaration_core()
    out = {}
    for k, v in d.items():
        out[k] = v
        if k == "template":
            out["station_lists"] = station_lists(prereg_dir)
    return out


def write_json(path: Path, obj: dict) -> None:
    text = json.dumps(obj, indent=1, ensure_ascii=False) + "\n"
    if _STATION.search(text):
        raise ValueError(f"{path.name} would hold a station id")
    Path(path).write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template")
    d = sub.add_parser("draft")
    d.add_argument("--prereg-dir", default=str(PREREG))
    c = sub.add_parser("check")
    c.add_argument("product_dir")
    a = ap.parse_args(argv)
    _paths()
    if a.cmd == "template":
        write_json(TEMPLATE, template())
    elif a.cmd == "draft":
        write_json(DRAFT, draft_declaration(Path(a.prereg_dir)))
    else:
        from common.manifest import load_manifest

        m, _ = load_manifest(a.product_dir)
        res = {"governance": check_governance(m) or "ok"}
        res.update({k: check_precondition(k, a.product_dir) for k in PRECONDITIONS})
        print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
