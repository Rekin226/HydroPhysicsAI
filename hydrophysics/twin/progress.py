"""Progress figures recomputed from the claims ledger by the rule of section 21.

``docs/TWIN_EVIDENCE_2026-10.md`` section 21 fixes how the project's progress figures come from
the claims ledger (``claims_v1.json``): a grade per claim from its status, a weight per claim
from where it is shown, a part score as the weighted mean grade in percent, science trust as
the mean of the five science parts and the headline as the mean of those five and claim
honesty. This module applies that rule. It reads nothing but the ledger and the committed
mapping ``progress_map_v1.json``; no observation value, report or registry file is opened.

What comes from where:

* PART: the mapping file (claim id -> one of the five science parts, or ``none`` for rows that
  make no science claim: UI rows, the model's configuration line, release and bookkeeping
  rows). A ``none`` row is listed but counted in no part.
* WEIGHT: derived from the ledger, never from free text (``derived_weight``). 0 for the
  ledger's non-claim rows (ids ``Z*``: UI, legends, 3D drawer, unread payload) and for rows
  the mapping marks as a mirror; 2 for a row on an answer tile (``display_map.tiles`` of
  the mapping's ``answer_tiles``, the template's ``TILE_IDS[:PRIMARY]``) or in a headline
  function (``display_map.functions['renderHeadline']``); 1 for any other row the page shows
  (any ``tile:``, ``i18n:`` or ``card:`` text key, or any page section of ``display_map``);
  0.5 for a row shown only in the README, the model card or the evidence update (``doc:``
  text keys or ``doc_anchors`` only); 0 for a row shown nowhere. The mapping may override a
  derived weight, and every override carries its reason (``weight_reason``).
* GRADE: the status code's grade (section 21); S and D codes take the qualifier the mapping
  gives for the row's exact status string (``grade[status][code]``). A D qualifier may be
  conditional, ``{"measured_by": REG, "otherwise": "D-unknown"}``: measured exactly when the
  ledger's ``registry_hashes`` cites REG at that ledger state with a real hash (a literal
  ``PENDING`` hash refuses). REG is the registration of a decidability reading; when the
  reading ships inside the test's own registration the spec also names it
  (``reading_report``, with its ``reading_sha256`` for the record), and the row must list
  that report in its ``cites.reports`` (E30). A reading with pre-declared outcomes
  (``reading_outcomes: {outcome: grade key}``, user ruling R9.4: E02.decidability.v1's
  measured / weak / unchanged) grades the row by the outcome the row itself records in
  ``decidability_reading: {registration, report, outcome}``, and only when the ledger cites the
  registration with a real hash and the row lists the report; until then ``otherwise``. A
  recorded outcome without both citations, both citations without a recorded outcome, or an
  undeclared outcome refuses, so a cited reading always decides, whichever way it went.
* SPLIT rows (user rulings R8.16b, R8.17a, R8.17d): a row the mapping marks
  ``split_of: {parent: P, parent_status: S0, piece_status_at_split: s_i}`` is a piece of the
  ledger row P, whose status was S0 when it was split; the piece's status then was s_i.
  Weight: the parent's display weight W_P is the largest display weight among the pieces
  present at the ledger state, and piece i weighs w_i = W_P * d_i / sum_j d_j (d: each
  piece's own display weight). Grade: g_P is P's grade at S0 under the composite-first rule,
  from P's own mapping entry; m0 is the pieces' weighted mean grade at their at-split
  statuses s_i (unconditional qualifiers, so m0 is fixed by the mapping) and m1 at their
  current statuses; the split's own uplift is u = max(0, m0 - g_P), and if u > 0 every
  piece's grade is multiplied by max(0, m1 - u) / m1, so the pieces' weighted mean is m1 - u.
  So the pieces together weigh exactly W_P; at the statuses of the split their mean grade is
  at most g_P (a split cannot raise its part's weight or grade); and a later status change of
  a piece moves the part by exactly that change's weighted grade (later evidence counts in
  full). This implements R8.17a's stated intent; the literal reading "never above g_P" would
  freeze later evidence too. The uniform scaling can show a piece below its own grade; that is
  display only, since the part score depends on the pieces' weighted sum alone, the same as
  capping only the piece above the parent. A split is in effect at a ledger state when a piece
  other than P itself is present; before that only P is present and is weighted and graded as
  P.
* COMPOSITE statuses (``S+D``, ``N+D``, ``W+D``, ``W+L``): a W anywhere grades the row W (a
  withdrawn row asserts nothing, and its pending decider does not restore it); otherwise the
  official rule (``first``, user ruling R8.16a) grades the first code, the status of what the
  row shows now (the trailing D is the pending decider). ``max`` (the highest code, the rule of
  mapping ``default-v1``) is kept for comparison. Every composite row is flagged.

The official mapping is the map file's base (user rulings 2026-11-07, fourteenth set,
R8.16a/b/c; fifteenth set, R8.17a-d); ``--mapping default-v1`` lays the recount's first default
over it for comparison.

Which ledger states can be recounted: a state refuses (``MappingError``) when a claim id has no
mapping entry, or when a claim's status carries an S or D code the mapping gives no qualifier
for at that exact status string (a new S/D-coded status without a qualifier refuses; a change
to an N, L or W status, or to an S/D status already mapped, does not), or when a
``measured_by`` registration or an end-goal ``requires`` registration is cited with a literal
``PENDING`` hash. Every main state from fd40796 (the 2026-11-02 audit state) to 08a6aa7 is
recounted without refusal (``tests/test_twin_progress.py``); earlier states are not covered.

Reported beside each other (never one alone): the five parts, science trust, the 6-part
headline (only when a claim-honesty figure is supplied: claim honesty is how well the page,
the ledger and the docs agree with the record, which the ledger alone cannot measure), the
intended-output variant (W and L graded 0) and the end-goal convention (rows the mapping marks
withdrawn-with-reason, registered-undecidable or no-target graded 1.0; bookkeeping, not
evidence, ruling R8.12b).

CLI: ``python -m hydrophysics.twin.progress [--ledger PATH] [--at COMMIT-or-file] [--json]
[--honesty PCT] [--mapping NAME] [--composite max|first]``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEDGER = HERE / "claims_v1.json"
MAP = HERE / "progress_map_v1.json"
LEDGER_IN_REPO = "hydrophysics/twin/claims_v1.json"
WEIGHTS = (2.0, 1.0, 0.5, 0.0)
NONE = "none"
D_PENDING_AFTER = "2030-01-15"
END_GOAL_KINDS = ("withdrawn_with_reason", "registered_undecidable", "no_target")
PAGE_SECTIONS = ("i18n", "card_rows", "functions", "arrays", "payload")


class MappingError(ValueError):
    """The mapping does not cover the ledger (missing id, missing qualifier, bad value)."""


# ---- inputs ---------------------------------------------------------------------------------
def load_map(path: Path = MAP) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_ledger(path: Path | None = None, at: str | None = None) -> tuple[dict, str]:
    """(ledger, label). ``at`` is a file path or a git commit (the ledger as committed there);
    otherwise ``path`` (default: the ledger beside this module). PENDING hashes are not
    refused here: the recount reads statuses only."""
    if at is not None:
        p = Path(at)
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8")), str(p)
        out = subprocess.run(["git", "show", f"{at}:{LEDGER_IN_REPO}"], cwd=REPO, check=True,
                             capture_output=True, text=True).stdout
        return json.loads(out), f"{at}:{LEDGER_IN_REPO}"
    p = Path(path) if path is not None else LEDGER
    return json.loads(p.read_text(encoding="utf-8")), str(p)


OFFICIAL = "official"


def with_mapping(pmap: dict, name: str | None) -> dict:
    """The mapping named ``name``: the map file's base (the official rule) when ``name`` is
    None or ``official``, else the base with that entry of ``mappings`` laid over it (its
    per-claim fields, its ``composite``, ``split_weights`` and ``split_cap``)."""
    if not name or name == OFFICIAL:
        return pmap
    alt = pmap.get("mappings", {}).get(name)
    if alt is None:
        raise MappingError(f"no mapping {name!r} in the map file")
    out = dict(pmap, claims={k: dict(v) for k, v in pmap["claims"].items()})
    for key in ("composite", "split_weights", "split_cap"):
        if key in alt:
            out[key] = alt[key]
    for cid, fields in alt["claims"].items():
        if cid not in out["claims"]:
            raise MappingError(f"mapping {name}: unknown claim id {cid}")
        out["claims"][cid].update(fields)
    return out


# ---- weight ---------------------------------------------------------------------------------
def _ids_in(section) -> set[str]:
    ids: set[str] = set()
    if isinstance(section, dict):
        for v in section.values():
            if isinstance(v, list):
                ids.update(x for x in v if isinstance(x, str))
            elif isinstance(v, dict):
                ids |= _ids_in(v.get("paths", {}))
    return ids


def answer_ids(ledger: dict, pmap: dict) -> set[str]:
    dm = ledger.get("display_map", {})
    ids = {c for t in pmap["answer_tiles"] for c in dm.get("tiles", {}).get(t, [])}
    for f in pmap["headline_functions"]:
        ids.update(dm.get("functions", {}).get(f, []))
    return ids


def page_ids(ledger: dict) -> set[str]:
    dm = ledger.get("display_map", {})
    ids = _ids_in(dm.get("tiles", {}))
    for s in PAGE_SECTIONS:
        ids |= _ids_in(dm.get(s, {}))
    ids |= _ids_in(dm.get("payload_conditional", {}))
    return ids


def derived_weight(claim: dict, ledger: dict, pmap: dict,
                   entry: dict | None = None) -> tuple[float, str]:
    """(weight, why) from the ledger alone (see the module docstring)."""
    cid = claim["id"]
    if cid.startswith("Z"):
        return 0.0, "non-claim row (Z*)"
    if entry and entry.get("mirror"):
        return 0.0, "mirror row"
    if cid in answer_ids(ledger, pmap):
        return 2.0, "answer tile or headline"
    keys = claim.get("text_keys", [])
    if cid in page_ids(ledger) or any(k.split(":", 1)[0] in ("tile", "i18n", "card")
                                      for k in keys):
        return 1.0, "page"
    if any(k.startswith("doc:") for k in keys) or claim.get("doc_anchors"):
        return 0.5, "docs only"
    return 0.0, "shown nowhere"


# ---- grade ----------------------------------------------------------------------------------
PENDING = "PENDING"


def _cited(reg: str, registry: dict) -> bool:
    """Is ``reg`` cited in ``registry_hashes``? A literal PENDING hash refuses."""
    if registry.get(reg) == PENDING:
        raise MappingError(f"{reg} is cited with a PENDING hash: register it before recounting")
    return reg in registry


def _outcome_qualifier(spec: dict, registry: dict, claim: dict) -> tuple[str, str]:
    """A reading with pre-declared outcomes (``reading_outcomes``): the grade key of the outcome
    the row records in ``decidability_reading`` ({registration, report, outcome}), which counts
    only when the ledger cites the registration with a real hash and the row lists the report
    in ``cites.reports``. A recorded outcome without those citations, a citation pair without a
    recorded outcome, another reading's record or an undeclared outcome refuse: once a reading
    is cited, its outcome (whichever it is) decides the qualifier."""
    cid, reg, rep = claim.get("id", "?"), spec["measured_by"], spec["reading_report"]
    cited = _cited(reg, registry)
    rep_cited = rep in ((claim.get("cites") or {}).get("reports") or [])
    rec = claim.get("decidability_reading")
    if rec is None:
        if cited and rep_cited:
            raise MappingError(f"{cid}: cites {reg} and its report but records no outcome "
                               "(decidability_reading)")
        return spec["otherwise"], f"{reg}: no outcome recorded at this ledger state"
    if rec.get("registration") != reg or rec.get("report") != rep:
        raise MappingError(f"{cid}: decidability_reading names another reading than {reg}")
    if not (cited and rep_cited):
        raise MappingError(f"{cid}: records the outcome of {reg} but the ledger does not cite "
                           "both the registration and its report")
    out = rec.get("outcome")
    if out not in spec["reading_outcomes"]:
        raise MappingError(f"{cid}: outcome {out!r} is not one {reg} declared")
    return spec["reading_outcomes"][out], f"{reg} outcome {out!r}"


def _qualifier(spec, registry: dict, claim: dict | None = None) -> tuple[str, str]:
    """(grade key, why) of one S/D qualifier, resolving a conditional ``measured_by``: measured
    when the registration is cited with a real hash and, if the spec names a
    ``reading_report``, the row's ``cites.reports`` lists that decidability reading. A spec
    with ``reading_outcomes`` grades by the recorded outcome (:func:`_outcome_qualifier`)."""
    if isinstance(spec, str):
        return spec, ""
    if spec.get("reading_outcomes") is not None:
        return _outcome_qualifier(spec, registry, claim or {})
    reg = spec["measured_by"]
    if not _cited(reg, registry):
        return spec["otherwise"], f"{reg} not cited at this ledger state"
    rep = spec.get("reading_report")
    if rep and rep not in ((claim or {}).get("cites") or {}).get("reports", []):
        return spec["otherwise"], f"{rep} not cited by the row at this ledger state"
    return "D-measured", f"measured by {rep or reg}"


def grade(claim: dict, entry: dict, registry: dict, grades: dict,
          composite: str = "max") -> dict:
    status = claim["status_current"]
    codes = status.split("+")
    per = []
    for code in codes:
        if code in ("S", "D"):
            spec = entry.get("grade", {}).get(status, {}).get(code)
            if spec is None:
                if "W" in codes:          # W grades the row; its D needs no qualifier
                    continue
                raise MappingError(f"{claim['id']}: status {status!r} carries an {code} code "
                                   "the mapping gives no qualifier for")
            key, why = _qualifier(spec, registry, claim)
        elif code in ("N", "L", "W"):
            key, why = code, ""
        else:
            raise MappingError(f"{claim['id']}: status code {code!r} cannot be graded")
        if key not in grades:
            raise MappingError(f"{claim['id']}: unknown grade key {key!r}")
        per.append((code, key, grades[key], why))
    if "W" in codes:
        pick = next(p for p in per if p[0] == "W")
    elif composite == "first":
        pick = per[0]
    elif composite == "max":
        pick = max(per, key=lambda p: p[2])
    else:
        raise ValueError(f"composite rule {composite!r}")
    label = pick[1]
    dnb = entry.get("decidable_not_before")
    if pick[0] == "D" and dnb and dnb > D_PENDING_AFTER:
        label += " (D-pending)"
    return {"grade": pick[2], "grade_key": label, "graded_code": pick[0],
            "why": pick[3], "codes": [(c, k, g) for c, k, g, _ in per],
            "composite": len(codes) > 1}


# ---- the recount ----------------------------------------------------------------------------
def check_mapping(ledger: dict, pmap: dict) -> list[str]:
    """Problems that make the recount undefined; an empty list means it is defined."""
    probs = []
    parts = set(pmap["parts"]) | {NONE}
    for c in ledger["claims"]:
        e = pmap["claims"].get(c["id"])
        if e is None:
            probs.append(f"{c['id']}: no mapping entry")
            continue
        if e.get("part") not in parts:
            probs.append(f"{c['id']}: part {e.get('part')!r} unknown")
        if "weight" in e and (e["weight"] not in WEIGHTS or not e.get("weight_reason")):
            probs.append(f"{c['id']}: weight override needs a value in {WEIGHTS} and a reason")
        eg = e.get("end_goal")
        if eg and eg.get("kind") not in END_GOAL_KINDS:
            probs.append(f"{c['id']}: end_goal kind {eg.get('kind')!r} unknown")
        for status, quals in e.get("grade", {}).items():
            for code, spec in quals.items():
                if not isinstance(spec, dict) or spec.get("reading_outcomes") is None:
                    continue
                ro = spec["reading_outcomes"]
                if not spec.get("reading_report") or not isinstance(ro, dict) or not ro or \
                        any(v not in pmap["grades"] for v in ro.values()):
                    probs.append(f"{c['id']}: {status}/{code} reading_outcomes needs a "
                                 "reading_report and grade keys")
    split_parts: dict[str, set] = {}
    for cid, e in pmap["claims"].items():
        so = e.get("split_of")
        if not so:
            continue
        if not isinstance(so, dict) or not so.get("parent") or not so.get("parent_status"):
            probs.append(f"{cid}: split_of needs a parent and a parent_status")
            continue
        parent = pmap["claims"].get(so["parent"])
        if parent is None or cid not in parent.get("split_into", []):
            probs.append(f"{cid}: its parent {so['parent']} must list it in split_into")
        if pmap.get("split_cap"):
            s0 = so.get("piece_status_at_split")
            if not s0:
                probs.append(f"{cid}: split_of needs piece_status_at_split under the split cap")
            elif any(not isinstance(q, str) for q in e.get("grade", {}).get(s0, {}).values()):
                probs.append(f"{cid}: the qualifier of its at-split status {s0!r} must be "
                             "unconditional, so that the at-split grade is fixed")
        split_parts.setdefault(so["parent"], set()).add(e.get("part"))
    probs += [f"split pieces of {p} sit in more than one part: {sorted(ps)}"
              for p, ps in split_parts.items() if len(ps) > 1]
    return probs


def _apply_splits(rows: list[dict], pmap: dict, registry: dict) -> dict[str, list[dict]]:
    """Share each split parent's weight among its pieces (R8.16b, R8.17d) and remove the grade
    uplift the split itself created (R8.17a); see the module docstring."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r["split_of"]:
            groups.setdefault(r["split_of"]["parent"], []).append(r)
    for parent in [p for p, ps in groups.items() if all(r["id"] == p for r in ps)]:
        for r in groups.pop(parent):      # before the split: only P is present, graded as P
            r["split_of"] = None
    for parent, pieces in groups.items():
        w_parent = max(r["display_weight"] for r in pieces)
        dsum = sum(r["display_weight"] for r in pieces)
        for r in pieces:
            r["weight"] = w_parent * r["display_weight"] / dsum if dsum else 0.0
        if not pmap.get("split_cap"):
            continue
        status0 = pieces[0]["split_of"]["parent_status"]
        g_parent = grade({"id": parent, "status_current": status0}, pmap["claims"][parent],
                         registry, pmap["grades"], "first")["grade"]
        wsum = sum(r["weight"] for r in pieces)
        if not wsum:
            continue
        g_split = {r["id"]: grade({"id": r["id"],
                                   "status_current": r["split_of"]["piece_status_at_split"]},
                                  pmap["claims"][r["id"]], registry, pmap["grades"],
                                  pmap.get("composite", "first"))["grade"] for r in pieces}
        m0 = sum(r["weight"] * g_split[r["id"]] for r in pieces) / wsum
        m1 = sum(r["weight"] * r["grade"] for r in pieces) / wsum
        uplift = max(0.0, m0 - g_parent)
        for r in pieces:
            r["parent_grade"], r["split_uplift"] = g_parent, uplift
        if uplift > 0 and m1 > 0:
            f = max(0.0, m1 - uplift) / m1
            for r in pieces:
                r["uncapped_grade"], r["grade"] = r["grade"], r["grade"] * f
                r["intended_grade"] = 0.0 if r["graded_code"] in ("W", "L") else r["grade"]
                if not r["end_goal"]:
                    r["end_goal_grade"] = r["grade"]
                r["grade_capped"] = f
    return groups


def recount(ledger: dict, pmap: dict, honesty: float | None = None,
            composite: str | None = None) -> dict:
    composite = composite or pmap.get("composite", "max")
    probs = check_mapping(ledger, pmap)
    if probs:
        raise MappingError("; ".join(probs))
    registry = dict(ledger.get("registry_hashes", {}))
    grades = pmap["grades"]
    rows = []
    for c in ledger["claims"]:
        e = pmap["claims"][c["id"]]
        dw, dwhy = derived_weight(c, ledger, pmap, e)
        w = float(e["weight"]) if "weight" in e else dw
        g = grade(c, e, registry, grades, composite)
        eg = e.get("end_goal")
        eg_on = bool(eg) and (not eg.get("requires") or _cited(eg["requires"], registry))
        code = g["graded_code"]
        rows.append({
            "id": c["id"], "status": c["status_current"], "part": e["part"],
            "weight": w, "display_weight": w, "derived_weight": dw, "weight_basis": dwhy,
            "split_of": e.get("split_of") if pmap.get("split_weights") else None,
            "weight_override": "weight" in e and w != dw,
            "grade": g["grade"], "grade_key": g["grade_key"], "graded_code": code,
            "grade_why": g["why"], "composite": g["composite"], "grade_capped": None,
            "intended_grade": 0.0 if code in ("W", "L") else g["grade"],
            "end_goal": eg["kind"] if eg_on else None,
            "end_goal_grade": float(pmap["end_goal_grade"]) if eg_on else g["grade"],
        })
    groups = _apply_splits(rows, pmap, registry)
    parts = {}
    for p in pmap["parts"]:
        rs = [r for r in rows if r["part"] == p and r["weight"] > 0]
        wsum = sum(r["weight"] for r in rs)

        def mean(field, rs=rs, wsum=wsum):
            return 100.0 * sum(r["weight"] * r[field] for r in rs) / wsum if wsum else None

        parts[p] = {"official": mean("grade"), "intended": mean("intended_grade"),
                    "end_goal": mean("end_goal_grade"), "rows": len(rs), "weight": wsum}

    def avg(field):
        vals = [parts[p][field] for p in pmap["parts"]]
        return sum(vals) / len(vals) if all(v is not None for v in vals) else None

    sci = {f: avg(f) for f in ("official", "intended", "end_goal")}
    six = ({f: (sci[f] * len(pmap["parts"]) + honesty) / (len(pmap["parts"]) + 1)
            for f in sci if sci[f] is not None} if honesty is not None else None)
    return {"rows": rows, "parts": parts, "science": sci, "six_part": six,
            "claim_honesty": honesty, "composite_rule": composite,
            "split_groups": {k: sorted(r["id"] for r in v) for k, v in sorted(groups.items())},
            "weight_overrides": sorted(r["id"] for r in rows if r["weight_override"]),
            "composite_rows": sorted(r["id"] for r in rows if r["composite"])}


# ---- output ---------------------------------------------------------------------------------
def _f(v) -> str:
    return "-" if v is None else f"{v:.1f}"


def render(res: dict, label: str, pmap: dict) -> str:
    splits = "; ".join(f"{k}: {', '.join(v)}" for k, v in res["split_groups"].items())
    out = [f"Progress recount (rule of section 21) of {label}",
           f"composite rule: {res['composite_rule']}; weight overrides: "
           f"{', '.join(res['weight_overrides']) or 'none'}; composite rows: "
           f"{', '.join(res['composite_rows']) or 'none'}; split rows sharing a parent's "
           f"weight: {splits or 'none'}", "",
           f"{'part':28s} {'official':>8s} {'intended':>8s} {'end-goal*':>9s} {'rows':>5s}"]
    for p in pmap["parts"]:
        s = res["parts"][p]
        out.append(f"{p:28s} {_f(s['official']):>8s} {_f(s['intended']):>8s} "
                   f"{_f(s['end_goal']):>9s} {s['rows']:>5d}")
    sc = res["science"]
    out.append(f"{'science trust (mean of 5)':28s} {_f(sc['official']):>8s} "
               f"{_f(sc['intended']):>8s} {_f(sc['end_goal']):>9s}")
    if res["six_part"]:
        sx = res["six_part"]
        out.append(f"{'6-part headline':28s} {_f(sx.get('official')):>8s} {'':>8s} "
                   f"{_f(sx.get('end_goal')):>9s}   (claim honesty {res['claim_honesty']:g}, "
                   "supplied, not computed from the ledger)")
    else:
        out.append("6-part headline: not computed (claim honesty is not a ledger quantity; "
                   "pass --honesty with the audit or estimate figure)")
    out += ["* end-goal convention (R8.12b): bookkeeping only, never shown alone", "",
            f"{'id':8s} {'status':6s} {'part':26s} {'w':>4s} {'grade':>5s}  basis"]
    for r in res["rows"]:
        flags = [f for f, on in (("composite", r["composite"]),
                                 ("override", r["weight_override"]),
                                 (f"split of {(r['split_of'] or {}).get('parent')}",
                                  r["split_of"]),
                                 (f"capped x{r['grade_capped'] or 0:.3f}", r["grade_capped"]),
                                 (f"end-goal:{r['end_goal']}", r["end_goal"])) if on]
        basis = r["grade_key"] + (f" [{r['grade_why']}]" if r["grade_why"] else "")
        out.append(f"{r['id']:8s} {r['status']:6s} {r['part']:26s} {r['weight']:>4.3g} "
                   f"{r['grade']:>5.2f}  {basis}" + (f"  ({'; '.join(flags)})" if flags else ""))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hydrophysics.twin.progress",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--ledger", type=Path, default=None, help="ledger file (default: beside "
                    "this module)")
    ap.add_argument("--at", default=None, help="a commit (the ledger as committed there) or "
                    "a ledger file")
    ap.add_argument("--map", type=Path, default=MAP, help="mapping file")
    ap.add_argument("--mapping", default=None, help="'official' (default: user rulings "
                    "R8.16a/b/c) or a named mapping of the map file, e.g. default-v1")
    ap.add_argument("--composite", choices=("max", "first"), default=None,
                    help="override the mapping's composite rule (sensitivity only)")
    ap.add_argument("--honesty", type=float, default=None, help="claim honesty in percent "
                    "(an audit or estimate figure; the ledger cannot give it)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    ledger, label = load_ledger(a.ledger, a.at)
    pmap = with_mapping(load_map(a.map), a.mapping)
    res = recount(ledger, pmap, a.honesty, a.composite)
    if a.json:
        json.dump(dict(res, ledger=label, mapping=a.mapping or OFFICIAL), sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        print(render(res, f"{label} (mapping {a.mapping or OFFICIAL})", pmap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
