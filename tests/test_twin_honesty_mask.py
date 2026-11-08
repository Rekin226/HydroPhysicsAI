"""The grader-view mask of user ruling R11.4 (hydrophysics/twin/honesty_mask.py), the masked
view (honesty.grader_view) and the grader workspace script
(dataprep_staging/honesty/make_grader_workspace.py), on synthetic statements and a synthetic
repository only.

CPU only, standard library plus the package; the workspace test needs git and tar and skips
without them. No real data is read: the real-view check skips when the gitignored pinned page
is absent (CI).
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from hydrophysics.twin import claims as C
from hydrophysics.twin import honesty as H
from hydrophysics.twin import honesty_mask as M

HON, SIX = M.PLACEHOLDERS["honesty"], M.PLACEHOLDERS["six_part"]
SCRIPT = C.REPO / "dataprep_staging" / "honesty" / "make_grader_workspace.py"


def mask(text: str) -> str:
    return M.mask_text(text)[0]


# ---- the rule on single statements ----------------------------------------------------------
@pytest.mark.parametrize("text, gone, kept", [
    ("The audited figure is **96.4** (page 97.1, README 95.5, evidence 99.2, ledger 91.8).",
     ["96.4", "97.1", "95.5", "99.2", "91.8"], []),
    ("Claim honesty: 93 % audited, about 96 estimated after step 0; now 96.4 (R9.3), section 27.",
     ["93 %", "96 ", "96.4"], ["step 0", "R9.3", "section 27"]),
    ("Progress figures (6-part 52.71 with claim honesty 96.4, science trust 44.0, head part 48.8).",
     ["52.71", "96.4"], ["44.0", "48.8"]),
    ("The official science trust (44.9) and 6-part mean (52.2) are within 0.3 of the audit's.",
     ["52.2", "0.3"], ["44.9"]),
    ("| **6-part mean** (headline) | 51.6 | **52.4** | 52.0-52.2 |", ["51.6", "52.4", "52.0"], []),
    ("| end-goal convention, science / 6-part | 55.6 / 62.0 |", ["55.6", "62.0"], []),
    ("It gives about **60-61 %** on the 6-part basis.", ["60-61"], []),
    ("Expected about -0.21 to +0.12 headline; science -0.30.", ["-0.21", "+0.12"], ["-0.30"]),
    ("The headline 52.7 stands.", ["52.7"], []),
    ("the primary graders give 97.7 and the second grader 96.2, so read it as about +-0.5",
     ["97.7", "96.2", "+-0.5"], []),
    ("聲明誠實度為 96.4%，六部分平均 52.7，科學可信度 44.0。", ["96.4", "52.7"], ["44.0"]),
    ("Official figures: 6-part 52.6 -> 52.1, science trust 44.6 -> 44.2 (from 53.0, 45.1 "
     "before the ruling).", ["52.6", "52.1", "53.0"], ["44.6", "44.2", "45.1"]),
])
def test_figures_in_context_are_masked(text, gone, kept):
    out = mask(text)
    for g in gone:
        assert g not in out, (g, out)
    for k in kept:
        assert k in out, (k, out)
    assert HON in out or SIX in out


@pytest.mark.parametrize("text", [
    "The run-spread band covered 96.4 % of 2023-26 observations at a nominal 90 %.",
    "Coverage 98.3 % of the window's well-months (model-only).",
    "Yunlin: 75 wells, RMSE 1.38 against 2.21 m; KGE 0.59.",
    "The headline lever cuts all pumping 30% (all30), 1.2 m by Dec 2030.",
    "The township leader of the earlier headline (no township is named), leads in 83 % of 36 "
    "runs.",
    "honesty = 100 x (1 - sum(w s) / (3 sum(w))), contradiction 3, overclaim 2, none 0",
    "Audited on 2026-11-07 under R9.3 (evidence section 27, sha256 a4907c37).",
    "the head part 46.9 at the audit state and science trust 44.2",
    "the 6-part headline and the intended-output variant (W and L graded 0), divided by 6",
    "this._startedAt=0,this._progress=0,this._connected=!1",
    "經檢討本方案執行進度、下陷情勢並依據 95 年 5 月 30 日會議決議",
    "the graders' rule: grade 0-3, an audit figure needs evidence",
])
def test_unrelated_numbers_are_left(text):
    assert mask(text) == text


def test_placeholder_names_the_owning_term():
    out = mask("6-part 52.7 with claim honesty 96.4, science trust 44.0")
    assert out == f"6-part {SIX} with claim honesty {HON}, science trust 44.0"


def test_block_context_reaches_a_later_sentence_of_the_same_block_only():
    text = ("- **Headline: the 6-part mean**, the five parts and claim honesty. Audited it is "
            "**52.6 %**.\n- Another item: 52.6 % of wells.\n\nA paragraph: 52.6 % of wells.")
    out, n = M.mask_blocks(text)
    assert n == 1 and out.count("52.6 %") == 2 and "**" + HON + "**" in out
    before = "- **Headline: the 6-part mean**, the five parts and claim honesty."
    assert M.mask_text("Audited it is **52.6 %**.", before + " ")[0] == f"Audited it is **{HON}**."


def test_json_masks_values_and_numbers_by_key_path_keeping_the_format():
    raw = ('{\n "claim_honesty": 96.4,\n "six_part": {"official": 52.71, "end_goal": 62.9},\n'
           ' "science": {"official": 44.0},\n "parts": [48.8, 50.1],\n'
           ' "decision": "re-audit central 96.4-96.9; science 0.",\n "n": 1669\n}\n')
    out, n = M.mask_json_text(raw)
    obj = json.loads(out)
    assert obj["claim_honesty"] == HON and obj["six_part"] == {"official": SIX, "end_goal": SIX}
    assert obj["science"] == {"official": 44.0} and obj["parts"] == [48.8, 50.1]
    assert obj["n"] == 1669 and "96.4" not in obj["decision"] and n == 4
    assert out.count("\n") == raw.count("\n")                  # formatting kept
    assert M.mask_json_text('{"a": "nothing here", "b": 3}')[1] == 0
    assert M.mask_file_text("x.bin", "honesty 96.4") is None


def test_rules_are_hashed_and_deterministic():
    assert M.rules_sha256() == H.sha_of(M.rules())
    assert len(M.rules_sha256()) == 64
    text = "The audited figure is 96.4 (6-part 52.7, science 44.0)."
    assert mask(text) == mask(text)
    assert set(M.rules()["terms"]) == {"H", "S", "K", "A", "O", "Q"}


def test_a_quoted_six_key_masks_only_honesty_range_figures_of_its_block():
    pinned = ('PIN = {"honesty": 93.0, "parts": (48.1, 51.2), "science": (44.18, 37.2),\n'
              '       "six": (52.88, 62.4)}')
    out = mask(pinned)
    assert "93.0" not in out and "52.88" not in out and "62.4" not in out and "51.2" not in out
    assert "44.18" in out and "48.1" in out
    words = 'NUM = {"five": 5, "six": 6, "seven": 7, "ten": 10, "twenty": 20}'
    assert mask(words) == words


# ---- the view cannot give honesty back ------------------------------------------------------
H_TRUE, SCI = 96.4, 44.18
SIXP = (5 * SCI + H_TRUE) / 6                                   # 52.8833...
H_OLD, SCI_OLD = 93.0, 45.27
SIX_OLD = (5 * SCI_OLD + H_OLD) / 6


def _nums(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text)]


def _synthetic_statements() -> list[str]:
    return [
        f"The audited figure is **{H_TRUE:.1f}** (page 97.0, README 95.9, ledger 92.1).",
        f"No status changes (6-part {SIXP:.1f} with claim honesty {H_TRUE:.1f}, science trust "
        f"{SCI:.1f}, forecast part 51.3).",
        f"| **science trust** (mean of the 5) | {SCI_OLD:.1f} | **{SCI:.2f}** |",
        f"| **claim honesty** | {H_OLD:.0f} | **{H_TRUE:.1f}** |",
        f"| **6-part mean** (headline) | {SIX_OLD:.1f} | **{SIXP:.2f}** |",
        f"The official science trust ({SCI:.2f}) and 6-part mean ({SIXP:.2f}) are within 0.1 "
        f"of the audit's ({SCI_OLD:.1f}, {SIX_OLD:.1f}).",
        f"The 6-part mean rises {SIXP - SIX_OLD:.2f}: +{(H_TRUE - H_OLD) / 6:.2f} is the "
        f"claim-honesty input and {5 * (SCI - SCI_OLD) / 6:+.2f} is science.",
        f"聲明誠實度 {H_TRUE:.1f}%，六部分 {SIXP:.2f}，科學 {SCI:.2f}。",
        "The band covered 75 % of observations; 158 wells; RMSE 2.05 m against 1.67 m.",
    ]


def test_masked_statements_cannot_give_honesty_back():
    plain = _synthetic_statements()
    nums_plain = [x for t in plain for x in _nums(t)]
    # the unmasked text gives it away: a 6-part figure beside a science figure
    assert any(abs(6 * a - 5 * b - H_TRUE) < 0.1
               for a, b in itertools.permutations(nums_plain, 2))
    masked = [mask(t) for t in plain]
    nums = [x for t in masked for x in _nums(t)]
    for target in (H_TRUE, H_OLD):
        assert all(abs(x - target) > 0.06 for x in nums)                     # not printed
        assert not any(abs(6 * a - 5 * b - target) < 0.15                    # not derivable
                       for a, b in itertools.permutations(nums, 2))
        assert not any(abs(6 * d - 5 * e - (H_TRUE - H_OLD)) < 0.05          # nor its change
                       for d, e in itertools.permutations(nums + [-x for x in nums], 2)
                       if abs(d) < 2 and abs(e) < 2)
    assert all(abs(x - SIXP) > 0.06 and abs(x - SIX_OLD) > 0.06 for x in nums)
    assert any(abs(x - SCI) < 0.01 for x in nums)                 # science trust stays checkable


# ---- the view of an inventory ---------------------------------------------------------------
def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


PAGE = """<html><body><h1 data-i18n="title"></h1>
<script id="payload" type="application/json">{"x": 1}</script>
<script>
const T = {
en: {
  title: "Head response", hon: "Claim honesty {h} (audited); 6-part mean 52.9 beside science 44.2.",
},
zh: {
  title: "水位反應", hon: "聲明誠實度 {h}（稽核）；六部分平均 52.9，科學 44.2。",
},
}};
function tr(k) { return T.en[k]; }
function r() { return tr("title") + tr("hon"); }
</script></body></html>
"""
README = """# Project

The model shows a head response of 1.2 m in Yunlin (C01).

- **Headline: the 6-part mean**, the five parts and claim honesty, as reported.
  Audited it is **52.6 %** (shown beside the official figure).
"""
EVIDENCE = """# Evidence

## 21. Rule

The audited figure is **96.4** (page 97.0, README 95.9); 75 wells were scored.

## 27. Round 9

No status changes (6-part 52.9 with claim honesty 96.4, science trust 44.2).
"""
LEDGER = {"status_note": "Claim-honesty audit: audited 96.4 on 2026-11-07.\n\nKGE 0.59.",
          "claims": [{"id": "C01", "status_current": "S", "text_keys": ["i18n:hon"],
                      "wording_en": "Head response supported in Yunlin (75 wells).",
                      "notes": "", "doc_anchors": {},
                      "cites": {"registrations": [], "tests": [], "reports": []}}],
          "display_map": {"i18n": {"hon": ["C01"]}}, "registry_hashes": {}}
BRIEF = "# Brief\n\nGrade 0-3; stay blind to any claim-honesty figure.\n"


def _tree(root: Path) -> None:
    _write(root / "README.md", README)
    _write(root / "MODEL_CARD.md", "# Card\n\nLevers are withdrawn under every outcome.\n")
    _write(root / H.EVIDENCE, EVIDENCE)
    _write(root / H.LEDGER_PATH, json.dumps(LEDGER, indent=1))
    _write(root / "map.json", json.dumps({"answer_tiles": [], "headline_functions": []}))
    _write(root / "dataprep_staging" / "honesty" / "GRADER_BRIEF.md", BRIEF)


def _inventory(root: Path, page: Path) -> dict:
    return H.build_inventory(page, root, map_path=root / "map.json")


def test_grader_view_masks_by_rule_and_keeps_everything_else(tmp_path):
    _tree(tmp_path)
    _write(tmp_path / "page.html", PAGE)
    inv = _inventory(tmp_path, tmp_path / "page.html")
    view = H.grader_view(inv, tmp_path)
    assert view["schema"] == H.VIEW_SCHEMA and view["sha256"] == inv["sha256"]
    assert view["mask_rules_sha256"] == M.rules_sha256() == H.sha_of(view["mask_rules"])
    assert view["view_sha256"] == H.sha_of(view["records"]) != inv["sha256"]
    assert [r["id"] for r in view["records"]] == [r["id"] for r in inv["records"]]
    for a, b in zip(inv["records"], view["records"], strict=True):
        assert {k: v for k, v in a.items() if k != "text"} == \
            {k: v for k, v in b.items() if k != "text"}
    text = json.dumps([r["text"] for r in view["records"]], ensure_ascii=False)
    for fig in ("96.4", "97.0", "95.9", "52.9", "52.6"):
        assert fig not in text, fig
    assert "44.2" in text and "75 wells" in text and "1.2 m" in text and "0.59" in text
    by = view["masked"]["by_source"]
    assert by["page"] == ["page:hon"] and len(by["evidence"]) == 2
    assert len(by["readme"]) == 1 and by["ledger"] == ["ledger:status_note:0"]
    # the README's second sentence is masked only through its list item's context
    alone = H.grader_view(inv)
    assert alone["masked"]["by_source"]["readme"] == []
    assert H.grader_view(inv, tmp_path) == view                      # deterministic
    with pytest.raises(H.FindingsError):                             # a view is not an inventory
        p = tmp_path / "view.json"
        p.write_text(json.dumps(view), encoding="utf-8")
        H.load_inventory(p)
    _write(tmp_path / "README.md", README + "\nMore.\n")
    with pytest.raises(H.InventoryError, match="not the inventory's input"):
        H.grader_view(inv, tmp_path)


def test_view_cli(tmp_path, capsys):
    _tree(tmp_path)
    _write(tmp_path / "page.html", PAGE)
    inv = _inventory(tmp_path, tmp_path / "page.html")
    ip = tmp_path / "inv.json"
    ip.write_text(json.dumps(inv), encoding="utf-8")
    assert H.main(["view", "--inventory", str(ip), "--out", str(tmp_path / "v.json"),
                   "--root", str(tmp_path)]) == 0
    assert "mask rules " + M.rules_sha256() in capsys.readouterr().out
    assert json.loads((tmp_path / "v.json").read_text())["sha256"] == inv["sha256"]


# ---- the workspace --------------------------------------------------------------------------
def _script():
    spec = importlib.util.spec_from_file_location("make_grader_workspace", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    *args], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None or shutil.which("tar") is None,
                    reason="needs git and tar")
def test_workspace_holds_the_view_brief_and_record_only(tmp_path):
    W = _script()
    repo = tmp_path / "repo"
    _tree(repo)
    _write(repo / ".gitignore", "*.local.md\nresults/\n")
    _write(repo / "docs" / "analyst" / "EVAL_2026-01-01.md", "analyst eval 98 %\n")
    _write(repo / "tests" / "test_x.py", "def test():\n    honesty = 96.4\n"
                                         "    six_part = (52.9, 44.2)\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "synthetic")
    # untracked, gitignored: local notes, the record, earlier audit outputs
    _write(repo / "PLAN.local.md", "audited 96.4\n")
    _write(repo / "docs" / "ROUND_PLAN.local.md", "plan\n")
    _write(repo / "results" / "twin" / "page.html", PAGE)
    _write(repo / "results" / "dataprep" / "rulings" / "r.json",
           json.dumps({"rulings": [{"id": "R1", "decision": "re-audit central 96.4-96.9."}]}))
    _write(repo / "results" / "dataprep" / "reports" / "rep.json",
           json.dumps({"six_part": {"official": 52.9}, "science": {"official": 44.2}}))
    _write(repo / "results" / "dataprep" / "reports" / "HONESTY.round9" / "score.json", "{}")
    _write(repo / "results" / "dataprep" / "reports" / "x" / "honesty_findings_a.json", "{}")
    _write(repo / "results" / "analyst" / "answers.json", "{}")
    inv = _inventory(repo, repo / "results" / "twin" / "page.html")
    ip = tmp_path / "inv.json"
    ip.write_text(json.dumps(inv), encoding="utf-8")
    recs = ("results/dataprep/rulings", "results/dataprep/reports", "results/twin",
            "results/analyst", "results/missing")

    with pytest.raises(W.WorkspaceError, match="outside the repository"):
        W.make_workspace(ip, repo / "ws", repo, records=recs)
    dest = tmp_path / "ws"
    res = W.make_workspace(ip, dest, repo, records=recs, check_literals=("96.4",))

    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    top = sorted(p.name for p in dest.iterdir())
    assert top == ["GRADER_BRIEF.md", "inventory.json", "repo"]
    assert not any(".git" in f.split("/") or f.endswith(".local.md") or "HONESTY" in f
                   or "honesty_findings" in f or "analyst" in f for f in files), files
    assert "repo/results/dataprep/rulings/r.json" in files
    assert "repo/results/twin/page.html" in files and "repo/README.md" in files
    assert res["records"]["results/missing"] == "missing"
    assert (dest / "GRADER_BRIEF.md").read_bytes() == BRIEF.encode()
    view = json.loads((dest / "inventory.json").read_text(encoding="utf-8"))
    assert view["schema"] == H.VIEW_SCHEMA and view["sha256"] == inv["sha256"]
    assert view["mask_rules_sha256"] == res["mask_rules_sha256"] == M.rules_sha256()
    assert res["literal_hits"]["96.4"] == {}, res["literal_hits"]
    rep = json.loads((dest / "repo/results/dataprep/reports/rep.json").read_text())
    assert rep == {"six_part": {"official": SIX}, "science": {"official": 44.2}}
    assert "52.9" not in (dest / "repo/tests/test_x.py").read_text()
    assert "repo/README.md" not in res["masked_files"]          # paths are relative to repo/
    assert {"README.md", H.EVIDENCE, H.LEDGER_PATH} <= set(res["masked_files"])
    with pytest.raises(W.WorkspaceError, match="not empty"):
        W.make_workspace(ip, dest, repo, records=recs)


# ---- the real inventory (local only) --------------------------------------------------------
REAL_PAGE = C.REPO / "results" / "twin" / "twin_app.html"


@pytest.mark.skipif(not REAL_PAGE.is_file(), reason="gitignored pinned page absent (CI)")
def test_real_view_leaves_no_six_part_beside_science():
    try:
        inv = H.build_inventory(REAL_PAGE)
        view = H.grader_view(inv, C.REPO)
    except H.InventoryError as e:                    # a working tree between pins
        pytest.skip(str(e))
    for r in view["records"]:
        for t in filter(None, r["text"].values()):
            assert not re.search(r"(?i)(6-part|six-part|honesty)[^.;|]{0,40}?\b(9\d|5\d)\.\d",
                                 t), (r["id"], t)


def test_anchored_term_scan_equals_the_full_scan():
    text = ("a department; 6-part and six_part, re-audited, the audited figures, \"six\": 1, "
            "claim_honesty, graders, headline head lever, headline 52, science-trust, parts, "
            "kappa, coverage, agreement, 誠實度 六部分 總分 頭條槓桿 科學")
    full = [(m.start(), m.end(), m.lastgroup[0]) for m in M._TERM.finditer(text)]
    assert M._terms(text) == full and len(full) == 17
    assert M._terms("x" * 100000 + " 6-part 52.1") == [(100001, 100007, "S")]
