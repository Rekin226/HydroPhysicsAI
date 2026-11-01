"""SD.cut_response_spread.v2 (round 7 rank 9, model-only declaration): v1 with the forward's
own rheology labelling rule, scored on v1's eight sealed forwards (reused, bound by sha256).

v1 (registered 2026-11-01T14:14:36Z, sha256 0d05eb53...) ran its eight forwards (exit 0,
stdout sealed). Its single scoring stopped at step 2 (layout) with an engineering failure
(exit 2) before any cut30 quantity was formed: ``sdlib.expected_labels`` assumed every
column is labelled by its directory name, but ``hydrophysics/twin/forward.py`` at main
4c2723e (the code the forwards ran) labels them otherwise (:func:`forward_labels`).

Only that changes here. The layout check maps every declared column to its rheology index by
forward.py's own rule, verified per forward against the npz ``rheology_labels`` and the
``--vep-json`` files of the command the runner passed, and refuses on any ambiguity
(:func:`label_mapping`). Every other check, estimand, threshold and wording is v1's code
(``sdlib``, ``score_sd``), imported read-only; v1's files stay byte-identical.

This module imports sdlib only (no harness, no torch) and reads no observation. The reuse
binding reads v1's attempts.jsonl (a log of hashes and times) and hashes the npz files; no
sealed log, no sd_* CSV and no subsidence or head array is opened by anything here.
"""
from __future__ import annotations

import json
import math
import os
import shlex
from pathlib import Path

import sdlib as L

SD2_ID = "SD.cut_response_spread.v2"
V1_ID = L.SD_ID
V1_SHA256 = "0d05eb53e281de873662dd79c29a809948066ddcf91f25525e5064e729974d48"
V1_CREATED = "2026-11-01T14:14:36.573335Z"
V1_DECL = f"results/dataprep/preregistration/{V1_ID}.json"
V1_REPORT = f"results/dataprep/reports/{V1_ID}/report.json"
#: the lead's closure record of v1 (status engineering_failure_v1_closed, 2026-11-01T15:15Z)
V1_CLOSURE = f"results/dataprep/reports/{V1_ID}/engineering_failure.json"
REPORT = f"results/dataprep/reports/{SD2_ID}/report.json"
#: v1's frozen code as registered (v2 imports sdlib/score_sd read-only; run_sd.sh ran the
#: forwards v2 reuses): these files must stay byte-identical
V1_FROZEN = {
    f"{L.X}/run_sd.sh": "576ea351eaed84248ccdace16c1d5aefb4ccde080d21515664c999ff86f3ebf0",
    f"{L.X}/score_sd.py": "805740b34fb5a87fa7444ec7f3b9aeddbfa59786a3e7ebd2e291f043a54d8ec1",
    f"{L.X}/sdlib.py": "901c07fe9ff60e5c300cad1e867d68604d1adbee4bd1104c0906a1bccac85ff8",
}
V1_ATTEMPTS = f"{L.LOG}/attempts.jsonl"
V2_ATTEMPTS = f"{L.LOG}/attempts_v2.jsonl"

# --------------------------------------------------------------------------- reuse binding
#: v1's eight forwards as recorded by run_sd.sh in attempts.jsonl (each key's first and only
#: forward, exit 0, worktree at main 4c2723e). v2 reuses exactly these outputs.
_WT_HEAD = "4c2723e87da3df822db4c73472c2fbf87796d533"
_V1_FWD_ROWS = {
    # key: (start, time, npz, summary.csv, members.csv, sealed stdout) sha256
    "page": ("2026-11-01T14:52:06Z", "2026-11-01T14:54:00Z",
             "09beddf71b2871ee9561a1f1eaf7a5e37c378dbac2f6d6be89bf5b3099760f5e",
             "244951e2f7e34e6da6c5c123101a3cc3867b3a8b98deadd4ceb3a90ab5b6e285",
             "3cc30b68d2ec56ba849fcd0b1839cea06dcf36ccdca3549cb4c5984b4ea20dfa",
             "ecb6e3188a63c648151783dbfc184d275c06ea2004cdecaa686b4386e6d3394d"),
    "q0": ("2026-11-01T14:54:00Z", "2026-11-01T14:55:55Z",
           "44ee0e6eb62001a67fd653ab7c45a86093ffa36513efbf08ca7a16c7d66ba858",
           "9df46d11cc5de49d28f685cac1d7367470b84f80ed07c54985a40f05fd51193d",
           "ff90134a323df3e33cbc1335e09f5f0fc01913407e4c265b134a9cfe8ced6be3",
           "b5b50fc24c55b12f48fc9ad3e0795e0538e4bf26b962fd8ee2a94f9a061f0e8b"),
    "q0cs": ("2026-11-01T14:55:55Z", "2026-11-01T14:57:50Z",
             "a9dd2e17e9eb99eb9c88c61cd826d66235b1a336f9bedfdf1dfdf279d4b7ac91",
             "079b741ebcd19c64f9ca7d1bd77b010dc675409d0fadeb073f69e038bfed6b4f",
             "3db4065db3a1b0b93e7dbd1ca77ed751b44f9594f1f266d20eddd68438b2a494",
             "90dece1c81c047a0fa20eaf2e9fef7ac48a31537efba76a7abf35d7122213a39"),
    "cwa": ("2026-11-01T14:57:50Z", "2026-11-01T14:59:44Z",
            "02ae2586940654aa59fadc69135f8041f8e4322bf7cb5c18bf42b170bf1c0079",
            "3385959cda498ef3d3d0b9457d93ab90979f5e19a47024da1c50da9ea1271464",
            "491a6911e3767dca77cf6cb8a3f18ac6a26f9150c916250b780f4e4b90dd5fee",
            "5abc519b05aa6c2bd6520d2bd85b01928604f598956e14ea7d688fe2ca4821cf"),
    "q0nh": ("2026-11-01T14:59:44Z", "2026-11-01T15:01:42Z",
             "722c5cd5fcd8fc842ca9eea6353d1685a624b98432cfa54fdce808d880cd8ea4",
             "f3ecc2a06a0df70c2fa0781e46d55dac1329c42a706bba1bdb7eb654c584c86c",
             "34d0840709a3c0122e56b39b15fff0eb9115f4a3c5e85d423b0e7413fa1a2ff9",
             "15702333a1e21f2b07856a10955a796a1c6885915cedf3ca20c34990cd87eb31"),
    "ccz": ("2026-11-01T15:01:42Z", "2026-11-01T15:04:21Z",
            "69f56c45e774a65f6b9780a3e4f179d50823ac8bdc8c6b70fb69dff940321e91",
            "8c7fa061b0f915e922c24c3266553288d44e4b1767d703a9ff343b417fbe142f",
            "e57e5c4f72dbc07f1821ef5ce4f5ed4c36b401ca250ef915215d61ad9c188f26",
            "0adba946c181ced66bb73ae6e9cf3aba0e1e48c3de8c3905931d3fe6947254db"),
    "q0c": ("2026-11-01T15:04:22Z", "2026-11-01T15:07:01Z",
            "8bb1129a79752bb798577dda02bc082219090d26060bc29b239aa21c833b772a",
            "f401de4fb4d3fcbff82e2c70e2648307d4d9b6780c19c9200415afbc2db30128",
            "758ed02339625855bf93f0af2a87c6e61d9367864c5eacccdc63f6dabee1f642",
            "e6581c24c68daa91db2de3a9544fa9fb85fcb098c56fa0d7e26a2c840e4c17c0"),
    "x5r1cs": ("2026-11-01T15:07:01Z", "2026-11-01T15:08:52Z",
               "ef56e696fb27850d7bb8b25f06f868d1dc223281ff22fc60a97132ec651c4c59",
               "48cc71fdcee70016eeccf0b262e72f0dedf78eada910e097bc2c8b8879975bc4",
               "bdd45d1b2ecb7eb1816c5ea3284593e8c6a76c86d5445a58084c09ec38d75a75",
               "dfe6d56aba7887aa849f4d6a9edbefa19de71859555326a87abce768cab1c7fc"),
}
V1_FORWARDS = {
    k: {"event": "forward", "fwd": k, "attempt": 1, "rc": 0, "code": L.CODE,
        "worktree_head": _WT_HEAD, "declaration_sha256": V1_SHA256, "start": r[0],
        "time": r[1], "npz_sha256": r[2], "summary_csv_sha256": r[3],
        "members_csv_sha256": r[4], "stdout_sealed_sha256": r[5]}
    for k, r in _V1_FWD_ROWS.items()}
#: v1's one scoring attempt (run_sd.sh score), the engineering failure
V1_SCORE_ATTEMPT = {"event": "score", "attempt": 1, "rc": 2, "time": "2026-11-01T15:09:42Z",
                    "report_sha256": "missing", "declaration_sha256": V1_SHA256}


def bind_v1_forwards(attempts: Path, repo: Path, records: dict | None = None
                     ) -> tuple[dict, list[str]]:
    """(forward key -> npz path, problems). Each key's pinned v1 record must be in
    ``attempts`` with exactly the pinned fields, no forward or archive event may touch the key
    after it, and the npz at ``sdlib.out_stem(key)`` must hash to the pinned sha256."""
    records = V1_FORWARDS if records is None else records
    if not attempts.exists():
        return {}, [f"{attempts} missing"]
    rows = [json.loads(s) for s in attempts.read_text().splitlines() if s.strip()]
    out, e = {}, []
    for k, want in records.items():
        at = [i for i, r in enumerate(rows)
              if all(r.get(f) == v for f, v in want.items())]
        if len(at) != 1:
            e.append(f"{k}: {len(at)} attempts.jsonl rows equal the pinned v1 forward record")
            continue
        later = [r for r in rows[at[0] + 1:]
                 if (r.get("event") == "forward" and r.get("fwd") == k)
                 or (r.get("event") == "archive" and k in str(r.get("keys", "")).split())]
        if later:
            e.append(f"{k}: a later {later[0]['event']} event touches the reused forward")
            continue
        p = repo / f"{L.out_stem(k)}.npz"
        got = L.sha(p)
        if got != want["npz_sha256"]:
            e.append(f"{k}: {p.name} sha256 {got[:12]} is not v1's {want['npz_sha256'][:12]}")
            continue
        out[k] = p
    return out, e


def reuse_plan(v1_forwards_doc: dict) -> dict:
    """Which v1 forwards v2 may reuse: a key whose registered v1 command (the command the v1
    runner passed, ``forwards[key]['command']`` of v1's declaration) is not exactly v2's
    command needs a rerun of that key only (v2 refuses to score until then)."""
    reuse, rerun = [], {}
    for k in L.FORWARDS:
        v1 = (v1_forwards_doc.get(k) or {}).get("command")
        v2 = L.forward_command(k)
        if v1 == v2:
            reuse.append(k)
        else:
            rerun[k] = "not in v1's declaration" if v1 is None else "v1 command differs"
    for k in v1_forwards_doc:
        if k not in L.FORWARDS:
            rerun[k] = "v1 key not declared by v2"
    return {"reuse": reuse, "rerun": rerun}


# --------------------------------------------------------------------------- label rule
#: the forward code the reused forwards ran (main 4c2723e; unchanged to main 11d6001)
FORWARD_PY = "hydrophysics/twin/forward.py"
FORWARD_PY_SHA256 = "f073eccc2126e908be968c10664d962591f048e3ebf8e5669ff782d6e6b628e6"
#: the source lines of that file that make the rule (checked verbatim by rule_source_problems)
RULE_SOURCE_LINES = (
    '        tmax = p.get("tau_max_years")',
    '        label = (f"tau{float(tmax):g}y" if tmax is not None',
    '                 else os.path.basename(os.path.dirname(path)) or os.path.basename(path))',
    "        if any(label == q[0] for q in out):",
    '            label = f"{label}_{len(out)}"',
    "    col = columns[0][1] if len(columns) == 1 else [(c[0], c[1]) for c in columns]",
    '    cols = ([(c[0], c[1]) for c in col] if isinstance(col, list) else [("column", col)])',
    '            "rheology_labels": [c[0] for c in cols],',
    "           if len(res[\"rheology_labels\"]) > 1 else {}),",
)
LABEL_RULE = (
    "forward.py at main 4c2723e: with ONE --vep-json file, main passes the bare column to run, "
    "which labels it 'column' (and the npz holds subs_mean only, no subs_mean_by_rheology); "
    "with several, load_columns labels each in --vep-json order: 'tau<tau_max_years:g>y' when "
    "the column JSON's tau_max_years is not null, else the column file's directory name (the "
    "file name if it has none); a label equal to an earlier one gets '_<its index>'. The "
    "rheology axis keeps the --vep-json order.")
AMBIGUITY_REFUSALS = (
    "the --vep-json list of the command the runner passed is not the declared columns' files "
    "in order",
    "a declared column JSON is missing or unreadable, or its tau_max_years is not a finite "
    "number or null",
    "the npz holds a different number of rheology labels than declared columns",
    "the npz labels or the rule's labels are not unique",
    "the npz labels are not the rule's labels in --vep-json order (a permutation included)",
    "a column's own label (before the index suffix) equals the final label of a column with "
    "a different own label",
    "one column labelled other than 'column', or with subs_mean_by_rheology; several columns "
    "without it")


def vep_paths(command: str) -> list[str]:
    """The ``--vep-json`` files of a forward command line, as forward.load_columns splits
    them (comma-separated, stripped, empties dropped)."""
    tok = shlex.split(command)
    at = [i for i, t in enumerate(tok) if t == "--vep-json"]
    if len(at) != 1 or at[0] + 1 >= len(tok):
        raise ValueError(f"{len(at)} --vep-json options in the command")
    return [p.strip() for p in tok[at[0] + 1].split(",") if p.strip()]


def natural_label(path: str, params: dict) -> str:
    """forward.load_columns' label of one column before the duplicate suffix."""
    tmax = params.get("tau_max_years")
    if tmax is not None:
        if isinstance(tmax, bool) or not isinstance(tmax, (int, float)) or not math.isfinite(
                float(tmax)):
            raise ValueError(f"{path}: tau_max_years {tmax!r} is not a finite number")
        return f"tau{float(tmax):g}y"
    return os.path.basename(os.path.dirname(path)) or os.path.basename(path)


def forward_labels(paths: list[str], params: list[dict]) -> list[str]:
    """The rheology labels forward.py (main 4c2723e) writes for these --vep-json files."""
    if len(paths) != len(params):
        raise ValueError("one params dict per path")
    if len(paths) == 1:
        return ["column"]
    out: list[str] = []
    for path, p in zip(paths, params, strict=True):
        label = natural_label(path, p)
        if label in out:
            label = f"{label}_{len(out)}"
        out.append(label)
    return out


def label_mapping(fwd: str, npz_labels: list[str], has_by_rheology: bool, repo: Path,
                  command: str | None = None) -> tuple[dict, list[str]]:
    """({declared column: rheology index}, problems) for one forward. ``command`` is the
    command the runner passed (v1's registered command; default v2's, identical); the
    column parameters are read from the files it names. Empty problems = unambiguous."""
    cols = list(L.FORWARDS[fwd]["columns"])
    e: list[str] = []
    try:
        paths = vep_paths(command or L.forward_command(fwd))
    except ValueError as x:
        return {}, [f"{fwd}: {x}"]
    if paths != [L.column_path(c) for c in cols]:
        return {}, [f"{fwd}: the command's --vep-json files are not the declared columns'"]
    params = []
    for c, p in zip(cols, paths, strict=True):
        try:
            params.append(json.loads((repo / p).read_text()))
        except (OSError, ValueError) as x:
            e.append(f"{fwd}: column {c} JSON unreadable ({type(x).__name__})")
    if e:
        return {}, e
    try:
        rule = forward_labels(paths, params)
        nat = [natural_label(p, q) for p, q in zip(paths, params, strict=True)]
    except ValueError as x:
        return {}, [f"{fwd}: {x}"]
    lab = [str(x) for x in npz_labels]
    if len(lab) != len(cols):
        e.append(f"{fwd}: {len(lab)} rheology labels for {len(cols)} declared columns")
    if len(set(lab)) != len(lab):
        e.append(f"{fwd}: rheology labels {lab} are not unique (ambiguous)")
    if len(set(rule)) != len(rule):
        e.append(f"{fwd}: the rule's labels {rule} are not unique (ambiguous)")
    if len(cols) > 1:
        # a shared directory name is the rule's normal case (suffix = index); a column whose
        # own label is the final label of a column with a DIFFERENT own label is not
        for i, n in enumerate(nat):
            if any(j != i and n == rule[j] and nat[j] != n for j in range(len(rule))):
                e.append(f"{fwd}: column {cols[i]}'s own label {n!r} is another column's "
                         "final label (ambiguous)")
    if lab != rule:
        e.append(f"{fwd}: rheology labels {lab} != forward.py's rule {rule}")
    if len(cols) == 1 and has_by_rheology:
        e.append(f"{fwd}: one column but the npz holds subs_mean_by_rheology")
    if len(cols) > 1 and not has_by_rheology:
        e.append(f"{fwd}: {len(cols)} columns but no subs_mean_by_rheology")
    if e:
        return {}, e
    index = {c: lab.index(rule[i]) for i, c in enumerate(cols)}
    if any(index[c] != i for i, c in enumerate(cols)):        # unreachable once lab == rule
        return {}, [f"{fwd}: rheology order differs from the --vep-json order"]
    return index, []


def check_forward_npz2(z: dict, fwd: str, repo: Path, command: str | None = None
                       ) -> list[str]:
    """v1's layout checks (sdlib.check_forward_npz, unchanged) with its one label line
    replaced by :func:`label_mapping` (empty = ok)."""
    v1_line = f"rheology labels {L.rheology_labels(z)} != {L.expected_labels(fwd)}"
    e = [x for x in L.check_forward_npz(z, fwd) if x != v1_line]
    _, le = label_mapping(fwd, L.rheology_labels(z), "subs_mean_by_rheology" in z, repo,
                          command)
    return e + le


def npz_label_metadata(path: Path) -> dict:
    """The npz's rheology_labels and whether it holds subs_mean_by_rheology (its key list):
    no subsidence or head array is read."""
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        return {"labels": [str(x) for x in z["rheology_labels"]],
                "has_by_rheology": "subs_mean_by_rheology" in z.files}


def rule_source_problems(text: str) -> list[str]:
    """Rule lines missing from forward.py's source text (empty = the rule is as pinned)."""
    lines = set(text.splitlines())
    return [f"{FORWARD_PY} lacks the rule line {s.strip()!r}" for s in RULE_SOURCE_LINES
            if s not in lines]


# --------------------------------------------------------------------------- v1's failure
#: the npz rheology labels of v1's forwards (metadata read by the v2 builder)
V1_OBSERVED_LABELS = {
    "page": ["coupled_leveling", "tau30y", "coupled_leveling_aquitard"],
    "q0": ["coupled_leveling", "coupled_leveling_1"],
    "q0cs": ["coupled_leveling", "coupled_leveling_1", "coupled_leveling_2"],
    "cwa": ["coupled_leveling", "tau30y"],
    "q0nh": ["column"], "ccz": ["column"], "q0c": ["column"], "x5r1cs": ["column"],
}
V1_FAILURE_REPORTED = (
    "forward layout/rain: page: rheology labels ['coupled_leveling','tau30y',"
    "'coupled_leveling_aquitard'] != ['coupled_leveling','coupled_leveling_tau30',"
    "'coupled_leveling_aquitard']; cwa: ['coupled_leveling','tau30y'] != "
    "['coupled_leveling','coupled_leveling_tau30']; q0nh/ccz/q0c/x5r1cs: ['column'] != "
    "['coupled_leveling'].")


def v1_failure_text(observed: dict | None = None) -> str:
    """score_sd's step-2 message rebuilt from its format and the observed labels (v1's
    scorer stdout was not archived; the lead reported it as V1_FAILURE_REPORTED)."""
    observed = V1_OBSERVED_LABELS if observed is None else observed
    bad = {k: [f"rheology labels {observed[k]} != {L.expected_labels(k)}"]
           for k in L.FORWARDS if observed[k] != L.expected_labels(k)}
    return f"ENGINEERING FAILURE (exit 2): forward layout/rain: {bad}"
