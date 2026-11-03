"""Member loader of COLF.column_family_2028.v1: SD v2 Tier A+B frozen base forwards, the
declared deduplication checked by bytes, and the prediction freeze.

Reads model outputs and registration metadata only: the SD v2 registration and report
(hash-bound; columns, forwards, tiers, checks), the eight-forward npz archives of SD v1/v2
(hash-bound; only the baseline scenario's subsidence at the unit cells, plus a whole-array
byte comparison for the dedup evidence), N3's mapping.json (pile cells) and N3S-L's frozen
cells.csv (township cell membership, geometry). No observation is read.

Usage (repository root; no GPU, no fit):
    $PY -I dataprep_staging/colfreeze/members.py check
        -> member list, dedup evidence and per-member prediction hashes (stdout, no values)
    $PY -I dataprep_staging/colfreeze/members.py freeze --out <dir>
        -> <dir>/predictions.npz and <dir>/freeze_summary.json, written once (mode 'x')
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import colf_config as C  # noqa: E402


class MemberError(RuntimeError):
    """A member, a binding or a dedup identity does not hold (no freeze)."""


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def array_sha(a: np.ndarray) -> str:
    """Canonical hash of a prediction array: little-endian float32, C order, shape-tagged."""
    a = np.ascontiguousarray(np.asarray(a, dtype="<f4"))
    h = hashlib.sha256(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def months(first: str, last: str) -> list[str]:
    y0, m0 = map(int, first.split("-"))
    y1, m1 = map(int, last.split("-"))
    out = []
    while (y0, m0) <= (y1, m1):
        out.append(f"{y0:04d}-{m0:02d}")
        y0, m0 = (y0 + 1, 1) if m0 == 12 else (y0, m0 + 1)
    return out


def _check_sha(repo: Path, rel: str, want: str) -> Path:
    p = Path(repo) / rel
    if not p.is_file():
        raise MemberError(f"{rel} is missing")
    got = sha256(p)
    if got != want:
        raise MemberError(f"{rel}: sha256 {got[:12]} != bound {want[:12]}")
    return p


# ----------------------------------------------------------------------------- SD v2
def verify_sd_v2(repo: Path = REPO) -> dict:
    """The SD v2 registration and report are the bound files, and every Tier A+B member of
    MEMBER_SPEC matches them field for field (registration metadata only)."""
    reg = json.loads(_check_sha(repo, C.SD_V2["registration"],
                                C.SD_V2["registration_sha256"]).read_text(encoding="utf-8"))
    rep = json.loads(_check_sha(repo, C.SD_V2["report"],
                                C.SD_V2["report_sha256"]).read_text(encoding="utf-8"))
    errs = []
    if rep.get("declaration_sha256") != C.SD_V2["registration_sha256"]:
        errs.append("SD v2 report does not name the bound registration")
    if not rep.get("evaluable") or rep.get("dropped_columns"):
        errs.append("SD v2 report: not evaluable or columns dropped")
    tier = sorted(rep.get("tiers", {}).get(C.SD_V2["tier_key"], []))
    if tier != list(C.TIER_AB):
        errs.append(f"SD v2 Tier A+B {tier} != MEMBER_SPEC {list(C.TIER_AB)}")
    for col, spec in C.MEMBER_SPEC.items():
        rc = reg["columns"].get(col, {})
        if not rc.get("included"):
            errs.append(f"{col}: not included in SD v2")
        for k in ("column_sha256", "tier"):
            if rc.get(k) != spec[k]:
                errs.append(f"{col}: registration {k} {rc.get(k)} != {spec[k]}")
        if rc.get("theta_sha256") != spec["theta_sha256"]:
            errs.append(f"{col}: theta_sha256 differs")
        if rc.get("forward") != spec["forward"]:
            errs.append(f"{col}: forward {rc.get('forward')} != {spec['forward']}")
        fw = rep["forwards"].get(spec["forward"], {})
        path, sha = C.FORWARDS[spec["forward"]]
        if fw.get("npz") != path or fw.get("npz_sha256") != sha or fw.get("problems"):
            errs.append(f"{col}: report forward {spec['forward']} binding differs")
        if fw.get("rheology_index", {}).get(col) != spec["rheology_index"]:
            errs.append(f"{col}: rheology index differs")
        cc = rep["columns_checks"].get(col, {})
        if not (cc.get("pass") and cc.get("paired_theta")):
            errs.append(f"{col}: SD v2 column checks did not pass")
        want_label = rep.get("unverified_columns", {}).get(col)
        if bool(want_label) != bool(spec["label"]):
            errs.append(f"{col}: unverified label mismatch")
    if rep.get("distinct_of", {}).get("q0_s29") != "q0":
        errs.append("SD v2 distinct_of no longer maps q0_s29 to q0")
    if errs:
        raise MemberError("; ".join(errs))
    return {"registration_sha256": C.SD_V2["registration_sha256"],
            "report_sha256": C.SD_V2["report_sha256"],
            "distinct_of": rep.get("distinct_of"),
            "unverified_columns": {k: v for k, v in rep.get("unverified_columns", {}).items()
                                   if k in C.MEMBER_SPEC}}


# ----------------------------------------------------------------------------- forwards
def load_forward(path, key: str) -> dict:
    """Layout checks and the baseline subsidence (rheology, cells, months), metres."""
    with np.load(path, allow_pickle=False) as z:
        names = [str(s) for s in z["scenario_names"]]
        labels = [str(d)[:7] for d in z["dates"]]
        origin = labels[int(z["origin"])]
        opts = json.loads(str(z["forward_options"])) if "forward_options" in z.files else {}
        errs = []
        if tuple(names) != C.SCENARIOS:
            errs.append(f"scenarios {names}")
        if (labels[0], labels[-1]) != C.FORWARD_DATES or len(labels) != 252:
            errs.append(f"dates {labels[0]}..{labels[-1]}")
        if origin != C.ORIGIN_MONTH:
            errs.append(f"origin {origin}")
        if float(z["gain"]) != 0.0 or int(z["n_members"]) != 1:
            errs.append("gain or member count")
        if opts.get("column_heads") != "free" or opts.get("extend_record_to"):
            errs.append(f"forward options {opts}")
        if key == "q0nh":
            if (float(z["hindcast_gain"]) != C.Q0NH_FORWARD["hindcast_gain"]
                    or int(z["hindcast_every"]) != C.Q0NH_FORWARD["hindcast_every"]):
                errs.append("q0nh: not the hindcast-nudged origin-2022-12 forward")
        elif float(z["hindcast_gain"]) != 0.0:
            errs.append("hindcast gain")
        if "subs_mean_by_rheology" in z.files:
            subs = np.asarray(z["subs_mean_by_rheology"][:, 0], dtype=np.float32)
            full = np.asarray(z["subs_mean_by_rheology"])
        else:
            subs = np.asarray(z["subs_mean"][0], dtype=np.float32)[None]
            full = np.asarray(z["subs_mean"])[None]
        if errs:
            raise MemberError(f"{path}: " + "; ".join(errs))
        return {"labels": labels, "subs": subs, "full": full, "mask": np.asarray(z["mask"]),
                "grid": (float(z["x0"]), float(z["y0"]), float(z["dx"])),
                "origin_month": origin, "options": opts}


# ----------------------------------------------------------------------------- cells
def pile_cells(repo: Path = REPO) -> dict:
    mp = json.loads(_check_sha(repo, C.N3["mapping"], C.N3["mapping_sha256"]).read_text(
        encoding="utf-8"))
    for k, spec in C.PILES.items():
        s = mp["piles"][k]["site"]
        if s["cell"] != spec["cell"] or s["depth_m"] != spec["depth_m"]:
            raise MemberError(f"N3 mapping: pile {k} cell/depth differ")
        fb = mp["piles"][k].get("fallback_site")
        if (fb or {}).get("depth_m") != spec["fallback_depth_m"] or (
                fb and fb["cell"] != spec["cell"]):
            raise MemberError(f"N3 mapping: pile {k} fallback differs")
    return {k: C.PILES[k]["cell"] for k in C.PILE_KEYS}


def town_cells(repo: Path, fw: dict) -> dict:
    """{township: cell indices} of N3S-L's frozen cells.csv for the 11 coastal townships,
    checked against the forward grid's cell centroids."""
    import pandas as pd

    c = pd.read_csv(_check_sha(repo, C.N3SL["cells"], C.N3SL["cells_sha256"]))
    x0, y0, dx = fw["grid"]
    ii, jj = np.nonzero(fw["mask"])
    cx, cy = x0 + (jj + 0.5) * dx, y0 + (ii + 0.5) * dx
    if not (np.allclose(cx[c["cell"]], c["cx"]) and np.allclose(cy[c["cell"]], c["cy"])):
        raise MemberError("N3S-L cells.csv does not match the forward grid")
    out = {}
    for t in C.COAST:
        g = c[(c["township"] == t) & (c["county"] == C.COAST_COUNTY[t])]
        if g.empty:
            raise MemberError(f"no cells for coastal township {t}")
        out[t] = np.sort(g["cell"].to_numpy(int))
    return out


# ----------------------------------------------------------------------------- loader
def load_members(repo: Path = REPO) -> dict:
    """Every Tier A+B member's frozen baseline predictions at the unit cells, with the
    bindings verified. Returns {"members", "piles_months", "towns_months", "town_cells",
    "pile_cells", "forwards", "sd_v2", "full_identity"}."""
    repo = Path(repo)
    sd = verify_sd_v2(repo)
    fws, mask0 = {}, None
    for key, (rel, sha) in C.FORWARDS.items():
        p = _check_sha(repo, rel, sha)
        fws[key] = load_forward(p, key)
        if mask0 is None:
            mask0, grid0 = fws[key]["mask"], fws[key]["grid"]
        elif not (np.array_equal(fws[key]["mask"], mask0) and fws[key]["grid"] == grid0):
            raise MemberError(f"{key}: grid differs from the other forwards")
    pc = pile_cells(repo)
    tc = town_cells(repo, fws["page"])
    labels = fws["page"]["labels"]
    pm, tm = months(*C.FREEZE_PILE_MONTHS), months(*C.FREEZE_TOWN_MONTHS)
    pi = [labels.index(m) for m in pm]
    ti = [labels.index(m) for m in tm]
    allc = np.concatenate([tc[t] for t in C.COAST])
    mem = {}
    for col, spec in C.MEMBER_SPEC.items():
        s = fws[spec["forward"]]["subs"][spec["rheology_index"]]
        mem[col] = {"piles": s[[pc[k] for k in C.PILE_KEYS]][:, pi].astype(np.float32),
                    "towns": s[allc][:, ti].astype(np.float32),
                    "spec": spec}
    full = {}
    for alias, canon in C.ALIASES.items():
        a, b = C.MEMBER_SPEC[alias], C.MEMBER_SPEC[canon]
        A = fws[a["forward"]]["full"][a["rheology_index"]]
        B = fws[b["forward"]]["full"][b["rheology_index"]]
        diff = np.flatnonzero((A != B).any(axis=(0, 2)))
        full[alias] = {"canonical": canon, "bytes_equal_all_scenarios_all_cells":
                       bool(A.tobytes() == B.tobytes()), "n_cells_differing": int(len(diff))}
    return {"members": mem, "piles_months": pm, "towns_months": tm, "pile_cells": pc,
            "town_cells": {t: tc[t].tolist() for t in C.COAST}, "forwards": {
                k: {"path": v[0], "sha256": v[1]} for k, v in C.FORWARDS.items()},
            "sd_v2": sd, "full_identity": full}


def arm_arrays(loaded: dict, col: str, arm: str) -> np.ndarray:
    """The member's predictions on an arm's cells and months (what the arm can see)."""
    m = loaded["members"][col]
    if C.ARMS[arm]["kind"] == "towns":
        return m["towns"]
    a = C.ARMS[arm]
    pm = loaded["piles_months"]
    lo, hi = pm.index(a["ref_month"]), pm.index(a["window"][1])
    return m["piles"][:, lo:hi + 1]


def dedup(loaded: dict) -> dict:
    """Per arm: the member list after collapsing each declared alias whose predictions are
    byte-identical to its canonical member on that arm's cells and months, the identity
    evidence, and any UNDECLARED byte-identical pair (reported, never collapsed)."""
    cols = sorted(loaded["members"])
    out = {}
    for arm in C.ARM_ORDER:
        arr = {c: arm_arrays(loaded, c, arm) for c in cols}
        ev, keep = {}, list(cols)
        for alias, canon in C.ALIASES.items():
            same = arr[alias].tobytes() == arr[canon].tobytes()
            ev[alias] = {"canonical": canon, "bytes_equal_on_arm": bool(same),
                         "sha_alias": array_sha(arr[alias]), "sha_canonical": array_sha(
                             arr[canon])}
            if same:
                keep.remove(alias)
        if C.ARMS[arm]["kind"] == "towns":
            for alias, canon in C.ALIASES.items():
                diff =[t for t in C.COAST if _town_slice(loaded, alias, t).tobytes()
                        != _town_slice(loaded, canon, t).tobytes()]
                ev[alias]["townships_differing"] = diff
        undeclared = [(a, b) for i, a in enumerate(keep) for b in keep[i + 1:]
                      if arr[a].tobytes() == arr[b].tobytes()]
        out[arm] = {"members": keep, "evidence": ev, "undeclared_identical": undeclared,
                    "n_pairs": len(keep) * (len(keep) - 1) // 2}
    return out


def _town_slice(loaded: dict, col: str, town: str) -> np.ndarray:
    off = 0
    for t in C.COAST:
        n = len(loaded["town_cells"][t])
        if t == town:
            return loaded["members"][col]["towns"][off:off + n]
        off += n
    raise KeyError(town)


def prediction_hashes(loaded: dict) -> dict:
    return {c: {"piles": array_sha(m["piles"]), "towns": array_sha(m["towns"])}
            for c, m in sorted(loaded["members"].items())}


def summary(loaded: dict) -> dict:
    """Everything the declaration binds about the members (hashes and identities, no
    values)."""
    return {"members": {c: {**{k: v for k, v in C.MEMBER_SPEC[c].items()},
                            "forward_npz": C.FORWARDS[C.MEMBER_SPEC[c]["forward"]][0],
                            "forward_npz_sha256": C.FORWARDS[C.MEMBER_SPEC[c]["forward"]][1]}
                        for c in sorted(loaded["members"])},
            "prediction_sha256": prediction_hashes(loaded),
            "dedup": dedup(loaded), "full_identity": loaded["full_identity"],
            "piles_months": [loaded["piles_months"][0], loaded["piles_months"][-1]],
            "towns_months": [loaded["towns_months"][0], loaded["towns_months"][-1]],
            "pile_cells": loaded["pile_cells"],
            "town_cells_sha256": hashlib.sha256(json.dumps(
                loaded["town_cells"], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "sd_v2": loaded["sd_v2"], "forwards": loaded["forwards"]}


# ----------------------------------------------------------------------------- freeze
def freeze(out_dir, repo: Path = REPO) -> dict:
    """Write predictions.npz and freeze_summary.json once (mode 'x'); returns the summary
    with both files' sha256. Never adds or removes a member: the set is MEMBER_SPEC."""
    out = Path(out_dir)
    npz, js = out / "predictions.npz", out / "freeze_summary.json"
    if npz.exists() or js.exists():
        raise FileExistsError(f"{out} already holds a freeze: the members are frozen once")
    loaded = load_members(repo)
    s = summary(loaded)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {f"{c}__piles": m["piles"] for c, m in loaded["members"].items()}
    arrays.update({f"{c}__towns": m["towns"] for c, m in loaded["members"].items()})
    arrays["piles_months"] = np.asarray(loaded["piles_months"])
    arrays["towns_months"] = np.asarray(loaded["towns_months"])
    arrays["town_cells_json"] = np.asarray(json.dumps(loaded["town_cells"],
                                                      ensure_ascii=False))
    with open(npz, "xb") as f:
        np.savez(f, **arrays)
    s["predictions_npz"] = {"path": npz.name, "sha256": sha256(npz)}
    with open(js, "x", encoding="utf-8") as f:
        json.dump(s, f, indent=1, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    s["freeze_summary_sha256"] = sha256(js)
    return s


def load_freeze(npz_path, want_sha: str) -> dict:
    """The frozen predictions (the only member input of the OSSE and the scorer)."""
    got = sha256(npz_path)
    if got != want_sha:
        raise MemberError(f"frozen predictions sha256 {got[:12]} != registered {want_sha[:12]}")
    with np.load(npz_path, allow_pickle=False) as z:
        cols = sorted({k.split("__")[0] for k in z.files if "__" in k})
        if cols != list(C.TIER_AB):
            raise MemberError(f"frozen members {cols} != {list(C.TIER_AB)} (never add members)")
        mem = {c: {"piles": np.asarray(z[f"{c}__piles"]), "towns": np.asarray(z[f"{c}__towns"]),
                   "spec": C.MEMBER_SPEC[c]} for c in cols}
        return {"members": mem, "piles_months": [str(m) for m in z["piles_months"]],
                "towns_months": [str(m) for m in z["towns_months"]],
                "town_cells": json.loads(str(z["town_cells_json"])),
                "pile_cells": {k: C.PILES[k]["cell"] for k in C.PILE_KEYS},
                "full_identity": {}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    f = sub.add_parser("freeze")
    f.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    s = summary(load_members(REPO)) if a.cmd == "check" else freeze(a.out, REPO)
    print(json.dumps({k: s[k] for k in s if k != "members"}, indent=1, ensure_ascii=False,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
