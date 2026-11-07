"""Synthetic decision pages for tests and the local parity run (no real data, no network).

``build_synthetic_page(tmp_path)`` builds a REAL page with ``viewer_app.build`` from tiny synthetic
inputs (the same recipe as tests/test_viewer_app.py), adds the side inputs the default test
builder leaves out (rail alignment, townships), then injects the blocks a real page gets from
files the synthetic run does not have: ``n2Unit`` (mirroring ``viewer_app._n2_unit_block``),
``classes[k].gwh``, ``townSkill`` and ``modelcard.pumping``. The result has the real template,
the real ``T`` and real payload shapes, so every preset has something to test.

``synthetic_ledger(tmp_path, page)`` writes a copy of the committed ledger whose page pins match
the synthetic page, so the full binding checks can run on it.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import socket
from pathlib import Path

import numpy as np
import pandas as pd

T_MONTHS = 252                     # Jan 2012 .. Dec 2032
ORIGIN = 131                       # Dec 2022
NX, NY, DX, X0, Y0 = 6, 5, 1000.0, 180000.0, 2620000.0
GWH = [120.0, 80.0, 10.0, 30.0, 25.0, 5.0]
TOWNS = ("土庫鎮", "虎尾鎮")        # cells 0-9 stay pooled (township 0, the Changhua panel)


def _mask():
    m = np.ones((NY, NX), dtype=bool)
    m[0, 0] = m[4, 5] = False
    return m


def _ramp(start_month: int) -> np.ndarray:
    t = np.arange(T_MONTHS, dtype="float64")
    return np.clip(t - start_month, 0, None) / 12.0


def write_inputs(tmp_path: Path, members: bool = True):
    """Forward run, response basis and members CSV (tests/test_viewer_app.py:_write_inputs)."""
    mask = _mask()
    A = int(mask.sum())
    rng = np.random.default_rng(0)
    dates = np.array([str(d)[:19] for d in pd.date_range("2012-01-01", periods=T_MONTHS, freq="MS")],
                     dtype="U19")
    w = 1.0 + rng.random(A)
    base_s = 0.01 * np.outer(w, _ramp(0))
    base_h = 10.0 - 0.2 * np.outer(w, _ramp(0))
    resp_irr = -0.002 * np.outer(w, _ramp(168))
    resp_aqua = -0.001 * np.outer(w[::-1], _ramp(168))
    resp_irr30 = -0.003 * np.outer(w, _ramp(216))

    def heads_of(s):
        return np.stack([base_h - 50 * (s - base_s) * (1 + 0.1 * k) for k in range(4)])

    fs = np.stack([base_s, base_s + 0.3 * resp_irr, base_s + resp_aqua])
    fh = np.stack([heads_of(x) for x in fs])
    gate = {"zone_boundaries": "205,182", "git_commit": "abc1234", "n_wells": 10,
            "fix_eta": 0.5, "fix_head_extra": 40.0, "return_flow": True, "learn_spread": True,
            "bounds_hit": {"mid": {"log_S": {"lo": 0, "hi": 2, "n": 4}}},
            "gate": {"r2_kfold": 0.8, "r2_idw": 0.7, "verdict": "PASS", "n_folds": 5}}
    fw = tmp_path / "fwd.npz"
    np.savez(fw, dates=dates, origin=np.int64(ORIGIN), heads_mean=fh.astype("float32"),
             heads_std=np.zeros_like(fh, dtype="float32"), subs_mean=fs.astype("float32"),
             subs_std=np.full_like(fs, 0.002, dtype="float32"),
             scenario_names=np.array(["baseline", "cut30", "retire_aqua"]),
             scenarios=np.array(["baseline: baseline (no change)",
                                 "cut30: irrigation x0.7 from 2026-01",
                                 "retire_aqua: aquaculture x0 from 2026-01"]),
             mask=mask, nx=np.int64(NX), ny=np.int64(NY), dx=np.float64(DX),
             x0=np.float64(X0), y0=np.float64(Y0), n_members=np.int64(4),
             member_labels=np.array(["m0", "m1"]), hindcast_r2=np.array([0.9, 0.91]),
             gate=np.array(json.dumps(gate)))
    names, rows = ["baseline"], [base_s]
    keys = ["irr", "aqua", "live", "dom", "ind", "oth"]
    for yr, start in (("2026", 168), ("2030", 216)):
        for k, key in enumerate(keys):
            names.append(f"{key}0_{yr}")
            if key == "irr":
                r = resp_irr if yr == "2026" else resp_irr30
            elif key == "aqua":
                r = resp_aqua if yr == "2026" else -0.0008 * np.outer(w[::-1], _ramp(start))
            else:
                r = -0.0001 * (k + 1) * np.outer(w, _ramp(start))
            rows.append(base_s + r)
    names.append("check_irr50_2026")
    rows.append(base_s + 0.5 * resp_irr)
    bsubs = np.stack(rows)
    bheads = np.stack([heads_of(x) for x in bsubs])
    bs = tmp_path / "basis.npz"
    np.savez(bs, dates=dates, origin=np.int64(ORIGIN), subs_mean=bsubs.astype("float32"),
             heads_mean=bheads.astype("float32"), scenario_names=np.array(names),
             mask=mask, nx=np.int64(NX), ny=np.int64(NY), dx=np.float64(DX),
             x0=np.float64(X0), y0=np.float64(Y0))
    if members:
        recs = []
        for i, lab in enumerate(["m0", "m1"]):
            for ic in (0, 1):
                j = 2 * i + ic
                fb = 10.0 + 0.1 * j
                recs.append({"scenario": "baseline", "member": lab, "ic": ic,
                             "subs_forward_cm": fb, "subs_end_cm": 40 + j, "head_L2_change_m": 1.0})
                recs.append({"scenario": "cut30", "member": lab, "ic": ic,
                             "subs_forward_cm": fb - (0.5 + 0.1 * j), "subs_end_cm": 39 + j,
                             "head_L2_change_m": 1.5 + 0.1 * j})
                recs.append({"scenario": "retire_aqua", "member": lab, "ic": ic,
                             "subs_forward_cm": fb - (1.0 + 0.2 * j), "subs_end_cm": 38 + j,
                             "head_L2_change_m": 2.0})
        pd.DataFrame(recs).to_csv(tmp_path / "fwd.members.csv", index=False)
    return fw, bs, A


def _side_inputs(tmp_path: Path, A: int) -> dict:
    hsr = tmp_path / "hsr.csv"
    ch = np.arange(0, 4.01, 0.25)
    pd.DataFrame({"chainage_km": ch, "x_twd97": X0 + 1000 + ch * 700,
                  "y_twd97": Y0 + 4500 - ch * 700}).to_csv(hsr, index=False)
    st = tmp_path / "st.csv"
    pd.DataFrame({"name_zh": ["甲"], "name_en": ["A"], "x_twd97": [X0 + 1000],
                  "y_twd97": [Y0 + 4500]}).to_csv(st, index=False)
    towns = tmp_path / "towns.csv"
    cell_town = [None] * 10 + [TOWNS[0]] * 10 + [TOWNS[1]] * (A - 20)
    pd.DataFrame({"cell": range(A), "town": cell_town}).to_csv(towns, index=False)
    return {"hsr_csv": str(hsr), "hsr_stations_csv": str(st), "townships_csv": str(towns)}


@contextlib.contextmanager
def _quiet(va):
    """No real-data side inputs and no network while building (like the ``quiet`` fixture)."""
    saved = (va._ground_idw, va._load_inputs, socket.socket.connect)

    def _no_net(*a, **k):
        raise AssertionError("the builder must not touch the network")
    va._ground_idw = lambda cent, log=print: None
    va._load_inputs = lambda log=print: None
    socket.socket.connect = _no_net
    try:
        yield
    finally:
        va._ground_idw, va._load_inputs, socket.socket.connect = saved


def _n2_unit(payload: dict, all30: bool = False) -> dict:
    """A plausible N2 unit in ``_n2_unit_block``'s shape: per solved run and per basis row, a
    head change that is zero before the policy's start year and grows after it. ``all30`` adds
    the headline lever's block (ruling R8.9; ``viewer_app --all30``), a little off the basis
    superposition, as a full-model run is."""
    years = payload["meta"]["years"]
    K = len(payload["classes"])
    starts = payload["meta"]["starts"]

    def ramp(start, amp):
        return [round(amp * max(0, y - start + 1) / 7.0, 4) for y in years]
    mag = [0.9, 0.5, 0.05, 0.12, 0.1, 0.02]
    solved = []
    for x in payload["solved"]:
        amp = sum((1 - f) * mag[k] for k, f in enumerate(x["factors"]))
        solved.append(ramp(int(x["start"]), amp * 1.07))
    basis = [ramp(s, mag[k] * (1.0 if si == 0 else 0.8)) for si, s in enumerate(starts)
             for k in range(K)]
    out = {"n": 84, "n_registered": 90, "solved": solved, "basis": basis}
    if all30:
        out["all30"] = {"name": "all30", "factors": [0.7] * K, "start": int(starts[0]),
                        "series": ramp(int(starts[0]), 0.3 * sum(mag) * 0.93),
                        "file": "all30_page.npz", "sha256": "0" * 64}
    return out


def _write_payload(page: Path, payload: dict) -> None:
    text = page.read_text(encoding="utf-8")
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    blob = blob.replace("</", "<\\/")
    m = re.search(r'(<script id="payload" type="application/json">)(.*?)(</script>)', text, re.S)
    page.write_text(text[:m.start(2)] + blob + text[m.end(2):], encoding="utf-8")


def build_synthetic_page(tmp_path: Path, *, n2: bool = True, basis: bool = True,
                         three: str = "none", all30: bool = False) -> Path:
    from hydrophysics.twin import viewer_app as va
    from hydrophysics.twin.release import read_payload
    tmp_path = Path(tmp_path)
    fw, bs, A = write_inputs(tmp_path)
    out = tmp_path / "page.html"
    opts = dict(leveling="none", wells="none", temporal_npz=None, column_csv=None,
                rivers_csv=None, theta_json=None, alt_npz=None, rheo_npz=None,
                log=lambda s: None, three=three)
    opts.update(_side_inputs(tmp_path, A))
    with _quiet(va):
        va.build(str(fw), str(bs) if basis else None, str(out), **opts)
    payload = read_payload(out)
    for k, c in enumerate(payload["classes"]):
        c["gwh"] = GWH[k]
    payload["modelcard"]["pumping"] = {"changhua": False}
    payload["townSkill"] = [{"n": 3, "bias": 6.1, "r2": None, "low": True}] + [
        {"n": 8, "bias": -1.2, "r2": 0.42, "low": False} for _ in payload["towns"][1:]]
    if n2:
        payload["n2Unit"] = _n2_unit(payload, all30)
        if not basis:
            payload["n2Unit"]["basis"] = None
    _write_payload(out, payload)
    return out


def synthetic_ledger(tmp_path: Path, page: Path) -> Path:
    """A ledger copy pinned to the synthetic page (forward sha) and the working-tree template."""
    from hydrophysics.twin.release import read_payload

    from .page import LEDGER, TEMPLATE
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    fwd = read_payload(page)["release"]["sources"]["forward"]["sha256"]
    led["page"]["forward_sha256"] = fwd
    led["page"]["template_sha256"] = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    out = Path(tmp_path) / "claims_synthetic.json"
    out.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


class RulesEchoBackend:
    """TEST DOUBLE ONLY: a fake LLM backend that answers each routing request with the rules
    parser's own reading, in the LLM's structured-output shape. Tests use it to reach a preset
    through the LLM path (ruling R9.2: the rules stage never routes a preset itself), so the
    preset still passes ``from_llm_json``, the enabled-preset check and ``inherit``. Never wire
    it into a deployment: that would bring back the rules routing presets."""

    name = "fake-rules-echo"

    def __init__(self, b):
        self.b, self.calls = b, []

    def complete(self, system, messages, *, output_format=None, effort="low", max_tokens=1024):
        from . import query as Q
        from .llm.backend import NormalizedResponse
        from .router.rules import RulesRouter
        req = json.loads(messages[0]["content"][0]["text"])
        self.calls.append(req)
        r = RulesRouter(self.b).route(req["question"], req.get("previous"))
        p = r.params
        town = p.get("town")
        out = {"preset": None if (r.refusal or r.clarify or r.preset == "refuse") else r.preset,
               "cuts": p.get("cuts"), "start": p.get("start"), "year": p.get("year"),
               "town": Q.ctx(self.b).towns[town]["en"] if isinstance(town, int) else town,
               "by": p.get("by"), "target_m": p.get("target_m"), "refusal": r.refusal,
               "clarify": (r.clarify or {}).get("reason"), "inherit": list(r.inherit),
               "answer_lang": r.lang}
        return NormalizedResponse(text=json.dumps(out, ensure_ascii=False), stop_reason="end_turn",
                                  model=self.name)


def echo_llm(b):
    """An LLMRouter over :class:`RulesEchoBackend` (tests only)."""
    from .router import routed_presets
    from .router.llm import LLMRouter
    return LLMRouter(b, RulesEchoBackend(b), presets=routed_presets())
