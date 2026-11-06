"""FW.rider.v1 (round-8 rank 13, rain-year rider; model-only, CPU): does E02c's strict level rule
stay decidable when the truth worlds carry the model's OWN recorded rain and pumping years?

FV.decidability.v2_postfailure read E02c's level rule 'not shown weak' (P = 0.96) with rain at
climatology in every truth world (its reading 1); its reading 2 (forcing_year.py) built forcing
worlds from FW.weather_replay.v1's pumping-only replays. This rider reruns reading 2's world
construction, unchanged, on FW.weather_replay.v2's eleven PRIMARY arms W2012..W2022 (rain and
detrended pumping replayed, the page ensemble, the same members, code and restart as arm C)
and scores every world with E02c's level rule exactly as reading 1 does
(``osse_fv.world_stats``). It reports P(E02c level pass) as a ONE-SIDED caveat only.

Truth world of sequence s (s = 0..10; W(2012 + s) replays the cyclic order of 2012..2022 from
2012 + s) and noise draw j, on 2026-11..2027-10 at E02c's 158 wells::

    obs(w, t) = ZR(w, t) + F_s(w, t) + e_(s, j)(w, t),    F_s = h_W(2012+s) - h_C

ZR is E02's registered prediction (forward cf2c7154); F_s is reading 2's forcing departure,
computed by ``forcing_year.load_replay`` itself (its gate E included: members and options equal
to C's, every replay's 2012-22 hindcast identical to C's within FV v2's 0.0 m, no two sequences
identical); e is FV v1's per-well stationary AR(1) noise (noise_ar1.csv), independent between
wells, on this rider's own fixed seed.

Code reused UNCHANGED by import and pinned by sha256 to FV.decidability.v2_postfailure's
registered frozen_code (:data:`REUSED`). A changed file refuses to run.

Model-only: no observed head is read. The rider reads model forwards, E02's registered
predictions and E02c's and FV v1's pre-origin observation-derived statistics (CL/CLT, the
observed-month mask, the AR(1) sd and lag-1).

Scoring is private: :func:`run` loads the registration itself; ``_score`` refuses a real run
unless the declaration in INDEX.jsonl loads intact (prereg3.load_prereg) with this code's
design, and a smoke unless every path lies outside results/.

Gates, in order, before any statistic:
  E  engineering: registration intact, frozen code and inputs unchanged, FW v2's report present
     and every world file at the sha256 that report and the declaration record; the replay
     index (forcing_year.build_index) with rain 'replayed' and pumping 'detrended_anomaly';
     forcing_year.load_replay's checks. A problem in the CONTENT of FW's forwards
     (forcing_year.ReplayGateE) maps to the pre-declared not-evaluable outcome 'replay worlds
     unusable'; any other failure is an engineering retry (free, before any statistic).
  N  null world (obs = CL + e, K_NULL draws): E02c level false-pass rate <= FV v1's tolerance
     recomputed from the same inputs; else not evaluable ('a possible bug').
  MC 1.645 x Monte Carlo SE of P <= 0.07: non-binding at K = 250 (largest possible 0.016).

Reading (one-sided, applied once): P = equal-weight mean over the 11 sequences of the pass
rate; P <= 0.5 -> 'weak_in_replayed_years', otherwise 'not_shown_weak_in_replayed_years'.
Neither moves E02c's rule, verdict or any status; the result reaches the claims ledger only as
a caveat line (E21, and beside C30-scen's post-2028 level wording), never the page.

CLI (repository root, hydro env):
  run_rider.py smoke --out <new dir outside results/>   synthetic end to end
  run_rider.py verify                                   registration, code and worlds intact
  run_rider.py run                                      once, after registration; report mode 'x'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_PATHS = (str(HERE), str(REPO / "dataprep_staging" / "fv_decidability_v2"),
          str(REPO / "dataprep_staging" / "fv_decidability"),
          str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"), str(REPO))


def _paths() -> None:
    for p in _PATHS:
        if p not in sys.path:
            sys.path.insert(0, p)


_paths()

import forcing_year as FY  # noqa: E402  (FV v2 reading 2, frozen; never edited)
import osse_fv as O  # noqa: E402  (FV v1, frozen)
import osse_fv2 as V2  # noqa: E402  (FV v2, frozen)
import score_e02c as S  # noqa: E402,F401  (E02c, frozen; behind osse_fv.world_stats)

ID = "FW.rider.v1"
WP = "FW"
VERSION = "1.1.0"
F = "dataprep_staging/fw_rider"
SELF = f"{F}/run_rider.py"
PREREG = f"{F}/prereg_rider.py"
TESTS = "tests/test_fw_rider.py"
EngineeringFailure = O.EngineeringFailure
ReplayGateE = FY.ReplayGateE

# ----------------------------------------------------------------------------- reused code
FV2_ID = V2.ID                                   # FV.decidability.v2_postfailure
FV2_DECL_SHA = "f7e35a17"                        # prefix, as cited by the claims ledger
#: FV v2's registered frozen_code for the files this rider imports (sha256 pins)
REUSED = {
    "dataprep_staging/fv_decidability_v2/forcing_year.py":
        "d5a122e321506b49e826c4317afeba6e014364fbeaa4a44f97403992f618a71d",
    "dataprep_staging/fv_decidability_v2/osse_fv2.py":
        "3fc5bc7f5a584e4be1276d5cde561610b2c870f846aaa5401b0a527a9a6b36c1",
    "dataprep_staging/fv_decidability/osse_fv.py":
        "8b0d5a3c110a9f31275a7d15bc42b4666f10907c11f37ab10147ebd2707a5b34",
    "dataprep_staging/e02c/score_e02c.py":
        "1c4e092a43ae42c9e57ec96ebf4120b0911155d1f7d3216b27c42a3f41b6bc3d",
    "dataprep_staging/e02c/export_baselines.py":
        "a9ee3610a042d6122a30be73ef14c0f0d373020064e804faf2654f9922a6be94",
    "dataprep_staging/harness/coverage3.py":
        "ffe13c84ee137920e239e69772b69bff30efb9005c050bb711c536c01bda7d17",
    "hydrophysics/twin/bands.py":
        "99a88018fc91824442222a5420ef701174b0b748787cdd9f5d0d464c3771b3d0",
}
FROZEN = (SELF, *REUSED)
#: FV v1's export as FV v2 registered it (inputs_frozen)
FV1_INPUTS = {
    "anomaly_pool.csv": "acab736764fd635ecd21eb62f77d26dd3ea8132a184678e7ac0a0fd1fc94f290",
    "noise_ar1.csv": "73965a08914149af2132c96f617268701277b49411cf17cfbf57d41fbeaa9d55",
    "obs_mask.csv": "605a35de5d07cab40cef3a1e86fae719502ff3afcd0f6ea36310ccf5425d0c89"}

# ----------------------------------------------------------------------------- worlds
FW_ID = "FW.weather_replay.v2"
N_SEQ = FY.N_SEQ                                 # 11
FIRST_YEAR = FY.FIRST_YEAR                       # 2012
W_ARMS = tuple(f"W{FIRST_YEAR + s}" for s in range(N_SEQ))   # FW v2's PRIMARY arms
CLIM_ARM = "C"
WANT_RAIN = "replayed"                           # FW v2 replay_fields 'energy+weather'
WANT_PUMPING = FY.PUMP_MODE                      # 'detrended_anomaly'
HINDCAST_TOL_M = V2.HINDCAST_TOL_M               # 0.0 m, inside forcing_year.load_replay
UNUSABLE = "replay worlds unusable"

# ----------------------------------------------------------------------------- design
K_WORLD = FY.K_RC                                # 250 noise draws per sequence (reading 2's K)
K_NULL = O.K_NULL                                # 500 null worlds (reading 1's)
SEED = 46190527                                  # this rider's own seed, fixed in code
SMOKE_SEED = 31750093                            # engineering smokes only; refused for a run
STREAM_WORLD, STREAM_NULL = 61, 62               # FV v1 used 1-4, FV v2 reading 2 11-13
P_WEAK = 0.5
MC_HALF_WIDTH = V2.MC_HALF_WIDTH                 # 0.07
READINGS = ("weak_in_replayed_years", "not_shown_weak_in_replayed_years")
READING_RULE = ("P = equal-weight mean over the 11 W sequences of the share of their worlds "
                "that pass E02c's strict level rule (osse_fv.world_stats 'level' == "
                "'level_forecast_supported'); P <= 0.5 -> 'weak_in_replayed_years', otherwise "
                "'not_shown_weak_in_replayed_years' (one-sided, applied once; a failed "
                "1.645 SE <= 0.07 Monte Carlo check gives 'not_evaluable'); a caveat only")

ATTEMPTS_NAME = "attempts.jsonl"
REPORT_DIR = f"results/dataprep/reports/{ID}"

_LBL = {"en": "model-only rain-year rider FW.rider.v1", "zh": "僅模型的降雨年附帶研究 FW.rider.v1"}
WORDING = {
    "weak_in_replayed_years": {
        "en": "Rain-year caveat ({lbl}): in the model's own replayed 2012-22 rain and detrended "
              "pumping years (FW.weather_replay.v2, the page ensemble), a model of the right "
              "kind passes E02c's level rule with probability {p:.2f} (90 % interval over the "
              "11 recorded years {t_lo:.2f}-{t_hi:.2f}; single sequences {lo:.2f}-{hi:.2f}), at "
              "or below 0.5: in a real weather year a 2028 level fail would not by itself "
              "refute the model's average-weather level. One-sided caveat; E02c's rule, its "
              "verdict and every status are unchanged.",
        "zh": "降雨年附註（{lbl}）：在模型自身重播的 2012-22 降雨與去趨勢抽水年份下"
              "（FW.weather_replay.v2，頁面系集），正確類型的模型通過 E02c 水位規則的機率為 "
              "{p:.2f}（11 個紀錄年的 90 % 區間 {t_lo:.2f}-{t_hi:.2f}；單一序列 {lo:.2f}-{hi:.2f}），"
              "不高於 0.5：在真實的天氣年份中，2028 年水位檢驗未通過本身不足以否定模型的平均天氣水位。"
              "此為單向附註；E02c 的規則、判定與所有狀態均不變。"},
    "not_shown_weak_in_replayed_years": {
        "en": "Rain-year caveat ({lbl}): in the model's own replayed 2012-22 rain and detrended "
              "pumping years (FW.weather_replay.v2, the page ensemble), a model of the right "
              "kind passes E02c's level rule with probability {p:.2f} (90 % interval over the "
              "11 recorded years {t_lo:.2f}-{t_hi:.2f}; single sequences {lo:.2f}-{hi:.2f}); "
              "the replayed rain years do not show the level test weak. One-sided: this adds no "
              "other reading, and P is not the chance of a 2028 pass.",
        "zh": "降雨年附註（{lbl}）：在模型自身重播的 2012-22 降雨與去趨勢抽水年份下"
              "（FW.weather_replay.v2，頁面系集），正確類型的模型通過 E02c 水位規則的機率為 "
              "{p:.2f}（11 個紀錄年的 90 % 區間 {t_lo:.2f}-{t_hi:.2f}；單一序列 {lo:.2f}-{hi:.2f}）；"
              "重播的降雨年份未顯示該水位檢驗為弱檢驗。此為單向附註：不增加其他解讀，"
              "且 P 並非 2028 年通過的機率。"},
    "not_evaluable": {
        "en": "The rain-year rider (FW.rider.v1, model-only) is not evaluable ({why}); E02c's "
              "level reading gains no caveat from it.",
        "zh": "降雨年附帶研究（FW.rider.v1，僅模型）無法評估（{why}）；E02c 的水位解讀不因此增加附註。"},
}


def wording_for(reading: str | None, *, p=None, t_lo=None, t_hi=None, lo=None, hi=None,
                why: str = "") -> dict:
    """The pre-written caveat of a result band, filled (EN and ZH)."""
    if reading is None:
        return {k: v.format(why=why) for k, v in WORDING["not_evaluable"].items()}
    if reading not in READINGS:
        raise EngineeringFailure(f"no declared wording for reading {reading!r}")
    return {k: v.format(lbl=_LBL[k], p=p, t_lo=t_lo, t_hi=t_hi, lo=lo, hi=hi)
            for k, v in WORDING[reading].items()}


def reading_of(p: float | None, mc_half: float | None) -> str | None:
    """One-sided: P <= 0.5 weak in replayed years, otherwise not shown weak; None (not
    evaluable) when P is missing or its Monte Carlo half-width exceeds 0.07."""
    if p is None or not np.isfinite(p) or mc_half is None or not np.isfinite(mc_half):
        return None
    if mc_half > MC_HALF_WIDTH:
        return None
    return "weak_in_replayed_years" if p <= P_WEAK else "not_shown_weak_in_replayed_years"


# ----------------------------------------------------------------------------- code pins
def reused_code_problems(root: Path = REPO) -> list[str]:
    """Reused files whose bytes are not FV v2's registered ones (empty = unchanged)."""
    out = []
    for p, h in REUSED.items():
        q = Path(root) / p
        if not q.is_file():
            out.append(f"{p} missing")
        elif O.sha256(q) != h:
            out.append(f"{p} differs from FV v2's registered frozen_code")
    return out


def check_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EngineeringFailure(f"seed must be an int, not {seed!r}")
    if seed == SMOKE_SEED:
        raise EngineeringFailure("the smoke seed is refused for a scored run")
    return seed


def rng_for(stream: int, s: int, j: int, *, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream, s, j]))


# ----------------------------------------------------------------------------- inputs
@dataclass
class RiderPaths:
    """Every file the rider or its declaration builder reads or writes (repo-relative by
    default; a synthetic smoke gives its own, all outside results/)."""
    v1: O.Paths = field(default_factory=O.Paths)
    outd: str = "results/twin_forward/fw_replay_v2"
    fw_report: str = f"results/dataprep/reports/{FW_ID}/report.json"
    fw_decl: str = f"results/dataprep/preregistration/{FW_ID}.json"
    fv2_decl: str = f"results/dataprep/preregistration/{FV2_ID}.json"
    rulings: str = "results/dataprep/rulings/user_rulings_2026-11-03.json"
    dataprep: str = "results/dataprep"
    index: str = f"{REPORT_DIR}/replay_index.json"
    report: str = f"{REPORT_DIR}/report.json"

    def clim(self) -> str:
        return f"{self.outd}/{CLIM_ARM}.npz"

    def seqs(self) -> list[str]:
        return [f"{self.outd}/{a}.npz" for a in W_ARMS]

    def e02_manifest(self) -> str:
        return str(Path(self.v1.pred).parent / "manifest.json")

    def all_paths(self) -> list[str]:
        v = self.v1
        return [self.outd, self.fw_report, self.fw_decl, self.fv2_decl, self.rulings,
                self.dataprep, self.index, self.report, v.pred, v.protocol, v.baselines,
                v.wells, v.inputs_dir, self.e02_manifest()]


def world_files(P: RiderPaths) -> dict:
    """npz and capture of C and every W arm (as given)."""
    out = {}
    for a, p in [(CLIM_ARM, P.clim()), *zip(W_ARMS, P.seqs(), strict=True)]:
        out[a] = {"npz": p, "capture": p[:-4] + ".fw.npz"}
    return out


def world_shas(P: RiderPaths) -> dict:
    """sha256 of every world file (no head value is read)."""
    out = {}
    for a, r in world_files(P).items():
        out[a] = {k: (O.sha256(O.rp(v)) if O.rp(v).is_file() else "MISSING")
                  for k, v in r.items()}
    return out


def fw_report_files(P: RiderPaths) -> dict | None:
    """{arm: {'npz': sha, 'capture': sha}} from FW v2's report.json 'files' (hashes only)."""
    q = O.rp(P.fw_report)
    if not q.is_file():
        return None
    files = json.loads(q.read_text(encoding="utf-8")).get("files") or {}
    return {a: {k: (files.get(a) or {}).get(k, {}).get("sha256") for k in ("npz", "capture")}
            for a in (CLIM_ARM, *W_ARMS)}


def world_problems(P: RiderPaths, declared: dict | None = None) -> list[str]:
    """World files whose sha256 differs from FW v2's report (and from the declaration)."""
    got, rep = world_shas(P), fw_report_files(P)
    if rep is None:
        return [f"{P.fw_report} missing: FW v2's forwards stay sealed until it exists"]
    out = [f"{a}: sha256 differs from FW v2's report.json files" for a in got
           if got[a] != rep.get(a)]
    if declared is not None:
        out += [f"{a}: sha256 differs from the declaration" for a in got
                if got[a] != declared.get(a)]
    return out


def ensure_index(P: RiderPaths) -> dict:
    """forcing_year.build_index on C and the W arms (written once), or the existing index if it
    names the same files at the same sha256. Metadata and sha256 only."""
    q = O.rp(P.index)
    if q.is_file():
        ix = json.loads(q.read_text(encoding="utf-8"))
        want = world_shas(P)
        got = {CLIM_ARM: ix["climatology"]["sha256"],
               **{a: r["sha256"] for a, r in zip(W_ARMS, ix["sequences"], strict=False)}}
        if any(got.get(a) != want[a]["npz"] for a in want):
            raise EngineeringFailure(f"{P.index} names other files than the declared worlds")
        return ix
    return FY.build_index(q, P.clim(), P.seqs(), source=FW_ID)


def load_worlds(P: RiderPaths, ctx: dict, ix: dict) -> dict:
    """F (11, W, 12) on the window: forcing_year.load_replay, unchanged (reading 2's F_s)."""
    if ix.get("source_declaration") != FW_ID:
        raise EngineeringFailure("replay index source declaration")
    if ix.get("rain") != WANT_RAIN or ix.get("pumping") != WANT_PUMPING:
        raise ReplayGateE(f"index rain/pumping {ix.get('rain')!r}/{ix.get('pumping')!r}, "
                          f"expected {WANT_RAIN!r}/{WANT_PUMPING!r} (FW v2 energy+weather)")
    rp = FY.load_replay(P.index, ctx["cells"], ctx["layers"])
    return {"F": rp["F"][:, :, O.WIN[0]:O.WIN[1]],
            "eng": {"replay": rp["eng"], "index_sha256": rp["index_sha256"],
                    "members": len(rp["members"]), "rain": rp["rain"],
                    "pumping": rp["pumping"], "hindcast_tol_m": HINDCAST_TOL_M}}


# ----------------------------------------------------------------------------- statistics
def null_gate(ctx: dict, *, k_null: int, seed: int) -> dict:
    tol = O.null_tolerance(ctx["zr"] - ctx["cl"], ctx["sd"], ctx["phi"], k_null)
    nulls = [O.world_stats(ctx, ctx["cl"] + O.simulate_ar1(
        ctx["sd"], ctx["phi"], O.N_WIN, rng_for(STREAM_NULL, 0, k, seed=seed)))
        for k in range(k_null)]
    fp = O._rate([w["level"] == "level_forecast_supported" for w in nulls])
    return {"tolerance": tol, "level_false_pass": fp, "pass": fp["p"] <= tol["tolerance"]}


def world_records(ctx: dict, F: np.ndarray, *, k: int, seed: int) -> list[list[dict]]:
    """E02c's verdicts on every world, by sequence (only the verdict and two flags kept)."""
    out = []
    for s in range(F.shape[0]):
        row = []
        for j in range(k):
            e = O.simulate_ar1(ctx["sd"], ctx["phi"], O.N_WIN, rng_for(STREAM_WORLD, s, j,
                                                                        seed=seed))
            w = O.world_stats(ctx, ctx["zr"] + F[s] + e)
            row.append({"level": w["level"], "beat_cl": bool(w.get("beat_cl", False)),
                        "beat_clt": bool(w.get("beat_clt", False))})
        out.append(row)
    return out


def summarise(records: list[list[dict]]) -> dict:
    """P, its Monte Carlo SE and the 11-year sampling uncertainty (forcing_year.summarise_rc,
    unchanged, on the 0/1 pass matrix), plus report-only rates."""
    lev = np.array([[r["level"] == "level_forecast_supported" for r in row] for row in records],
                   float)
    sm = FY.summarise_rc(lev)
    return {"p": sm["rc"], "by_sequence": sm["by_sequence"],
            "range_over_sequences": sm["range_over_sequences"], "mc_se": sm["mc_se"],
            "mc_half_width_90": sm["mc_half_width_90"], "sampling_11yr": sm["sampling_11yr"],
            "k_per_sequence": int(lev.shape[1]),
            "report_only": {
                "beat_cl": float(np.mean([[r["beat_cl"] for r in row] for row in records])),
                "beat_clt": float(np.mean([[r["beat_clt"] for r in row] for row in records])),
                "not_evaluable_worlds": int(sum(r["level"] == "not_evaluable" for row in records
                                                for r in row))}}


def close(rep: dict, why: str) -> dict:
    rep.update(status="not_evaluable", why=why, reading=None,
               wording=wording_for(None, why=why))
    return rep


def _registry_guard(P: RiderPaths, prereg_dir=None, rows_dir=None) -> None:
    """A run on real paths (any under results/) uses the real registry and ledger rows only:
    a temporary registry holding a copy of the declaration must never unlock the real worlds.
    Synthetic paths (all outside results/) may use a temporary registry (tests)."""
    from harness import prereg3

    if not any(_under_results(p) for p in P.all_paths()):
        return
    for given, real, name in ((prereg_dir, prereg3.PREREG_DIR, "prereg_dir"),
                              (rows_dir, prereg3.LEDGER_ROWS, "rows_dir")):
        if given is not None and Path(given).resolve() != Path(real).resolve():
            raise EngineeringFailure(f"{ID} on real paths uses the real registry only: "
                                     f"{name} {given} is not {real}")


def _score(P: RiderPaths, *, seed: int, k: int = K_WORLD, k_null: int = K_NULL,
           smoke: bool = False, prereg_dir=None, rows_dir=None, log=print) -> dict:
    """Gates E, N and MC, then the reading. Returns the report (not written).

    A scored run loads the registration itself (load_registered: INDEX.jsonl entry, intact
    document at that sha256, frozen code, inputs and design) and uses only the registered
    seed and K; a smoke needs SMOKE_SEED and every path outside results/."""
    decl = dsha = None
    if smoke:
        if seed != SMOKE_SEED or any(_under_results(p) for p in P.all_paths()):
            raise EngineeringFailure("a smoke runs on SMOKE_SEED and synthetic worlds outside "
                                     "results/ only")
    else:
        if seed != SEED or k != K_WORLD or k_null != K_NULL:
            raise EngineeringFailure(f"{ID} scores only on its registered seed and K")
        _registry_guard(P, prereg_dir, rows_dir)
        decl, dsha = load_registered(ID, prereg_dir, rows_dir)
    rep: dict = {"id": ID, "version": VERSION, "smoke": smoke, "created": O._now(),
                 "declaration_sha256": dsha, "seed": seed, "k_per_sequence": k,
                 "k_null": k_null, "label": "model-only rain-year rider; one-sided caveat only",
                 "worlds": {"source": FW_ID, "arms": list(W_ARMS), "reference": CLIM_ARM},
                 "mc_gate_note": V2.mc_note(N_SEQ, k), "model_only": True}
    if decl is not None:
        for name, h in FV1_INPUTS.items():
            if O.sha256(O.rp(P.v1.inputs_dir) / name) != h:
                raise EngineeringFailure(f"{name} differs from FV v2's registered sha256")
    bad = world_problems(P, ((decl or {}).get("worlds") or {}).get("files")
                         if decl is not None else None)
    if bad:
        raise EngineeringFailure("world files: " + "; ".join(bad))
    rep["fw_report_sha256"] = O.sha256(O.rp(P.fw_report))
    ctx = O.load_context(P.v1)
    ix = ensure_index(P)
    try:
        W = load_worlds(P, ctx, ix)
    except ReplayGateE as err:
        rep["engineering"] = {"replay_content_failure": str(err)[:500]}
        return close(rep, f"{UNUSABLE}: {str(err)[:200]}")
    rep["engineering"] = W["eng"]
    log("gate E passed (engineering)")
    nul = null_gate(ctx, k_null=k_null, seed=seed)
    if decl is not None:
        dtol = ((decl.get("gates") or {}).get("N") or {}).get("tolerance")
        if dtol is None or abs(dtol - nul["tolerance"]["tolerance"]) > 1e-12:
            raise EngineeringFailure("null tolerance differs from the declared one")
    rep["null_world"] = nul
    if not nul["pass"]:
        return close(rep, "null-world level false pass above its tolerance (a possible bug)")
    log("gate N passed")
    sm = summarise(world_records(ctx, W["F"], k=k, seed=seed))
    rd = reading_of(sm["p"], sm["mc_half_width_90"])
    rep["P"] = sm
    rep["mc_gate"] = {"max_half_width": MC_HALF_WIDTH,
                      "pass": sm["mc_half_width_90"] <= MC_HALF_WIDTH,
                      "p_minus_half_in_mc_se": (sm["p"] - P_WEAK) / sm["mc_se"]
                      if sm["mc_se"] else None}
    if rd is None:
        return close(rep, "Monte Carlo half-width above 0.07")
    t_lo, t_hi = (min(max(x, 0.0), 1.0) for x in sm["sampling_11yr"]["t90"])
    lo, hi = sm["range_over_sequences"]
    rep.update(status="scored", reading=rd, rule=READING_RULE,
               wording=wording_for(rd, p=sm["p"], t_lo=t_lo, t_hi=t_hi, lo=lo, hi=hi),
               ledger_mapping="caveat line only (E21 notes; beside C30-scen's post-2028 level "
                              "wording); no page line, no status change, never S")
    if decl is not None:
        rep["frozen_code"] = decl["frozen_code"]
    return rep


# ----------------------------------------------------------------------------- registration
def index_record(prereg_id: str = ID, prereg_dir=None) -> dict | None:
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    p = d / "INDEX.jsonl"
    if not p.is_file():
        return None
    rows = [r for r in prereg3.read_chained(p) if r["id"] == prereg_id]
    return rows[-1] if rows else None


def design() -> dict:
    return {"seed": SEED, "K_WORLD": K_WORLD, "K_NULL": K_NULL, "P_weak": P_WEAK,
            "streams": {"world": STREAM_WORLD, "null": STREAM_NULL},
            "hindcast_tol_m": HINDCAST_TOL_M, "arms": list(W_ARMS), "reference": CLIM_ARM}


def load_registered(prereg_id: str = ID, prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    """Refuse unless the declaration is in INDEX.jsonl, loads intact through
    prereg3.load_prereg at that entry's sha256, its frozen code is byte-identical and its
    design is this code's."""
    from harness import prereg3

    if prereg_id != ID:
        raise EngineeringFailure(f"this runner scores {ID} only, not {prereg_id!r}")
    rec = index_record(prereg_id, prereg_dir)
    if rec is None:
        raise EngineeringFailure(f"{prereg_id} is not in INDEX.jsonl: register it first")
    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    try:
        doc, sha = prereg3.load_prereg(prereg_id, **kw)
    except Exception as err:      # noqa: BLE001 - any failure is a refusal
        raise EngineeringFailure(f"{prereg_id} is not registered or not intact: {err}") from err
    if sha != rec.get("sha256"):
        raise EngineeringFailure(f"{prereg_id}: the document's sha256 is not INDEX.jsonl's")
    fc = doc.get("frozen_code") or {}
    if set(fc) != set(FROZEN):
        raise EngineeringFailure("frozen_code does not list exactly FW.rider.v1's files")
    bad = [p for p, h in fc.items() if not O.rp(p).is_file() or O.sha256(O.rp(p)) != h]
    if bad:
        raise EngineeringFailure(f"code sha256 differs from the registration: {bad}")
    bad = reused_code_problems()
    if bad:
        raise EngineeringFailure(f"reused FV v2 code changed: {bad}")
    changed = prereg3.changed_inputs(doc)
    if changed:
        raise EngineeringFailure(f"registered inputs changed: {changed}")
    d = doc.get("design") or {}
    bad = sorted(k for k, v in design().items() if d.get(k) != v)
    if bad:
        raise EngineeringFailure(f"the registered design differs from this code on {bad}")
    return doc, sha


def attempts_path(report: str) -> Path:
    return O.rp(report).parent / ATTEMPTS_NAME


def _log_attempt(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": ID, "time": O._now(), **rec}, sort_keys=True,
                            ensure_ascii=False, default=O._jsonable) + "\n")


def _attempt_no(path: Path) -> int:
    if not path.is_file():
        return 1
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return 1 + sum(r.get("event") == "start" for r in rows)


def run(P: RiderPaths | None = None, *, prereg_dir=None, rows_dir=None, log=print) -> dict:
    """Score once. Refuses (nothing logged, nothing read) unless registered and intact; then
    logs 'start' and 'engineering_failure', 'error' or 'success' (with the report sha256) to
    attempts.jsonl beside the report, and writes report.json once (mode 'x'). No statistic
    is logged."""
    P = P or RiderPaths()
    _registry_guard(P, prereg_dir, rows_dir)          # real paths: the real registry only
    load_registered(ID, prereg_dir, rows_dir)         # before anything is logged or read
    outp = O.rp(P.report)
    ap = attempts_path(P.report)
    n = _attempt_no(ap)
    _log_attempt(ap, {"event": "start", "attempt": n, "report": str(P.report),
                      "code_sha256": O.sha256(Path(__file__))})
    try:
        if outp.exists():
            raise EngineeringFailure(f"{outp} exists: {ID} is scored once")
        rep = _score(P, seed=check_seed(SEED), prereg_dir=prereg_dir, rows_dir=rows_dir,
                     log=log)
        O._write_json_once(outp, rep)
    except EngineeringFailure as err:
        _log_attempt(ap, {"event": "engineering_failure", "attempt": n, "message": str(err)})
        raise
    except Exception as err:
        _log_attempt(ap, {"event": "error", "attempt": n,
                          "message": f"{type(err).__name__}: {err}"})
        raise
    _log_attempt(ap, {"event": "success", "attempt": n, "report_sha256": O.sha256(outp),
                      "declaration_sha256": rep.get("declaration_sha256")})
    return rep


# ----------------------------------------------------------------------------- synthetic
def _under_results(p) -> bool:
    """True for any path that is, or may be, part of the real results tree: lexically under
    <repo>/results, resolved under it, or with a 'results' directory anywhere in its resolved
    form (results/ entries are symlinks into another checkout in build worktrees, so the
    resolved path alone misses them). Synthetic smokes live outside every such path."""
    q = Path(p)
    q = q if q.is_absolute() else REPO / q
    root = REPO / "results"
    for cand in (Path(os.path.abspath(q)), q.resolve()):
        for r in (root, root.resolve()):
            try:
                cand.relative_to(r)
                return True
            except ValueError:
                pass
    return "results" in q.resolve().parts


SYNTH_PRODUCTS = {"twin_rain/preorigin/v2": ["forcing_frozen", "design_time_evidence"],
                  "heads/L1_clean/v3": ["baseline", "design_time_evidence"],
                  "pump_electricity/preorigin/v3": ["design_time_evidence"]}


def build_synthetic(out: Path, seed: int = 5, amp: float = 0.6) -> RiderPaths:
    """SYNTHETIC inputs only (no real data, nothing under results/): FV v1's synthetic set
    (osse_fv.build_synthetic) plus an arm C and eleven W arms with the same members, their
    capture sidecars, an FW v2-style report and declaration, an FV v2-style declaration,
    rulings and product MANIFESTs, in the real layouts."""
    import pandas as pd

    out = Path(out)
    if _under_results(out):
        raise EngineeringFailure("synthetic inputs never live under results/")
    v1 = O.build_synthetic(out / "v1", nonlinear=0.01)
    with np.load(v1.zr_forward, allow_pickle=False) as z:
        H = np.asarray(z["heads_mean"], float)[0]
    L_, C_, T = H.shape
    rng = np.random.default_rng(seed)
    year_eff = rng.normal(0.0, 1.0, N_SEQ)
    t = np.arange(T)
    outd = out / "fw_replay_v2"
    outd.mkdir(parents=True)
    labels = np.array(["stage3_datum_gate", "fold0", "fold1"])

    def save(arm, heads, pf, years=None):
        rec = {"projection_forcing": pf,
               "replay": None if years is None else {"years": years, "pumping": "detrended",
                                                     "fields": "energy+weather"}}
        np.savez_compressed(outd / f"{arm}.fw.npz", record=json.dumps(rec),
                            inelastic=np.zeros(3))
        np.savez_compressed(
            outd / f"{arm}.npz", heads_mean=heads[None].astype(np.float32),
            heads_std=np.zeros((1, L_, C_, T), np.float32),
            scenario_names=np.array(["baseline"]),
            dates=np.array([str(d) for d in pd.date_range("2012-01-01", periods=T,
                                                          freq="MS")]),
            origin=O.ORIGIN_IDX, horizon=T - O.T_FIT, n_members=2 * len(labels),
            member_labels=labels, gain=1.0, hindcast_gain=0.0,
            gate=json.dumps({"git_commit": "synthetic"}),
            forward_options=json.dumps({"update_method": "innovation", "column_heads": "free",
                                        "restart_taper_km": 5.0, "apex_hold": "calibrated",
                                        "column_hpc0_fast_days": 365.0}))

    save(CLIM_ARM, H, "climatology")
    for s in range(N_SEQ):
        yrs = FY.seq_years(s, N_SEQ)
        eff = np.zeros(T)
        for i in range(O.T_FIT, T):
            eff[i] = year_eff[yrs[((i - O.T_FIT) // 12) % N_SEQ] - FIRST_YEAR] * (
                1.0 - np.exp(-(i - O.T_FIT + 1) / 6.0))
        save(W_ARMS[s], H + amp * eff[None, None, :] * (t >= O.T_FIT),
             f"replay:{yrs[0]}-...", yrs)
    P = RiderPaths(v1=v1, outd=str(outd), fw_report=str(out / "fw_report.json"),
                   fw_decl=str(out / "fw_decl.json"), fv2_decl=str(out / "fv2_decl.json"),
                   rulings=str(out / "rulings.json"), dataprep=str(out / "dataprep"),
                   index=str(out / "replay_index.json"), report=str(out / "report.json"))
    files = {a: {k: {"path": v, "sha256": O.sha256(Path(v))} for k, v in r.items()}
             for a, r in world_files(P).items()}
    Path(P.fw_report).write_text(json.dumps({"prereg_id": FW_ID, "verdict": "scored",
                                             "files": files}) + "\n")
    Path(P.fw_decl).write_text(json.dumps({"id": FW_ID, "recipe": {
        "replay_fields": "energy+weather"}}) + "\n")
    Path(P.fv2_decl).write_text(json.dumps({"id": FV2_ID, "frozen_code": dict(REUSED)}) + "\n")
    Path(P.rulings).write_text(json.dumps({"rulings": [
        {"id": "R8.7", "decision": "synthetic stand-in"}]}) + "\n")
    for prod, uses in SYNTH_PRODUCTS.items():
        d = Path(P.dataprep) / prod
        d.mkdir(parents=True)
        (d / "MANIFEST.json").write_text(json.dumps({"allowed_uses": uses,
                                                     "forbidden_uses": []}) + "\n")
    Path(P.e02_manifest()).write_text(json.dumps({"synthetic": True}) + "\n")
    return P


def run_smoke(out: Path, k: int = 40, k_null: int = 20) -> dict:
    """Synthetic end to end on SMOKE_SEED; writes out/smoke_report.json once. Numbers mean
    nothing about the twin."""
    out = Path(out)
    if _under_results(out):
        raise EngineeringFailure("the smoke never writes under results/")
    P = build_synthetic(out / "inputs_synthetic")
    rep = _score(P, seed=SMOKE_SEED, k=k, k_null=k_null, smoke=True, log=lambda *_: None)
    rep["note"] = "SMOKE: synthetic inputs only; numbers mean nothing about the twin"
    O._write_json_once(out / "smoke_report.json", rep)
    return rep


# ----------------------------------------------------------------------------- CLI
def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke", help="synthetic end to end")
    sm.add_argument("--out", required=True, help="a NEW directory outside results/")
    sub.add_parser("verify", help="registration, code and worlds intact (no statistic)")
    sub.add_parser("run", help="score once, after registration")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "smoke":
            rep = run_smoke(Path(a.out))
            print(f"smoke {rep['status']} (synthetic)")
            return 0
        if a.cmd == "verify":
            _registry_guard(RiderPaths(), prereg_dir, rows_dir)
            doc, dsha = load_registered(ID, prereg_dir, rows_dir)
            bad = world_problems(RiderPaths(), (doc.get("worlds") or {}).get("files"))
            if bad:
                raise EngineeringFailure("; ".join(bad))
            print(f"{ID} registered (sha256 {dsha[:16]}); code, inputs and worlds intact")
            return 0
        rep = run(prereg_dir=prereg_dir, rows_dir=rows_dir)
        print(f"wrote {RiderPaths().report} (status {rep['status']}; reading "
              f"{rep.get('reading')})")
        return 0
    except EngineeringFailure as err:
        print(f"REFUSED / ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
