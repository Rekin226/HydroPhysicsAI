"""FW.rider.v1 (round-8 rank 13, rain-year rider; model-only, CPU): does E02c's strict level rule
stay decidable when the truth worlds carry the model's OWN recorded rain and pumping years?

FV.decidability.v2_postfailure read E02c's level rule 'not shown weak' (P = 0.96) with rain at
climatology in every truth world (its reading 1), and its reading 2 (forcing_year.py, the
forcing-year completeness of E02's interval) ran on FW.weather_replay.v1's pumping-only replays.
This rider reruns reading 2's world construction on FW.weather_replay.v2's eleven RD arms (rain
and detrended pumping replayed, the page parameter set alone, --ic-members 0) as the truth
worlds, and scores each world with E02c's level rule exactly as reading 1 does
(``osse_fv.world_stats``). It reports P(E02c level pass) as a ONE-SIDED caveat only.

Truth world of sequence s (s = 0..10; RD(2012 + s) replays the cyclic order of 2012..2022
starting at 2012 + s) and noise draw j, on 2026-11..2027-10 at E02c's 158 wells::

    obs(w, t) = ZR(w, t) + F_s(w, t) + e_(s, j)(w, t)
    F_s = osse_fv.departure(h_RD_s, h_C, mask)

ZR is E02's registered prediction (forward cf2c7154). h_C is FW v2's arm C, the page ensemble
under climatology run by the same code as the RD arms. ``departure`` is FV v1's frozen function:
the window difference minus its mean over each well's observed 2012-22 months (the months E02's
datum offset used), so F_s is the difference of the two datum-corrected series; it equals
reading 2's F_s = h_s - h_clim whenever the two hindcasts are identical (reading 2's W arms).
Here C is an 18-set ensemble and RD one parameter set, so F_s also holds that set's own
departure from the ensemble, common to every sequence (reported, report-only, never removed):
the truth is one page parameter set living through a recorded weather year. e is FV v1's
per-well stationary AR(1) noise (noise_ar1.csv), independent between wells, on this rider's own
fixed seed.

Code reused UNCHANGED by import and pinned by sha256 to FV.decidability.v2_postfailure's
registered frozen_code (:data:`REUSED`): forcing_year.py (reading 2: build_index,
capture_record, check_capture, seq_years, _heads, _opts, summarise_rc), osse_fv.py
(load_context, departure, simulate_ar1, world_stats, null_tolerance, _rate), osse_fv2.py and
score_e02c.py (E02c's level rule t_level behind world_stats). A changed file refuses to run.

Model-only: no observation is read. The inputs are model forwards, E02's registered
predictions and protocol, E02c's registered pre-origin baselines and well table, and FV v1's
pre-origin noise and mask export (``inputs`` in the declaration; ``assert_model_only``).

Gates, in order, before any statistic:
  E  engineering: registration intact, frozen code and inputs unchanged, FW v2's report present,
     the replay index (forcing_year.build_index, metadata and sha256 only) with rain 'replayed'
     and pumping 'detrended_anomaly', every RD npz and capture at its declared sha256, axis,
     origin, members (C the page ensemble, each RD the page set alone), forward options and gain
     equal, the eleven RD hindcasts identical to each other within 1e-6 m at the wells, no two
     sequences identical, finite heads. A problem in the CONTENT of FW's forwards maps to the
     pre-declared not-evaluable outcome 'replay worlds unusable' (the rider cannot rerun them);
     any other failure is an engineering retry (free, before any statistic, logged).
  N  null world (obs = CL + e, K_NULL draws): E02c level false-pass rate <= FV v1's tolerance
     recomputed from the same inputs; else not evaluable ('a possible bug').
  MC 1.645 x Monte Carlo SE of P <= 0.07: non-binding at K = 250 (largest possible 0.016).

Reading (one-sided, applied once): P = equal-weight mean over the 11 sequences of the pass
rate; P <= 0.5 -> 'weak_in_replayed_years', otherwise 'not_shown_weak_in_replayed_years'.
Neither moves E02c's rule, verdict or any status; the result reaches the claims ledger only as
a caveat line (E21, and beside C30-scen's post-2028 level wording), never the page.

CLI (repository root, hydro env):
  run_rider.py smoke --out <new dir outside results/>   synthetic end to end, writes nothing real
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
import score_e02c as S  # noqa: E402  (E02c, frozen)

ID = "FW.rider.v1"
WP = "FW"
VERSION = "1.0.0"
F = "dataprep_staging/fw_rider"
SELF = f"{F}/run_rider.py"
PREREG = f"{F}/prereg_rider.py"
TESTS = "tests/test_fw_rider.py"
EngineeringFailure = O.EngineeringFailure

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
#: FV v1's export as FV v2 registered it (inputs_frozen), and E02c's registered baselines
FV1_INPUTS = {
    "anomaly_pool.csv": "acab736764fd635ecd21eb62f77d26dd3ea8132a184678e7ac0a0fd1fc94f290",
    "noise_ar1.csv": "73965a08914149af2132c96f617268701277b49411cf17cfbf57d41fbeaa9d55",
    "obs_mask.csv": "605a35de5d07cab40cef3a1e86fae719502ff3afcd0f6ea36310ccf5425d0c89"}
E02C_BASE_SHA = dict(O.E02C_BASE_SHA)
E02_MANIFEST = f"{S.E02_DIR}/manifest.json"
E02_MANIFEST_SHA = "d7286551245644b7932fac98751bcd0e837611ffdfbc9a403dd096c8c51af056"

# ----------------------------------------------------------------------------- worlds
FW_ID = "FW.weather_replay.v2"
FW_OUTD = "results/twin_forward/fw_replay_v2"
FW_REPORT = f"results/dataprep/reports/{FW_ID}/report.json"
FW_DECL = f"results/dataprep/preregistration/{FW_ID}.json"
N_SEQ = FY.N_SEQ                                 # 11
FIRST_YEAR = FY.FIRST_YEAR                       # 2012
RD_ARMS = tuple(f"RD{FIRST_YEAR + s}" for s in range(N_SEQ))
CLIM_ARM = "C"
RD_MEMBERS = ("stage3_datum_gate",)              # the page parameter set alone (--ic-members 0)
WANT_RAIN = "replayed"                           # FW v2 replay_fields 'energy+weather'
WANT_PUMPING = FY.PUMP_MODE                      # 'detrended_anomaly'
SAME_OPTIONS = ("update_method", "column_heads", "restart_taper_km", "apex_hold",
                "column_hpc0_fast_days")
HINDCAST_TOL_M = 1e-6                            # RD hindcasts identical to each other
UNUSABLE = "replay worlds unusable"


def arm_npz(arm: str, outd: str = FW_OUTD) -> str:
    return f"{outd}/{arm}.npz"


def capture_npz(arm: str, outd: str = FW_OUTD) -> str:
    return f"{outd}/{arm}.fw.npz"


# ----------------------------------------------------------------------------- design
K_WORLD = FY.K_RC                                # 250 noise draws per sequence (reading 2's K)
K_NULL = O.K_NULL                                # 500 null worlds (reading 1's)
SEED = 46190527                                  # this rider's own seed, fixed in code
SMOKE_SEED = 31750093                            # engineering smokes only; refused for a run
STREAM_WORLD, STREAM_NULL = 61, 62               # FV v1 used 1-4, FV v2 reading 2 11-13
P_WEAK = 0.5
MC_HALF_WIDTH = V2.MC_HALF_WIDTH                 # 0.07
READINGS = ("weak_in_replayed_years", "not_shown_weak_in_replayed_years")
READING_RULE = ("P = equal-weight mean over the 11 RD sequences of the share of their worlds "
                "that pass E02c's strict level rule (osse_fv.world_stats 'level' == "
                "'level_forecast_supported'); P <= 0.5 -> 'weak_in_replayed_years', otherwise "
                "'not_shown_weak_in_replayed_years' (one-sided, applied once; a failed "
                "1.645 SE <= 0.07 Monte Carlo check gives 'not_evaluable'); a caveat only")

REPORT_DIR = f"results/dataprep/reports/{ID}"
REPORT = f"{REPORT_DIR}/report.json"
INDEX_PATH = f"{REPORT_DIR}/replay_index.json"
ATTEMPTS_NAME = "attempts.jsonl"

_LBL = {"en": "model-only rain-year rider FW.rider.v1", "zh": "僅模型的降雨年附帶研究 FW.rider.v1"}
WORDING = {
    "weak_in_replayed_years": {
        "en": "Rain-year caveat ({lbl}): in the model's own replayed 2012-22 rain and detrended "
              "pumping years (FW.weather_replay.v2, the page parameter set alone), a model of "
              "the right kind passes E02c's level rule with probability {p:.2f} (90 % interval "
              "over the 11 recorded years {t_lo:.2f}-{t_hi:.2f}; single sequences "
              "{lo:.2f}-{hi:.2f}), at or below 0.5: in a real weather year a 2028 level fail "
              "would not by itself refute the model's average-weather level. One-sided caveat; "
              "E02c's rule, its verdict and every status are unchanged.",
        "zh": "降雨年附註（{lbl}）：在模型自身重播的 2012-22 降雨與去趨勢抽水年份下"
              "（FW.weather_replay.v2，僅用頁面參數組），正確類型的模型通過 E02c 水位規則的機率為 "
              "{p:.2f}（11 個紀錄年的 90 % 區間 {t_lo:.2f}-{t_hi:.2f}；單一序列 {lo:.2f}-{hi:.2f}），"
              "不高於 0.5：在真實的天氣年份中，2028 年水位檢驗未通過本身不足以否定模型的平均天氣水位。"
              "此為單向附註；E02c 的規則、判定與所有狀態均不變。"},
    "not_shown_weak_in_replayed_years": {
        "en": "Rain-year caveat ({lbl}): in the model's own replayed 2012-22 rain and detrended "
              "pumping years (FW.weather_replay.v2, the page parameter set alone), a model of "
              "the right kind passes E02c's level rule with probability {p:.2f} (90 % interval "
              "over the 11 recorded years {t_lo:.2f}-{t_hi:.2f}; single sequences "
              "{lo:.2f}-{hi:.2f}); the replayed rain years do not show the level test weak. "
              "One-sided: this adds no other reading, and P is not the chance of a 2028 pass.",
        "zh": "降雨年附註（{lbl}）：在模型自身重播的 2012-22 降雨與去趨勢抽水年份下"
              "（FW.weather_replay.v2，僅用頁面參數組），正確類型的模型通過 E02c 水位規則的機率為 "
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
    v1: O.Paths = field(default_factory=O.Paths)
    outd: str = FW_OUTD
    fw_report: str = FW_REPORT
    index: str = INDEX_PATH
    report: str = REPORT

    def clim(self) -> str:
        return arm_npz(CLIM_ARM, self.outd)

    def seqs(self) -> list[str]:
        return [arm_npz(a, self.outd) for a in RD_ARMS]


def world_files(P: RiderPaths) -> dict:
    """npz and capture of C and every RD arm (repo-relative as given)."""
    out = {}
    for a, p in [(CLIM_ARM, P.clim()), *zip(RD_ARMS, P.seqs(), strict=True)]:
        out[a] = {"npz": p, "capture": p[:-4] + ".fw.npz"}
    return out


def world_shas(P: RiderPaths) -> dict:
    """sha256 of every world file (metadata only; no head value is read)."""
    out = {}
    for a, r in world_files(P).items():
        out[a] = {k: (O.sha256(O.rp(v)) if O.rp(v).is_file() else "MISSING")
                  for k, v in r.items()}
    return out


def ensure_index(P: RiderPaths, code_commit: str = "") -> dict:
    """forcing_year.build_index on C and the RD arms (written once), or the existing index if
    it names the same files at the same sha256. Metadata and sha256 only."""
    q = O.rp(P.index)
    if q.is_file():
        ix = json.loads(q.read_text(encoding="utf-8"))
        want = world_shas(P)
        got = {CLIM_ARM: ix["climatology"]["sha256"],
               **{a: r["sha256"] for a, r in zip(RD_ARMS, ix["sequences"], strict=False)}}
        if any(got.get(a) != want[a]["npz"] for a in want):
            raise EngineeringFailure(f"{P.index} names other files than the declared worlds")
        return ix
    return FY.build_index(q, P.clim(), P.seqs(), source=FW_ID, code_commit=code_commit)


class ReplayContent(EngineeringFailure):
    """A problem in the CONTENT of FW's forwards (the rider cannot rerun them): maps to the
    pre-declared not-evaluable outcome 'replay worlds unusable'."""


def load_worlds(P: RiderPaths, ctx: dict, ix: dict) -> dict:
    """F (11, W, 12) on the window, with the engineering checks of gate E."""
    probs = []
    if ix.get("schema") != FY.INDEX_SCHEMA or ix.get("source_declaration") != FW_ID:
        raise EngineeringFailure("replay index schema or source declaration")
    if ix.get("rain") != WANT_RAIN or ix.get("pumping") != WANT_PUMPING:
        raise ReplayContent(f"index rain/pumping {ix.get('rain')!r}/{ix.get('pumping')!r}, "
                            f"expected {WANT_RAIN!r}/{WANT_PUMPING!r} (FW v2 energy+weather)")
    seqs = ix.get("sequences") or []
    if [r.get("seq") for r in seqs] != list(range(N_SEQ)):
        raise EngineeringFailure("index sequences are not s = 0..10 in order")
    for r in [ix["climatology"], *seqs]:
        for k, hk in (("npz", "sha256"), ("capture", "capture_sha256")):
            if O.sha256(O.rp(r[k])) != r[hk]:
                raise EngineeringFailure(f"{r[k]}: sha256 differs from the index")
    cells, layers = ctx["cells"], ctx["layers"]
    try:
        cm, hc, _ = FY._heads(O.rp(ix["climatology"]["npz"]), layers, cells, O.T_RUN)
    except EngineeringFailure as err:
        raise ReplayContent(str(err)) from err
    copts = FY._opts(cm)
    clabels = [str(x) for x in cm["member_labels"]]
    if len(clabels) < 2 or RD_MEMBERS[0] not in clabels:
        probs.append(f"arm C members {clabels[:3]}... are not the page ensemble")
    H = np.empty((N_SEQ, ctx["W"], O.T_RUN))
    eng = []
    for r in seqs:
        s = r["seq"]
        try:
            m, hs, _ = FY._heads(O.rp(r["npz"]), layers, cells, O.T_RUN)
        except EngineeringFailure as err:
            raise ReplayContent(str(err)) from err
        o = FY._opts(m)
        dates = [str(x)[:7] for x in m["dates"]]
        if dates[0] != O.AXIS_START or int(m["origin"]) != O.ORIGIN_IDX:
            probs.append(f"seq {s}: axis or origin differs")
        if tuple(str(x) for x in m["member_labels"]) != RD_MEMBERS:
            probs.append(f"seq {s}: members are not the page parameter set alone")
        for k in SAME_OPTIONS:
            if o.get(k) != copts.get(k):
                probs.append(f"seq {s}: option {k} differs from arm C")
        if float(m["gain"]) != float(cm["gain"]):
            probs.append(f"seq {s}: gain differs from arm C")
        if r.get("years") != FY.seq_years(s, len(r.get("years") or [])) or \
                len(r.get("years") or []) < 7:
            probs.append(f"seq {s}: declared years are not the cyclic sequence")
        if not (np.isfinite(hs).all() and np.isfinite(hc).all()):
            probs.append(f"seq {s}: non-finite heads at a well")
        H[s] = hs
        eng.append({"seq": s, "years": r["years"], "arm": RD_ARMS[s]})
    hind = [float(np.max(np.abs(H[s][:, :O.T_FIT] - H[0][:, :O.T_FIT]))) for s in range(N_SEQ)]
    if max(hind) > HINDCAST_TOL_M:
        probs.append(f"RD hindcasts differ from each other by up to {max(hind):.3g} m")
    for s in range(N_SEQ):
        for q in range(s):
            if np.array_equal(H[s][:, O.T_FIT:], H[q][:, O.T_FIT:]):
                probs.append(f"seq {s} and seq {q} are the same run")
    if probs:
        raise ReplayContent("replay forwards: " + "; ".join(probs))
    F = np.stack([O.departure(H[s], hc, ctx["mask"]) for s in range(N_SEQ)])
    common = F.mean(axis=0)
    return {"F": F, "eng": {"sequences": eng, "rd_hindcast_max_abs_m": max(hind),
                            "hindcast_tol_m": HINDCAST_TOL_M, "members_clim": len(clabels),
                            "members_rd": list(RD_MEMBERS),
                            "index_sha256": O.sha256(O.rp(P.index)) if O.rp(P.index).is_file()
                            else None},
            "decomposition_report_only": {
                "common_part_rms_m": float(np.sqrt(np.mean(common ** 2))),
                "sequence_part_rms_m": float(np.sqrt(np.mean((F - common[None]) ** 2))),
                "note": "F_s = common part (the parameter set's departure from the ensemble "
                        "plus the mean replayed forcing) + sequence part (the recorded year); "
                        "report-only, never removed from a world"}}


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
    k = lev.shape[1]
    out = {"p": sm["rc"], "by_sequence": sm["by_sequence"],
           "range_over_sequences": sm["range_over_sequences"], "mc_se": sm["mc_se"],
           "mc_half_width_90": sm["mc_half_width_90"], "sampling_11yr": sm["sampling_11yr"],
           "k_per_sequence": int(k),
           "report_only": {
               "beat_cl": float(np.mean([[r["beat_cl"] for r in row] for row in records])),
               "beat_clt": float(np.mean([[r["beat_clt"] for r in row] for row in records])),
               "not_evaluable_worlds": int(sum(r["level"] == "not_evaluable" for row in records
                                               for r in row))}}
    return out


def close(rep: dict, why: str) -> dict:
    rep.update(status="not_evaluable", why=why, reading=None,
               wording=wording_for(None, why=why))
    return rep


def score(P: RiderPaths, *, seed: int, k: int = K_WORLD, k_null: int = K_NULL,
          decl: dict | None = None, decl_sha: str | None = None, smoke: bool = False,
          log=print) -> dict:
    """Gates E, N and MC, then the reading. Returns the report (not written). A scored run
    needs the registered declaration and its sha256 and this rider's seed; a smoke needs the
    smoke seed and worlds outside results/ (no real world is ever scored without the
    registration)."""
    if smoke:
        if seed != SMOKE_SEED or any(_under_results(p) for p in _smoke_paths(P)):
            raise EngineeringFailure("a smoke runs on SMOKE_SEED and synthetic worlds outside "
                                     "results/ only")
    elif decl is None or not decl_sha or len(str(decl_sha)) != 64 or seed != SEED:
        raise EngineeringFailure(f"{ID} scores only under its registration (declaration, its "
                                 "sha256 and the registered seed)")
    rep: dict = {"id": ID, "version": VERSION, "smoke": smoke, "created": O._now(),
                 "declaration_sha256": decl_sha, "seed": seed, "k_per_sequence": k,
                 "k_null": k_null, "label": "model-only rain-year rider; one-sided caveat only",
                 "worlds": {"source": FW_ID, "arms": list(RD_ARMS), "reference": CLIM_ARM},
                 "mc_gate_note": V2.mc_note(N_SEQ, k), "model_only": True}
    ctx = O.load_context(P.v1)
    if decl is not None:
        for name, h in FV1_INPUTS.items():
            if O.sha256(O.rp(P.v1.inputs_dir) / name) != h:
                raise EngineeringFailure(f"{name} differs from FV v2's registered sha256")
        if not O.rp(P.fw_report).is_file():
            raise EngineeringFailure(f"{P.fw_report} missing: FW v2's forwards stay sealed")
        rep["fw_report_sha256"] = O.sha256(O.rp(P.fw_report))
        want = (decl.get("worlds") or {}).get("files") or {}
        got = world_shas(P)
        bad = [a for a in got if got[a] != want.get(a)]
        if bad:
            raise EngineeringFailure(f"world files differ from the declared sha256: {bad}")
    ix = ensure_index(P)
    try:
        W = load_worlds(P, ctx, ix)
    except ReplayContent as err:
        rep["engineering"] = {"replay_content_failure": str(err)[:500]}
        return close(rep, f"{UNUSABLE}: {str(err)[:200]}")
    rep["engineering"] = W["eng"]
    rep["decomposition_report_only"] = W["decomposition_report_only"]
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
    recs = world_records(ctx, W["F"], k=k, seed=seed)
    sm = summarise(recs)
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


def load_registered(prereg_id: str = ID, prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    """Refuse unless the declaration is registered, intact, its frozen code byte-identical and
    its design this run's."""
    from harness import prereg3

    if prereg_id != ID:
        raise EngineeringFailure(f"this runner scores {ID} only, not {prereg_id!r}")
    if index_record(prereg_id, prereg_dir) is None:
        raise EngineeringFailure(f"{prereg_id} is not in INDEX.jsonl: register it first")
    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    try:
        doc, sha = prereg3.load_prereg(prereg_id, **kw)
    except Exception as err:      # noqa: BLE001 - any failure is a refusal
        raise EngineeringFailure(f"{prereg_id} is not registered or not intact: {err}") from err
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
    want = {"seed": SEED, "K_WORLD": K_WORLD, "K_NULL": K_NULL, "P_weak": P_WEAK,
            "streams": {"world": STREAM_WORLD, "null": STREAM_NULL},
            "hindcast_tol_m": HINDCAST_TOL_M}
    bad = sorted(k for k, v in want.items() if d.get(k) != v)
    if bad:
        raise EngineeringFailure(f"the registered design differs from this code on {bad}")
    return doc, sha


def attempts_path(out: str = REPORT) -> Path:
    return O.rp(out).parent / ATTEMPTS_NAME


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
    """Score once: refuses unless registered and intact; writes report.json once (mode 'x').
    Every attempt appends 'start' and then 'engineering_failure', 'error' or 'success' (with
    the report sha256) to attempts.jsonl beside the report; no statistic is logged."""
    P = P or RiderPaths()
    outp = O.rp(P.report)
    ap = attempts_path(P.report)
    n = _attempt_no(ap)
    _log_attempt(ap, {"event": "start", "attempt": n, "report": str(P.report),
                      "code_sha256": O.sha256(Path(__file__))})
    try:
        if outp.exists():
            raise EngineeringFailure(f"{outp} exists: {ID} is scored once")
        doc, dsha = load_registered(ID, prereg_dir, rows_dir)
        rep = score(P, seed=check_seed(SEED), decl=doc, decl_sha=dsha, log=log)
        rep["declaration_sha256"] = dsha
        rep["frozen_code"] = doc["frozen_code"]
        O._write_json_once(outp, rep)
    except EngineeringFailure as err:
        _log_attempt(ap, {"event": "engineering_failure", "attempt": n, "message": str(err)})
        raise
    except Exception as err:
        _log_attempt(ap, {"event": "error", "attempt": n,
                          "message": f"{type(err).__name__}: {err}"})
        raise
    _log_attempt(ap, {"event": "success", "attempt": n, "report_sha256": O.sha256(outp),
                      "declaration_sha256": dsha})
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


def _smoke_paths(P: RiderPaths) -> list[str]:
    v = P.v1
    return [P.outd, P.index, P.report, P.fw_report, v.pred, v.protocol, v.baselines, v.wells,
            v.inputs_dir]


def build_synthetic(out: Path, seed: int = 5, amp: float = 0.6, common: float = 0.2
                    ) -> RiderPaths:
    """SYNTHETIC inputs only (no real data): FV v1's synthetic set (osse_fv.build_synthetic)
    plus an arm C (several members) and eleven RD arms (one member) with their capture
    sidecars, in FW v2's real layout and metadata."""
    import pandas as pd

    out = Path(out)
    v1 = O.build_synthetic(out / "v1", nonlinear=0.01)
    with np.load(v1.zr_forward, allow_pickle=False) as z:
        H = np.asarray(z["heads_mean"], float)[0]
    L_, C_, T0 = H.shape
    T = 252
    H = np.concatenate([H, np.repeat(H[:, :, -1:], T - T0, axis=2)], axis=2)
    rng = np.random.default_rng(seed)
    year_eff = rng.normal(0.0, 1.0, N_SEQ)
    set_off = common * rng.normal(0.0, 1.0, (L_, C_))
    t = np.arange(T)
    outd = out / "fw_replay_v2"
    outd.mkdir(parents=True)
    labels_c = np.array(list(RD_MEMBERS) + ["fold0", "fold1"])

    def save(arm, heads, pf, years=None, labels=RD_MEMBERS):
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
            origin=O.ORIGIN_IDX, horizon=T - O.T_FIT, n_members=len(labels),
            member_labels=np.array(list(labels)), gain=1.0, hindcast_gain=0.0,
            gate=json.dumps({"git_commit": "synthetic"}),
            forward_options=json.dumps({"update_method": "innovation", "column_heads": "free",
                                        "restart_taper_km": 5.0, "apex_hold": "calibrated",
                                        "column_hpc0_fast_days": 365.0}))

    save(CLIM_ARM, H, "climatology", labels=labels_c)
    base = H + set_off[:, :, None]          # one parameter set: its own offset at every month
    for s in range(N_SEQ):
        yrs = FY.seq_years(s, N_SEQ)
        eff = np.zeros(T)
        for i in range(O.T_FIT, T):
            eff[i] = year_eff[yrs[((i - O.T_FIT) // 12) % N_SEQ] - FIRST_YEAR] * (
                1.0 - np.exp(-(i - O.T_FIT + 1) / 6.0))
        save(RD_ARMS[s], base + amp * eff[None, None, :] * (t >= O.T_FIT),
             f"replay:{yrs[0]}-...", yrs)
    rep = out / "fw_report.json"
    rep.write_text(json.dumps({"prereg_id": FW_ID, "verdict": "scored"}) + "\n")
    return RiderPaths(v1=v1, outd=str(outd), fw_report=str(rep),
                      index=str(out / "replay_index.json"), report=str(out / "report.json"))


def run_smoke(out: Path, k: int = 40, k_null: int = 20) -> dict:
    """Synthetic end to end on SMOKE_SEED; writes out/smoke_report.json once. Numbers mean
    nothing about the twin."""
    out = Path(out)
    if _under_results(out):
        raise EngineeringFailure("the smoke never writes under results/")
    P = build_synthetic(out / "inputs_synthetic")
    rep = score(P, seed=SMOKE_SEED, k=k, k_null=k_null, smoke=True, log=lambda *_: None)
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
            doc, dsha = load_registered(ID, prereg_dir, rows_dir)
            got = world_shas(RiderPaths())
            want = (doc.get("worlds") or {}).get("files") or {}
            bad = [k for k in got if got[k] != want.get(k)]
            if bad:
                raise EngineeringFailure(f"world files differ from the declared sha256: {bad}")
            print(f"{ID} registered (sha256 {dsha[:16]}); code, inputs and worlds intact")
            return 0
        rep = run(prereg_dir=prereg_dir, rows_dir=rows_dir)
        print(f"wrote {REPORT} (status {rep['status']}; reading {rep.get('reading')})")
        return 0
    except EngineeringFailure as err:
        print(f"REFUSED / ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
