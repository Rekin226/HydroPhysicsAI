"""Reading 2 of FV.decidability.v2_postfailure (NEW, declared as new, not a v1 reading):
does the frozen combined 90 % head interval of E02 include the model's OWN forcing-year
variation?

Worlds: round-7 rank 11's replay forwards (FW.weather_replay.v1, ``python -m
hydrophysics.twin.replay --projection-forcing replay:<years> -- <forward arguments>``): its
primary arms W2012..W2022, the page ensemble (datum_gate_innov flags, page column,
--column-heads free) from the 2022-12 origin, sequence s replaying the recorded calendar
years 2012+s, 2013+s, ... (cyclic over 2012..2022), pumping as anomalies about a declared
2012-22 trend (detrended), plus its arm C, the same ensemble and code under climatology. The
capture sidecars' records (<arm>.fw.npz 'record') name each run's replay; FW's runner seals
the outputs until FW's report exists, so this reading is scored only after it. The forcing
departure of sequence s is F_s = h_s - h_clim (same code, members and restart, so F holds the
weather only). Model-only: no observation is read; rank 11's forwards are read only here, by
this scorer, after this declaration is registered.

* RC (per county): the share of window well-months (2026-11..2027-10, the 158 wells) with
  ZR + F_s + e inside E02's frozen [lower, upper], over 11 sequences x K_RC AR(1) noise draws
  (v1's noise inputs). A perfect-model-unknown-weather coverage. Rule, per county:
  RC < 0.85 -> 'omits' (the interval leaves out the model's own forcing-year variation);
  RC >= 0.88 -> 'not_shown_to_omit'; otherwise no line. Point estimate; the Monte Carlo SE
  and the 11-year sampling uncertainty (t interval and bootstrap over sequences) are
  reported, the t interval is quoted in the wording, and neither moves a threshold.
  Pumping-only replay (FW ran --replay-fields energy, index rain 'climatology'): the model
  then holds NO Changhua forcing-year variation (rain at climatology, no Changhua pumping in
  the page model), so Changhua gets no C14a line ('not_represented') and the Changhua
  sentence says so; with rain replayed, Changhua's line carries 'rain-year range only'.
* Lead ratio (per county): median over wells of sd_s(F_s(w, L)) / sigma_w(L) at leads 12,
  24, 36, 47-58 (mean over the window leads), 70 and 82 months; sigma_w(L) is the frozen
  recipe sqrt(max(sigma_ens, 1e-3)^2 + sd_w^2), sigma_ens from the registered forward's
  heads_std and sd_w solved from the window rows (pre-declared alternative if that
  reconstruction misses by > 1e-4 m: the window-mean sigma at every lead, labelled).
  C14b -> L if the 47-58 ratio exceeds 1.2, read per the declared trigger: 'either_county'
  (per county, either one; the cautious choice) or 'pooled' (all wells of both counties),
  chosen explicitly by the lead at declaration (prereg_fv2.py --c14b-trigger).
* ICC_year (REPORT-ONLY): the county-common share of the window-mean forcing departure
  across sequences (forcing only, and with the AR(1) noise added).
* Aggregate forcing range (C14c): min..max over sequences of the N2-unit mean (the 84 Yunlin
  wells of the 158, N2's primary) and of the fan layer-2 mean (every active cell), by lead.

Gates: E (index, shas, metadata, hindcast identity 0.0 m between every replay and the
climatology run at the wells, sigma reconstruction or the declared alternative);
N-cov (null world F = 0: Monte Carlo coverage within 0.02 of the analytic value per county,
else not evaluable, a possible bug); MC (1.645 SE of RC <= 0.07; non-binding at K_RC = 250,
the largest possible half-width being 0.016).

Gate E on another declaration's forwards: an engineering retry cannot rerun FW's forwards.
A gate-E failure found in their CONTENT (capture records, metadata, axis, members, months,
hindcast identity, identical sequences: ``ReplayGateE``) is logged in FV v2's attempts.jsonl
and maps to the pre-declared 'replay worlds unusable' not-evaluable outcome. FV-side problems
(index schema, sha mismatch, a missing file) stay ordinary engineering failures (retry free).
``--replay-unavailable`` is accepted only on a pre-declared ground: FW's report states it
not evaluable; or FW's report is absent on or after 2027-01-31; or a ReplayGateE is logged.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "fv_decidability"),
           str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osse_fv as O  # noqa: E402
import osse_fv2 as V2  # noqa: E402
import score_e02c as S  # noqa: E402

N_SEQ = 11
FIRST_YEAR = 2012
LEAD_MAX = 82
T_NEED = O.ORIGIN_IDX + LEAD_MAX + 1              # months 2012-01..2029-10 (214)
LEADS = {"12": (12,), "24": (24,), "36": (36,), "47-58": tuple(range(47, 59)), "70": (70,),
         "82": (82,)}
WINDOW_LEADS = "47-58"
RC_LOW = 0.85
RC_HIGH = 0.88
LEAD_RATIO_L = 1.2
NULL_COV_TOL = 0.02
K_RC = 250
K_RC_NULL = 500
STREAM_RC, STREAM_RC_NULL, STREAM_BOOT = 11, 12, 13
B_BOOT = 2000
T90_DF10 = 1.8124611228107335                     # Student t, 0.95 quantile, 10 df
SIGMA_FLOOR = 1e-3                                # export_baselines.SIGMA_FLOOR (E02 recipe)
SIGMA_REC_TOL_M = 1e-4
DAMPING = 0.73
AFTER_THE_FACT_COV = 0.926
SOURCE_DECL = "FW.weather_replay.v1"
# rank 11's layout (dataprep_staging/fw_replay/run_fw.sh): arm C = the page ensemble under
# climatology (reference), arm W<y> = the page ensemble replaying the cyclic order of
# 2012..2022 starting at y (sequence s = W(2012+s)); each <arm>.npz has a capture sidecar
# <arm>.fw.npz whose 'record' names the replay (years, pumping, fields). Its runner seals the
# outputs until FW's report exists, so reading 2 is scored only after that report.
FW_OUTD = "results/twin_forward/fw_replay"
FW_REPORT = f"results/dataprep/reports/{SOURCE_DECL}/report.json"
FIELDS_TO_RAIN = {"energy+weather": "replayed", "energy": "climatology"}
INDEX_SCHEMA = "fv2-replay-index/1"
RAIN_MODES = ("replayed", "climatology")
PUMP_MODE = "detrended_anomaly"
N2_UNIT = "Yunlin"                                # N2's primary: the 84 Yunlin wells of the 158
NOT_REPRESENTED = "not_represented"               # Changhua under a pumping-only replay
C14B_TRIGGERS = ("either_county", "pooled")
C14B_DEFAULT = "either_county"                    # smoke only; the declaration states it
UNAVAILABLE_FROM = dt.datetime(2027, 1, 31, tzinfo=dt.timezone.utc)
GATE_E_EVENT = "forcing_replay_gate_e"
EngineeringFailure = O.EngineeringFailure


class ReplayGateE(EngineeringFailure):
    """Gate E failed on the CONTENT of FW's replay forwards (not FV's own index): FV cannot
    rerun them, so this maps to the pre-declared 'replay worlds unusable' outcome."""


def log_replay_gate_e(attempts: Path, index: str | None, err: str) -> None:
    attempts = Path(attempts)
    attempts.parent.mkdir(parents=True, exist_ok=True)
    rec = {"event": GATE_E_EVENT, "time": O._now(), "index": index, "error": err}
    with open(attempts, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def fw_report_status(path=None) -> str | None:
    """FW's report verdict ('scored' / 'not_evaluable'), or None if absent (metadata only)."""
    q = O.rp(path or FW_REPORT)
    if not q.is_file():
        return None
    r = json.loads(q.read_text(encoding="utf-8"))
    return str(r.get("verdict") or r.get("status"))


def unavailable_grounds(attempts=None, fw_report=None, now=None) -> list[str]:
    """The pre-declared grounds that allow 'replay worlds unavailable' (empty: none)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    st = fw_report_status(fw_report)
    out = []
    if st == "not_evaluable":
        out.append(f"{SOURCE_DECL} report states not_evaluable")
    if st is None and now >= UNAVAILABLE_FROM:
        out.append(f"{SOURCE_DECL} report absent on {now:%Y-%m-%d} (>= 2027-01-31)")
    a = O.rp(attempts) if attempts else None
    if a is not None and a.is_file():
        ev = [json.loads(x) for x in a.read_text(encoding="utf-8").splitlines() if x.strip()]
        out += [f"FV gate E failed on FW's replay forwards at {e.get('time')}: "
                f"{str(e.get('error'))[:200]}" for e in ev if e.get("event") == GATE_E_EVENT]
    return out


def seq_years(s: int, n: int) -> list[int]:
    """Calendar years replayed by sequence ``s`` for projection years 1..n (cyclic)."""
    return [FIRST_YEAR + (s + j) % N_SEQ for j in range(n)]


def lead_index(lead: int) -> int:
    return O.ORIGIN_IDX + lead


# ----------------------------------------------------------------------------- wording
_LBL_EN = "post-failure exploratory, FV.decidability.v2"
_LBL_ZH = "失敗後探索性研究，FV.decidability.v2"
WORDING = {
    "rc": {
        "omits": {
            "en": "{county}: in the model's own replayed forcing years ({forcing}; {lbl}), "
                  "the frozen 90 % head interval would cover {rc:.0%} of the window's "
                  "well-months even if the model were exactly right (90 % interval over the "
                  "11 recorded years {t_lo:.0%}-{t_hi:.0%}; single years {lo:.0%}-{hi:.0%}). "
                  "The interval leaves out the model's own forcing-year variation, so some "
                  "under-coverage is expected from forcing-year variation alone. This is "
                  "model-only and the model's amplitude is damped (about 0.73); the 92.6 % "
                  "coverage measured after the fact on 2023-26 observations is a different "
                  "quantity and is shown separately.",
            "zh": "{county}：在模型自身重播的逐年驅動下（{forcing}；{lbl}），即使模型完全正確，"
                  "凍結的 90% 水位區間也只涵蓋試驗期井月的 {rc:.0%}"
                  "（11 個紀錄年的 90% 區間為 {t_lo:.0%}-{t_hi:.0%}；"
                  "單一年份為 {lo:.0%}-{hi:.0%}）。"
                  "此區間未納入模型自身的逐年驅動變異，因此僅逐年驅動變異即可能造成涵蓋不足。"
                  "此為純模型結果，且模型振幅偏小（約 0.73）；事後以 2023-26 觀測量得的 92.6% "
                  "涵蓋率是不同的量，另行顯示。"},
        "not_shown_to_omit": {
            "en": "{county}: in the model's own replayed forcing years ({forcing}; {lbl}), "
                  "the frozen 90 % head interval would cover {rc:.0%} of the window's "
                  "well-months if the model were exactly right (90 % interval over the 11 "
                  "recorded years {t_lo:.0%}-{t_hi:.0%}; single years {lo:.0%}-{hi:.0%}); its "
                  "width is not shown to omit forcing-year variation. Model-only and "
                  "optimistic: the model's amplitude is damped (about 0.73).",
            "zh": "{county}：在模型自身重播的逐年驅動下（{forcing}；{lbl}），若模型完全正確，"
                  "凍結的 90% 水位區間涵蓋試驗期井月的 {rc:.0%}"
                  "（11 個紀錄年的 90% 區間為 {t_lo:.0%}-{t_hi:.0%}；"
                  "單一年份為 {lo:.0%}-{hi:.0%}）；"
                  "未顯示其寬度遺漏逐年驅動變異。"
                  "此為純模型結果且偏樂觀：模型振幅偏小（約 0.73）。"}},
    "changhua": {
        "replayed": {
            "en": "Changhua: rain-year range only; the page model carries no Changhua pumping.",
            "zh": "彰化：僅含降雨年際範圍；頁面模型不含彰化抽水。"},
        "climatology": {
            "en": "Changhua: no Changhua forcing-year variation is represented in this reading "
                  "(pumping-only replay with rain at climatology, and the page model carries "
                  "no Changhua pumping), so Changhua gets no line from it.",
            "zh": "彰化：本解讀未呈現任何彰化逐年驅動變異（僅重播抽水、降雨取氣候值，且頁面模型"
                  "不含彰化抽水），因此彰化不因此增加任何說明。"}},
    "forcing": {
        "replayed": {"en": "rain and detrended pumping replayed from 2012-22",
                     "zh": "重播 2012-22 的降雨與去趨勢抽水"},
        "climatology": {"en": "pumping years only, rain at climatology",
                        "zh": "僅抽水年際變化，降雨取氣候值"}},
    "c14b": {
        "L": {"en": "Outside the window the interval keeps its recipe while the model's own "
                    "forcing-year spread at 47-58 months is {lr:.2f} times sigma in {county} "
                    "({lbl}); the recipe is narrowed (L) outside the window (model-only).",
              "zh": "試驗期之外，區間沿用同一作法，但模型自身在 47-58 個月的逐年驅動離散度為 "
                    "{county} sigma 的 {lr:.2f} 倍（{lbl}）；"
                    "該作法在試驗期外限縮為 L（純模型結果）。"},
        "stays_N": {"en": "Lead curve (model-only, {lbl}): the model's own forcing-year spread "
                          "relative to sigma is {curve}; outside the window the recipe stays "
                          "untested.",
                    "zh": "前置時間曲線（純模型，{lbl}）："
                          "模型自身逐年驅動離散度相對 sigma 為 {curve}；"
                          "試驗期之外此作法仍未經檢驗。"}},
    "c14c": {
        "en": "run spread, a parameter spread, plus the model's own forcing-year range "
              "({n2_lo:+.2f} to {n2_hi:+.2f} m for the N2-unit mean and {l2_lo:+.2f} to "
              "{l2_hi:+.2f} m for the fan layer-2 mean in 2026-11..2027-10; {forcing}); not a "
              "predictive interval; the model's amplitude is damped (about 0.73) ({lbl})",
        "zh": "執行離散度（參數離散）加上模型自身的逐年驅動範圍（2026-11..2027-10 期間 N2 單元平均 "
              "{n2_lo:+.2f} 至 {n2_hi:+.2f} m，扇地第 2 層平均 {l2_lo:+.2f} 至 {l2_hi:+.2f} m；"
              "{forcing}）；並非預測區間；模型振幅偏小（約 0.73）（{lbl}）"},
    "not_evaluable": {
        "en": "The forcing-year reading of FV.decidability.v2 (post-failure, exploratory) is "
              "not evaluable ({why}); C14a, C14b and C14c gain nothing from it.",
        "zh": "FV.decidability.v2（失敗後探索性）的逐年驅動解讀無法評估（{why}）；C14a、C14b 與 "
              "C14c 不因此改變。"},
}


def rc_reading(rc: float | None) -> str | None:
    if rc is None or not np.isfinite(rc):
        return None
    if rc < RC_LOW:
        return "omits"
    if rc >= RC_HIGH:
        return "not_shown_to_omit"
    return "no_line"


# ----------------------------------------------------------------------------- index
def _meta(npz) -> dict:
    with np.load(npz, allow_pickle=False) as z:
        return {k: z[k] for k in ("scenario_names", "dates", "origin", "member_labels",
                                  "forward_options", "gain")
                if k in z.files} | {"shape": tuple(z["heads_mean"].shape)
                                    if "heads_mean" in z.files else None}


def _opts(meta: dict) -> dict:
    return json.loads(str(meta["forward_options"]))


def replay_tag(opts: dict) -> str | None:
    """The forward option that names the projection forcing, if the run records one (rank
    11's engine is opt-in; its key name is not fixed here): the first key containing
    'projection' or 'replay', as 'key=value'."""
    for k, v in opts.items():
        if "projection" in k or "replay" in k:
            return f"{k}={v}"
    return None


def capture_record(path) -> dict:
    """The 'record' JSON of a replay capture sidecar (metadata only; no array is loaded)."""
    with np.load(O.rp(path), allow_pickle=False) as z:
        return json.loads(str(z["record"]))


def check_capture(rec: dict, s: int | None) -> list[str]:
    """The capture names a climatology run (s None) or sequence s's cyclic replay with
    detrended pumping."""
    pf, rp = rec.get("projection_forcing"), rec.get("replay")
    if s is None:
        return [] if pf == "climatology" and rp is None else [f"climatology capture: {pf!r}"]
    out = []
    years = (rp or {}).get("years") or []
    if not str(pf).startswith("replay:") or rp is None:
        out.append(f"seq {s}: capture {pf!r} is not a replay")
    if len(years) < 7 or list(years) != seq_years(s, len(years)):
        out.append(f"seq {s}: capture years {years[:3]}... are not the cyclic sequence")
    if (rp or {}).get("pumping") != "detrended":
        out.append(f"seq {s}: capture pumping {(rp or {}).get('pumping')!r}")
    if (rp or {}).get("fields") not in FIELDS_TO_RAIN:
        out.append(f"seq {s}: capture fields {(rp or {}).get('fields')!r}")
    return out


def default_layout() -> tuple[str, list[str]]:
    return (f"{FW_OUTD}/C.npz", [f"{FW_OUTD}/W{FIRST_YEAR + s}.npz" for s in range(N_SEQ)])


def _capture_of(npz: str) -> str:
    return npz[:-4] + ".fw.npz" if npz.endswith(".npz") else npz + ".fw.npz"


def build_index(out: Path, clim: str, seqs: list[str], source: str = SOURCE_DECL,
                code_commit: str = "") -> dict:
    """Index of rank 11's replay forwards and their capture sidecars, written once (lead,
    after FW's report exists). Metadata and sha256 only: no head value is read. The rain
    mode follows the captures' replayed fields (energy+weather -> replayed; energy -> rain
    at climatology, 'pumping years only')."""
    if len(seqs) != N_SEQ:
        raise EngineeringFailure(f"{len(seqs)} sequences given, expected {N_SEQ} (s = 0..10)")
    probs = check_capture(capture_record(_capture_of(clim)), None)
    seq_recs = []
    for s, p in enumerate(seqs):
        cr = capture_record(_capture_of(p))
        probs += check_capture(cr, s)
        seq_recs.append({"seq": s, "years": list((cr.get("replay") or {}).get("years") or []),
                         "fields": (cr.get("replay") or {}).get("fields"),
                         "npz": str(p), "sha256": O.sha256(O.rp(p)),
                         "capture": _capture_of(str(p)),
                         "capture_sha256": O.sha256(O.rp(_capture_of(str(p)))),
                         "projection_forcing": cr.get("projection_forcing")})
    fields = {r["fields"] for r in seq_recs}
    if len(fields) != 1:
        probs.append(f"replayed fields differ between sequences: {sorted(map(str, fields))}")
    if probs:
        raise EngineeringFailure("replay captures: " + "; ".join(probs))
    rec = {"schema": INDEX_SCHEMA, "created": O._now(), "source_declaration": source,
           "code_commit": code_commit, "rain": FIELDS_TO_RAIN[fields.pop()],
           "pumping": PUMP_MODE,
           "climatology": {"npz": str(clim), "sha256": O.sha256(O.rp(clim)),
                           "capture": _capture_of(str(clim)),
                           "capture_sha256": O.sha256(O.rp(_capture_of(str(clim))))},
           "sequences": seq_recs}
    O._write_json_once(Path(out), rec)
    return rec


def add_index_args(ap) -> None:
    ap.add_argument("--out", required=True)
    ap.add_argument("--clim", default=None, help=f"climatology run (default {FW_OUTD}/C.npz)")
    ap.add_argument("--seq", action="append", default=None,
                    help=f"replay run of sequence s, in order s = 0..10 (default "
                         f"{FW_OUTD}/W2012.npz .. W2022.npz)")
    ap.add_argument("--code-commit", default="")


def build_index_cli(a) -> dict:
    if not O.rp(FW_REPORT).is_file():
        raise EngineeringFailure(f"{FW_REPORT} missing: rank 11's outputs stay sealed until "
                                 "its report exists")
    clim, seqs = default_layout()
    return build_index(O.rp(a.out), a.clim or clim, a.seq or seqs, code_commit=a.code_commit)


def _heads(npz, layers, cells, t_need: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """(meta, baseline heads at the wells (W, t_need), baseline layer-2 field (C, t_need))."""
    with np.load(npz, allow_pickle=False) as z:
        names = [str(x) for x in z["scenario_names"]]
        if "baseline" not in names:
            raise EngineeringFailure(f"{npz}: no baseline scenario")
        sc = names.index("baseline")
        hm = z["heads_mean"]
        if hm.shape[-1] < t_need:
            raise EngineeringFailure(f"{npz}: {hm.shape[-1]} months < {t_need} (to 2029-10)")
        w = np.asarray(hm[sc][layers, cells, :t_need], float)
        l2 = np.asarray(hm[sc][1, :, :t_need], float)
        meta = {k: z[k] for k in z.files if k not in ("heads_mean", "heads_std", "subs_mean",
                                                       "subs_std", "subs_mean_by_rheology")}
    return meta, w, l2


def load_replay(index_path, cells, layers, *, t_need: int = T_NEED) -> dict:
    """F (11, W, t_need) at the wells and F_l2 (11, C, t_need) on layer 2, with the
    engineering checks. Raises EngineeringFailure on an FV-side problem (index schema, order,
    sha) and ReplayGateE on a problem in the content of FW's forwards."""
    ix = json.loads(O.rp(index_path).read_text(encoding="utf-8"))
    probs = []
    if ix.get("schema") != INDEX_SCHEMA:
        probs.append("index schema")
    if ix.get("rain") not in RAIN_MODES or ix.get("pumping") != PUMP_MODE:
        probs.append("index rain/pumping modes")
    seqs = ix.get("sequences") or []
    if [r.get("seq") for r in seqs] != list(range(N_SEQ)):
        probs.append("sequences are not s = 0..10 in order")
    for r in [ix.get("climatology") or {}, *seqs]:
        for k in ("npz", "capture"):
            if not r.get(k) or O.sha256(O.rp(r[k])) != r.get(f"{k}_sha256" if k == "capture"
                                                             else "sha256"):
                probs.append(f"{r.get(k)}: sha256 differs from the index")
    if probs:
        raise EngineeringFailure("replay index: " + "; ".join(probs))
    probs += check_capture(capture_record(ix["climatology"]["capture"]), None)
    for r in seqs:
        probs += check_capture(capture_record(r["capture"]), r["seq"])
    if probs:
        raise ReplayGateE("replay captures: " + "; ".join(probs))
    try:
        cm, hc, lc = _heads(O.rp(ix["climatology"]["npz"]), layers, cells, t_need)
    except EngineeringFailure as err:
        raise ReplayGateE(str(err)) from err
    copts = _opts(cm)
    ctag = replay_tag(copts)
    if ctag is not None and "climatology" not in ctag:
        probs.append(f"climatology run records {ctag!r}")
    labels = [str(x) for x in cm["member_labels"]]
    W, C = hc.shape[0], lc.shape[0]
    F = np.empty((N_SEQ, W, t_need))
    Fl2 = np.empty((N_SEQ, C, t_need))
    eng = []
    for r in seqs:
        s = r["seq"]
        try:
            m, hs, ls = _heads(O.rp(r["npz"]), layers, cells, t_need)
        except EngineeringFailure as err:
            raise ReplayGateE(str(err)) from err
        o = _opts(m)
        pf = replay_tag(o)
        dates = pd.DatetimeIndex(pd.to_datetime(m["dates"])).to_period("M").to_timestamp()
        if dates[0] != pd.Timestamp(f"{O.AXIS_START}-01") or int(m["origin"]) != O.ORIGIN_IDX:
            probs.append(f"seq {s}: axis or origin differs")
        if [str(x) for x in m["member_labels"]] != labels:
            probs.append(f"seq {s}: members differ from the climatology run")
        if pf is not None and "replay" not in pf:
            probs.append(f"seq {s}: records {pf!r}, not a replay")
        for k in ("update_method", "column_heads", "restart_taper_km", "apex_hold",
                  "column_hpc0_fast_days"):
            if o.get(k) != copts.get(k):
                probs.append(f"seq {s}: option {k} differs from the climatology run")
        if float(m["gain"]) != float(cm["gain"]):
            probs.append(f"seq {s}: gain differs")
        if r.get("years") != seq_years(s, len(r.get("years") or [])) or \
                len(r.get("years") or []) < 7:
            probs.append(f"seq {s}: declared years are not the cyclic sequence")
        if not (np.isfinite(hs).all() and np.isfinite(hc).all()):
            probs.append(f"seq {s}: non-finite heads at a well")
        hind = float(np.max(np.abs(hs[:, :O.T_FIT] - hc[:, :O.T_FIT])))
        if hind > V2.HINDCAST_TOL_M:
            probs.append(f"seq {s}: hindcast differs from the climatology run by {hind:.3g} m")
        F[s] = hs - hc
        Fl2[s] = ls - lc
        if not np.any(F[s][:, O.T_FIT:] != 0.0):
            probs.append(f"seq {s}: projection identical to the climatology run")
        eng.append({"seq": s, "years": r["years"], "replay_tag": pf or "not recorded",
                    "hindcast_max_abs_m": hind})
    for s in range(N_SEQ):
        for q in range(s):
            if np.array_equal(F[s], F[q]):
                probs.append(f"seq {s} and seq {q} are the same run")
    if probs:
        raise ReplayGateE("replay forwards: " + "; ".join(probs))
    active = np.isfinite(Fl2).all(axis=(0, 2))
    return {"F": F, "F_l2": Fl2[:, active], "n_active_l2": int(active.sum()), "eng": eng,
            "rain": ix["rain"], "pumping": ix["pumping"], "members": labels,
            "index_sha256": O.sha256(O.rp(index_path)), "source": ix.get("source_declaration")}


# ----------------------------------------------------------------------------- sigma by lead
def registered_sigma_ens(npz, cells, layers, t_need: int = T_NEED) -> np.ndarray:
    with np.load(npz, allow_pickle=False) as z:
        sc = [str(x) for x in z["scenario_names"]].index("baseline")
        return np.asarray(z["heads_std"][sc][layers, cells, :t_need], float)


def sigma_by_lead(lo: np.ndarray, hi: np.ndarray, sig_ens: np.ndarray) -> dict:
    """The frozen recipe at every lead: sqrt(max(sigma_ens, floor)^2 + sd_w^2), sd_w^2 the
    window mean of sigma_pred^2 - max(sigma_ens, floor)^2 (sigma_pred = (upper - lower) / 2z).
    If it misses sigma_pred by more than SIGMA_REC_TOL_M in the window, the declared
    alternative 'sigma_const' (window-mean sigma_pred at every lead) is used, labelled."""
    sp = (hi - lo) / (2.0 * O.Z90)
    se = np.maximum(sig_ens, SIGMA_FLOOR)
    win = se[:, O.WIN[0]:O.WIN[1]]
    sd2 = np.mean(sp ** 2 - win ** 2, axis=1)
    rec = np.sqrt(se ** 2 + np.maximum(sd2, 0.0)[:, None])
    err = float(np.max(np.abs(rec[:, O.WIN[0]:O.WIN[1]] - sp)))
    ok = err <= SIGMA_REC_TOL_M and bool((sd2 >= 0).all())
    sigma = rec if ok else np.repeat(sp.mean(axis=1)[:, None], se.shape[1], axis=1)
    return {"sigma": sigma, "method": "recipe" if ok else "sigma_const",
            "reconstruction_max_abs_m": err, "tolerance_m": SIGMA_REC_TOL_M}


# ----------------------------------------------------------------------------- statistics
def _ncdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.vectorize(math.erfc)(-np.asarray(x, float) / math.sqrt(2.0))


def analytic_coverage(zr, lo, hi, sd, sel) -> float:
    """Expected share of the selected wells' window months inside [lo, hi] when
    obs = zr + e, e ~ N(0, sd_w) (the stationary AR(1) marginal)."""
    s = np.asarray(sd, float)[sel, None]
    p = _ncdf((hi[sel] - zr[sel]) / s) - _ncdf((lo[sel] - zr[sel]) / s)
    return float(p.mean())


def _rng(stream: int, a: int, b: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([O.SEED, stream, a, b]))


def coverage_draws(zr, lo, hi, Fw, sd, phi, county, k: int, stream: int) -> dict:
    """{county: (n_seq, k)} coverage of window well-months, obs = zr + Fw[s] + e_(s, k)."""
    out = {c: np.empty((Fw.shape[0], k)) for c in S.COUNTIES_COV}
    sel = {c: county == c for c in S.COUNTIES_COV}
    for s in range(Fw.shape[0]):
        for j in range(k):
            e = O.simulate_ar1(sd, phi, O.N_WIN, _rng(stream, s, j))
            obs = zr + Fw[s] + e
            hit = (obs >= lo) & (obs <= hi)
            for c in S.COUNTIES_COV:
                out[c][s, j] = float(hit[sel[c]].mean())
    return out


def summarise_rc(cov: np.ndarray) -> dict:
    """Point estimate, Monte Carlo SE and the 11-year sampling uncertainty of RC."""
    n, k = cov.shape
    rc_s = cov.mean(axis=1)
    rc = float(rc_s.mean())
    mc_se = float(math.sqrt(np.sum(cov.var(axis=1, ddof=1)) / k) / n) if k > 1 else float("nan")
    se11 = float(rc_s.std(ddof=1) / math.sqrt(n))
    rng = _rng(STREAM_BOOT, 0, 0)
    boot = rc_s[rng.integers(0, n, size=(B_BOOT, n))].mean(axis=1)
    return {"rc": rc, "by_sequence": [float(x) for x in rc_s],
            "range_over_sequences": [float(rc_s.min()), float(rc_s.max())],
            "mc_se": mc_se, "mc_half_width_90": V2.Z_MC * mc_se,
            "sampling_11yr": {"se": se11, "t90": [rc - T90_DF10 * se11, rc + T90_DF10 * se11],
                              "bootstrap90_over_sequences": [float(np.quantile(boot, 0.05)),
                                                             float(np.quantile(boot, 0.95))],
                              "note": "uncertainty from the 11 recorded years; never used to "
                                      "move a threshold"}}


def lead_ratio(F: np.ndarray, sigma: np.ndarray, county: np.ndarray) -> dict:
    """Per county (and 'pooled' over the wells of both counties) and lead set: median over
    wells of mean_L sd_s(F_s(w, L)) / sigma_w(L), with a bootstrap over sequences (5-95 %)."""
    rng = _rng(STREAM_BOOT, 1, 0)
    pick = rng.integers(0, F.shape[0], size=(200, F.shape[0]))
    out: dict = {}
    groups = {c: county == c for c in S.COUNTIES_COV}
    groups["pooled"] = np.isin(county, S.COUNTIES_COV)
    for c, sel in groups.items():
        out[c] = {}
        for name, leads in LEADS.items():
            idx = [lead_index(L) for L in leads]
            r = (F[:, sel][:, :, idx].std(axis=0, ddof=1) / sigma[sel][:, idx]).mean(axis=1)
            bs = [float(np.median((F[p][:, sel][:, :, idx].std(axis=0, ddof=1)
                                   / sigma[sel][:, idx]).mean(axis=1))) for p in pick]
            out[c][name] = {"median_ratio": float(np.median(r)),
                            "bootstrap90_over_sequences": [float(np.quantile(bs, 0.05)),
                                                           float(np.quantile(bs, 0.95))]}
    return out


def icc_year(Fw: np.ndarray, sd: np.ndarray, phi: np.ndarray, county: np.ndarray) -> dict:
    """REPORT-ONLY: county-common share of the window-mean forcing departure across
    sequences; 'with_noise' adds the window-mean AR(1) noise variance to the denominator."""
    out = {}
    lag = np.abs(np.arange(O.N_WIN)[:, None] - np.arange(O.N_WIN)[None])
    for c in S.COUNTIES_COV:
        sel = county == c
        f = Fw[:, sel].mean(axis=2)                                  # (n_seq, W_c)
        v_tot = float(f.var(axis=0, ddof=1).mean())
        v_c = float(f.mean(axis=1).var(ddof=1))
        v_e = float(np.mean([sd[w] ** 2 * float((phi[w] ** lag).sum()) / O.N_WIN ** 2
                             for w in np.flatnonzero(sel)]))
        out[c] = {"icc_forcing_only": v_c / v_tot if v_tot > 0 else None,
                  "icc_with_noise": v_c / (v_tot + v_e) if (v_tot + v_e) > 0 else None,
                  "var_common_m2": v_c, "var_well_m2": v_tot, "var_noise_window_mean_m2": v_e,
                  "report_only": True}
    return out


def aggregates(F: np.ndarray, Fl2: np.ndarray, n2_sel: np.ndarray) -> dict:
    """Forcing-year range of the N2-unit mean and the fan layer-2 mean by lead set."""
    out: dict = {}
    for name, leads in LEADS.items():
        idx = [lead_index(L) for L in leads]
        n2 = F[:, n2_sel][:, :, idx].mean(axis=(1, 2))
        l2 = Fl2[:, :, idx].mean(axis=(1, 2))
        out[name] = {k: {"min_m": float(v.min()), "max_m": float(v.max()),
                         "sd_m": float(v.std(ddof=1)), "by_sequence_m": [float(x) for x in v]}
                     for k, v in (("n2_unit_mean", n2), ("fan_layer2_mean", l2))}
    out["window_years_by_sequence"] = [seq_years(s, 5)[3:5] for s in range(N_SEQ)]
    return out


# ----------------------------------------------------------------------------- score
@dataclass
class ForcingPaths:
    v1: O.Paths = field(default_factory=O.Paths)
    report: str = V2.REPORT_FORCING


def close(rep: dict, why: str) -> dict:
    rep["status"] = "not_evaluable"
    rep["why"] = why
    rep["readings"] = None
    rep["wording"] = {k: v.format(why=why) for k, v in WORDING["not_evaluable"].items()}
    return rep


def declared_c14b_trigger(decl: dict | None) -> str:
    """The C14b trigger the lead declared (prereg_fv2.py --c14b-trigger); smoke default."""
    if decl is None:
        return C14B_DEFAULT
    t = ((decl.get("readings") or {}).get("reading_2") or {}).get("c14b_trigger")
    if t not in C14B_TRIGGERS:
        raise EngineeringFailure(f"declaration has no valid c14b_trigger ({t!r})")
    return t


def score_forcing(FP: ForcingPaths, *, index: str | None = None, unavailable: str | None = None,
                  decl: dict | None = None, decl_sha: str | None = None, k_rc: int = K_RC,
                  k_null: int = K_RC_NULL, smoke: bool = False, attempts=None, now=None,
                  log=print) -> dict:
    """Gates E, N-cov and MC, then reading 2. Returns the report (not written). With a
    declaration, 'unavailable' is accepted only on a pre-declared ground."""
    rep: dict = {"id": V2.ID, "version": V2.VERSION, "part": "reading 2 (NEW: forcing-year "
                 "completeness of the frozen combined sigma)", "smoke": smoke,
                 "created": O._now(), "declaration_sha256": decl_sha, "k_rc": k_rc,
                 "k_null": k_null, "label": "post-failure, exploratory",
                 "mc_gate_note": V2.mc_note(N_SEQ, k_rc)}
    trigger = declared_c14b_trigger(decl)
    if unavailable:
        if decl is not None:
            grounds = unavailable_grounds(attempts, now=now)
            if not grounds:
                raise EngineeringFailure(
                    "--replay-unavailable needs a pre-declared ground (FW's report not "
                    "evaluable; FW's report absent on or after 2027-01-31; or a logged FV "
                    "gate-E failure on FW's replay forwards); none holds")
            rep["unavailable_grounds"] = grounds
        return close(rep, f"replay worlds unavailable: {unavailable}")
    if not index:
        raise EngineeringFailure("no replay index given")
    ctx = O.load_context(FP.v1)
    if decl is not None:
        for name, h in (decl.get("inputs_frozen") or {}).items():
            if O.sha256(O.rp(FP.v1.inputs_dir) / name) != h:
                raise EngineeringFailure(f"{name} differs from the declared sha256")
        if not O.rp(FW_REPORT).is_file():
            raise EngineeringFailure(f"{FW_REPORT} missing: rank 11's forwards stay sealed")
        rep["fw_report_sha256"] = O.sha256(O.rp(FW_REPORT))
    rp = load_replay(index, ctx["cells"], ctx["layers"])
    sig = sigma_by_lead(ctx["lo"], ctx["hi"],
                        registered_sigma_ens(O.rp(FP.v1.zr_forward), ctx["cells"],
                                             ctx["layers"]))
    rep["engineering"] = {"replay": rp["eng"], "index_sha256": rp["index_sha256"],
                          "source_declaration": rp["source"], "members": len(rp["members"]),
                          "n_active_layer2_cells": rp["n_active_l2"],
                          "sigma": {k: v for k, v in sig.items() if k != "sigma"}}
    log("gate E passed (engineering)")
    county = ctx["county"]
    # ---- gate N-cov
    nul = coverage_draws(ctx["zr"], ctx["lo"], ctx["hi"], np.zeros((1, ctx["W"], O.N_WIN)),
                         ctx["sd"], ctx["phi"], county, k_null, STREAM_RC_NULL)
    ncov = {}
    for c in S.COUNTIES_COV:
        mc = float(nul[c].mean())
        an = analytic_coverage(ctx["zr"], ctx["lo"], ctx["hi"], ctx["sd"], county == c)
        ncov[c] = {"monte_carlo": mc, "analytic": an, "abs_diff": abs(mc - an),
                   "pass": abs(mc - an) <= NULL_COV_TOL}
    rep["null_world"] = {"tolerance": NULL_COV_TOL, "by_county": ncov,
                         "pass": all(v["pass"] for v in ncov.values())}
    if not rep["null_world"]["pass"]:
        return close(rep, "null-world coverage differs from its analytic value (a possible "
                          "bug)")
    log("gate N-cov passed")
    # ---- RC
    Fw = rp["F"][:, :, O.WIN[0]:O.WIN[1]]
    cov = coverage_draws(ctx["zr"], ctx["lo"], ctx["hi"], Fw, ctx["sd"], ctx["phi"], county,
                         k_rc, STREAM_RC)
    rc = {c: summarise_rc(cov[c]) for c in S.COUNTIES_COV}
    mc = {c: rc[c]["mc_half_width_90"] <= V2.MC_HALF_WIDTH for c in S.COUNTIES_COV}
    lr = lead_ratio(rp["F"], sig["sigma"], county)
    agg = aggregates(rp["F"], rp["F_l2"], county == N2_UNIT)
    forcing = WORDING["forcing"][rp["rain"]]
    lbl = {"en": _LBL_EN, "zh": _LBL_ZH}
    rd = {c: (rc_reading(rc[c]["rc"]) if mc[c] else None) for c in S.COUNTIES_COV}
    if rp["rain"] == "climatology":
        # pumping-only replay: no Changhua forcing-year variation exists in the model
        rd["Changhua"] = NOT_REPRESENTED
    wr = WINDOW_LEADS
    groups = list(S.COUNTIES_COV) if trigger == "either_county" else ["pooled"]
    over = [c for c in groups if lr[c][wr]["median_ratio"] > LEAD_RATIO_L]
    c14b = "L" if over else "stays_N"
    words: dict = {"C14a": {}, "C14b": {}, "C14c": {}}
    for c in S.COUNTIES_COV:
        if rd[c] in WORDING["rc"]:
            lo_, hi_ = rc[c]["range_over_sequences"]
            t_lo, t_hi = (min(max(x, 0.0), 1.0) for x in rc[c]["sampling_11yr"]["t90"])
            words["C14a"][c] = {k: v.format(county=c, rc=rc[c]["rc"], lo=lo_, hi=hi_,
                                            t_lo=t_lo, t_hi=t_hi, forcing=forcing[k],
                                            lbl=lbl[k])
                                for k, v in WORDING["rc"][rd[c]].items()}
    words["changhua"] = dict(WORDING["changhua"][rp["rain"]])
    if c14b == "L":
        c = max(over, key=lambda x: lr[x][wr]["median_ratio"])
        where = {"pooled": {"en": "both counties (pooled)", "zh": "兩縣合併"}}.get(
            c, {"en": c, "zh": c})
        words["C14b"] = {k: v.format(lr=lr[c][wr]["median_ratio"], county=where[k], lbl=lbl[k])
                         for k, v in WORDING["c14b"]["L"].items()}
    else:
        curve = ", ".join(f"{n} mo {lr['Yunlin'][n]['median_ratio']:.2f} (Yunlin) / "
                          f"{lr['Changhua'][n]['median_ratio']:.2f} (Changhua)" for n in LEADS)
        words["C14b"] = {k: v.format(curve=curve, lbl=lbl[k])
                         for k, v in WORDING["c14b"]["stays_N"].items()}
    a = agg[wr]
    words["C14c"] = {k: v.format(n2_lo=a["n2_unit_mean"]["min_m"],
                                 n2_hi=a["n2_unit_mean"]["max_m"],
                                 l2_lo=a["fan_layer2_mean"]["min_m"],
                                 l2_hi=a["fan_layer2_mean"]["max_m"], forcing=forcing[k],
                                 lbl=lbl[k])
                     for k, v in WORDING["c14c"].items()}
    rep.update({
        "forcing": {"rain": rp["rain"], "pumping": rp["pumping"]},
        "rc": rc, "mc_gate": {"max_half_width": V2.MC_HALF_WIDTH, **mc},
        "lead_ratio": lr, "sigma_method": sig["method"],
        "icc_year_report_only": icc_year(Fw, ctx["sd"], ctx["phi"], county),
        "aggregates": agg,
        "readings": {"rule": f"per county RC < {RC_LOW} -> omits; RC >= {RC_HIGH} -> "
                             f"not_shown_to_omit; otherwise no line; Changhua "
                             f"'{NOT_REPRESENTED}' (no line) under a pumping-only replay. "
                             f"C14b -> L if the {wr} lead ratio exceeds {LEAD_RATIO_L} "
                             f"({trigger})",
                     "rc": rd, "c14b": c14b, "c14b_trigger": trigger, "c14b_groups_over": over,
                     "c14c": "reworded with the aggregate forcing-year range",
                     "label": "post-failure, exploratory"},
        "wording": words, "status": "scored"})
    return rep


# ----------------------------------------------------------------------------- synthetic
def build_synthetic(out: Path, amp_yunlin: float = 1.5, amp_changhua: float = 0.3,
                    seed: int = 5, fields: str = "energy+weather") -> tuple[ForcingPaths, Path]:
    """SYNTHETIC v1 inputs (O.build_synthetic) plus 11 replay forwards and a climatology run
    with the real layout (no real data), and their index."""
    out = Path(out)
    v1 = O.build_synthetic(out / "v1", nonlinear=0.01)
    wells = pd.read_csv(v1.wells, dtype={"station_id": str})
    with np.load(v1.zr_forward, allow_pickle=False) as z:
        H = np.asarray(z["heads_mean"], float)[0][:, :, :T_NEED + 6]
    rng = np.random.default_rng(seed)
    L, C, T = H.shape
    amp = np.full((L, C), 0.05)
    for _, r in wells.iterrows():
        amp[int(r["layer"]), int(r["cell"])] = (amp_yunlin if S.county_of(r["station_id"])
                                                == "Yunlin" else amp_changhua)
    year_eff = rng.normal(0.0, 1.0, N_SEQ)                       # one effect per 2012..2022
    t = np.arange(T)

    def save(path, heads, pf, years=None):
        rec = {"projection_forcing": pf,
               "replay": None if years is None else {"years": years, "pumping": "detrended",
                                                     "fields": fields}}
        np.savez_compressed(_capture_of(str(path)), record=json.dumps(rec),
                            inelastic=np.zeros(3))
        np.savez_compressed(
            path, heads_mean=heads[None].astype(np.float32),
            heads_std=np.zeros((1, L, C, T), np.float32), scenario_names=np.array(["baseline"]),
            dates=np.array([str(d) for d in pd.date_range("2012-01-01", periods=T, freq="MS")]),
            origin=O.ORIGIN_IDX, horizon=T - O.T_FIT, n_members=1,
            member_labels=np.array(["stage3_datum_gate"]), gain=1.0, hindcast_gain=0.0,
            gate=json.dumps({"git_commit": "synthetic"}),
            forward_options=json.dumps({"update_method": "innovation", "column_heads": "free",
                                        "gain": 1.0}))

    save(out / "replay_clim.npz", H, "climatology")
    seqs = []
    for s in range(N_SEQ):
        yrs = seq_years(s, (T - O.T_FIT) // 12 + 2)
        eff = np.zeros(T)
        for i in range(O.T_FIT, T):
            eff[i] = year_eff[yrs[(i - O.T_FIT) // 12] - FIRST_YEAR] * (
                1.0 - np.exp(-(i - O.T_FIT + 1) / 6.0))
        p = out / f"replay_s{s}.npz"
        save(p, H + amp[:, :, None] * eff[None, None, :] * (t >= O.T_FIT),
             f"replay:{yrs[0]}-...", yrs[:10])
        seqs.append(str(p))
    idx = out / "replay_index.json"
    build_index(idx, str(out / "replay_clim.npz"), seqs, code_commit="synthetic")
    return ForcingPaths(v1=v1, report=str(out / "report_forcing_year.json")), idx


def run_smoke(out: Path, k_rc: int = 20, k_null: int = 200, **kw) -> dict:
    out = Path(out)
    FP, idx = build_synthetic(out / "inputs_synthetic", **kw)
    rep = score_forcing(FP, index=str(idx), k_rc=k_rc, k_null=k_null, smoke=True)
    rep["note"] = "SMOKE: synthetic inputs only; numbers mean nothing about the twin"
    O._write_json_once(out / "smoke_forcing.json", rep)
    return rep
