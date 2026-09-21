"""Build the twin's decision application: one self-contained page, no server.

    python -m hydrophysics.twin.viewer_app --forward results/twin_forward/<run>.npz \\
        --basis results/twin_forward/response_basis.npz --out results/twin/twin_app.html

``explorer3d`` draws a Plotly figure: four surfaces and a slider. This builds the thing a
decision maker can be handed instead. The fan is a block model -- four aquifers at their
per-zone screen depths with the aquitards between them, standing on the real polygon --
and the controls are the levers a water authority actually has: how much each class of
user pumps, and from when.

Where the interactivity comes from. A solver run per policy is minutes, too slow to put
behind a slider, so the page ships a **response basis** instead: the baseline projection
plus one run per water-use class with that class retired. The page forms

    subsidence(policy) = baseline + sum_c (1 - factor_c) * (retired_c - baseline)

which is exact for a linear system and close for this one (``--basis`` carries a
half-cut run so the builder can report the error, which it prints and the page shows). A
later start year is the same response shifted in time; the builder checks that against a
real delayed run when the basis has one.

Everything is quantised to int16 and base64'd into the page: about 8 MB for the fan at
1 km, 252 months, six classes. That is what keeps it one file with nothing to install.
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np
import pandas as pd

from .zones import ZONE_NAMES, fan_zones

# Per-zone aquifer geometry, metres below ground, from the median screen depth of the
# zone-50 wells in each layer (AMP_V2 station metadata; see the module docstring of
# heads.py for the QC). The proximal fan has no layer-4 wells and no aquitards -- the
# confining muds pinch out there, which is the published geology and the reason the
# zonal parameterisation merges its aquifers.
# The proximal layer-4 depth is extrapolated, not measured: no zone-50 well is screened
# in layer 4 there, which is itself the geology (the confining muds pinch out and the
# aquifers merge). It only places a slab in the drawing; nothing numerical uses it.
LAYER_DEPTHS = {
    "proximal": [67.0, 125.0, 205.0, 250.0],
    "mid": [35.5, 119.8, 214.0, 289.0],
    "distal": [66.4, 104.9, 203.9, 276.0],
}
AQUIFER_THICK = {"proximal": [40.0, 45.0, 45.0, 40.0],
                 "mid": [30.0, 45.0, 45.0, 40.0],
                 "distal": [35.0, 40.0, 45.0, 40.0]}
CLASS_LABELS = {"irrigation": "Irrigation", "aquaculture": "Aquaculture",
                "livestock": "Livestock", "domestic": "Domestic", "industry": "Industry",
                "other": "Other"}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _q(a: np.ndarray, scale: float) -> np.ndarray:
    """Quantise to int16 at ``scale`` units per count, clipped to the int16 range."""
    return np.clip(np.round(np.asarray(a) / scale), -32768, 32767).astype("int16")


def build(forward_npz: str, basis_npz: str | None, out_html: str,
          townships_csv: str | None = None, quarter: int = 3, delta_step: int = 12,
          log=print) -> dict:
    fw = np.load(forward_npz, allow_pickle=False)
    mask = fw["mask"].astype(bool)
    nx, ny, dx = int(fw["nx"]), int(fw["ny"]), float(fw["dx"])
    x0, y0 = float(fw["x0"]), float(fw["y0"])
    dates = [str(d)[:7] for d in fw["dates"]]
    origin = int(fw["origin"])
    T = len(dates)
    A = int(mask.sum())
    rows, cols = np.nonzero(mask)


    cent = np.column_stack([x0 + (cols + 0.5) * dx, y0 + (rows + 0.5) * dx])
    zone = fan_zones(cent)

    # ground elevation: the same IDW field the solver used, recomputed here so the page
    # stands the block on the real surface rather than a flat plane
    from .calibrate_flow import DEFAULT_PATHS, _idw_field
    from .heads import _station_xy
    stn = pd.read_parquet(DEFAULT_PATHS["stations"])
    stn = stn[stn.GroundwaterZoneIdentifier == 50]
    gxy, gval = [], []
    for _, r in stn.iterrows():
        p = _station_xy(r)
        g = pd.to_numeric(r.get("GroundHeight"), errors="coerce")
        if p is not None and np.isfinite(g):
            gxy.append(p)
            gval.append(float(g))

    class _G:  # _idw_field only needs centroids()
        def centroids(self):
            return cent

    ground = _idw_field(_G(), np.array(gxy), np.array(gval))

    town_idx = np.zeros(A, dtype="uint8")
    town_names = ["unlabelled"]
    if townships_csv and os.path.exists(townships_csv):
        tw = pd.read_csv(townships_csv).set_index("cell")["town"].dropna()
        names = sorted(tw.unique())
        town_names = ["unlabelled"] + list(names)
        lookup = {n: i + 1 for i, n in enumerate(names)}
        for c, n in tw.items():
            if 0 <= int(c) < A:
                town_idx[int(c)] = lookup[n]

    subs = fw["subs_mean"]          # (S, A, T) metres
    heads = fw["heads_mean"]        # (S, L, A, T) metres
    sd = fw["subs_std"]
    base_s = subs[0] * 100.0        # cm
    base_h = heads[0]               # m
    qsel = np.arange(0, T, quarter)
    if qsel[-1] != T - 1:
        qsel = np.r_[qsel, T - 1]
    # Policy deltas are smooth ramps once the policy starts, so they are sampled far more
    # coarsely than the seasonal baseline and interpolated in the page. That is most of
    # the difference between a 19 MB file and an 8 MB one.
    dsel = np.arange(0, T, delta_step)
    if dsel[-1] != T - 1:
        dsel = np.r_[dsel, T - 1]

    # ---- response basis -------------------------------------------------------------
    classes, dsub, dhead, start_month, lin_err, shift_err = [], [], [], None, None, None
    if basis_npz and os.path.exists(basis_npz):
        bs = np.load(basis_npz, allow_pickle=False)
        names = [str(n) for n in bs["scenario_names"]]
        bsub, bhead = bs["subs_mean"] * 100.0, bs["heads_mean"]
        base_i = names.index("baseline")
        starts = {}
        for i, n in enumerate(names):
            if "_" not in n or n.startswith("check"):
                continue
            key, yr = n.rsplit("_", 1)
            starts.setdefault(yr, {})[key.replace("0", "")] = i
        year = sorted(starts)[0]
        start_month = dates.index(f"{year}-01")
        for key, i in starts[year].items():
            cname = {"irr": "irrigation", "aqua": "aquaculture", "live": "livestock",
                     "dom": "domestic", "ind": "industry", "oth": "other"}.get(key, key)
            classes.append(cname)
            dsub.append(bsub[i] - bsub[base_i])
            dhead.append(bhead[i] - bhead[base_i])
        # linearity: a half cut should be half the response
        if "check_irr50_2026" in names and "irrigation" in classes:
            j = classes.index("irrigation")
            pred = bsub[base_i] + 0.5 * dsub[j]
            act = bsub[names.index("check_irr50_2026")]
            lin_err = float(np.abs(pred[:, -1] - act[:, -1]).mean())
            log(f"linearity check: a 50% irrigation cut predicted by superposition differs "
                f"from the solved run by {lin_err:.3f} cm at the horizon "
                f"(response itself {abs(dsub[j][:, -1].mean()) * 0.5:.2f} cm)")
        # time shift: a later start should be the same response, moved
        if len(starts) > 1 and "irrigation" in classes:
            y2 = sorted(starts)[1]
            m2 = dates.index(f"{y2}-01")
            j = classes.index("irrigation")
            real = bsub[starts[y2]["irr"]] - bsub[base_i]
            lag = m2 - start_month
            shifted = np.zeros_like(real)
            shifted[:, lag:] = dsub[j][:, : T - lag]
            shift_err = float(np.abs(shifted[:, -1] - real[:, -1]).mean())
            log(f"time-shift check: delaying to {y2} by shifting the response differs from "
                f"the solved run by {shift_err:.3f} cm at the horizon")
        dsub = np.stack(dsub) if dsub else np.zeros((0, A, T))
        dhead = np.stack(dhead) if dhead else np.zeros((0, 4, A, T))
        log(f"response basis: {len(classes)} classes from {year}-01 -> "
            f"{[f'{c} {dsub[i][:, -1].mean():+.2f} cm' for i, c in enumerate(classes)]}")

    # energy share per class, for the slider subtitles
    share = {}
    try:
        from .inputs import load_twin_inputs
        inp = load_twin_inputs(verbose=False)
        tot = sum(v.sum() for v in inp.E_by_class.values())
        share = {k: float(v.sum() / tot) for k, v in inp.E_by_class.items()}
    except Exception as e:  # the page still works without it
        log(f"energy shares unavailable ({type(e).__name__}); sliders omit them")

    gate = {}
    try:
        gate = json.loads(str(fw["gate"])).get("gate") or {}
    except Exception:
        gate = {}

    data = {
        "nx": nx, "ny": ny, "dx": dx, "nA": A, "months": dates, "origin": origin,
        "quarters": [int(q) for q in qsel], "dQuarters": [int(q) for q in dsel],
        "cols": _b64(cols.astype("int16")), "rows": _b64(rows.astype("int16")),
        "zone": _b64(zone.astype("uint8")), "town": _b64(town_idx),
        "townNames": town_names,
        "ground": _b64(_q(ground, 0.01)),                       # m, 1 cm counts
        "subsBase": _b64(_q(base_s, 0.01)),                     # cm, 0.01 cm counts
        "subsStd": _b64(_q(sd[0][:, qsel] * 100.0, 0.01)),
        "headBase": _b64(_q(base_h[:, :, qsel], 0.01)),         # m, 1 cm counts
        "classes": classes,
        "classLabels": [CLASS_LABELS.get(c, c.title()) for c in classes],
        "classShare": [round(share.get(c, 0.0), 4) for c in classes],
        "dSub": _b64(_q(dsub[:, :, dsel] if len(classes) else np.zeros((0,)), 0.01)),
        "dHead": _b64(_q(dhead[:, :, :, dsel] if len(classes) else np.zeros((0,)), 0.01)),
        "startMonth": start_month if start_month is not None else origin,
        "layerDepths": LAYER_DEPTHS, "layerThick": AQUIFER_THICK, "zoneNames": list(ZONE_NAMES),
        "gate": gate, "linErr": lin_err, "shiftErr": shift_err,
        "scenarios": [str(s) for s in fw["scenarios"]],
        "nMembers": int(fw["n_members"]),
    }

    html = _PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    with open(out_html, "w") as fh:
        fh.write(html)
    log(f"wrote {out_html} ({os.path.getsize(out_html) / 1e6:.1f} MB, {A} cells, "
        f"{T} months, {len(classes)} policy classes)")
    return data


_PAGE = r"""<title>Choushui Fan Twin</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.128.0/three.min.js"></script>
<style>
:root{
  --paper:#f4f2ed; --surface:#fbfaf7; --surface-2:#eceae4; --line:#d8d5cc;
  --ink:#171c1f; --ink-2:#4b5560; --ink-3:#7d8791;
  --accent:#1f6f8b; --accent-soft:#d7e6ec;
  --good:#3f7d5a; --warn:#b4741f; --bad:#a53d2c;
  --stage:#e8e6e0; --stage-2:#d5d2ca;
  --rail:360px;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --paper:#101416; --surface:#161b1e; --surface-2:#1d2428; --line:#2c3439;
  --ink:#eef1f2; --ink-2:#aab4ba; --ink-3:#78848b;
  --accent:#57b6d4; --accent-soft:#1b3b47;
  --good:#6fb98c; --warn:#d9a34a; --bad:#e2735e;
  --stage:#0b0e10; --stage-2:#141a1d;
}}
:root[data-theme="dark"]{
  --paper:#101416; --surface:#161b1e; --surface-2:#1d2428; --line:#2c3439;
  --ink:#eef1f2; --ink-2:#aab4ba; --ink-3:#78848b;
  --accent:#57b6d4; --accent-soft:#1b3b47;
  --good:#6fb98c; --warn:#d9a34a; --bad:#e2735e;
  --stage:#0b0e10; --stage-2:#141a1d;
}
html,body{height:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:13px;overflow:hidden}
#app{display:grid;grid-template-columns:var(--rail) 1fr;grid-template-rows:auto 1fr auto;height:100%}
header{grid-column:1/-1;display:flex;align-items:center;gap:14px;padding:10px 16px;padding-top:calc(10px + env(safe-area-inset-top,0px));background:var(--surface);border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{font-size:15px;font-weight:600;margin:0;letter-spacing:-.01em}
.sub{color:var(--ink-3);font-size:11.5px}
.spacer{flex:1}
.verdict{font-family:"IBM Plex Mono",monospace;font-size:11px;padding:3px 8px;border-radius:3px;background:var(--accent-soft);color:var(--accent);border:1px solid color-mix(in srgb,var(--accent) 35%,transparent)}
.modes{display:flex;border:1px solid var(--line);border-radius:4px;overflow:hidden}
.modes button{border:0;background:var(--surface);color:var(--ink-2);padding:5px 11px;font:inherit;font-size:11.5px;cursor:pointer}
.modes button[aria-pressed="true"]{background:var(--accent);color:#fff}
#rail{grid-column:1;grid-row:2/4;background:var(--surface);border-right:1px solid var(--line);overflow-y:auto;padding:14px 16px 20px;display:flex;flex-direction:column;gap:18px}
#stage{grid-column:2;grid-row:2;position:relative;background:linear-gradient(180deg,var(--stage) 0%,var(--stage-2) 100%);overflow:hidden}
canvas{display:block;width:100%;height:100%;touch-action:none}
section h2{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--ink-3);margin:0 0 9px;font-weight:600}
.slider{display:grid;grid-template-columns:1fr auto;gap:2px 8px;align-items:baseline;margin-bottom:11px}
.slider label{font-size:12.5px}
.slider .val{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--accent);font-variant-numeric:tabular-nums}
.slider .note{grid-column:1/-1;font-size:10.5px;color:var(--ink-3)}
.slider input[type=range]{grid-column:1/-1;width:100%;accent-color:var(--accent);margin:3px 0 0}
.presets{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
.presets button{border:1px solid var(--line);background:var(--surface-2);color:var(--ink-2);border-radius:3px;padding:4px 9px;font:inherit;font-size:11.5px;cursor:pointer}
.presets button:hover{border-color:var(--accent);color:var(--accent)}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.tile{background:var(--surface);padding:9px 11px}
.tile .k{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3)}
.tile .v{font-family:"IBM Plex Mono",monospace;font-size:19px;font-variant-numeric:tabular-nums;margin-top:2px}
.tile .d{font-size:11px;color:var(--ink-2)}
.layers{display:flex;flex-direction:column;gap:2px}
.lrow{display:grid;grid-template-columns:auto 12px 1fr auto;gap:9px;align-items:center;padding:4px 6px;border-radius:3px;cursor:pointer}
.lrow:hover{background:var(--surface-2)}
.lrow .sw{width:12px;height:12px;border-radius:2px;border:1px solid rgba(0,0,0,.18)}
.lrow .dep{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--ink-3)}
.lrow input{accent-color:var(--accent)}
#timeline{grid-column:2;grid-row:3;display:flex;align-items:center;gap:12px;padding:9px 16px;padding-bottom:calc(9px + env(safe-area-inset-bottom,0px));background:var(--surface);border-top:1px solid var(--line)}
#timeline input[type=range]{flex:1;accent-color:var(--accent)}
#play{border:1px solid var(--line);background:var(--surface-2);color:var(--ink);border-radius:3px;width:34px;height:28px;font-size:13px;cursor:pointer}
.stamp{font-family:"IBM Plex Mono",monospace;font-size:13px;min-width:112px;font-variant-numeric:tabular-nums}
.stamp small{color:var(--ink-3)}
#legend{position:absolute;left:14px;bottom:14px;background:color-mix(in srgb,var(--surface) 92%,transparent);border:1px solid var(--line);border-radius:4px;padding:9px 11px;backdrop-filter:blur(6px);max-width:215px}
#legend .bar{height:9px;border-radius:2px;margin:5px 0 3px}
#legend .ends{display:flex;justify-content:space-between;font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--ink-2)}
#legend .cap{font-size:10.5px;color:var(--ink-3);margin-top:5px;line-height:1.35}
#probe{position:absolute;right:14px;top:14px;width:250px;background:color-mix(in srgb,var(--surface) 95%,transparent);border:1px solid var(--line);border-radius:4px;padding:11px 12px;backdrop-filter:blur(6px)}
#probe h3{margin:0 0 2px;font-size:13px}
#probe .loc{font-size:11px;color:var(--ink-3);margin-bottom:8px}
#probe .row{display:flex;justify-content:space-between;font-size:11.5px;padding:2px 0}
#probe .row b{font-family:"IBM Plex Mono",monospace;font-weight:500;font-variant-numeric:tabular-nums}
#probe svg{width:100%;height:74px;margin-top:7px}
#probe .hint{font-size:10.5px;color:var(--ink-3);line-height:1.4}
.tabs{display:flex;gap:2px;margin:-3px -3px 9px;border-bottom:1px solid var(--line)}
.tabs button{flex:1;border:0;border-bottom:2px solid transparent;background:none;color:var(--ink-3);
  font:inherit;font-size:11px;padding:5px 2px;cursor:pointer}
.tabs button[aria-pressed="true"]{color:var(--accent);border-bottom-color:var(--accent)}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
#probe.wide{width:390px}
#secWrap svg,#cmpWrap svg{width:100%;display:block}
.tbl{width:100%;border-collapse:collapse;font-size:11px;font-variant-numeric:tabular-nums}
.tbl th{text-align:left;font-weight:500;color:var(--ink-3);font-size:10px;text-transform:uppercase;
  letter-spacing:.06em;padding:3px 4px;cursor:pointer;white-space:nowrap}
.tbl th[aria-sort]{color:var(--accent)}
.tbl td{padding:3px 4px;border-top:1px solid var(--line)}
.tbl tr:hover td{background:var(--surface-2)}
.tbl .num{font-family:"IBM Plex Mono",monospace;text-align:right}
.tbl .bar{height:5px;border-radius:2px;background:var(--accent);opacity:.75}
.tblwrap{max-height:230px;overflow-y:auto}
.mini{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}
.mini figure{margin:0}
.mini figcaption{font-size:10px;color:var(--ink-3);margin-bottom:3px}
.mini canvas{width:100%;image-rendering:pixelated;border:1px solid var(--line);border-radius:3px;background:var(--surface-2)}
.btnrow{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
.btnrow button{flex:1;min-width:92px;border:1px solid var(--line);background:var(--surface-2);color:var(--ink-2);
  border-radius:3px;padding:5px 7px;font:inherit;font-size:11px;cursor:pointer}
.btnrow button:hover{border-color:var(--accent);color:var(--accent)}
.btnrow button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.said{font-size:10.5px;color:var(--good);min-height:13px;margin-top:5px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
.view-btns{display:flex;gap:6px;flex-wrap:wrap}
.view-btns button{flex:1;border:1px solid var(--line);background:var(--surface-2);color:var(--ink-2);border-radius:3px;padding:5px;font:inherit;font-size:11.5px;cursor:pointer}
.view-btns button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}
.caveat{font-size:11px;color:var(--ink-2);line-height:1.45;border-left:2px solid var(--warn);padding-left:9px}
.rng{display:grid;grid-template-columns:1fr auto;gap:2px 8px;align-items:baseline;margin-bottom:9px}
.rng input{grid-column:1/-1;width:100%;accent-color:var(--accent)}
.rng label{font-size:12px}
.rng .val{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-3)}
@media (max-width:860px){
  #app{grid-template-columns:1fr;grid-template-rows:auto 44vh auto 1fr}
  #stage{grid-column:1;grid-row:2}
  #timeline{grid-column:1;grid-row:3}
  #rail{grid-column:1;grid-row:4;border-right:0;border-top:1px solid var(--line)}
  #probe{position:static;width:auto;margin:0 0 4px}
  :root{--rail:auto}
}
</style>

<div id="app">
  <header>
    <div>
      <h1>Choushui Fan Twin</h1>
      <div class="sub">Pumping policy to groundwater heads to land subsidence, Yunlin and Changhua, Taiwan</div>
    </div>
    <div class="spacer"></div>
    <div class="verdict" id="verdict"></div>
    <div class="modes">
      <button id="mDecide" aria-pressed="true">Decision</button>
      <button id="mAnalyst" aria-pressed="false">Analyst</button>
    </div>
  </header>

  <aside id="rail">
    <section>
      <h2>Pumping policy</h2>
      <div id="sliders"></div>
      <div class="rng">
        <label for="startYear">Policy starts</label><span class="val" id="startVal"></span>
        <input type="range" id="startYear" min="0" max="1" step="1" value="0">
      </div>
      <div class="presets" id="presets"></div>
    </section>

    <section>
      <h2>Outcome by <span id="horizonYear"></span></h2>
      <div class="tiles">
        <div class="tile"><div class="k">Fan mean</div><div class="v" id="tMean">—</div><div class="d" id="tMeanD">&nbsp;</div></div>
        <div class="tile"><div class="k">Worst township</div><div class="v" id="tWorst" style="font-size:14px">—</div><div class="d" id="tWorstD">&nbsp;</div></div>
        <div class="tile"><div class="k">Area past 5 cm</div><div class="v" id="tArea">—</div><div class="d">km² of 2,144</div></div>
        <div class="tile"><div class="k">Peak cell</div><div class="v" id="tPeak">—</div><div class="d" id="tPeakD">&nbsp;</div></div>
      </div>
    </section>

    <section class="analyst">
      <h2>Aquifers</h2>
      <div class="layers" id="layers"></div>
      <div class="view-btns" style="margin-top:9px">
        <button id="vHead" aria-pressed="true">Head</button>
        <button id="vDraw" aria-pressed="false">Drawdown</button>
      </div>
    </section>

    <section class="analyst">
      <h2>View</h2>
      <div class="rng"><label for="explode">Separate layers</label><span class="val" id="explodeVal">0</span>
        <input type="range" id="explode" min="0" max="100" value="0"></div>
      <div class="rng"><label for="clip">Cut away, west to east</label><span class="val" id="clipVal">off</span>
        <input type="range" id="clip" min="0" max="100" value="100"></div>
      <div class="rng"><label for="exag">Vertical exaggeration</label><span class="val" id="exagVal">40×</span>
        <input type="range" id="exag" min="10" max="120" value="40"></div>
      <div class="view-btns"><button id="vReset">Reset camera</button><button id="vTop">Map view</button></div>
    </section>

    <section>
      <h2>What this is</h2>
      <p class="caveat" id="caveat"></p>
    </section>
  </aside>

  <div id="stage">
    <canvas id="c"></canvas>
    <div id="legend">
      <div style="font-size:11px;font-weight:500" id="legTitle">Cumulative subsidence</div>
      <div class="bar" id="legBar"></div>
      <div class="ends"><span id="legLo">0</span><span id="legHi"></span></div>
      <div class="cap" id="legCap"></div>
    </div>
    <div id="probe">
      <div class="tabs" role="tablist">
        <button id="tabCell" aria-pressed="true">Cell</button>
        <button id="tabSec" aria-pressed="false">Section</button>
        <button id="tabCmp" aria-pressed="false">Compare</button>
        <button id="tabTown" aria-pressed="false">Townships</button>
      </div>

      <div id="paneCell">
        <h3 id="pTitle">Click the ground</h3>
        <div class="loc" id="pLoc"></div>
        <div id="pRows"></div>
        <svg id="pChart" viewBox="0 0 250 74" aria-label="subsidence over time at the selected cell"></svg>
        <div class="hint" id="pHint">Pick a cell to read its heads, its sinking, and the ensemble spread.</div>
      </div>

      <div id="paneSec" hidden>
        <h3>Cross-section</h3>
        <div class="loc" id="secHint">Click two points on the ground to cut a line through the fan.</div>
        <div id="secWrap"><svg id="secSvg" viewBox="0 0 380 210" aria-label="vertical section through the aquifers"></svg></div>
        <div class="btnrow">
          <button id="secDraw">Draw section</button>
          <button id="secWE">West to east</button>
          <button id="secCopy">Copy CSV</button>
        </div>
        <div class="said" id="secSaid"></div>
      </div>

      <div id="paneCmp" hidden>
        <h3>Compare policies</h3>
        <div class="loc">Your policy against a reference, at <span id="cmpYear"></span>.</div>
        <div class="mini">
          <figure><figcaption id="cmpAName">Your policy</figcaption><canvas id="cmpA" width="59" height="77"></canvas></figure>
          <figure><figcaption id="cmpBName">Reference</figcaption><canvas id="cmpB" width="59" height="77"></canvas></figure>
        </div>
        <label class="loc" for="cmpRef">Reference</label>
        <select id="cmpRef" style="width:100%;font:inherit;font-size:11.5px;padding:4px;border:1px solid var(--line);border-radius:3px;background:var(--surface);color:var(--ink)">
          <option value="base">Baseline, no change</option>
          <option value="irr30">Irrigation −30%</option>
          <option value="aqua0">Retire aquaculture</option>
          <option value="all20">All users −20%</option>
        </select>
        <div id="cmpRows" style="margin-top:8px"></div>
        <div class="btnrow"><button id="cmpOn" aria-pressed="false">Colour the block by difference</button></div>
      </div>

      <div id="paneTown" hidden>
        <h3>Township report</h3>
        <div class="loc">Mean subsidence at <span id="townYear"></span> and the change your policy makes.</div>
        <div class="tblwrap"><table class="tbl" id="townTbl"></table></div>
        <div class="btnrow">
          <button id="townCopy">Copy CSV</button>
          <button id="townDl">Download CSV</button>
          <button id="shotPng">Save image</button>
        </div>
        <div class="said" id="townSaid"></div>
      </div>
    </div>
  </div>

  <div id="timeline">
    <button id="play" aria-label="play">▶</button>
    <input type="range" id="time" min="0" max="1" value="0">
    <div class="stamp" id="stamp"></div>
  </div>
</div>

<script>
const D = __DATA__;
const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i);
  return new T(u.buffer); };
const cols = dec(D.cols, Int16Array), rows = dec(D.rows, Int16Array);
const zone = dec(D.zone, Uint8Array), town = dec(D.town, Uint8Array);
const ground = dec(D.ground, Int16Array);          // 1 cm counts
const subsBase = dec(D.subsBase, Int16Array);      // cm, 0.01 counts, (A,T)
const subsStd = dec(D.subsStd, Int16Array);        // (A,Q)
const headBase = dec(D.headBase, Int16Array);      // (L,A,Q), 1 cm counts
const dSub = dec(D.dSub, Int16Array);              // (C,A,Q)
const dHead = dec(D.dHead, Int16Array);            // (C,L,A,Q)
const A = D.nA, T = D.months.length, Q = D.quarters.length, C = D.classes.length, L = 4;
const P = D.dQuarters.length;
const qIndex = new Int32Array(T);                  // month -> nearest stored head sample
{ let j = 0; for (let t = 0; t < T; t++) { while (j + 1 < Q && D.quarters[j + 1] <= t) j++; qIndex[t] = j; } }
// deltas are stored sparsely and interpolated: slot + weight toward the next slot
const dSlot = new Int32Array(T), dFrac = new Float32Array(T);
{ let j = 0;
  for (let t = 0; t < T; t++) {
    while (j + 1 < P && D.dQuarters[j + 1] <= t) j++;
    dSlot[t] = j;
    const a = D.dQuarters[j], b = j + 1 < P ? D.dQuarters[j + 1] : a;
    dFrac[t] = b > a ? (t - a) / (b - a) : 0;
  } }
function dLerp(arr, stride, base, t) {          // arr[(base)*P + slot], interpolated
  const j = dSlot[t], f = dFrac[t];
  const v0 = arr[base * P + j];
  const v1 = j + 1 < P ? arr[base * P + j + 1] : v0;
  return v0 + (v1 - v0) * f;
}

const state = {
  t: T - 1, playing: false, mode: "decide", field: "subs", headMode: "head",
  factors: D.classes.map(() => 1), start: 0, layers: [true, true, true, true],
  aquitards: true, explode: 0, clip: 1, exag: 40, sel: -1,
  tab: "cell", secMode: false, secA: null, secB: null, cmpRef: "base", cmpOn: false
};
// (col,row) -> active cell index, so a section can walk the grid
const cellAt = new Int32Array(D.nx * D.ny).fill(-1);
for (let i = 0; i < A; i++) cellAt[rows[i] * D.nx + cols[i]] = i;
const presetFactors = p => D.classes.map(c =>
  p === "base" ? 1 : p === "irr30" ? (c === "irrigation" ? 0.7 : 1)
  : p === "aqua0" ? (c === "aquaculture" ? 0 : 1) : 0.8);
function subsWith(factors, cell, t) {
  let v = subsBase[cell * T + t] * 0.01;
  if (!C) return v;
  const lag = startMonths[state.start] - startMonths[0], ts = t - lag;
  if (ts < 0) return v;
  const tc = Math.min(T - 1, ts);
  for (let c = 0; c < C; c++) {
    const f = factors[c]; if (f === 1) continue;
    v += (1 - f) * dLerp(dSub, 1, c * A + cell, tc) * 0.01;
  }
  return v;
}
const startMonths = [D.startMonth, Math.min(T - 1, D.startMonth + 48)];
const startLabels = [D.months[startMonths[0]].slice(0, 4), D.months[startMonths[1]].slice(0, 4)];

// ---- policy response ---------------------------------------------------------------
function subsAt(cell, t) {
  let v = subsBase[cell * T + t] * 0.01;
  if (!C) return v;
  const lag = startMonths[state.start] - startMonths[0];
  const ts = t - lag; if (ts < 0) return v;
  const tc = Math.min(T - 1, ts);
  for (let c = 0; c < C; c++) {
    const f = state.factors[c]; if (f === 1) continue;
    v += (1 - f) * dLerp(dSub, 1, c * A + cell, tc) * 0.01;
  }
  return v;
}
function headAt(layer, cell, t) {
  const q = qIndex[t];
  let v = headBase[((layer * A) + cell) * Q + q] * 0.01;
  if (!C) return v;
  const lag = startMonths[state.start] - startMonths[0];
  const ts = t - lag; if (ts < 0) return v;
  const tc = Math.min(T - 1, ts);
  for (let c = 0; c < C; c++) {
    const f = state.factors[c]; if (f === 1) continue;
    v += (1 - f) * dLerp(dHead, 1, (c * L + layer) * A + cell, tc) * 0.01;
  }
  return v;
}

// ---- colour ramps ------------------------------------------------------------------
const subsRamp = [[0.00,[0.96,0.94,0.87]],[0.25,[0.91,0.78,0.52]],[0.50,[0.85,0.55,0.26]],
                  [0.75,[0.68,0.27,0.18]],[1.00,[0.36,0.10,0.10]]];
const headRamp = [[0.00,[0.13,0.20,0.29]],[0.35,[0.16,0.42,0.56]],[0.65,[0.42,0.68,0.78]],
                  [1.00,[0.83,0.92,0.94]]];
function ramp(r, u) {
  u = Math.max(0, Math.min(1, u));
  for (let i = 1; i < r.length; i++) if (u <= r[i][0]) {
    const a = r[i-1], b = r[i], k = (u - a[0]) / (b[0] - a[0]);
    return [a[1][0]+(b[1][0]-a[1][0])*k, a[1][1]+(b[1][1]-a[1][1])*k, a[1][2]+(b[1][2]-a[1][2])*k];
  }
  return r[r.length-1][1];
}
const CLAY = [0.55,0.50,0.42], ROCK = {0:[0.72,0.66,0.55],1:[0.78,0.71,0.57],2:[0.64,0.63,0.57]};
// difference: one cool pole, a neutral midpoint, one warm pole
const diffRamp = [[0.0,[0.10,0.42,0.42]],[0.35,[0.55,0.74,0.72]],[0.5,[0.88,0.87,0.83]],
                  [0.65,[0.88,0.66,0.44]],[1.0,[0.60,0.20,0.14]]];
const rgb = c => "rgb(" + c.map(v => Math.round(v*255)).join(",") + ")";

// ---- scene -------------------------------------------------------------------------
const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
renderer.localClippingEnabled = true;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 1, 8000);
const KM = 1000 / D.dx;                       // one cell = 1 unit
const W = D.nx, H = D.ny;
const target = new THREE.Vector3(0, 0, 0);
let camR = 105, camTheta = -0.9, camPhi = 0.92;
function place() {
  camera.position.set(target.x + camR*Math.sin(camPhi)*Math.cos(camTheta),
                      target.y + camR*Math.cos(camPhi),
                      target.z + camR*Math.sin(camPhi)*Math.sin(camTheta));
  camera.lookAt(target);
}
scene.add(new THREE.HemisphereLight(0xffffff, 0x5a5348, 0.95));
const key = new THREE.DirectionalLight(0xfff4e2, 0.85); key.position.set(-60, 90, 40); scene.add(key);
const fill = new THREE.DirectionalLight(0xbcd6e4, 0.35); fill.position.set(50, 30, -50); scene.add(fill);

const clipPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 1e6);
const boxGeom = new THREE.BoxGeometry(1, 1, 1);
const meshes = [];      // {mesh, kind:'aquifer'|'aquitard'|'ground', layer}
const DEPTH = {}, THICK = {};
D.zoneNames.forEach((z, i) => { DEPTH[i] = D.layerDepths[z]; THICK[i] = D.layerThick[z]; });

function makeLayer(kind, layer) {
  const m = new THREE.MeshLambertMaterial({vertexColors:true, clippingPlanes:[clipPlane],
    transparent: kind === "aquitard", opacity: kind === "aquitard" ? 0.93 : 1});
  const inst = new THREE.InstancedMesh(boxGeom, m, A);
  inst.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const c0 = new THREE.Color(0.6, 0.6, 0.6);
  for (let i = 0; i < A; i++) inst.setColorAt(i, c0);   // creates instanceColor properly
  scene.add(inst);
  meshes.push({mesh:inst, kind, layer});
  return inst;
}
const ground3 = makeLayer("ground", -1);
for (let l = 0; l < L; l++) { makeLayer("aquifer", l); if (l < L-1) makeLayer("aquitard", l); }

const mtx = new THREE.Matrix4(), pos = new THREE.Vector3(), scl = new THREE.Vector3(1,1,1), qt = new THREE.Quaternion();
function layerY(kind, layer, cell) {
  const z = zone[cell], g = ground[cell] * 0.01;
  if (kind === "ground") return {y: g, h: 2.0};
  const d = DEPTH[z][layer], th = THICK[z][layer];
  if (kind === "aquifer") return {y: g - d, h: th};
  const d2 = DEPTH[z][layer+1], th2 = THICK[z][layer+1];
  const top = g - d - th/2, bot = g - d2 + th2/2;
  return {y: (top + bot) / 2, h: Math.max(4, top - bot)};
}
function rebuildGeometry(onlyGround) {
  const ex = state.explode;
  for (const rec of meshes) {
    if (onlyGround && rec.kind !== "ground") continue;
    const {mesh, kind, layer} = rec;
    let n = 0;
    for (let i = 0; i < A; i++) {
      const {y, h} = layerY(kind, layer, i);
      const li = kind === "ground" ? 0 : (kind === "aquifer" ? layer + 1 : layer + 1.5);
      const lift = ex * li * 1.1;
      const sink = kind === "ground" ? -subsAt(i, state.t) * state.exag : 0;
      pos.set(cols[i] - W/2 + 0.5, (y + sink) * state.exag / 40 + lift, -(rows[i] - H/2 + 0.5));
      scl.set(0.98, Math.max(0.6, h * state.exag / 40), 0.98);
      mtx.compose(pos, qt, scl);
      mesh.setMatrixAt(n++, mtx);
    }
    mesh.count = n;
    mesh.instanceMatrix.needsUpdate = true;
  }
}
let subsMax = 1;
function recolour() {
  const t = state.t;
  // fan-wide scale for the subsidence ramp, stable across time so colours mean one thing
  for (const rec of meshes) {
    const {mesh, kind, layer} = rec;
    const col = mesh.instanceColor.array;
    for (let i = 0; i < A; i++) {
      let c;
      if (kind === "ground") {
        if (state.cmpOn) {
          const d = subsAt(i, t) - subsWith(presetFactors(state.cmpRef), i, t);
          c = ramp(diffRamp, 0.5 + d / (2 * cmpScale));
        } else c = ramp(subsRamp, subsAt(i, t) / subsMax);
      } else if (kind === "aquifer") {
        if (state.field === "subs") {
          c = ROCK[zone[i]].slice();
        } else if (state.headMode === "draw") {
          const d = headAt(layer, i, 0) - headAt(layer, i, t);
          c = ramp(headRamp, 1 - Math.max(0, Math.min(1, d / 12)));
        } else {
          c = ramp(headRamp, (headAt(layer, i, t) + 20) / 90);
        }
      } else {
        c = CLAY;
      }
      col[i*3] = c[0]; col[i*3+1] = c[1]; col[i*3+2] = c[2];
    }
    mesh.instanceColor.needsUpdate = true;
  }
}
function applyVisibility() {
  for (const rec of meshes) {
    if (rec.kind === "ground") rec.mesh.visible = true;
    else if (rec.kind === "aquifer") rec.mesh.visible = state.layers[rec.layer];
    else rec.mesh.visible = state.aquitards && state.layers[rec.layer] && state.layers[rec.layer+1];
  }
  const x = -W/2 + state.clip * W;
  clipPlane.constant = state.clip >= 1 ? 1e6 : x;
}

// ---- camera interaction ------------------------------------------------------------
let drag = null;
canvas.addEventListener("pointerdown", e => {
  drag = {x:e.clientX, y:e.clientY, b:e.button, t:Date.now()};
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY; drag.moved = (drag.moved || 0) + Math.abs(dx) + Math.abs(dy);
  if (drag.b === 2 || e.shiftKey) {
    const s = camR * 0.0016;
    target.x -= dx * s * Math.sin(camTheta + Math.PI/2);
    target.z -= dx * s * -Math.cos(camTheta + Math.PI/2);
    target.y += dy * s;
  } else {
    camTheta -= dx * 0.006;
    camPhi = Math.max(0.12, Math.min(1.5, camPhi - dy * 0.005));
  }
  place();
});
canvas.addEventListener("pointerup", e => {
  if (drag && (drag.moved || 0) < 5 && Date.now() - drag.t < 400) pick(e);
  drag = null;
});
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  camR = Math.max(20, Math.min(400, camR * (1 + Math.sign(e.deltaY) * 0.08)));
  place();
}, {passive:false});

const ray = new THREE.Raycaster();
function pick(e) {
  const r = canvas.getBoundingClientRect();
  const m = new THREE.Vector2(((e.clientX-r.left)/r.width)*2-1, -((e.clientY-r.top)/r.height)*2+1);
  ray.setFromCamera(m, camera);
  const hits = ray.intersectObjects(meshes.filter(x => x.mesh.visible).map(x => x.mesh));
  if (!hits.length) return;
  const id = hits[0].instanceId;
  if (state.secMode) {
    if (!state.secA || state.secB) { state.secA = id; state.secB = null; }
    else state.secB = id;
    if (state.secA && state.secB) { state.secMode = false; $("secDraw").setAttribute("aria-pressed", false); }
    drawSectionLine(); drawSection();
  } else { state.sel = id; drawProbe(); }
}

// ---- UI ----------------------------------------------------------------------------
const $ = id => document.getElementById(id);
const fmt = (v, n=2) => (v >= 0 ? "" : "") + v.toFixed(n);

function buildSliders() {
  $("sliders").innerHTML = D.classes.map((c, i) => {
    const pct = D.classShare[i] ? ` · ${(D.classShare[i]*100).toFixed(0)}% of pumped energy` : "";
    return `<div class="slider">
      <label for="f${i}">${D.classLabels[i]}</label><span class="val" id="fv${i}">no change</span>
      <div class="note">${pct.replace(/^ · /,"")}</div>
      <input type="range" id="f${i}" min="0" max="150" value="100">
    </div>`;
  }).join("");
  D.classes.forEach((c, i) => $("f"+i).addEventListener("input", e => {
    state.factors[i] = (+e.target.value) / 100; refresh(true);
  }));
  $("presets").innerHTML = `
    <button data-p="base">Baseline</button>
    <button data-p="irr30">Irrigation −30%</button>
    <button data-p="aqua0">Retire aquaculture</button>
    <button data-p="all20">All users −20%</button>`;
  $("presets").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    const p = b.dataset.p;
    state.factors = D.classes.map(c => {
      if (p === "base") return 1;
      if (p === "irr30") return c === "irrigation" ? 0.7 : 1;
      if (p === "aqua0") return c === "aquaculture" ? 0 : 1;
      return 0.8;
    });
    D.classes.forEach((c, i) => $("f"+i).value = Math.round(state.factors[i]*100));
    refresh(true);
  }));
}
function buildLayers() {
  const names = ["Aquifer 1, shallow", "Aquifer 2, main production", "Aquifer 3", "Aquifer 4, deep"];
  $("layers").innerHTML = names.map((n, i) => {
    const d = D.layerDepths.mid[i];
    return `<label class="lrow"><input type="checkbox" id="L${i}" checked>
      <span class="sw" style="background:rgb(${ramp(headRamp,0.35+i*0.18).map(v=>Math.round(v*255)).join(",")})"></span>
      <span>${n}</span><span class="dep">${d.toFixed(0)} m</span></label>`;
  }).join("") + `<label class="lrow"><input type="checkbox" id="Lclay" checked>
      <span class="sw" style="background:rgb(${CLAY.map(v=>Math.round(v*255)).join(",")})"></span>
      <span>Aquitards, clay</span><span class="dep">between</span></label>`;
  for (let i = 0; i < L; i++) $("L"+i).addEventListener("change", e => {
    state.layers[i] = e.target.checked; applyVisibility(); });
  $("Lclay").addEventListener("change", e => { state.aquitards = e.target.checked; applyVisibility(); });
}
function drawProbe() {
  const i = state.sel;
  if (i < 0) return;
  const t = state.t;
  $("pTitle").textContent = D.townNames[town[i]] === "unlabelled"
    ? "Cell " + i : D.townNames[town[i]];
  $("pLoc").textContent = `${D.zoneNames[zone[i]]} fan · ground ${(ground[i]*0.01).toFixed(1)} m · ${D.months[t]}`;
  const s = subsAt(i, t), sd = subsStd[i*Q + qIndex[t]] * 0.01;
  const b = subsBase[i*T + t] * 0.01;
  let rows = `<div class="row"><span>Subsidence</span><b>${s.toFixed(1)} ± ${sd.toFixed(1)} cm</b></div>`;
  if (Math.abs(s - b) > 0.005)
    rows += `<div class="row"><span>Versus baseline</span><b>${(s-b>=0?"+":"")}${(s-b).toFixed(2)} cm</b></div>`;
  for (let l = 0; l < L; l++)
    rows += `<div class="row"><span>Head, aquifer ${l+1}</span><b>${headAt(l,i,t).toFixed(1)} m</b></div>`;
  $("pRows").innerHTML = rows;
  $("pHint").textContent = "Band is the spread across " + D.nMembers + " ensemble members.";
  // sparkline of subsidence with spread
  let up = "", dn = "", ln = "";
  const X = t2 => 4 + 242 * t2 / (T - 1), Y = v => 70 - 62 * Math.min(1, v / subsMax);
  for (let t2 = 0; t2 < T; t2 += 2) {
    const v = subsAt(i, t2), e = subsStd[i*Q + qIndex[t2]] * 0.01;
    up += `${X(t2).toFixed(1)},${Y(v+e).toFixed(1)} `;
    dn = `${X(t2).toFixed(1)},${Y(Math.max(0,v-e)).toFixed(1)} ` + dn;
    ln += `${X(t2).toFixed(1)},${Y(v).toFixed(1)} `;
  }
  const ox = X(D.origin);
  $("pChart").innerHTML =
    `<polygon points="${up}${dn}" fill="var(--accent)" opacity=".16"></polygon>
     <polyline points="${ln}" fill="none" stroke="var(--accent)" stroke-width="1.6"></polyline>
     <line x1="${ox}" y1="4" x2="${ox}" y2="70" stroke="var(--ink-3)" stroke-dasharray="2 2" stroke-width="1"></line>
     <line x1="${X(t)}" y1="2" x2="${X(t)}" y2="72" stroke="var(--ink)" stroke-width="1"></line>
     <text x="4" y="10" font-size="8" fill="var(--ink-3)">cm</text>
     <text x="${ox+3}" y="10" font-size="8" fill="var(--ink-3)">projection</text>`;
}
function stats() {
  const t = state.t;
  let sum = 0, peak = 0, peakI = 0, area = 0, base = 0;
  const byTown = new Map();
  for (let i = 0; i < A; i++) {
    const v = subsAt(i, t);
    sum += v; base += subsBase[i*T+t] * 0.01;
    if (v > peak) { peak = v; peakI = i; }
    if (v > 5) area++;
    const k = town[i];
    if (k) { const e = byTown.get(k) || [0,0]; e[0] += v; e[1]++; byTown.set(k, e); }
  }
  let wn = "—", wv = 0;
  byTown.forEach((e, k) => { const m = e[0]/e[1]; if (m > wv) { wv = m; wn = D.townNames[k]; } });
  return {mean: sum/A, baseMean: base/A, peak, peakI, area, worst: wn, worstV: wv};
}
function refresh(geom) {
  const s = stats();
  $("tMean").textContent = s.mean.toFixed(1) + " cm";
  const d = s.mean - s.baseMean;
  $("tMeanD").textContent = Math.abs(d) < 0.005 ? "baseline policy"
    : `${d>=0?"+":""}${d.toFixed(2)} cm vs baseline`;
  $("tWorst").textContent = s.worst;
  $("tWorstD").textContent = s.worstV.toFixed(1) + " cm average";
  $("tArea").textContent = s.area;
  $("tPeak").textContent = s.peak.toFixed(1) + " cm";
  $("tPeakD").textContent = D.townNames[town[s.peakI]] === "unlabelled" ? "single cell" : D.townNames[town[s.peakI]];
  D.classes.forEach((c, i) => {
    const f = state.factors[i];
    $("fv"+i).textContent = f === 1 ? "no change" : (f < 1 ? `−${Math.round((1-f)*100)}%` : `+${Math.round((f-1)*100)}%`);
  });
  $("stamp").innerHTML = D.months[state.t] + (state.t > D.origin ? " <small>projected</small>" : " <small>observed forcing</small>");
  $("startVal").textContent = "January " + startLabels[state.start];
  $("legHi").textContent = subsMax.toFixed(0) + " cm";
  rebuildGeometry(geom === "ground");
  recolour();
  if (state.sel >= 0 && state.tab === "cell") drawProbe();
  if (state.tab === "sec") { drawSectionLine(); drawSection(); }
  if (state.tab === "cmp") drawCompare();
  if (state.tab === "town") drawTownships();
}
function setMode(m) {
  state.mode = m;
  $("mDecide").setAttribute("aria-pressed", m === "decide");
  $("mAnalyst").setAttribute("aria-pressed", m === "analyst");
  document.querySelectorAll(".analyst").forEach(el => el.hidden = (m === "decide"));
  state.field = m === "analyst" ? "head" : "subs";
  recolour();
}

// ---- cross-section -----------------------------------------------------------------
let secLine = null;
function drawSectionLine() {
  if (secLine) { scene.remove(secLine); secLine.geometry.dispose(); secLine = null; }
  if (!(state.secA >= 0 && state.secB >= 0 && state.secB !== null)) return;
  const pt = i => new THREE.Vector3(cols[i] - W/2 + 0.5,
    (ground[i]*0.01 - subsAt(i, state.t)) * state.exag / 40 + 3, -(rows[i] - H/2 + 0.5));
  const g = new THREE.BufferGeometry().setFromPoints([pt(state.secA), pt(state.secB)]);
  secLine = new THREE.Line(g, new THREE.LineBasicMaterial({color: 0x1f6f8b, linewidth: 2}));
  scene.add(secLine);
}
function sectionCells(n) {
  if (!(state.secA >= 0 && state.secB >= 0 && state.secB !== null)) return [];
  const c0 = [cols[state.secA], rows[state.secA]], c1 = [cols[state.secB], rows[state.secB]];
  const out = [];
  for (let k = 0; k < n; k++) {
    const u = k / (n - 1);
    const cx = Math.round(c0[0] + (c1[0]-c0[0]) * u), cy = Math.round(c0[1] + (c1[1]-c0[1]) * u);
    const cell = cellAt[cy * D.nx + cx];
    out.push({u, cell, km: Math.hypot((c1[0]-c0[0]) * u, (c1[1]-c0[1]) * u) * D.dx / 1000});
  }
  return out;
}
function drawSection() {
  const pts = sectionCells(120).filter(p => p.cell >= 0);
  const svg = $("secSvg");
  if (pts.length < 3) { svg.innerHTML = ""; $("secHint").textContent =
    "Click two points on the ground to cut a line through the fan."; return; }
  const t = state.t, len = pts[pts.length-1].km;
  const X = u => 34 + 336 * u;
  const yMin = -320, yMax = 60;                          // metres, depth axis
  const Y = m => 16 + 168 * (yMax - m) / (yMax - yMin);
  let out = "";
  // aquifers and aquitards as filled bands, drawn deep to shallow
  for (let l = L - 1; l >= 0; l--) {
    let top = "", bot = "";
    for (const p of pts) {
      const z = zone[p.cell], g = ground[p.cell] * 0.01;
      const d = D.layerDepths[D.zoneNames[z]][l], th = D.layerThick[D.zoneNames[z]][l];
      top += X(p.u).toFixed(1) + "," + Y(g - d + th/2).toFixed(1) + " ";
      bot = X(p.u).toFixed(1) + "," + Y(g - d - th/2).toFixed(1) + " " + bot;
    }
    const mid = pts[Math.floor(pts.length/2)];
    const hv = headAt(l, mid.cell, t);
    out += `<polygon points="${top}${bot}" fill="${rgb(ramp(headRamp,(hv+20)/90))}" stroke="rgba(0,0,0,.18)" stroke-width=".5"></polygon>`;
    if (l < L - 1) {                                     // the clay between this and the next
      let a = "", b = "";
      for (const p of pts) {
        const z = zone[p.cell], g = ground[p.cell] * 0.01, zn = D.zoneNames[z];
        const d1 = D.layerDepths[zn][l], t1 = D.layerThick[zn][l];
        const d2 = D.layerDepths[zn][l+1], t2 = D.layerThick[zn][l+1];
        a += X(p.u).toFixed(1) + "," + Y(g - d1 - t1/2).toFixed(1) + " ";
        b = X(p.u).toFixed(1) + "," + Y(g - d2 + t2/2).toFixed(1) + " " + b;
      }
      out += `<polygon points="${a}${b}" fill="${rgb(CLAY)}" opacity=".9"></polygon>`;
    }
  }
  // head line per aquifer
  for (let l = 0; l < L; l++) {
    let ln = "";
    for (const p of pts) ln += X(p.u).toFixed(1) + "," + Y(headAt(l, p.cell, t)).toFixed(1) + " ";
    out += `<polyline points="${ln}" fill="none" stroke="var(--ink)" stroke-width="1" opacity=".55" stroke-dasharray="${l?"3 2":""}"></polyline>`;
  }
  // ground and the subsidence profile above it
  let gl = "", sl = "", smax = 0;
  for (const p of pts) smax = Math.max(smax, subsAt(p.cell, t));
  const sScale = Math.max(5, Math.ceil(smax/5)*5);
  for (const p of pts) {
    gl += X(p.u).toFixed(1) + "," + Y(ground[p.cell]*0.01).toFixed(1) + " ";
    sl += X(p.u).toFixed(1) + "," + (14 - 11 * subsAt(p.cell, t) / sScale).toFixed(1) + " ";
  }
  out += `<polyline points="${gl}" fill="none" stroke="var(--ink)" stroke-width="1.4"></polyline>`;
  out += `<polyline points="${sl}" fill="none" stroke="${rgb(subsRamp[3][1])}" stroke-width="1.6"></polyline>`;
  for (const m of [0, -100, -200, -300])
    out += `<text x="4" y="${(Y(m)+3).toFixed(1)}" font-size="8" fill="var(--ink-3)">${m}</text>`;
  out += `<text x="4" y="10" font-size="8" fill="var(--ink-3)">m</text>`;
  out += `<text x="${X(0)}" y="205" font-size="8" fill="var(--ink-3)">0 km</text>`;
  out += `<text x="${X(1)-26}" y="205" font-size="8" fill="var(--ink-3)">${len.toFixed(0)} km</text>`;
  out += `<text x="${X(0.5)-52}" y="205" font-size="8" fill="var(--ink-3)">subsidence to ${sScale} cm, top line</text>`;
  svg.innerHTML = out;
  const a = D.townNames[town[state.secA]], b = D.townNames[town[state.secB]];
  $("secHint").textContent = `${len.toFixed(1)} km, ${a === "unlabelled" ? "cell " + state.secA : a} to ${b === "unlabelled" ? "cell " + state.secB : b}, at ${D.months[t]}.`;
}
function sectionCsv() {
  const pts = sectionCells(120).filter(p => p.cell >= 0), t = state.t;
  let csv = "km,cell,township,zone,ground_m,subsidence_cm,head_L1_m,head_L2_m,head_L3_m,head_L4_m\n";
  for (const p of pts) {
    const h = [0,1,2,3].map(l => headAt(l, p.cell, t).toFixed(2)).join(",");
    csv += `${p.km.toFixed(3)},${p.cell},${D.townNames[town[p.cell]]},${D.zoneNames[zone[p.cell]]},` +
           `${(ground[p.cell]*0.01).toFixed(2)},${subsAt(p.cell,t).toFixed(3)},${h}\n`;
  }
  return csv;
}

// ---- compare ------------------------------------------------------------------------
let cmpScale = 2;
function miniMap(id, factors) {
  const cv = $(id), ctx = cv.getContext("2d");
  const img = ctx.createImageData(D.nx, D.ny);
  img.data.fill(0);
  for (let i = 0; i < A; i++) {
    const c = ramp(subsRamp, subsWith(factors, i, state.t) / subsMax);
    const o = ((D.ny - 1 - rows[i]) * D.nx + cols[i]) * 4;
    img.data[o] = c[0]*255; img.data[o+1] = c[1]*255; img.data[o+2] = c[2]*255; img.data[o+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
function drawCompare() {
  const ref = presetFactors(state.cmpRef);
  miniMap("cmpA", state.factors); miniMap("cmpB", ref);
  let mA = 0, mB = 0, worst = 0, best = 0;
  for (let i = 0; i < A; i++) {
    const a = subsAt(i, state.t), b = subsWith(ref, i, state.t);
    mA += a; mB += b; worst = Math.max(worst, a - b); best = Math.min(best, a - b);
  }
  mA /= A; mB /= A;
  cmpScale = Math.max(0.5, Math.max(Math.abs(worst), Math.abs(best)));
  const label = {base:"Baseline, no change", irr30:"Irrigation −30%", aqua0:"Retire aquaculture", all20:"All users −20%"}[state.cmpRef];
  $("cmpBName").textContent = label;
  $("cmpRows").innerHTML =
    `<div class="row" style="display:flex;justify-content:space-between;font-size:11.5px;padding:2px 0"><span>Your policy</span><b>${mA.toFixed(2)} cm</b></div>
     <div class="row" style="display:flex;justify-content:space-between;font-size:11.5px;padding:2px 0"><span>${label}</span><b>${mB.toFixed(2)} cm</b></div>
     <div class="row" style="display:flex;justify-content:space-between;font-size:11.5px;padding:2px 0;border-top:1px solid var(--line);margin-top:3px;padding-top:5px">
       <span>Difference</span><b style="color:${mA <= mB ? "var(--good)" : "var(--bad)"}">${mA-mB>=0?"+":""}${(mA-mB).toFixed(2)} cm</b></div>
     <div class="hint" style="margin-top:6px">Best cell ${best.toFixed(2)} cm, worst ${worst>=0?"+":""}${worst.toFixed(2)} cm.</div>`;
}

// ---- townships ----------------------------------------------------------------------
let townSort = {key: "now", dir: -1};
function townshipRows() {
  const t = state.t, base = presetFactors("base");
  const m = new Map();
  for (let i = 0; i < A; i++) {
    const k = town[i]; if (!k) continue;
    const e = m.get(k) || {n:0, now:0, base:0, over:0, peak:0};
    const v = subsAt(i, t);
    e.n++; e.now += v; e.base += subsWith(base, i, t); e.over += v > 5 ? 1 : 0;
    e.peak = Math.max(e.peak, v); m.set(k, e);
  }
  const out = [];
  m.forEach((e, k) => out.push({name: D.townNames[k], n: e.n, now: e.now/e.n,
    delta: (e.now - e.base)/e.n, over: e.over, peak: e.peak}));
  out.sort((a, b) => (a[townSort.key] > b[townSort.key] ? 1 : -1) * townSort.dir);
  return out;
}
function drawTownships() {
  const r = townshipRows();
  const mx = Math.max(...r.map(x => x.now), 1);
  const th = (k, l) => `<th data-k="${k}"${townSort.key===k?' aria-sort="descending"':''}>${l}</th>`;
  $("townTbl").innerHTML = `<thead><tr>${th("name","Township")}${th("now","cm")}${th("delta","Δ policy")}${th("over","km² >5cm")}${th("peak","Peak")}</tr></thead><tbody>` +
    r.map(x => `<tr><td>${x.name}<div class="bar" style="width:${(x.now/mx*100).toFixed(0)}%"></div></td>
      <td class="num">${x.now.toFixed(1)}</td>
      <td class="num" style="color:${x.delta < -0.005 ? "var(--good)" : x.delta > 0.005 ? "var(--bad)" : "var(--ink-3)"}">${x.delta>=0?"+":""}${x.delta.toFixed(2)}</td>
      <td class="num">${x.over}</td><td class="num">${x.peak.toFixed(1)}</td></tr>`).join("") + "</tbody>";
  $("townTbl").querySelectorAll("th").forEach(h => h.addEventListener("click", () => {
    const k = h.dataset.k;
    townSort = {key: k, dir: townSort.key === k ? -townSort.dir : (k === "name" ? 1 : -1)};
    drawTownships();
  }));
}
function townCsv() {
  const r = townshipRows();
  let csv = `township,cells_km2,subsidence_cm,delta_vs_baseline_cm,cells_over_5cm,peak_cm,month,policy\n`;
  const pol = D.classes.map((c, i) => `${c}=${state.factors[i].toFixed(2)}`).join(" ");
  for (const x of r) csv += `${x.name},${x.n},${x.now.toFixed(3)},${x.delta.toFixed(3)},${x.over},${x.peak.toFixed(3)},${D.months[state.t]},${pol}\n`;
  return csv;
}

// ---- export -------------------------------------------------------------------------
function say(id, msg) { const e = $(id); e.textContent = msg; setTimeout(() => { e.textContent = ""; }, 2600); }
async function copyText(text, sayId) {
  try { await navigator.clipboard.writeText(text); say(sayId, "Copied to the clipboard."); }
  catch (e) { say(sayId, "Could not copy here; use Download instead."); }
}
// Saving a file: on claude.ai the page has to ask the viewer through the downloads
// capability, which resolves null where it is not granted; opened as a local file there
// is no such shell, so fall back to an anchor. Either way the button only ever fires on
// a deliberate click, and the page says what happened.
let downloadsApi = null, downloadsReady = false;
(async () => {
  try { downloadsApi = await window.claude?.use?.("downloads"); } catch (e) { downloadsApi = null; }
  downloadsReady = true;
})();
async function download(name, text, sayId) {
  if (downloadsApi) {
    try { await downloadsApi.save({filename: name, data: text}); say(sayId, "Saved " + name + "."); }
    catch (e) { say(sayId, e && e.code === "cancelled" ? "Save cancelled." : "Could not save; use Copy CSV."); }
    return;
  }
  try {
    const url = URL.createObjectURL(new Blob([text], {type:"text/csv"}));
    const a = document.createElement("a");
    a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
    say(sayId, "Saved " + name + ".");
  } catch (e) { say(sayId, "Download blocked here; use Copy CSV."); }
}
async function savePng() {
  renderer.render(scene, camera);
  const url = canvas.toDataURL("image/png");
  const name = "choushui-twin-" + D.months[state.t] + ".png";
  if (downloadsApi) {
    try {
      const blob = await (await fetch(url)).blob();
      await downloadsApi.save({filename: name, data: blob});
      say("townSaid", "Saved " + name + "."); return;
    } catch (e) {
      say("townSaid", e && e.code === "cancelled" ? "Save cancelled." : "Could not save the image.");
      return;
    }
  }
  try {
    const blob = await (await fetch(url)).blob();
    await navigator.clipboard.write([new ClipboardItem({"image/png": blob})]);
    say("townSaid", "Image copied to the clipboard.");
  } catch (e) {
    try {
      const a = document.createElement("a");
      a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
      say("townSaid", "Saved the image.");
    } catch (e2) { say("townSaid", "Blocked here; screenshot the window instead."); }
  }
}

// ---- tabs ---------------------------------------------------------------------------
function setTab(name) {
  state.tab = name;
  const panes = {cell:"paneCell", sec:"paneSec", cmp:"paneCmp", town:"paneTown"};
  for (const [k, id] of Object.entries(panes)) {
    $(id).hidden = k !== name;
    $("tab" + k[0].toUpperCase() + k.slice(1)).setAttribute("aria-pressed", k === name);
  }
  $("probe").classList.toggle("wide", name !== "cell");
  if (name === "sec") drawSection();
  if (name === "cmp") drawCompare();
  if (name === "town") drawTownships();
}

// wiring
buildSliders(); buildLayers();
$("time").max = T - 1; $("time").value = T - 1;
$("time").addEventListener("input", e => { state.t = +e.target.value; refresh("ground"); });
$("play").addEventListener("click", () => {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "❚❚" : "▶";
});
$("startYear").addEventListener("input", e => { state.start = +e.target.value; refresh(true); });
$("explode").addEventListener("input", e => { state.explode = +e.target.value * 0.32; $("explodeVal").textContent = e.target.value; rebuildGeometry(); });
$("clip").addEventListener("input", e => { state.clip = +e.target.value / 100;
  $("clipVal").textContent = state.clip >= 1 ? "off" : Math.round(state.clip*100) + "%"; applyVisibility(); });
$("exag").addEventListener("input", e => { state.exag = +e.target.value; $("exagVal").textContent = e.target.value + "×"; rebuildGeometry(); });
$("vHead").addEventListener("click", () => { state.headMode = "head"; state.field = "head";
  $("vHead").setAttribute("aria-pressed", true); $("vDraw").setAttribute("aria-pressed", false); recolour(); });
$("vDraw").addEventListener("click", () => { state.headMode = "draw"; state.field = "head";
  $("vHead").setAttribute("aria-pressed", false); $("vDraw").setAttribute("aria-pressed", true); recolour(); });
$("vReset").addEventListener("click", () => { camR = 105; camTheta = -0.9; camPhi = 0.92; target.set(0,0,0); place(); });
$("vTop").addEventListener("click", () => { camPhi = 0.14; camR = 95; place(); });
$("mDecide").addEventListener("click", () => setMode("decide"));
$("mAnalyst").addEventListener("click", () => setMode("analyst"));

$("tabCell").addEventListener("click", () => setTab("cell"));
$("tabSec").addEventListener("click", () => setTab("sec"));
$("tabCmp").addEventListener("click", () => setTab("cmp"));
$("tabTown").addEventListener("click", () => setTab("town"));
$("secDraw").addEventListener("click", e => {
  state.secMode = !state.secMode; state.secA = state.secB = null;
  e.target.setAttribute("aria-pressed", state.secMode);
  drawSectionLine();
  $("secHint").textContent = state.secMode
    ? "Click the start of the line, then its end."
    : "Click two points on the ground to cut a line through the fan.";
});
$("secWE").addEventListener("click", () => {          // the fan's own axis, apex to coast
  const mid = Math.round(D.ny / 2);
  let west = -1, east = -1;
  for (let c = 0; c < D.nx; c++) { const i = cellAt[mid * D.nx + c]; if (i >= 0) { if (west < 0) west = i; east = i; } }
  state.secA = west; state.secB = east; state.secMode = false;
  $("secDraw").setAttribute("aria-pressed", false);
  drawSectionLine(); drawSection();
});
$("secCopy").addEventListener("click", () => copyText(sectionCsv(), "secSaid"));
$("cmpRef").addEventListener("change", e => { state.cmpRef = e.target.value; drawCompare(); if (state.cmpOn) recolour(); });
$("cmpOn").addEventListener("click", e => {
  state.cmpOn = !state.cmpOn;
  e.target.setAttribute("aria-pressed", state.cmpOn);
  e.target.textContent = state.cmpOn ? "Colour the block by subsidence" : "Colour the block by difference";
  $("legTitle").textContent = state.cmpOn ? "Difference from the reference" : "Cumulative subsidence";
  $("legBar").style.background = "linear-gradient(90deg," +
    (state.cmpOn ? diffRamp : subsRamp).map(s2 => rgb(s2[1]) + " " + s2[0]*100 + "%").join(",") + ")";
  $("legLo").textContent = state.cmpOn ? "−" + cmpScale.toFixed(1) + " cm" : "0";
  $("legHi").textContent = state.cmpOn ? "+" + cmpScale.toFixed(1) + " cm" : subsMax.toFixed(0) + " cm";
  $("legCap").textContent = state.cmpOn
    ? "Teal is less sinking than the reference, warm is more."
    : "Ground colour. Aquifer colour is head, dark is low.";
  recolour();
});
$("townCopy").addEventListener("click", () => copyText(townCsv(), "townSaid"));
$("townDl").addEventListener("click", () => download("choushui-townships-" + D.months[state.t] + ".csv", townCsv(), "townSaid"));
$("shotPng").addEventListener("click", savePng);

const g = D.gate || {};
$("verdict").textContent = g.verdict
  ? `Flow model ${g.verdict} · held-out wells R² ${g.r2_kfold.toFixed(3)} vs ${g.r2_idw.toFixed(3)}`
  : "Gate verdict not recorded";
$("horizonYear").textContent = D.months[T-1].slice(0,4);
$("cmpYear").textContent = D.months[T-1].slice(0,7);
$("townYear").textContent = D.months[T-1].slice(0,7);
$("legBar").style.background = "linear-gradient(90deg," +
  subsRamp.map(s => `rgb(${s[1].map(v=>Math.round(v*255)).join(",")}) ${s[0]*100}%`).join(",") + ")";
$("legCap").textContent = "Ground colour. Aquifer colour is head, dark is low.";
$("caveat").innerHTML = `Heads and subsidence are validated against wells and 798 leveling
  benchmarks. The response to a policy is a model consequence, not a validated forecast:
  a free-running continuation drifts within three years, so read the shape and the
  ranking of policies, not the third decimal.` +
  (D.linErr != null ? ` Mixing policies is linear to within ${D.linErr.toFixed(2)} cm.` : "");

subsMax = (() => { let m = 0; for (let i = 0; i < A; i++) m = Math.max(m, subsBase[i*T + T-1] * 0.01); return Math.ceil(m/5)*5; })();
$("startYear").max = 1;
place(); applyVisibility(); setMode("decide"); setTab("cell"); refresh();

let last = 0;
function loop(ts) {
  if (state.playing && ts - last > 55) {
    last = ts;
    state.t = state.t >= T - 1 ? 0 : state.t + 1;
    $("time").value = state.t;
    refresh("ground");
  }
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
function resize() {
  const r = canvas.getBoundingClientRect();
  renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(1, r.height);
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize); resize();
requestAnimationFrame(loop);
</script>
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="build the twin's decision application")
    ap.add_argument("--forward", required=True, help="a twin.forward .npz (the deliverable run)")
    ap.add_argument("--basis", default=None, help="the per-class response basis .npz")
    ap.add_argument("--townships", default="results/twin_runs/cell_townships.csv")
    ap.add_argument("--quarter", type=int, default=3, help="months per stored head sample")
    ap.add_argument("--delta-step", type=int, default=12,
                    help="months per stored policy-response sample (interpolated in the page)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    build(args.forward, args.basis, args.out, townships_csv=args.townships,
          quarter=args.quarter, delta_step=args.delta_step)


if __name__ == "__main__":
    main()
