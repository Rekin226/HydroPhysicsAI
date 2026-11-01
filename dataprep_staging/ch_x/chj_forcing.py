"""CH-J (I1.CHX.v1 secondary): the pieces that live outside the twin modules.

CH-J is ONE pre-declared arm: Changhua census pumping at the X3 measured layer shares
(0.84/0.11/0.025/0.025), co-located with ALL of Changhua's surface water in layer 0 (W1 Jiji
deliveries and percolation plus x5_changhua_water's non-Jiji volume), Yunlin and every
shared parameter frozen at Q0, only the Changhua copies fitted (--county-copy-sd 0.5), seeds
2027 and 2029, both county columns refitted on the new heads with Changhua's Ske/Skv <= 0.1.
The hypothesis is the co-location of canal/paddy recharge with shallow pumping (not the pump
layer: fan-wide shallow runs already overdrove Changhua). A fitted Changhua sw fraction at its
0.01 bound is a registered outcome ('co-located recharge rejected').

What this module provides (pure functions are tested on synthetic arrays; ``build_sw`` is a
lead-run product step):

* ``q0_rest_fractions`` / ``county_layer_fractions``: the per-cell layer fractions ``(L, A)``:
  Changhua cells the X3 shares, every other cell EXACTLY Q0's own placement (Q0's meta
  pump_split is on: a sigmoid(pump_split_logit) share in layer 0, the rest in Q0's pump layer
  1; the hook computes it from Q0's theta, this module never reads a theta value);
* ``build_sw``: the merged surface-water npz (``inputs.load_sw_recharge`` schema) =
  surface_water_v4 model_npz v1 (W1 deliveries ``sw_m_per_day`` + percolation
  ``perc_m_per_day``) with x5_changhua_water preorigin v1 ``x5_t2_service`` (the non-Jiji
  volume placed on canal_redistribution_x4's service-area shares) ADDED to the deliveries,
  then ZEROED outside R1c's Changhua cells (R1c's county map, sha verified against R1c's
  meta): Q0 has no surface water, so Yunlin stays exactly as in Q0; all inputs opened through
  the loader guard with use ``forcing_frozen`` (none of them allows selection, so the merge
  is frozen forcing, never an arm choice);
* ``freeze_partition`` / ``pin_frozen``: the freeze-hook rule (which theta keys, and which
  entries of ``log_sw_scale``, are free) and the pinned values from Q0's theta;
* ``CALIBRATE_FLOW_PATCH``: the exact opt-in change ``calibrate_flow.fit_flow`` needs. It is
  NOT applied here (round-7 rule: existing twin modules are not edited by builders); until a
  reviewed opt-in hook exists at a frozen commit, CH-J cannot run and I1.CHX.v1's declared
  fallback applies (primary scored alone; CH-J external gates report-only).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chx_const as K  # noqa: E402

COUNTY_SUFFIX = "_changhua"                 # hydrophysics.twin.zones.COUNTY_SUFFIX
SW_W1 = "results/dataprep/surface_water_v4/model_npz/v1"
SW_W1_FILE = "surface_water_v4_model.npz"
SW_X5 = "results/dataprep/x5_changhua_water/preorigin/v1"
SW_X5_FILE = "x5_t2_service.npz"
SW_KEYS = ("sw_m_per_day", "perc_m_per_day", "dates", "dx", "n_active")
Q0_THETA = K.Q0_THETA
SW_KEY = "log_sw_scale"

#: theta keys CH-J fits: every Changhua copy (suffix _changhua) and the Changhua entries of
#: the county-split sw scale; everything else is pinned at Q0
FREE_RULE = ("keys ending in '_changhua' (the zonal Changhua copies of --zone-county-split) "
             "and the Changhua entries of log_sw_scale under --sw-county-split (entries 0 and "
             "2 of [c0 Changhua, c0 rest, c1 Changhua, c1 rest]); every other key, and the "
             "rest-of-fan log_sw_scale entries, pinned (Q0's value, or for the sw entries Q0 "
             "lacks the calibration's initial value: their field is identically zero); pump "
             "layer fractions fixed (not fitted)")

CALIBRATE_FLOW_PATCH = """\
Opt-in, all default off (a run without them builds the historical dict):
1. --freeze-from THETA_JSON: after theta is built (and after --county-copy-sd setup), every
   key k of theta NOT matched by FREE_RULE is moved to `fixed` with the value
   THETA_JSON['theta'][k] (or its parent's value for a copy-less key); a key FREE_RULE marks
   free that THETA_JSON lacks starts at its parent's Q0 value (epoch 0 = Q0 exactly).
   _view() already merges `fixed`, so only the pop/assign is new.
2. --pump-layer-shares-county CH0,CH1,CH2,CH3: fixed["pump_layer_frac_cells"] = the (L, A)
   matrix of chj_forcing.county_layer_fractions(changhua, q0_rest_fractions(Q0
   theta['pump_split_logit'], Q0 meta['pump_layer'])): Changhua cells the shares, every
   other cell EXACTLY Q0's placement (sigmoid(pump_split_logit) in layer 0, the rest in
   layer 1). _forward passes it as pump_layer_frac (that path replaces the pump_split path;
   for a non-Changhua cell it subtracts the same two terms and 0 x rate elsewhere, so it is
   identical). pump_split_logit stays pinned at Q0's value. Mutually exclusive with
   --pump-layer-fractions / --pump-layer-shares.
3. --sw-recharge <chj_sw.npz> --sw-components both --sw-layer 0 --sw-county-split (existing
   flags). chj_sw.npz is zero outside R1c's Changhua cells (build_sw), so the rest-of-fan
   components contribute exactly 0 whatever their fraction; their log_sw_scale entries (1
   and 3) are pinned at the initial value, the Changhua entries (0 and 2) are free and
   bounded to fractions in [0.01, 1].
Engineering pre-gate (synthetic heads, CPU): with every Changhua copy pinned at Q0, the
Changhua shares set to q0_rest_fractions (Q0's own placement) and the sw field zero, the
patched fit_flow's epoch-0 rollout reproduces Q0's rollout to 1e-6 m
(chx_const.CHJ_FREEZE_REPRO_TOL_M). Its result, the hook's commit and sha256, the rank-8
forcing version and sha256, the panel producer's sha256 and the frozen time go into
results/twin_runs/chx/chj/chj_engineering.json (score_chx.ENGINEERING_KEYS) before any fit;
the producer writes chj_panel.json (score_chx.PANEL_SUMMARY_KEYS) beside chj_panel.npz.
Without them, or if the record is dated after the 21-day deadline, CH-J is report-only.
"""


# ----------------------------------------------------------------------------- shares
def q0_rest_fractions(pump_split_logit: float | None, pump_layer: int = K.Q0_PUMP_LAYER,
                      n_layers: int = 4) -> np.ndarray:
    """``(L,)``: Q0's own placement of every cell's pumped volume, as calibrate_flow applies
    it: with ``pump_split_logit`` and ``pump_layer != 0``, sigmoid(logit) in layer 0 and the
    rest in ``pump_layer``; otherwise everything in ``pump_layer``."""
    if not 0 <= pump_layer < n_layers:
        raise ValueError("pump_layer outside the layers")
    f = np.zeros(n_layers)
    if pump_split_logit is None or pump_layer == 0:
        f[pump_layer] = 1.0
        return f
    s = 1.0 / (1.0 + np.exp(-float(pump_split_logit)))
    f[0], f[pump_layer] = s, 1.0 - s
    return f


def county_layer_fractions(changhua: np.ndarray, rest_fractions,
                           shares=K.CHJ_SHARES_CHANGHUA) -> np.ndarray:
    """``(L, A)``: Changhua cells take ``shares``, every other cell ``rest_fractions`` (Q0's
    placement, ``q0_rest_fractions``). Columns sum to one."""
    ch = np.asarray(changhua, bool)
    sh = np.asarray(shares, float)
    rest = np.asarray(rest_fractions, float)
    n_layers = sh.size
    if (sh <= 0).any() or abs(sh.sum() - 1.0) > 1e-9:
        raise ValueError("shares must be positive values summing to 1")
    if rest.shape != (n_layers,) or (rest < 0).any() or abs(rest.sum() - 1.0) > 1e-9:
        raise ValueError(f"rest_fractions must be {n_layers} values >= 0 summing to 1")
    F = np.empty((n_layers, ch.size))
    F[:, ~ch] = rest[:, None]
    F[:, ch] = sh[:, None]
    return F


# ----------------------------------------------------------------------------- freeze
def sw_entries(n_components: int = K.CHJ_SW_COMPONENTS) -> tuple[list[int], list[int]]:
    """(Changhua, rest) entries of log_sw_scale under --sw-county-split: the field is
    [c0 Changhua, c0 rest, c1 Changhua, c1 rest, ...]."""
    return ([2 * i for i in range(n_components)], [2 * i + 1 for i in range(n_components)])


def freeze_partition(keys, n_components: int = K.CHJ_SW_COMPONENTS
                     ) -> tuple[list[str], list[str]]:
    """(free, pinned) under FREE_RULE; ``log_sw_scale`` is split by entry
    (``log_sw_scale[i]``): its Changhua entries free, its rest-of-fan entries pinned."""
    keys = list(keys)
    free = sorted(k for k in keys if k.endswith(COUNTY_SUFFIX))
    pinned = sorted(k for k in keys if k not in free and k != SW_KEY)
    if SW_KEY in keys:
        ch, rest = sw_entries(n_components)
        free += [f"{SW_KEY}[{i}]" for i in ch]
        pinned += [f"{SW_KEY}[{i}]" for i in rest]
    return free, pinned


def pin_frozen(keys, q0_theta: dict, n_components: int = K.CHJ_SW_COMPONENTS,
               sw_init=K.CHJ_SW_INIT) -> dict:
    """{'pinned': {key: value}, 'start_free': {key: value}}: pinned keys from Q0's theta (the
    rest-of-fan sw entries at the initial value, since Q0 has no sw and their field is zero);
    a free Changhua copy starts at Q0's value of its parent; a free Changhua sw entry at the
    initial value."""
    free, pinned = freeze_partition(keys, n_components)
    plain = [k for k in pinned if not k.startswith(f"{SW_KEY}[")]
    miss = [k for k in plain if k not in q0_theta]
    if miss:
        raise KeyError(f"Q0's theta lacks pinned keys {miss}")
    init = [float(np.log(v)) for v in sw_init]
    out = {k: q0_theta[k] for k in plain}
    start = {}
    for k in free + [k for k in pinned if k.startswith(f"{SW_KEY}[")]:
        if k.startswith(f"{SW_KEY}["):
            v = init[(int(k[len(SW_KEY) + 1:-1]) // 2) % len(init)]
            (start if k in free else out)[k] = v
        elif k in q0_theta:
            start[k] = q0_theta[k]
        elif k[:-len(COUNTY_SUFFIX)] in q0_theta:
            start[k] = q0_theta[k[:-len(COUNTY_SUFFIX)]]
    return {"pinned": out, "start_free": start}


# ----------------------------------------------------------------------------- sw merge
def merge_fields(w1: dict, x5: dict, changhua: np.ndarray | None = None) -> dict:
    """The merged field: deliveries W1 + x5 non-Jiji (same axis, grid), percolation W1; with
    ``changhua`` (A,) bool, both fields are zero outside the Changhua cells."""
    for k in ("dates", "dx", "n_active"):
        if k not in w1 or k not in x5:
            raise ValueError(f"missing {k}")
    d1 = [str(d)[:7] for d in w1["dates"]]
    d5 = [str(d)[:7] for d in x5["dates"]]
    if d1 != d5:
        raise ValueError("W1 and x5 axes differ")
    if float(w1["dx"]) != float(x5["dx"]) or int(w1["n_active"]) != int(x5["n_active"]):
        raise ValueError("W1 and x5 grids differ")
    a, b = np.asarray(w1["sw_m_per_day"], float), np.asarray(x5["sw_m_per_day"], float)
    p = np.asarray(w1["perc_m_per_day"], float)
    for name, f in (("w1", a), ("x5", b), ("perc", p)):
        if f.shape != (int(w1["n_active"]), len(d1)):
            raise ValueError(f"{name} field shape {f.shape}")
        if not np.isfinite(f).all() or (f < 0).any():
            raise ValueError(f"{name} field must be finite and >= 0")
    sw = a + b
    if changhua is not None:
        ch = np.asarray(changhua, bool)
        if ch.shape != (sw.shape[0],):
            raise ValueError("county flags do not match the field's cells")
        sw = np.where(ch[:, None], sw, 0.0)
        p = np.where(ch[:, None], p, 0.0)
    return {"sw_m_per_day": sw, "perc_m_per_day": p,
            "dates": np.asarray([str(d)[:10] for d in w1["dates"]]),
            "dx": np.float64(w1["dx"]), "n_active": np.int64(w1["n_active"])}


def build_sw(out: Path) -> dict:
    """Lead-run: open the two forcing products (use forcing_frozen), merge, write
    ``out/chj_sw.npz`` (mode 'x') and a provenance JSON. Values are never printed."""
    import chx_io as IO
    from common import guard

    from hydrophysics.twin.zone_county import county_sha256, load_county_npz

    meta = json.loads(IO.rp(K.R1C_THETA).read_text())["meta"]
    cmap = load_county_npz(str(IO.rp(K.COUNTY_MAP)))
    changhua = cmap["changhua"]
    if county_sha256(changhua) != meta.get("zone_county_sha256"):
        raise SystemExit("the county map differs from the one R1c was fitted on")
    p1 = guard.open_product(IO.rp(SW_W1), "forcing_frozen")
    p5 = guard.open_product(IO.rp(SW_X5), "forcing_frozen")
    with np.load(p1.path(SW_W1_FILE), allow_pickle=False) as z:
        w1 = {k: z[k] for k in z.files}
    with np.load(p5.path(SW_X5_FILE), allow_pickle=False) as z:
        x5 = {k: z[k] for k in z.files}
    merged = merge_fields(w1, x5, changhua)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "chj_sw.npz"
    with target.open("xb") as fh:
        np.savez(fh, sw_version=np.str_("CH-J: W1 v4 deliveries + x5 non-Jiji (service) ; "
                                        "W1 percolation; zero outside Changhua"),
                 source=np.str_(f"{SW_W1}/{SW_W1_FILE} + {SW_X5}/{SW_X5_FILE}"), **merged)
    prov = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "inputs": {f"{SW_W1}/{SW_W1_FILE}": IO.sha256(p1.path(SW_W1_FILE)),
                       f"{SW_X5}/{SW_X5_FILE}": IO.sha256(p5.path(SW_X5_FILE))},
            "county_map": K.COUNTY_MAP, "county_sha256": county_sha256(changhua),
            "use": "forcing_frozen", "guard_notes": list(p1.notes) + list(p5.notes),
            "rule": "deliveries = W1 + x5 service placement; percolation = W1; both zero "
                    "outside R1c's Changhua cells; layer 0, county-split fractions (Changhua "
                    "free in [0.01, 1]; rest pinned, their field is zero)",
            "output": str(target), "output_sha256": IO.sha256(target)}
    with (out / "chj_sw.json").open("x") as f:
        json.dump(prov, f, indent=1)
    return prov


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-sw")
    b.add_argument("--out", type=Path, required=True)
    sub.add_parser("patch", help="print the calibrate_flow opt-in patch CH-J needs")
    a = ap.parse_args(argv)
    if a.cmd == "patch":
        print(CALIBRATE_FLOW_PATCH)
        print("FREE_RULE:", FREE_RULE)
        return 0
    prov = build_sw(a.out)
    print(f"wrote {prov['output']} sha256 {prov['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
