"""CH-J hook (I1.CHX.v1 secondary; ruling R8.1): --freeze-from and --pump-layer-shares-county,
the opt-in change chj_forcing.CALIBRATE_FLOW_PATCH registers.

- (a) default off: without the new arguments fit_flow builds the historical theta and fixed
  dicts bit for bit (against the pre-patch module when git can show it; and the new code
  path is never entered);
- (b) --freeze-from partitions and pins exactly as chj_forcing.freeze_partition/pin_frozen;
  epoch 0 is the parent; only the free keys move;
- (c) --pump-layer-shares-county: Changhua cells take the shares, every other cell exactly
  the parent's placement; the theta file form rebuilds the same fractions;
- (d) mutual exclusions and refusals, in fit_flow and on the CLI.

CPU, synthetic grid and heads, no data."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import hydrophysics.twin.calibrate_flow as CF  # noqa: E402
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.inputs import split_sw_by_county  # noqa: E402
from hydrophysics.twin.zones import (  # noqa: E402
    COUNTY_SUFFIX,
    county_split,
    fan_zones,
    zone_labels,
)

D = torch.float64
ROOT = Path(__file__).resolve().parents[1]
#: the last commit before the hook (round-7 merge, PR #68)
PREPATCH = "af97301196a6ee816567f6e7c8a2fabedf6bf413"
X3 = (0.84, 0.11, 0.025, 0.025)


# ----------------------------------------------------------------------------- fixtures
def _grid():
    # 8 x 4 cells of 2 km: centroids at x = 200..214 km
    return FanGrid(nx=8, ny=4, dx=2000.0, x0=199_000.0, y0=2_640_000.0,
                   mask=np.ones((4, 8), dtype=bool))


def _changhua(g):
    return g.centroids()[:, 1] > 2_644_000.0


def _problem(g, T=5, seed=0):
    rng = np.random.default_rng(seed)
    A = g.n_active
    x = torch.tensor(g.centroids()[:, 0], dtype=D)
    h0 = (10.0 + (x - x.mean()) / 2000.0).expand(4, A).clone()
    idx = torch.tensor([0, 5, 10, 15, 20, 25, 30, 7, 12, 27])
    lay = torch.tensor([0, 1, 2, 3, 0, 1, 2, 1, 0, 3])
    north = torch.tensor(_changhua(g)[idx.numpy()], dtype=D)
    obs = (h0[lay, idx][:, None] - 0.3 * (1.0 - 0.6 * north)[:, None]
           * torch.arange(1, T + 1, dtype=D)[None, :])
    E = torch.tensor(rng.uniform(1e3, 4e3, (A, T)), dtype=D)
    ge = torch.full((A,), 20.0, dtype=D)
    rf = torch.full((A, T), 4e-4, dtype=D)
    rch = torch.zeros(4, A, T, dtype=D)
    sw = torch.tensor(rng.uniform(0.0, 2e-3, (2, A, T)), dtype=D)
    return dict(h0=h0, idx=idx, lay=lay, obs=obs, E=E, ge=ge, rf=rf, rch=rch, sw=sw)


def _zones(g, split=False):
    z = fan_zones(g.centroids(), 205.0, 201.0, 209.0 if split else None)
    n = 3 + int(split)
    return z, county_split(z, _changhua(g), n), zone_labels(split), zone_labels(split, True)


def _fit(mod, g, P, *, county=True, sw=True, epochs=3, lr=0.05, split=False, **kw):
    z, zc, names, names_c = _zones(g, split)
    ch = _changhua(g)
    args = dict(E=P["E"], ground_elev=P["ge"], epochs=epochs, lr=lr, param_mode="zonal",
                h0=P["h0"], recharge_field=P["rf"], pump_split=True, return_flow=True,
                learn_spread=True, fix_eta=0.5, fix_head_extra=40.0, well_datum="fit")
    if county:
        args.update(zone_of_cell=zc, zonation=names_c,
                    changhua_cells=torch.as_tensor(ch))
        if sw:
            args.update(sw_field=split_sw_by_county(P["sw"], ch), sw_layer=0)
    else:
        args.update(zone_of_cell=z, zonation=names)
        if sw:
            args.update(sw_field=P["sw"], sw_layer=0)
    args.update(kw)
    m = mod.FlowModel(g, n_layers=4, dt_days=30.0)
    return m, mod.fit_flow(m, P["obs"], P["idx"], P["lay"], P["rch"], **args)


def _parent(g, P):
    """A parent theta in Q0's form: unsplit zonation, no surface water, pump_split."""
    _, f = _fit(CF, g, P, county=False, sw=False, epochs=4, lr=0.1)
    return f["theta"]


def _chj_module():
    """The frozen registered helpers (dataprep_staging/ch_x/chj_forcing.py), sys.path kept."""
    path = ROOT / "dataprep_staging" / "ch_x" / "chj_forcing.py"
    if not path.is_file():
        pytest.skip("chj_forcing.py not present")
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("_chj_forcing_frozen", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def _prepatch_module():
    """calibrate_flow as of PREPATCH. Skips only outside a git checkout, or in a shallow
    clone off CI; CI checks out the full history (ci.yml fetch-depth: 0), so there, and in
    any full checkout, a missing PREPATCH FAILS instead of skipping."""
    try:
        shallow = _git("rev-parse", "--is-shallow-repository").strip() == "true"
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout: the pre-patch module cannot be shown")
    if shallow:
        if os.environ.get("CI"):
            pytest.fail("CI checked out a shallow clone: set fetch-depth: 0 (ci.yml) so the "
                        "pre-patch bit-identity test runs")
        pytest.skip("shallow clone (local): the pre-patch commit cannot be shown")
    try:
        src = _git("show", f"{PREPATCH}:hydrophysics/twin/calibrate_flow.py")
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.fail(f"full git history but the pre-patch commit {PREPATCH[:7]} is missing: "
                    f"{exc}")
    name = "hydrophysics.twin._calibrate_flow_prepatch"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hydrophysics.twin"
    mod.__file__ = CF.__file__          # used only for the repository root (git provenance)
    exec(compile(src, f"<{PREPATCH[:7]}:calibrate_flow.py>", "exec"), mod.__dict__)
    # the module-level options other tests may have set on calibrate_flow, mirrored
    mod.BOUNDS.clear()
    mod.BOUNDS.update(CF.BOUNDS)
    mod.ZONE_LOWER_BOUNDS.clear()
    mod.ZONE_LOWER_BOUNDS.update(CF.ZONE_LOWER_BOUNDS)
    mod.RETURN_RICE = CF.RETURN_RICE
    return mod


def _same(a: dict, b: dict) -> None:
    """Bit-for-bit equality of two fit results (theta values, key order, fixed, loss)."""
    assert list(a["theta"]) == list(b["theta"])
    for k in a["theta"]:
        assert np.array_equal(np.asarray(a["theta"][k]), np.asarray(b["theta"][k])), k
    assert a["fixed"] == b["fixed"]
    assert a["loss"] == b["loss"] and a["r2"] == b["r2"] and a["n_params"] == b["n_params"]
    assert a["bounds_hit"] == b["bounds_hit"]
    assert a.get("county_copy_departure") == b.get("county_copy_departure")
    assert set(a) == set(b)


# ----------------------------------------------------------------------------- helpers
def test_helpers_equal_the_registered_ones():
    chj = _chj_module()
    keys = ["log_T_mid", "log_S_mid", "log_T_mid_changhua", "log_S_mid_changhua",
            "pump_split_logit", "log_C_coast", "log_sw_scale", "log_spread_km"]
    assert CF.freeze_partition(keys, 2) == chj.freeze_partition(keys, 2)
    assert CF.freeze_partition(keys[:-2], 2) == chj.freeze_partition(keys[:-2], 2)
    q0 = {"log_T_mid": [1.0, 2.0], "log_S_mid": [-3.0, -4.0], "pump_split_logit": -2.3,
          "log_C_coast": [5.0], "log_spread_km": -0.69}
    assert CF.pin_frozen(keys, q0, 2) == chj.pin_frozen(keys, q0, 2)
    with pytest.raises(KeyError, match="lacks pinned"):
        CF.pin_frozen(keys + ["log_L_mid"], q0, 2)
    assert tuple(CF.SW_SCALE_INIT_2) == tuple(chj.K.CHJ_SW_INIT)
    assert CF.FREEZE_SW_KEY == chj.SW_KEY and COUNTY_SUFFIX == chj.COUNTY_SUFFIX
    for logit, layer in ((-2.346840771613989, 1), (0.7, 2), (None, 1), (1.0, 0)):
        assert np.array_equal(CF.q0_rest_fractions(logit, layer),
                              chj.q0_rest_fractions(logit, layer))
    ch = np.array([True, False, True, False, False])
    rest = chj.q0_rest_fractions(-2.346840771613989, 1)
    assert np.array_equal(CF.county_layer_fractions(ch, rest, X3),
                          chj.county_layer_fractions(ch, rest, X3))
    with pytest.raises(ValueError, match="positive"):
        CF.county_layer_fractions(ch, rest, rest)          # chj_forcing's rule by default
    with pytest.raises(ValueError, match="positive"):
        chj.county_layer_fractions(ch, rest, rest)
    F = CF.county_layer_fractions(ch, rest, rest, zero_shares_ok=True)
    assert np.array_equal(F, np.repeat(rest[:, None], ch.size, axis=1))


# ----------------------------------------------------------------------------- (a) default
@pytest.mark.filterwarnings("ignore:_cg did not converge")
@pytest.mark.parametrize("cfg", [
    dict(county=True, sw=True, county_copy_sd=0.5),
    dict(county=True, sw=True, split=True, county_tie=("proximal",)),
    dict(county=False, sw=True),
    dict(county=True, sw=False, pump_split=False, pump_layer_fractions_on=True,
         pump_fractions_county=True),
])
def test_default_run_is_bit_identical_to_the_prepatch_code(cfg):
    old = _prepatch_module()
    g = _grid()
    P = _problem(g)
    _, a = _fit(old, g, P, **cfg)
    _, b = _fit(CF, g, P, **cfg)
    _same(a, b)
    assert "freeze_partition" not in b


@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_default_run_never_enters_the_hook(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("CH-J hook entered on a default run")

    g = _grid()
    P = _problem(g)
    _, ref = _fit(CF, g, P, county_copy_sd=0.5)
    for name in ("freeze_partition", "pin_frozen", "county_layer_fractions", "share_logits"):
        monkeypatch.setattr(CF, name, boom)
    _, f = _fit(CF, g, P, county_copy_sd=0.5)
    _same(ref, f)
    assert "pump_layer_frac_cells" not in f["fixed"]
    assert not any(k.startswith("pump_layer_logits") for k in f["theta"])
    # the CLI helpers: nothing set -> no options, no meta
    from types import SimpleNamespace

    assert CF._chj_check_args(SimpleNamespace(freeze_from=None,
                                              pump_layer_shares_county=None)) is None
    assert CF._chj_fit_options(None) == {} and CF._chj_meta(None, f) == {}
    defaults = _parse_defaults()
    assert defaults.freeze_from is None and defaults.pump_layer_shares_county is None


def _parse_defaults(argv=(), mod=None):
    """main's parsed namespace for ``argv``, without running it (parse_args is intercepted
    right after parsing; nothing after it runs)."""
    import argparse

    captured = {}
    real = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        ns = real(self, args, namespace)
        captured["ns"] = ns
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = grab
    try:
        with pytest.raises(SystemExit):
            (mod or CF).main(list(argv))
    finally:
        argparse.ArgumentParser.parse_args = real
    return captured["ns"]


#: Q0's recipe (results/twin_runs/par20/jobs.sh) and R1c's county split on top of it
Q0_ARGV = ["--param-mode", "zonal", "--epochs", "500", "--n-folds", "5", "--fix-eta", "0.5",
           "--fix-head-extra", "40", "--return-flow", "--l-min", "1e-4", "--learn-spread",
           "--spread-max-km", "10", "--ic-merged-proximal", "--ground-elev", "dem",
           "--zone-boundaries", "205,182,208", "--log-t-min-proximal", "58", "--well-datum",
           "fit", "--well-datum-sd", "5", "--device", "cuda", "--compile-matvec",
           "--log-every", "25", "--seed", "2027", "--fold-site-tol-m", "200", "--pump-split",
           "--day-weighted-months"]
COUNTY_ARGV = ["--zone-county-split", "--county-copy-sd", "0.5"]


@pytest.mark.parametrize("extra", [[], COUNTY_ARGV, ["--fit-only"], ["--only-fold", "3"]])
def test_default_recipe_is_byte_identical_to_the_prepatch_one(extra):
    """The new options never enter a default run's cfg (merge_folds compares it), so a fold
    of a pre-patch run and one of a patched run still merge."""
    old = _prepatch_module()
    args = (12, 205.0, 182.0, 208.0, {"mid": 3}, None, "abc1234")
    a = old._flow_cfg(_parse_defaults(Q0_ARGV + extra, old), *args)
    b = CF._flow_cfg(_parse_defaults(Q0_ARGV + extra), *args)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert "freeze_from" not in b["recipe"] and "pump_layer_shares_county" not in b["recipe"]
    assert "chj_hook" not in b


# ----------------------------------------------------------------------------- (b) freeze
@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_freeze_partitions_and_pins_like_chj_forcing():
    chj = _chj_module()
    g = _grid()
    P = _problem(g)
    q0 = _parent(g, P)
    _, start = _fit(CF, g, P, epochs=0, county_copy_sd=0.5, freeze_from=q0)
    _, f = _fit(CF, g, P, epochs=8, lr=0.1, county_copy_sd=0.5, freeze_from=q0)
    # the fit's own parameter keys (fix_eta / fix_head_extra already fixed them)
    keys = [k for k in _fit(CF, g, P, epochs=0)[1]["theta"]
            if CF._base_param_name(k) in _PARAMS and k not in ("log_eta", "log_head_extra")]
    free, pinned = chj.freeze_partition(keys)
    assert f["freeze_partition"] == {"free": free, "pinned": pinned}
    vals = chj.pin_frozen(keys, q0)
    th = f["theta"]
    for k, v in vals["pinned"].items():
        if k.startswith("log_sw_scale["):
            assert th["log_sw_scale"][int(k[13:-1])] == v
        else:
            assert np.array_equal(np.ravel(th[k]), np.ravel(v)), k
            assert k in f["fixed"]
    for k, v in vals["start_free"].items():
        got = (start["theta"]["log_sw_scale"][int(k[13:-1])] if k.startswith("log_sw_scale[")
               else start["theta"][k])
        assert np.array_equal(np.ravel(got), np.ravel(v)), k
    # only the free keys move; the Changhua sw fractions stay inside [0.01, 1]
    moved = [k for k in free if not k.startswith("log_sw_scale[")
             and not np.array_equal(np.ravel(th[k]), np.ravel(start["theta"][k]))]
    assert moved
    sw = np.asarray(th["log_sw_scale"])
    assert (sw[[0, 2]] >= math.log(0.01) - 1e-12).all() and (sw[[0, 2]] <= 0.0).all()
    assert th["pump_split_logit"] == q0["pump_split_logit"]
    # the two pinned rest-of-fan sw entries are not parameters; the two Changhua ones are
    assert f["n_params"] == sum(np.size(start["theta"][k]) for k in free
                                if not k.startswith("log_sw_scale[")) + 2
    assert set(f["county_copy_departure"]) == {k for k in free if k.endswith(COUNTY_SUFFIX)}


@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_freeze_bounds_hit_reads_the_changhua_sw_fractions_against_the_hook_bounds():
    """(minor) under --freeze-from, bounds_hit's log_sw_scale entry covers the two free
    Changhua entries against [0.01, 1] (the hook's ceiling, not BOUNDS' 2); a default fit
    keeps the historical entry over every component."""
    sw = torch.log(torch.tensor([1.0, 0.15, 0.01, 0.25], dtype=D))
    pinned = torch.tensor([False, True, False, True])
    hits = {"global": {"log_sw_scale": {"lo": 1, "hi": 0, "n": 4}}, "mid": {}}
    CF._freeze_sw_hits(hits, sw, pinned)
    assert hits["global"]["log_sw_scale"] == {"lo": 1, "hi": 1, "n": 2}
    CF._freeze_sw_hits(hits, torch.log(torch.tensor([0.5, 0.15, 0.3, 0.25], dtype=D)),
                       pinned)
    assert hits["global"]["log_sw_scale"] == {"lo": 0, "hi": 0, "n": 2}
    assert abs(CF.FREEZE_SW_FRACTION_MIN - math.exp(CF.BOUNDS["log_sw_scale"][0])) < 1e-15
    g = _grid()
    P = _problem(g)
    q0 = _parent(g, P)
    _, f = _fit(CF, g, P, epochs=2, county_copy_sd=0.5, freeze_from=q0)
    assert f["bounds_hit"]["global"]["log_sw_scale"]["n"] == 2
    _, d = _fit(CF, g, P, epochs=2, county_copy_sd=0.5)
    assert d["bounds_hit"]["global"]["log_sw_scale"]["n"] == 4


def test_chj_meta_marks_the_fixed_keys_and_refuses_posterior_tools(tmp_path):
    """(finding 10) a CH-J theta's meta names every key it holds fixed (the pinned parent
    values and the county shares, whose zero shares are PUMP_FRAC_LOGIT_ZERO logits) and
    marks posterior tools unsupported; ``fixed`` is the list uncertainty.py and n1_gaps'
    identifiability already hold, so a stray posterior run would not clip a -1000 logit."""
    p = tmp_path / "q0.json"
    p.write_text(json.dumps({"theta": {"pump_split_logit": -2.0}, "meta": {"pump_layer": 1}}))
    x3 = ",".join(map(str, X3))
    ns = _parse_defaults(Q0_ARGV + COUNTY_ARGV + ["--fit-only", "--freeze-from", str(p),
                                                  "--pump-layer-shares-county", x3])
    chj = CF._chj_check_args(ns)
    fit = {"freeze_partition": {
        "free": ["log_T_mid_changhua", "log_sw_scale[0]", "log_sw_scale[2]"],
        "pinned": ["log_T_mid", "pump_split_logit", "log_sw_scale[1]", "log_sw_scale[3]"]},
        "fixed": ["log_T_mid", "log_eta", "log_head_extra", "pump_layer_frac_cells"]}
    m = CF._chj_meta(chj, fit)
    hook = m["chj_hook"]
    assert hook["fixed_keys"] == sorted(["log_T_mid", "pump_split_logit", "pump_layer_logits",
                                         "pump_layer_logits_ch", "log_eta",
                                         "log_head_extra"])
    assert "log_sw_scale[1]" in hook["sw_pinned_values"]
    assert "sw_recharge_sha256" not in m                     # no --sw-recharge file here
    sw = tmp_path / "chj_sw.npz"
    sw.write_bytes(b"field")
    ns = _parse_defaults(Q0_ARGV + COUNTY_ARGV + ["--fit-only", "--freeze-from", str(p),
                                                  "--sw-recharge", str(sw),
                                                  "--sw-county-split"])
    chj_sw = CF._chj_check_args(ns)
    assert CF._chj_meta(chj_sw, fit)["sw_recharge_sha256"] == CF._file_sha256(str(sw))
    rec = CF._chj_fold_record(chj_sw, {"fold": 3})
    assert rec["chj_parent"]["sw_recharge_sha256"] == CF._file_sha256(str(sw))
    assert m["fixed"] == hook["fixed_keys"]
    assert hook["pinned_sw_entries"] == ["log_sw_scale[1]", "log_sw_scale[3]"]
    assert hook["posterior_tools_unsupported"] is True and hook["fixed_pump_layer_shares"]
    assert hook["pump_frac_logit_zero"] == CF.PUMP_FRAC_LOGIT_ZERO
    assert m["freeze_from_sha256"] == CF._file_sha256(str(p))
    from hydrophysics.twin.uncertainty import flatten

    theta = {"log_T_mid": [1.0], "log_T_mid_changhua": [1.5], "pump_split_logit": -2.0,
             "pump_layer_logits": [0.0, -1000.0], "pump_layer_logits_ch": [-0.1, -2.0]}
    _, index = flatten(theta, fixed=tuple(m["fixed"]))
    assert [k for k, _ in index] == ["log_T_mid_changhua"]


_PARAMS = {"log_T", "log_S", "log_L", "log_C_coast", "log_C_apex", "log_sw_scale", "log_eta",
           "log_head_extra", "recharge_frac_logit", "pump_split_logit", "return_frac_logit",
           "log_spread_km"}


@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_epoch0_of_the_frozen_fit_is_the_parent(monkeypatch):
    g = _grid()
    P = _problem(g, T=6)
    q0 = _parent(g, P)
    z, zc, names, names_c = _zones(g)
    ch = _changhua(g)
    # the parent's rollout through the evaluation path, on a model carrying its parameters
    th = {k: torch.as_tensor(np.asarray(v, dtype="float64"), dtype=D)
          for k, v in q0.items() if isinstance(v, (int, float, list))}
    for k in list(th):
        if k.startswith(("log_T_", "log_S_", "log_L_")):
            th[k] = th[k].reshape(-1, 1)
    m_ref = FlowModel(g, n_layers=4, dt_days=30.0)
    with torch.no_grad():
        lT, lS, lL = CF._expand_zonal(th, torch.as_tensor(z, dtype=torch.long), 4)
        m_ref.log_T.copy_(lT)
        m_ref.log_S.copy_(lS)
        m_ref.log_L.copy_(lL)
    m_ref.set_apex_heads(P["h0"])
    ref = CF._predict_homogeneous(m_ref, {"theta": q0}, P["h0"], 6, recharge_field=P["rf"],
                                  E=P["E"], ground_elev=P["ge"])
    seen = []
    orig = CF._rollout

    def cap(*a, **k):
        out = orig(*a, **k)
        seen.append(out.detach().clone())
        return out

    monkeypatch.setattr(CF, "_rollout", cap)
    rest = CF.q0_rest_fractions(q0["pump_split_logit"], 1)
    zero_sw = torch.zeros(4, g.n_active, 6, dtype=D)
    _fit(CF, g, P, epochs=1, county_copy_sd=0.5, sw=False, sw_field=zero_sw, sw_layer=0,
         freeze_from=q0, pump_layer_shares_county=(rest, rest))
    assert float((seen[0] - ref).abs().max()) <= 1e-9
    seen.clear()
    _fit(CF, g, P, epochs=1, county_copy_sd=0.5, sw=False, sw_field=zero_sw, sw_layer=0,
         freeze_from=q0, pump_layer_shares_county=(rest, np.asarray(X3)))
    assert float((seen[0] - ref).abs().max()) > 1e-3       # the gate can see a change
    assert ch.any() and (~ch).any()


# ----------------------------------------------------------------------------- (c) shares
@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_county_shares_on_changhua_cells_and_the_parent_elsewhere(monkeypatch):
    g = _grid()
    P = _problem(g)
    q0 = _parent(g, P)
    ch = _changhua(g)
    rest = CF.q0_rest_fractions(q0["pump_split_logit"], 1)
    seen = []
    orig = CF._rollout

    def cap(*a, **k):
        seen.append(k.get("pump_layer_frac"))
        return orig(*a, **k)

    monkeypatch.setattr(CF, "_rollout", cap)
    m, f = _fit(CF, g, P, epochs=2, county_copy_sd=0.5, freeze_from=q0,
                pump_layer_shares_county=(rest, np.asarray(X3)))
    fr = seen[0].cpu().numpy()
    assert fr.shape == (4, g.n_active)
    assert np.array_equal(fr[:, ~ch], np.repeat(rest[:, None], (~ch).sum(), axis=1))
    assert np.array_equal(fr[:, ch], np.repeat(np.asarray(X3)[:, None], ch.sum(), axis=1))
    assert all(np.array_equal(s.cpu().numpy(), fr) for s in seen)    # fixed, never fitted
    assert "pump_layer_frac_cells" in f["fixed"]
    assert "pump_layer_frac_cells" not in f["theta"]
    # the theta file form: pump_layer_logits (rest) / _ch (shares), rebuilt by the consumers
    th = f["theta"]
    rb = CF.pump_layer_fractions(torch.tensor(th["pump_layer_logits"], dtype=D),
                                 torch.tensor(th["pump_layer_logits_ch"], dtype=D),
                                 torch.as_tensor(ch)).numpy()
    assert np.abs(rb - fr).max() <= 1e-15 and (rb[2:, ~ch] == 0.0).all()
    assert th["pump_layer_shares"] == pytest.approx(list(rest), abs=1e-15)
    assert th["pump_layer_shares_ch"] == pytest.approx(list(X3), abs=1e-15)
    # k-fold scoring's rebuild of the fitted model equals the fit's own final rollout
    seen.clear()
    monkeypatch.setattr(CF, "_rollout", orig)
    sw = split_sw_by_county(P["sw"], ch)
    h_eval = CF._predict_homogeneous(m, f, P["h0"], P["obs"].shape[1],
                                     recharge_field=P["rf"], E=P["E"], ground_elev=P["ge"],
                                     sw_field=sw, sw_layer=0, changhua_cells=torch.as_tensor(ch))
    pred = h_eval[P["lay"], P["idx"], 1:].numpy()
    assert np.isfinite(pred).all()
    # the same model without the shares (Q0's placement everywhere) differs in Changhua only
    m2, f2 = _fit(CF, g, P, epochs=0, county_copy_sd=0.5, freeze_from=q0)
    m3, f3 = _fit(CF, g, P, epochs=0, county_copy_sd=0.5, freeze_from=q0,
                  pump_layer_shares_county=(rest, rest))
    h2 = CF._predict_homogeneous(m2, f2, P["h0"], 5, recharge_field=P["rf"], E=P["E"],
                                 ground_elev=P["ge"], sw_field=sw, sw_layer=0)
    h3 = CF._predict_homogeneous(m3, f3, P["h0"], 5, recharge_field=P["rf"], E=P["E"],
                                 ground_elev=P["ge"], sw_field=sw, sw_layer=0,
                                 changhua_cells=torch.as_tensor(ch))
    assert float((h2 - h3).abs().max()) <= 1e-9


def test_theta_file_form_is_rebuilt_by_forward(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyproj")
    from hydrophysics.twin.forward import Member, build_model, check_supported
    from hydrophysics.twin.zone_county import county_meta, save_county_npz

    g = _grid()
    P = _problem(g)
    q0 = _parent(g, P)
    ch = _changhua(g)
    path = str(tmp_path / "zone_county_map.npz")
    save_county_npz(path, g, ch, polygon="synthetic", gpkg="synthetic.gpkg")
    rest = CF.q0_rest_fractions(q0["pump_split_logit"], 1)
    _, f = _fit(CF, g, P, epochs=1, county_copy_sd=0.5, freeze_from=q0,
                pump_layer_shares_county=(rest, np.asarray(X3)))
    meta = {"param_mode": "zonal", "zone_boundaries": "205,201", "pump_layer": 1,
            **county_meta(path, ch), "pump_split": True, "return_flow": True,
            "sw_layer": 0}
    check_supported(f["theta"], meta)                       # no key forward cannot rebuild
    _, scalars, _ = build_model(g, Member("chj", json.loads(json.dumps(f["theta"])), meta),
                                "cpu")
    want = CF.county_layer_fractions(ch, rest, X3)
    assert np.abs(scalars["pump_layer_frac"].numpy() - want).max() <= 1e-15


# ----------------------------------------------------------------------------- (d) refusals
@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_refusals_in_fit_flow():
    g = _grid()
    P = _problem(g)
    q0 = _parent(g, P)
    rest = CF.q0_rest_fractions(q0["pump_split_logit"], 1)
    sh = (rest, np.asarray(X3))
    with pytest.raises(ValueError, match="cannot be combined"):
        _fit(CF, g, P, epochs=0, pump_split=False, pump_layer_fractions_on=True,
             pump_layer_shares_county=sh)
    with pytest.raises(ValueError, match="cannot be combined"):
        _fit(CF, g, P, epochs=0, pump_split=False, pump_layer_fractions_on=True,
             fix_pump_layer_shares=X3, pump_layer_shares_county=sh)
    with pytest.raises(ValueError, match="county flags"):
        _fit(CF, g, P, epochs=0, changhua_cells=None, pump_layer_shares_county=sh)
    with pytest.raises(ValueError, match="county-split"):
        _fit(CF, g, P, epochs=0, county=False, freeze_from=q0)
    with pytest.raises(ValueError, match="sw-county-split"):
        _fit(CF, g, P, epochs=0, sw=False, sw_field=P["sw"][:1], sw_layer=0,
             freeze_from=q0)
    with pytest.raises(ValueError, match="sw-county-split"):
        _fit(CF, g, P, epochs=0, fix_sw_scale=0.2, freeze_from=q0)
    with pytest.raises(KeyError, match="lacks pinned"):
        _fit(CF, g, P, epochs=0, freeze_from={k: v for k, v in q0.items()
                                               if k != "return_frac_logit"})


def test_refusals_on_the_cli(tmp_path):
    p = tmp_path / "parent.json"
    p.write_text(json.dumps({"theta": {"pump_split_logit": -2.0}, "meta": {"pump_layer": 1}}))
    x3 = ",".join(str(v) for v in X3)
    with pytest.raises(SystemExit, match="pass --freeze-from"):
        CF.main(["--pump-layer-shares-county", x3])
    with pytest.raises(SystemExit, match="zone-county-split"):
        CF.main(["--freeze-from", str(p), "--fit-only"])
    base = ["--param-mode", "zonal", "--zone-county-split", "--freeze-from", str(p),
            "--fit-only"]
    for extra in (["--pump-layer-fractions"], ["--pump-layer-shares", x3],
                  ["--pump-layer-fractions", "--pump-fractions-county"]):
        with pytest.raises(SystemExit, match="excludes"):
            CF.main(base + ["--pump-layer-shares-county", x3] + extra)
    with pytest.raises(SystemExit, match="sw-county-split"):
        CF.main(base + ["--sw-recharge", "sw.npz"])
    with pytest.raises(SystemExit, match="fix-sw-scale"):
        CF.main(base + ["--fix-sw-scale", "0.2"])
    with pytest.raises(SystemExit, match="positive"):
        CF.main(base + ["--pump-layer-shares-county", "0.9,0.1,0,0"])
    with pytest.raises(SystemExit, match="differs from the parent"):
        CF.main(base + ["--pump-layer-shares-county", x3, "--pump-layer", "2"])
    with pytest.raises(SystemExit, match="no 'theta'"):
        q = tmp_path / "bad.json"
        q.write_text("{}")
        CF.main(["--param-mode", "zonal", "--zone-county-split", "--freeze-from", str(q),
                 "--fit-only"])


def _fold_parent(tmp_path, k, seed=2027, n_folds=5, name=None):
    p = tmp_path / (name or f"q0_s{seed}_fold{k}_theta.json")
    p.write_text(json.dumps({"theta": {"pump_split_logit": -2.0},
                             "meta": {"pump_layer": 1, CF.CHJ_FOLD_PARENT_KEY: {
                                 "fold": k, "seed": seed, "n_folds": n_folds,
                                 "source": "q0/stage3_fold{k}.json", "source_sha256": "x"}}}))
    return p


def test_cli_refuses_a_sequential_kfold_and_a_parent_of_another_fold(tmp_path):
    """(finding 2) --freeze-from with neither --fit-only nor --only-fold would run every
    fold from the full-data parent: refused. --only-fold K needs Q0's fold-K parent of the
    same partition; --fit-only refuses a fold parent."""
    full = tmp_path / "q0_theta.json"
    full.write_text(json.dumps({"theta": {"pump_split_logit": -2.0},
                                "meta": {"pump_layer": 1}}))
    cs = ["--param-mode", "zonal", "--zone-county-split", "--n-folds", "5", "--seed", "2027"]
    with pytest.raises(SystemExit, match="never as a sequential k-fold"):
        CF.main(cs + ["--freeze-from", str(full)])
    with pytest.raises(SystemExit, match="never as a sequential k-fold"):
        CF.main(cs + ["--freeze-from", str(_fold_parent(tmp_path, 2))])
    with pytest.raises(SystemExit, match="own fold theta of this partition"):
        CF.main(cs + ["--only-fold", "2", "--freeze-from", str(full)])
    for bad in (_fold_parent(tmp_path, 1, name="a.json"),
                _fold_parent(tmp_path, 2, seed=2029, name="b.json"),
                _fold_parent(tmp_path, 2, n_folds=10, name="c.json")):
        with pytest.raises(SystemExit, match="own fold theta of this partition"):
            CF.main(cs + ["--only-fold", "2", "--freeze-from", str(bad)])
    # the Python API refuses the same leak
    g = _grid()
    P = _problem(g)
    with pytest.raises(ValueError, match="freeze_from needs only_fold"):
        CF.kfold_wells(g, P["obs"], P["idx"], P["lay"], P["rch"], n_layers=4, epochs=0,
                       n_folds=2, freeze_from={"pump_split_logit": -2.0})
    with pytest.raises(SystemExit, match="a fold parent"):
        CF.main(cs + ["--fit-only", "--freeze-from", str(_fold_parent(tmp_path, 2))])
    # the matching parent passes the hook's checks (main then stops at the missing data)
    ns = _parse_defaults(cs + ["--only-fold", "2", "--freeze-from",
                               str(_fold_parent(tmp_path, 2))])
    rec = CF._chj_check_args(ns)
    assert rec["fold_parent"]["fold"] == 2 and rec["sha256"] == CF._file_sha256(
        str(_fold_parent(tmp_path, 2)))


@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_fit_only_and_fold_jobs_with_their_own_parents_merge(tmp_path):
    """(finding 1) the fit-only job (parent Q0's full theta) and each --only-fold K job
    (parent Q0's fold-K theta) build cfgs merge_folds accepts: the parent path is
    provenance (RECIPE_FREE_ARGS, recorded with its sha256 in the fold file), while the
    hook flag and the shares stay in the compared cfg; a fold run without the hook is
    refused. The same parser (main's) and the same comparison (merge_folds.merge)."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyproj")
    from test_twin_kfold_parallel import SIDS, _case, _full_fit, _seed_dir

    from hydrophysics.twin.merge_folds import merge

    g, obs, idx, lay, rch, rf, kw = _case("off")
    ins, _ = _full_fit(g, obs, idx, lay, rch, rf, "off")
    fit = CF._flow_fit_cols(ins, None, None)
    full = tmp_path / "q0_theta.json"
    full.write_text(json.dumps({"theta": {"pump_split_logit": -2.0},
                                "meta": {"pump_layer": 1}}))
    x3 = ",".join(map(str, X3))
    hook = Q0_ARGV + COUNTY_ARGV + ["--n-folds", "2", "--pump-layer-shares-county", x3]
    cargs = (g.n_active, 205.0, 182.0, 208.0, {"mid": 3}, None, "abc1234")
    run = tmp_path / "run"
    _seed_dir(run, ins, None)
    ns = _parse_defaults(hook + ["--fit-only", "--freeze-from", str(full), "--out", str(run)])
    cfg_fit = CF._flow_cfg(ns, *cargs)
    assert cfg_fit["chj_hook"] == "freeze_from" and "freeze_from" not in cfg_fit["recipe"]
    assert cfg_fit["recipe"]["pump_layer_shares_county"] == x3
    CF.write_fit_summary(str(run), cfg_fit, fit, (0, 0.0), 1.0, SIDS)
    parents = {}
    for k in (0, 1):
        par = _fold_parent(tmp_path, k, n_folds=2)
        ns_k = _parse_defaults(hook + ["--only-fold", str(k), "--freeze-from", str(par),
                                       "--out", str(run), "--dump-predictions"])
        chj = CF._chj_check_args(ns_k)
        parents[k] = chj["sha256"]
        gk = CF.kfold_wells(g, obs, idx, lay, rch, only_fold=k, **kw)
        CF.write_fold_file(str(run), gk, CF._flow_cfg(ns_k, *cargs), (0, 0.0), 1.0, SIDS,
                           extra=CF._chj_fold_record(chj, gk["per_fold"][0]))
    assert merge(str(run), log=lambda *_: None)["n_folds"] == 2
    for k in (0, 1):
        rec = json.loads((run / f"stage3_fold{k}.json").read_text())
        assert rec["chj_parent"]["sha256"] == parents[k]
        assert rec["chj_parent"][CF.CHJ_FOLD_PARENT_KEY]["fold"] == k
        assert rec["chj_hook"]["posterior_tools_unsupported"] is True
    # a fold run WITHOUT the hook does not merge with the hooked full fit
    ns_plain = _parse_defaults(Q0_ARGV + COUNTY_ARGV + ["--n-folds", "2", "--only-fold", "1",
                                                         "--out", str(run)])
    g1 = CF.kfold_wells(g, obs, idx, lay, rch, only_fold=1, **kw)
    CF.write_fold_file(str(run), g1, CF._flow_cfg(ns_plain, *cargs), (0, 0.0), 1.0, SIDS)
    with pytest.raises(ValueError, match=r"different configuration.*chj_hook"):
        merge(str(run), log=lambda *_: None)


def test_main_passes_the_hook_to_the_fit_and_every_fold():
    """main cannot run here (it reads the head data); its wiring is checked on the source:
    the options reach the full fit and the k-fold arguments, the county flags reach the
    fit (the shares need them), and the meta records the parent."""
    import inspect

    src = inspect.getsource(CF.main)
    assert src.count("**chj_kw") == 2
    assert "chj_kw = _chj_fit_options(chj_doc)" in src
    assert "args.pump_layer_shares_county):" in src          # r4_kw carries changhua_cells
    assert "**_chj_meta(chj_doc, ins)" in src
    assert "_chj_fold_record(chj_doc, gate[\"per_fold\"][0])" in src     # fold provenance
    kf = inspect.signature(CF.kfold_wells).parameters
    assert kf["freeze_from"].default is None and kf["pump_layer_shares_county"].default is None
    assert "freeze_from=freeze_from" in inspect.getsource(CF.kfold_wells)
