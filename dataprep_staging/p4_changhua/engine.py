"""P4.0 linear-response engine: a sparse-LU replica of the calibrated twin (plan WP P4.0).

Port of the physics scout's code (pilot ``_pilots/physics_inversion_hooks``: ``b_operator.py``,
``c_greens.py``, ``inv_lib.py``) with the additions P4.0 asks for. CPU only, numpy/scipy; the
repository's torch model is read (never edited) to take the calibrated fields from it and to
verify against it.

What the twin is, in the deliverable recipe (``stage3_datum_gate_cwa``): backward Euler,
monthly steps, ``M h_t = D h_{t-1} + q_t + b_bnd`` with ``M = D + K + leakage + GHB diagonal``,
``D = S area / dt``. ``M`` is symmetric positive definite and time-invariant (no rivers, delay
bed or aquitard store), so heads are exactly linear in the volumetric sources ``q``. The only
non-linearity is the energy-to-volume conversion: ``rate = eta E 3.6e6 / (rho g max(z_g - h_1
+ head_extra, 2 m)) / dt`` uses the previous step's layer-1 head (lift feedback).

Contents

* :func:`assemble_operator` - ``M`` as ``scipy.sparse`` from a ``FlowModel``, with cell-masked
  multipliers on T, S and the leakance (:func:`region_multiplier`), verified against the
  model's own matvec (:func:`verify_operator`).
* :class:`SparseTwin` - LU factorisation and a stepwise rollout with lift feedback, extra
  (added) abstraction that feeds back through the same lift, direct volumetric sources,
  canal recharge, and MODFLOW-RIV switching (operator re-factorised per connection state,
  cached). :func:`verify_against_torch` compares it with ``forward.rollout`` on CPU.
* :func:`adjoint_greens` - exact Green's functions by one adjoint rollout per observation
  well (``M`` symmetric), all layers; :func:`pump_kernel` combines them with the pump split
  and return flow; :func:`steady_gain` gives the infinite-lag sum (kernel truncation error).
* :func:`convolve` - the convolution engine (per-well FFT), valid only where the response is
  linear: fixed lift, no RIV switching. Use the stepwise rollout otherwise.
* Datum - :func:`profile_datum`, :func:`fit_flow_level_loss` replicate ``fit_flow``'s per-well
  datum prior (sd 5 m, 12 months per independent observation); :func:`solve_with_datum` and
  :func:`fisher_with_datum` use the same convention in linear inversions (free datum = sd inf).
* Forcing uncertainty (plan 4.8) - ``Var h = K Sigma_q K^T``: :func:`head_var_independent`
  (independent cell-month noise), :func:`factor_responses` + :func:`head_var_factors`
  (low-rank / separable covariances), :func:`separable_factors`,
  :func:`energy_rate_jacobian` (kWh -> m3/day linearised at the base head path).
* Bound-pinned parameters - :func:`bound_pinned` lists entries of a theta at their bounds;
  :func:`perturb_theta`, :func:`theta_vector` and :func:`profile_loss` (in ``build_engine``)
  profile them instead of differentiating at the bound.
"""
from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if __package__ in (None, ""):
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hydrophysics.twin import pumping as _pm  # noqa: E402  (read-only import)

RHO_G = float(_pm.RHO_G)
J_PER_KWH = float(_pm.J_PER_KWH)
MIN_LIFT_M = float(_pm.MIN_LIFT_M)
RETURN_FRAC_MAX = 0.7          # calibrate_flow.RETURN_FRAC_MAX (checked in build_engine)
WELL_DATUM_SD = 5.0            # calibrate_flow.WELL_DATUM_SD_DEFAULT
WELL_DATUM_MONTHS_PER_OBS = 12.0
PERMC = "MMD_AT_PLUS_A"


def _np(x) -> np.ndarray:
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


# ============================================================================ operator
def region_multiplier(cells: np.ndarray, factor: float, n_rows: int,
                      rows: tuple[int, ...] | None = None) -> np.ndarray:
    """``(n_rows, A)`` multiplier: ``factor`` on the masked cells (bool ``(A,)``) of the given
    rows (layers, or layer interfaces for the leakance), 1 elsewhere."""
    cells = np.asarray(cells, bool)
    m = np.ones((n_rows, cells.size))
    for r in (range(n_rows) if rows is None else rows):
        m[r, cells] = float(factor)
    return m


def _mult(mult, shape: tuple[int, int], name: str) -> np.ndarray:
    if mult is None:
        return np.ones(shape)
    m = np.asarray(mult, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    m = np.array(np.broadcast_to(m, shape), dtype=np.float64)
    if not np.all(np.isfinite(m)) or np.any(m <= 0):
        raise ValueError(f"{name} multiplier must be finite and > 0")
    return m


@dataclass
class Operator:
    """The backward-Euler operator and its pieces (all float64, layer-major flattening)."""

    M: sp.csc_matrix          # (L*A, L*A)
    stor: np.ndarray          # (L, A) S area / dt
    bdiag: np.ndarray         # (L, A) GHB diagonal
    T: np.ndarray             # (L, A) m2/day, multipliers applied
    S: np.ndarray             # (L, A)
    Lk: np.ndarray            # (L-1, A) leakance x area, m2/day
    ia: np.ndarray
    ib: np.ndarray
    area: float
    dt: float

    @property
    def n_layers(self) -> int:
        return self.T.shape[0]

    @property
    def n_active(self) -> int:
        return self.T.shape[1]


def _build_M(T, stor_b, Lk, ia, ib) -> sp.csc_matrix:
    L, A = T.shape
    rows, cols, vals = [], [], []
    diag = stor_b.copy()
    for k in range(L):
        Tf = 2.0 * T[k, ia] * T[k, ib] / np.maximum(T[k, ia] + T[k, ib], 1e-30)
        np.add.at(diag[k], ia, Tf)
        np.add.at(diag[k], ib, Tf)
        off = k * A
        rows += [off + ia, off + ib]
        cols += [off + ib, off + ia]
        vals += [-Tf, -Tf]
    i = np.arange(A)
    for k in range(L - 1):
        diag[k] += Lk[k]
        diag[k + 1] += Lk[k]
        rows += [k * A + i, (k + 1) * A + i]
        cols += [(k + 1) * A + i, k * A + i]
        vals += [-Lk[k], -Lk[k]]
    rows.append(np.arange(L * A))
    cols.append(np.arange(L * A))
    vals.append(diag.reshape(-1))
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(L * A, L * A))


def assemble_operator(model, *, T_mult=None, S_mult=None, L_mult=None) -> Operator:
    """``M`` of a calibrated ``FlowModel`` (``forward.build_model``) as a sparse matrix.

    ``T_mult``/``S_mult`` (``(L, A)`` or ``(A,)``) multiply transmissivity and storativity
    cell by cell; ``L_mult`` (``(L-1, A)`` or ``(A,)``) multiplies the leakance. A masked
    region multiplier (E3's region-masked {log T, log Lk}) is :func:`region_multiplier`."""
    import torch

    L, A = model.n_layers, model.grid.n_active
    with torch.no_grad():
        T = np.exp(_np(model.log_T)) * _mult(T_mult, (L, A), "T")
        S = np.exp(_np(model.log_S)) * _mult(S_mult, (L, A), "S")
        lam = np.exp(_np(model.log_L)) * _mult(L_mult, (L - 1, A), "L")
        C_coast, C_apex = model._C_from_params()
        bdiag, _ = model.boundary_terms(C_coast, C_apex)
    area, dt = float(model.area), float(model.dt)
    stor = S * area / dt
    bdiag = _np(bdiag)
    ia, ib = model.ia.cpu().numpy(), model.ib.cpu().numpy()
    Lk = lam * area
    M = _build_M(T, stor + bdiag, Lk, ia, ib)
    return Operator(M=M, stor=stor, bdiag=bdiag, T=T, S=S, Lk=Lk, ia=ia, ib=ib, area=area,
                    dt=dt)


def verify_operator(op: Operator, model, *, T_mult=None, S_mult=None, L_mult=None,
                    seed: int = 0) -> dict:
    """Relative max |M x - mv(x)| of the sparse operator against ``FlowModel._matvec_from``
    (with the same multipliers applied to the torch fields), the symmetry defect, and the
    diagonal check."""
    import torch

    L, A = op.n_layers, op.n_active
    with torch.no_grad():
        T = torch.exp(model.log_T) * torch.as_tensor(_mult(T_mult, (L, A), "T"))
        S = torch.exp(model.log_S) * torch.as_tensor(_mult(S_mult, (L, A), "S"))
        lam = torch.exp(model.log_L) * torch.as_tensor(_mult(L_mult, (L - 1, A), "L"))
        mv, diag_t = model._matvec_from(T, S, lam, bdiag=torch.as_tensor(op.bdiag))
        x = np.random.default_rng(seed).standard_normal((L, A))
        ref = mv(torch.as_tensor(x)).numpy().reshape(-1)
    got = op.M @ x.reshape(-1)
    return {"rel_max_matvec": float(np.abs(got - ref).max() / np.abs(ref).max()),
            "max_asymmetry": float(abs(op.M - op.M.T).max()),
            "rel_max_diag": float(np.abs(op.M.diagonal() - diag_t.numpy().reshape(-1)).max()
                                  / np.abs(diag_t.numpy()).max()),
            "nnz": int(op.M.nnz), "unknowns": int(L * A)}


# ============================================================================ sparse twin
class SparseTwin:
    """The calibrated twin as a sparse direct solver.

    ``model, scalars`` come from ``hydrophysics.twin.forward.build_model`` (read-only), so the
    fields, boundaries, rivers, spread matrix and driver scalars are exactly the ones the
    torch rollout uses. Models with a delay bed or an aquitard store (extra state) are
    refused. ``T_mult``/``S_mult``/``L_mult`` perturb the operator (see
    :func:`assemble_operator`); the torch model is not modified.
    """

    def __init__(self, model, scalars: dict, ground_elev, *, pump_layer: int = 1,
                 recharge_layer: int = 0, T_mult=None, S_mult=None, L_mult=None,
                 permc_spec: str = PERMC):
        if "delay_Sd" in scalars or "aqt_Sa" in scalars:
            raise NotImplementedError("delay-bed / aquitard-store models carry extra state; "
                                      "the sparse twin does not replicate them")
        self.model = model
        self.op = assemble_operator(model, T_mult=T_mult, S_mult=S_mult, L_mult=L_mult)
        self.mults = {"T": T_mult, "S": S_mult, "L": L_mult}
        self.ge = np.asarray(_np(ground_elev), dtype=np.float64).reshape(-1)
        self.pump_layer = int(pump_layer)
        self.recharge_layer = int(recharge_layer)
        self.permc_spec = permc_spec
        le = _np(scalars.get("log_eta"))
        if le is not None and np.size(le) != 1:
            raise NotImplementedError("per-class efficiencies (eta_classes) are not replicated")
        self.eta = None if le is None else math.exp(float(np.ravel(le)[0]))
        lh = scalars.get("log_head_extra")
        self.hx = None if lh is None else math.exp(float(_np(lh)))
        ps = scalars.get("pump_split_logit")
        self.f_sh = None if ps is None else _sigmoid(float(_np(ps)))
        rr = scalars.get("return_frac_logit")
        self.r_ret = None if rr is None else RETURN_FRAC_MAX * _sigmoid(float(_np(rr)))
        rf = scalars.get("recharge_frac_logit")
        self.rfrac = None if rf is None else _sigmoid(float(_np(rf)))
        W = scalars.get("spread_W")
        self.W = None if W is None else _np(W)
        sws = scalars.get("log_sw_scale")
        self.sw_scale = None if sws is None else np.exp(np.ravel(_np(sws)))
        self.sw_layer = int(scalars.get("sw_layer", self.recharge_layer) or 0)
        self.has_rivers = bool(getattr(model, "has_rivers", False))
        if self.has_rivers:
            self.C_riv = np.exp(np.ravel(_np(scalars["log_C_riv"])))
            self.riv_idx = model.riv_idx.cpu().numpy()
            self.riv_w = _np(model.riv_w)
            self.riv_group = model.riv_group.cpu().numpy()
            self.riv_h = _np(model.riv_h)
            self.riv_rbot = _np(model.riv_rbot)
            self.riv_layer = int(model.river_layer)
            self.riv_mode = str(model.river_mode)
            self.riv_season = _np(model.river_season)
        self._lu: dict = {}
        self.n_factorisations = 0

    # ------------------------------------------------------------------ basics
    @property
    def n_layers(self) -> int:
        return self.op.n_layers

    @property
    def n_active(self) -> int:
        return self.op.n_active

    def _factor(self, M: sp.csc_matrix):
        self.n_factorisations += 1
        return spla.splu(M.tocsc(), permc_spec=self.permc_spec)

    def lu(self):
        """LU of the base operator (no river terms)."""
        if "base" not in self._lu:
            self._lu["base"] = self._factor(self.op.M)
        return self._lu["base"]

    def boundary_rhs(self, apex_from) -> np.ndarray:
        """GHB right-hand side ``(L, A)`` with the apex held at ``apex_from[:, apex]`` (the
        torch rollout pins the apex to its own ``h0``, ``set_apex_heads``)."""
        import torch

        m = self.model
        if not m.has_boundaries:
            return np.zeros((self.n_layers, self.n_active))
        with torch.no_grad():
            m.set_apex_heads(torch.as_tensor(np.asarray(apex_from, dtype=np.float64)))
            _, brhs = m.boundary_terms(*m._C_from_params())
        return _np(brhs)

    def spread(self, E: np.ndarray) -> np.ndarray:
        E = np.asarray(E, dtype=np.float64)
        return E if self.W is None else self.W @ E

    def rate_factor(self, h_pump: np.ndarray) -> np.ndarray:
        """m3/day of abstraction per kWh/month of (spread) energy at pump-layer head
        ``h_pump`` ``(A,)``: the lift-dependent conversion of ``pumping.energy_to_volume``."""
        head = self.ge - np.asarray(h_pump, dtype=np.float64)
        if self.hx is not None:
            head = head + self.hx
        return self.eta * J_PER_KWH / (RHO_G * np.maximum(head, MIN_LIFT_M)) / self.op.dt

    def layer_sources(self, rate: np.ndarray) -> np.ndarray:
        """``(L, A)`` volumetric sources (m3/day, + = into the aquifer) of an abstraction
        ``rate`` ``(A,)``: pump split between layer 0 and the pump layer, and return flow."""
        q = np.zeros((self.n_layers, self.n_active))
        if self.f_sh is not None and self.pump_layer != 0:
            q[0] -= self.f_sh * rate
            q[self.pump_layer] -= (1.0 - self.f_sh) * rate
        else:
            q[self.pump_layer] -= rate
        if self.r_ret is not None:
            q[0] += self.r_ret * rate
        return q

    def pump_weights(self) -> dict[int, float]:
        """Source weight per layer of a unit abstraction (``layer_sources`` of 1)."""
        q = self.layer_sources(np.ones(1 if self.n_active == 0 else self.n_active))
        return {k: float(q[k, 0]) for k in range(self.n_layers) if q[k, 0] != 0.0}

    # ------------------------------------------------------------------ rivers
    def river_terms(self, mask: np.ndarray | None, month: int | None):
        L, A = self.n_layers, self.n_active
        c = self.C_riv[self.riv_group] * self.riv_w
        if self.riv_season is not None and month is not None:
            c = c * self.riv_season[self.riv_group, int(month) % 12]
        conn = np.ones_like(c) if mask is None else mask.astype(np.float64)
        d = np.zeros((L, A))
        r = np.zeros((L, A))
        np.add.at(d[self.riv_layer], self.riv_idx, c * conn)
        np.add.at(r[self.riv_layer], self.riv_idx,
                  c * conn * self.riv_h + c * (1.0 - conn) * (self.riv_h - self.riv_rbot))
        return d, r

    def _river_lu(self, mask, month):
        key = ("riv", None if mask is None else mask.tobytes(),
               (int(month) % 12) if self.riv_season is not None else None)
        if key not in self._lu:
            d, _ = self.river_terms(mask, month)
            self._lu[key] = self._factor(self.op.M + sp.diags(d.reshape(-1)))
        return self._lu[key]

    # ------------------------------------------------------------------ rollouts
    def rollout(self, h0, *, E=None, R=None, n_steps: int | None = None, E_extra=None,
                spread_extra: bool = True, q_extra=None, sw=None, month0: int = 1,
                apex_from=None, frozen_head=None, return_rates: bool = False):
        """Stepwise backward-Euler rollout -> heads ``(L, A, T+1)`` (and rates).

        ``E`` ``(A, T)`` kWh per month (spread by the calibrated kernel, then converted with
        the lift from the previous step's pump-layer head); ``R`` ``(A, T)`` m/day recharge
        field (times the recharge fraction, into ``recharge_layer``); ``E_extra`` an added
        abstraction's energy, converted with the same evolving lift (lift feedback on the
        added abstraction; spread too unless ``spread_extra=False``); ``q_extra``
        ``(L, A, T)`` direct sources in m3/day (+ = injection); ``sw`` ``(A, T)`` or
        ``(K, A, T)`` canal deliveries (m/day) times ``exp(log_sw_scale)``. ``frozen_head``
        ``(A, T)`` replaces the evolving pump-layer head in the lift (no feedback: the
        response is then linear in the energy). Rivers in ``riv`` mode switch per step on
        the previous head; each connection state is factorised once and cached.
        ``return_rates`` also returns the abstraction rate ``(A, T)`` m3/day."""
        h0 = np.asarray(h0, dtype=np.float64)
        L, A = self.n_layers, self.n_active
        if n_steps is None:
            for x in (E, R, E_extra, sw):
                if x is not None:
                    n_steps = np.shape(x)[-1]
                    break
            else:
                n_steps = np.shape(q_extra)[-1]
        Es = None
        if E is not None:
            Es = self.spread(E)
        if E_extra is not None:
            Ex = self.spread(E_extra) if spread_extra else np.asarray(E_extra, np.float64)
            Es = Ex if Es is None else Es + Ex
        if Es is not None and self.eta is None:
            raise ValueError("energy forcing needs log_eta in the scalars")
        if R is not None and self.rfrac is None:
            raise ValueError("a recharge field needs recharge_frac_logit in the scalars")
        if sw is not None and self.sw_scale is None:
            raise ValueError("canal recharge needs log_sw_scale in the scalars")
        brhs = self.boundary_rhs(h0 if apex_from is None else apex_from)
        D = self.op.stor
        area = self.op.area
        h = h0.copy()
        out = [h0.copy()]
        rates = np.zeros((A, n_steps)) if return_rates else None
        lu = None if self.has_rivers else self.lu()
        for t in range(n_steps):
            q = np.zeros((L, A))
            if R is not None:
                q[self.recharge_layer] += self.rfrac * R[:, t] * area
            if Es is not None:
                hp = h[self.pump_layer] if frozen_head is None else frozen_head[:, t]
                rate = self.rate_factor(hp) * Es[:, t]
                q += self.layer_sources(rate)
                if return_rates:
                    rates[:, t] = rate
            if sw is not None:
                sw = np.asarray(sw)
                s_t = ((self.sw_scale[:, None] * sw[:, :, t]).sum(axis=0) if sw.ndim == 3
                       else self.sw_scale[0] * sw[:, t])
                q[self.sw_layer] += s_t * area
            if q_extra is not None:
                q += q_extra[..., t]
            b = D * h + q + brhs
            if self.has_rivers:
                mask = (h[self.riv_layer, self.riv_idx] > self.riv_rbot
                        if self.riv_mode == "riv" else None)
                _, rr = self.river_terms(mask, month0 + t)
                b = b + rr
                lu = self._river_lu(mask, month0 + t)
            h = lu.solve(b.reshape(-1)).reshape(L, A)
            out.append(h)
        H = np.stack(out, axis=-1)
        return (H, rates) if return_rates else H

    def rollout_linear(self, q: np.ndarray, h0=None) -> np.ndarray:
        """Pure linear response ``h_t = M^-1 (D h_{t-1} + q_t)`` from ``h0`` (default 0) with
        no boundary right-hand side and no rivers: the system the Green's functions solve.
        ``q`` ``(L, A, T)`` m3/day (+ = injection) -> ``(L, A, T+1)``."""
        L, A = self.n_layers, self.n_active
        lu = self.lu()
        D = self.op.stor
        h = np.zeros((L, A)) if h0 is None else np.asarray(h0, dtype=np.float64).copy()
        out = [h.copy()]
        for t in range(q.shape[-1]):
            h = lu.solve((D * h + q[..., t]).reshape(-1)).reshape(L, A)
            out.append(h)
        return np.stack(out, axis=-1)


def verify_against_torch(twin: SparseTwin, scalars: dict, h0, E, R, *, E_extra=None,
                         sw=None, month0: int | None = None, H_sparse=None) -> dict:
    """Run the repository's torch rollout (``forward.rollout``, CPU) on the same forcing and
    compare heads with the sparse rollout: ``max |dh|`` and rms in metres.

    ``E_extra`` is added to ``E`` for torch (the twin spreads and converts the sum), which
    is what the sparse rollout's ``E_extra`` does with ``spread_extra=True``."""
    import time

    import torch

    from hydrophysics.twin.forward import rollout

    E_t = np.asarray(E, np.float64) + (0.0 if E_extra is None else np.asarray(E_extra))
    t0 = time.perf_counter()
    H_t = rollout(twin.model, scalars, torch.as_tensor(np.asarray(h0, np.float64)),
                  torch.as_tensor(E_t), torch.as_tensor(np.asarray(R, np.float64)),
                  torch.as_tensor(twin.ge), pump_layer=twin.pump_layer,
                  recharge_layer=twin.recharge_layer,
                  sw_field=None if sw is None else torch.as_tensor(np.asarray(sw)),
                  month0=month0).numpy()
    t_torch = time.perf_counter() - t0
    t0 = time.perf_counter()
    if H_sparse is None:
        H_sparse = twin.rollout(h0, E=E, R=R, E_extra=E_extra, sw=sw,
                                month0=1 if month0 is None else month0)
    t_sparse = time.perf_counter() - t0
    dev = np.abs(H_t - H_sparse)
    return {"max_abs_dh_m": float(dev.max()), "rms_dh_m": float(np.sqrt((dev ** 2).mean())),
            "argmax": [int(i) for i in np.unravel_index(int(dev.argmax()), dev.shape)],
            "t_torch_s": round(t_torch, 2), "t_sparse_s": round(t_sparse, 3),
            "n_steps": int(np.shape(E)[-1]), "H_torch": H_t}


# ============================================================================ Green's functions
def adjoint_greens(twin: SparseTwin, obs_idx, obs_layer, n_lags: int,
                   layers: tuple[int, ...] = (0, 1, 2, 3), dtype=np.float32,
                   out: dict | None = None) -> dict[int, np.ndarray]:
    """Exact Green's functions of the time-invariant operator by adjoint rollouts.

    ``G[l][w, c, tau]`` (m per m3/day) is the head change at observation well ``w`` (its
    cell and layer) at step ``s + tau`` caused by a source of 1 m3/day during step ``s`` in
    cell ``c`` of layer ``l``. With ``M`` symmetric and ``D`` diagonal, ``z_0 = M^-1 e_w`` and
    ``z_tau = M^-1 D z_{tau-1}``; then ``G[l][w, :, tau] = z_tau[l]``. ``out`` may hold
    preallocated (memmap) arrays."""
    op = twin.op
    asym = float(abs(op.M - op.M.T).max())
    if asym > 1e-9 * float(abs(op.M).max()):
        raise ValueError(f"operator is not symmetric (max |M - M^T| {asym:.3e})")
    L, A = op.n_layers, op.n_active
    oi, ol = np.asarray(obs_idx), np.asarray(obs_layer)
    nW = oi.size
    G = out if out is not None else {lay: np.zeros((nW, A, n_lags), dtype=dtype)
                                     for lay in layers}
    lu = twin.lu()
    D = op.stor.reshape(-1)
    buf = np.empty((n_lags, L * A))
    for w in range(nW):
        e = np.zeros(L * A)
        e[int(ol[w]) * A + int(oi[w])] = 1.0
        z = lu.solve(e)
        for tau in range(n_lags):
            buf[tau] = z
            z = lu.solve(D * z)
        for lay in layers:
            G[lay][w] = buf[:, lay * A:(lay + 1) * A].T
    return G


def steady_gain(twin: SparseTwin, obs_idx, obs_layer,
                layers: tuple[int, ...] = (0, 1)) -> dict[int, np.ndarray]:
    """Infinite-lag sum ``sum_tau G[l][w, :, tau] = ((M - D)^-1 e_w)[l]`` per well (the steady
    response; needs the GHB boundaries, which make ``M - D`` non-singular)."""
    op = twin.op
    L, A = op.n_layers, op.n_active
    lu = spla.splu((op.M - sp.diags(op.stor.reshape(-1))).tocsc(), permc_spec=PERMC)
    oi, ol = np.asarray(obs_idx), np.asarray(obs_layer)
    out = {lay: np.zeros((oi.size, A)) for lay in layers}
    for w in range(oi.size):
        e = np.zeros(L * A)
        e[int(ol[w]) * A + int(oi[w])] = 1.0
        z = lu.solve(e)
        for lay in layers:
            out[lay][w] = z[lay * A:(lay + 1) * A]
    return out


def pump_kernel(G: dict[int, np.ndarray], weights: dict[int, float], dtype=np.float32,
                chunk: int = 16) -> np.ndarray:
    """Drawdown kernel ``K[w, c, tau]`` (m of drawdown per m3/day abstracted for one step):
    ``K = -sum_l weights[l] G[l]`` with ``weights`` the per-layer source of a unit
    abstraction (:meth:`SparseTwin.pump_weights`: ``-(1 - f_sh)`` in the pump layer,
    ``r_ret - f_sh`` in layer 0). Computed in float64 per chunk of wells."""
    first = G[next(iter(weights))]
    K = np.empty(first.shape, dtype=dtype)
    for s in range(0, first.shape[0], chunk):
        acc = np.zeros(first[s:s + chunk].shape)
        for lay, wgt in weights.items():
            acc -= float(wgt) * G[lay][s:s + chunk].astype(np.float64)
        K[s:s + chunk] = acc
    return K


def forward_impulse(twin: SparseTwin, q0: np.ndarray, n_lags: int, obs_idx, obs_layer
                    ) -> np.ndarray:
    """Head at the wells over ``n_lags`` steps after a one-step source ``q0`` ``(L, A)`` at
    step 0 (linear system, zero initial state) -> ``(W, n_lags)``."""
    L, A = twin.n_layers, twin.n_active
    q = np.zeros((L, A, n_lags))
    q[..., 0] = q0
    H = twin.rollout_linear(q)[..., 1:]
    return H[np.asarray(obs_layer), np.asarray(obs_idx), :]


# ============================================================================ convolution
def convolve(K: np.ndarray, q: np.ndarray, chunk_cells: int | None = None) -> np.ndarray:
    """Convolution engine: ``y[..., w, t] = sum_c sum_{tau<=t} K[w, c, tau] q[..., c, t-tau]``.

    ``K`` ``(W, A, NT)`` (float32 memmap is fine; read one well at a time), ``q`` ``(A, T)``
    or ``(n, A, T)`` -> ``(W, T)`` or ``(n, W, T)``, float64, by FFT along time. Lags beyond
    ``NT`` are truncated (see :func:`steady_gain` for the size of the tail). Valid only for
    a time-invariant linear response: fixed lift, no RIV switching."""
    q = np.asarray(q, dtype=np.float64)
    squeeze = q.ndim == 2
    if squeeze:
        q = q[None]
    nW, A, NT = K.shape
    T = q.shape[-1]
    nfft = 1 << int(math.ceil(math.log2(T + NT)))
    Qf = np.fft.rfft(q, n=nfft, axis=-1)                    # (n, A, F)
    y = np.empty((q.shape[0], nW, T))
    for w in range(nW):
        Kf = np.fft.rfft(np.asarray(K[w], dtype=np.float64), n=nfft, axis=-1)   # (A, F)
        y[:, w, :] = np.fft.irfft(np.einsum("af,naf->nf", Kf, Qf), n=nfft, axis=-1)[:, :T]
    return y[0] if squeeze else y


# ============================================================================ datum prior
def profile_datum(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None,
                  sd: float = WELL_DATUM_SD, m: float = WELL_DATUM_MONTHS_PER_OBS):
    """numpy replica of ``calibrate_flow._profile_well_datum`` -> ``(d, kappa, s2, n)``.

    ``d_i = n_i rbar_i / (n_i + kappa_i)``, ``kappa_i = min(n_i, m) s^2 / sd^2``, ``s^2`` the
    within-well residual variance pooled over the masked cells. ``sd = inf`` is the fully
    free datum (``kappa = 0``). Unobserved cells of ``obs`` may be NaN when masked out."""
    pred = np.asarray(pred, np.float64)
    mf = np.ones_like(pred) if mask is None else np.asarray(mask, bool).astype(np.float64)
    r = np.where(mf > 0, np.asarray(obs, np.float64) - pred, 0.0)
    n = mf.sum(axis=1, keepdims=True)
    rbar = (r * mf).sum(axis=1, keepdims=True) / np.maximum(n, 1.0)
    s2 = float((((r - rbar) ** 2) * mf).sum() / max(mf.sum(), 1.0))
    kappa = (np.zeros_like(n) if not np.isfinite(sd)
             else np.minimum(n, m) * s2 / float(sd) ** 2)
    d = np.where(n > 0, n * rbar / np.maximum(n + kappa, 1e-12), 0.0)
    return d[:, 0], kappa[:, 0], s2, n[:, 0]


def fit_flow_level_loss(pred: np.ndarray, target: np.ndarray, datum_mask: np.ndarray | None,
                        sd: float = WELL_DATUM_SD, loss_mask: np.ndarray | None = None):
    """``fit_flow``'s level loss with the datum profiled (``_loss_datum``, level mode):
    ``MSE(pred + d, target) + sum_i kappa_i d_i^2 / N`` -> ``(loss, d, s2)``. ``target`` is the
    calibration target (back-filled heads, no NaN unless ``loss_mask`` is given)."""
    d, kappa, s2, _ = profile_datum(pred, target, datum_mask, sd)
    res = pred + d[:, None] - target
    if loss_mask is None:
        mse, N = float((res ** 2).mean()), float(res.size)
    else:
        lm = np.asarray(loss_mask, bool)
        N = float(lm.sum())
        mse = float((np.where(lm, res, 0.0) ** 2).sum() / max(N, 1.0))
    return mse + float((kappa * d ** 2).sum()) / N, d, s2


def _datum_transform(mask: np.ndarray, kappa: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Per well: observed rows and ``gamma_w = 1 - sqrt(kappa / (n + kappa))``. Subtracting
    ``gamma_w`` times the well mean from every observed row turns ``sum_t (u - d)^2 +
    kappa d^2`` minimised over ``d`` into a plain sum of squares (exact profiling)."""
    out = []
    for w in range(mask.shape[0]):
        rows = np.flatnonzero(mask[w])
        n = rows.size
        if n == 0:
            out.append((rows, 0.0))
            continue
        k = float(kappa[w])
        out.append((rows, 1.0 - math.sqrt(k / (n + k)) if np.isfinite(k) else 0.0))
    return out


def _profiled_rows(X: np.ndarray, r: np.ndarray, tr) -> tuple[np.ndarray, np.ndarray]:
    Xs, rs = [], []
    for w, (rows, g) in enumerate(tr):
        if rows.size == 0:
            continue
        xw, rw = X[w, rows], r[w, rows]
        Xs.append(xw - g * xw.mean(axis=0, keepdims=True))
        rs.append(rw - g * rw.mean())
    return np.concatenate(Xs, axis=0), np.concatenate(rs)


def solve_with_datum(X: np.ndarray, resid: np.ndarray, mask: np.ndarray | None = None, *,
                     sd: float = WELL_DATUM_SD, m: float = WELL_DATUM_MONTHS_PER_OBS,
                     prior=None, prior_sd=None, lower=None, upper=None,
                     max_iter: int = 50, tol: float = 1e-12) -> dict:
    """Linear inversion ``resid ~ X theta + d_w`` with ``fit_flow``'s datum prior.

    ``X`` ``(W, T, P)`` head response per unit theta (drawdown responses enter with a minus
    sign), ``resid`` ``(W, T)`` observed minus base heads (NaN unobserved), ``mask`` the
    months that enter (default: finite ``resid``). Objective, in ``fit_flow``'s own units:
    ``sum_obs (resid - X theta - d)^2 + sum_w kappa_w d_w^2 + s^2 sum_p ((theta_p -
    prior_p) / prior_sd_p)^2`` with ``kappa_w = min(n_w, m) s^2 / sd^2`` and ``s^2`` the
    within-well residual variance at the solution (fixed-point iteration, as ``fit_flow``
    evaluates it at the current prediction). ``sd = inf`` is the free datum. Optional box
    bounds on theta (``lower``/``upper``) use ``scipy.optimize.lsq_linear``."""
    from scipy.optimize import lsq_linear

    X = np.asarray(X, np.float64)
    resid = np.asarray(resid, np.float64)
    mask = np.isfinite(resid) if mask is None else (np.asarray(mask, bool) & np.isfinite(resid))
    r0 = np.where(mask, resid, 0.0)
    W_, T_, P = X.shape
    n = mask.sum(axis=1)
    theta = np.zeros(P) if prior is None else np.asarray(prior, np.float64).copy()
    s2 = None
    n_it = 0
    for _ in range(max_iter):
        n_it += 1
        u = r0 - X @ theta
        _, _, s2_new, _ = profile_datum(np.zeros_like(u), u, mask, sd, m)
        if s2 is not None and abs(s2_new - s2) <= tol * max(s2, 1e-300):
            s2 = s2_new
            break
        s2 = s2_new
        kappa = (np.zeros(W_) if not np.isfinite(sd) else np.minimum(n, m) * s2 / sd ** 2)
        tr = _datum_transform(mask, kappa)
        A_, b_ = _profiled_rows(X, r0, tr)
        if prior is not None:
            psd = np.asarray(prior_sd, np.float64)
            A_ = np.vstack([A_, np.diag(math.sqrt(s2) / psd)])
            b_ = np.concatenate([b_, math.sqrt(s2) * np.asarray(prior, np.float64) / psd])
        if lower is None and upper is None:
            theta = np.linalg.lstsq(A_, b_, rcond=None)[0]
        else:
            lo = np.full(P, -np.inf) if lower is None else np.asarray(lower, np.float64)
            hi = np.full(P, np.inf) if upper is None else np.asarray(upper, np.float64)
            theta = lsq_linear(A_, b_, bounds=(lo, hi), method="bvls").x
    u = r0 - X @ theta
    d, kappa, s2, n = profile_datum(np.zeros_like(u), u, mask, sd, m)
    return {"theta": theta, "datum": d, "kappa": kappa, "s2": s2, "n": n,
            "iterations": n_it, "fitted": X @ theta + d[:, None]}


def fisher_with_datum(X: np.ndarray, mask: np.ndarray, s2: float, kappa: np.ndarray,
                      prior_sd=None, fixed: np.ndarray | None = None) -> dict:
    """Fisher information of theta after profiling the per-well datum with ``fit_flow``'s
    prior (exact Schur complement, iid months of variance ``s2`` - ``fit_flow``'s own,
    overconfident convention; label results uncalibrated). ``fixed`` (bool ``(P,)``)
    excludes parameters that are held fixed (bound-pinned: profiled, never given a Laplace
    sd); their rows of ``sd`` are NaN."""
    X = np.asarray(X, np.float64)
    P = X.shape[-1]
    keep = np.ones(P, bool) if fixed is None else ~np.asarray(fixed, bool)
    tr = _datum_transform(np.asarray(mask, bool), np.asarray(kappa, np.float64))
    A_, _ = _profiled_rows(X[..., keep], np.zeros(X.shape[:2]), tr)
    F = A_.T @ A_ / s2
    if prior_sd is not None:
        F = F + np.diag(1.0 / np.asarray(prior_sd, np.float64)[keep] ** 2)
    C = np.linalg.inv(F)
    sd_k = np.sqrt(np.diag(C))
    sd = np.full(P, np.nan)
    sd[keep] = sd_k
    return {"F": F, "cov": C, "sd": sd, "corr": C / np.outer(sd_k, sd_k), "keep": keep,
            "cond": float(np.linalg.cond(F))}


# ============================================================================ forcing variance
def energy_rate_jacobian(twin: SparseTwin, H: np.ndarray, pump_layer: int | None = None
                         ) -> np.ndarray:
    """``c[:, t]`` (m3/day per kWh of spread energy) at the head path ``H`` ``(L, A, T+1)``:
    step ``t`` converts with the head at its start, ``H[pump_layer, :, t]``. A kWh
    perturbation ``dE`` ``(A, T)`` then maps to an abstraction perturbation ``dq = c *
    (W @ dE)`` at frozen lift (the feedback of the perturbation on its own lift is
    second order; the stepwise rollout keeps it)."""
    pl = twin.pump_layer if pump_layer is None else pump_layer
    return np.stack([twin.rate_factor(H[pl, :, t]) for t in range(H.shape[-1] - 1)], axis=1)


def head_var_independent(K: np.ndarray, var_q: np.ndarray) -> np.ndarray:
    """``Var h[w, t] = sum_c sum_tau K[w, c, tau]^2 var_q[c, t - tau]`` for independent
    cell-month abstraction errors ``var_q`` ``(A, T)`` ((m3/day)^2) -> ``(W, T)`` m^2."""
    var_q = np.asarray(var_q, np.float64)
    nW, A, NT = K.shape
    T = var_q.shape[-1]
    nfft = 1 << int(math.ceil(math.log2(T + NT)))
    Vf = np.fft.rfft(var_q, n=nfft, axis=-1)
    out = np.empty((nW, T))
    for w in range(nW):
        Kf = np.fft.rfft(np.asarray(K[w], np.float64) ** 2, n=nfft, axis=-1)
        out[w] = np.fft.irfft((Kf * Vf).sum(axis=0), n=nfft)[:T]
    return out


def factor_responses(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Head responses ``U`` ``(J, W, T)`` (drawdown, m) of factor fields ``F`` ``(J, A, T)``
    (m3/day): the abstraction error is ``sum_j F_j xi_j``, ``xi ~ N(0, C)``."""
    return convolve(K, F)


def head_var_factors(U: np.ndarray, C: np.ndarray | None = None) -> np.ndarray:
    """``Var h = diag(U C U^T)`` per well-month -> ``(W, T)``; ``C`` defaults to identity."""
    J = U.shape[0]
    Uf = U.reshape(J, -1)
    if C is None:
        return (Uf ** 2).sum(axis=0).reshape(U.shape[1:])
    return np.einsum("ix,ij,jx->x", Uf, np.asarray(C, np.float64), Uf).reshape(U.shape[1:])


def head_cov_well(U: np.ndarray, w: int, C: np.ndarray | None = None) -> np.ndarray:
    """Full ``(T, T)`` head covariance of well ``w`` from factor responses."""
    Uw = U[:, w, :]
    return Uw.T @ (Uw if C is None else np.asarray(C, np.float64) @ Uw)


def separable_factors(cov_space: np.ndarray, cov_time: np.ndarray,
                      var_frac: float = 0.999) -> np.ndarray:
    """Factor fields ``F`` ``(J, A, T)`` with ``sum_j vec(F_j) vec(F_j)^T`` approximating
    ``cov_space (A, A) kron cov_time (T, T)``: leading eigenpairs of each, products kept
    until ``var_frac`` of the trace is reached."""
    ls, vs = np.linalg.eigh(np.asarray(cov_space, np.float64))
    lt, vt = np.linalg.eigh(np.asarray(cov_time, np.float64))
    ls, lt = np.clip(ls, 0, None), np.clip(lt, 0, None)
    prod = np.outer(ls, lt)
    order = np.argsort(prod.ravel())[::-1]
    tot = prod.sum()
    cum = np.cumsum(prod.ravel()[order])
    J = int(np.searchsorted(cum, var_frac * tot) + 1)
    F = []
    for k in order[:J]:
        i, j = np.unravel_index(k, prod.shape)
        F.append(math.sqrt(prod[i, j]) * np.outer(vs[:, i], vt[:, j]))
    return np.stack(F)


# ============================================================================ bound-pinned
@dataclass
class PinnedEntry:
    name: str
    index: int
    side: str
    value: float
    bound: float
    zone: str = ""
    extra: dict = field(default_factory=dict)


def theta_bounds(meta: dict) -> dict[str, tuple[float, float]]:
    """``calibrate_flow.BOUNDS`` as the run saw them: ``l_min`` raises the leakance floor,
    ``spread_max_km`` sets the spread ceiling (copies; the module globals are untouched)."""
    from hydrophysics.twin import calibrate_flow as cf

    b = {k: tuple(v) for k, v in cf.BOUNDS.items()}
    if meta.get("l_min") is not None:
        b["log_L"] = (math.log(float(meta["l_min"])), b["log_L"][1])
    if meta.get("spread_max_km") is not None:
        b["log_spread_km"] = (b["log_spread_km"][0], math.log(float(meta["spread_max_km"])))
    return b


def zone_lower_bounds(meta: dict) -> dict[tuple[str, str], float]:
    from hydrophysics.twin import calibrate_flow as cf

    out = {}
    v = meta.get("log_t_min_proximal")
    if v is not None:
        lo = max(math.log(float(v)), cf.BOUNDS["log_T"][0])
        out[("log_T", "proximal")] = lo
        out[("log_T", "proximal_w")] = lo
    return out


def _split_name(name: str) -> tuple[str, str]:
    from hydrophysics.twin.zones import ZONE_NAMES_SPLIT

    for z in sorted(ZONE_NAMES_SPLIT, key=len, reverse=True):
        if name.endswith(f"_{z}"):
            return name[: -len(z) - 1], z
    return name, "global"


FREE_KEYS = ("log_T_proximal", "log_S_proximal", "log_T_mid", "log_S_mid", "log_T_distal",
             "log_S_distal", "log_L_mid", "log_L_distal", "log_T_proximal_w",
             "log_S_proximal_w", "log_C_coast", "log_C_apex", "pump_split_logit",
             "return_frac_logit", "log_spread_km", "recharge_frac_logit")


def theta_vector(theta: dict, meta: dict) -> list[dict]:
    """The fitted (free) entries of a zonal theta, one record per scalar entry:
    ``{key, index, name, value, lo, hi}`` (``lo``/``hi`` None for unbounded logits). Fixed
    conversions (``fix_eta``/``fix_head_extra``) are excluded."""
    b = theta_bounds(meta)
    zl = zone_lower_bounds(meta)
    out = []
    for key in FREE_KEYS:
        if key not in theta:
            continue
        if key == "log_eta" and meta.get("fix_eta") is not None:
            continue
        vals = np.ravel(np.asarray(theta[key], dtype=np.float64))
        base, zone = _split_name(key)
        lo, hi = b.get(base, (None, None))
        if (base, zone) in zl:
            lo = zl[(base, zone)]
        for i, v in enumerate(vals):
            out.append({"key": key, "index": i, "name": f"{key}[{i}]", "value": float(v),
                        "lo": lo, "hi": hi, "zone": zone, "base": base})
    return out


def bound_pinned(theta: dict, meta: dict, atol: float = 1e-9) -> list[PinnedEntry]:
    """Entries of the theta sitting on a bound (``calibrate_flow._clamp_``'s 1e-9 test)."""
    out = []
    for e in theta_vector(theta, meta):
        if e["lo"] is not None and e["value"] <= e["lo"] + atol:
            out.append(PinnedEntry(e["key"], e["index"], "lo", e["value"], e["lo"], e["zone"]))
        elif e["hi"] is not None and e["value"] >= e["hi"] - atol:
            out.append(PinnedEntry(e["key"], e["index"], "hi", e["value"], e["hi"], e["zone"]))
    return out


def pinned_counts(pinned: list[PinnedEntry]) -> dict:
    """Pinned entries counted like ``meta['bounds_hit']`` (``{zone: {base: {lo, hi}}}``)."""
    out: dict = {}
    for p in pinned:
        base, zone = _split_name(p.name)
        z = out.setdefault(zone, {}).setdefault(base, {"lo": 0, "hi": 0})
        z[p.side] += 1
    return out


def perturb_theta(theta: dict, key: str, index: int, value: float) -> dict:
    """Copy of ``theta`` with entry ``key[index]`` set to ``value`` (and the derived
    readouts ``build_model`` reads kept consistent: ``spread_km``)."""
    th = copy.deepcopy(theta)
    v = th[key]
    if isinstance(v, list):
        v[index] = float(value)
    else:
        if index != 0:
            raise IndexError(f"{key} is a scalar")
        th[key] = float(value)
    if key == "log_spread_km":
        th["spread_km"] = float(math.exp(float(value)))
    return th
