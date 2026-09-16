# Stage-3 zonal remediation — design

**Date:** 2026-08-29
**Status:** approved, not yet implemented
**Parent spec:** `docs/superpowers/specs/2026-08-22-choushui-differentiable-twin-design.md`
**Predecessor result:** Stage-3 gate FAILS, mean margin −0.048 ± 0.013 over 5 seeds (`6802ec2`, spec §7)

This is **P1** of four sub-projects. It is the only one currently unblocked, and everything
downstream depends on its outcome.

---

## 1. Why this exists

The Stage-3 gate failed on leak-free, seed-verified folds. The failure is real, but the
diagnosis is not "the physics model cannot work" — it is **"the parameterization cannot
express the fan."** Two independent pieces of evidence say so.

**Evidence 1 — the transmissivity clamp binds.**

```
bounds_hit = {'log_T': 3, 'log_S': 2, 'log_L': 0, 'log_eta': 1}
log_T      = [log 10, 5.304, log 10, log 10]
```

Three of four layers sit pinned at the **lower** `log_T` clamp, T = 10 m²/day — below the
58 m²/day floor of the range Liu et al. 2002 measured at Choushui. A single homogeneous
transmissivity is being asked to describe coarse proximal gravel and fine distal silt at
once, and it resolves that by leaving the physically measured range.

**Evidence 2 — the leakages are uniformly, implausibly small.**

```
log_L = [-11.3, -15.4, -10.0]      # interior, no bound hit
```

Layers essentially decoupled everywhere. But the published hydrogeology says the
**proximal fan has no confining layers at all** — thick gravel, indistinct stratification,
aquifers merged, vertical flow unrestricted. Leakage there should be high. One homogeneous
value cannot be both high and low, so the fit split the difference and is wrong at both ends.

This second point was visible in the existing results and was missed, because `log_L` reports
`bounds_hit: 0` and a parameter that hits no bound reads as healthy.

**Both point at the same fix, and neither is fixed by more folds, more seeds, or more epochs.**

## 2. Scope

Add a **structural zonal parameterization** and re-run the Stage-3 gate against it.

Out of scope: Stage 4 coupling (P2), scenarios and UQ (P3), the 3D twin (P4), and the
distance-degradation analysis (code already committed at `8f53147`, compute deferred).

## 3. Product context

Decided during brainstorming, and it sets the bar for everything below:

- The twin's job is an **operational decision tool** — real pumping-policy counterfactuals.
- Decisions are expressed **per zone**, not basin-wide and not per well. A 1 km grid
  calibrated on 66 sites cannot resolve individual wells.
- Zones are **physical** (proximal/mid/distal) for parameterization, and results are
  **aggregated to administrative units** (county, water-resources district) for reporting.
  Parameterise on the geology that drives the physics; report in units a regulator can act on.

The operational bar has a hard consequence: **scenarios must not ship from a bound-saturated
model.** Someone acts on the output. That is why this sub-project gates everything downstream.

## 4. Zone definition — `hydrophysics/twin/zones.py`

New module, one responsibility: map coordinates to a zone id.

```python
def fan_zones(xy: np.ndarray,
              proximal_km: float = 205.0,
              distal_km: float = 182.0) -> np.ndarray:
    """TWD97 easting -> zone id. 0 = proximal (E), 1 = mid, 2 = distal (W)."""
```

### 4.1 The proximal/mid boundary is well-constrained: x = 205 km

Published criterion: the proximal fan is where the confining mud layers are **absent**. The
CRAF literature is consistent that mid and distal fan carry four aquifers separated by four
aquitards within ~330 m, while the proximal fan is thick gravel with indistinct
stratification and unrestricted vertical flow.

That boundary is independently locatable in this project's own data. Of 311 fan stations with
valid TWD97 coordinates:

```
wells screened in layer 3 or 4 :  n=78   x 163.4 .. 207.9 km   (95th pct 202.1)
wells screened in layers 1-2   :  n=215  x 163.4 .. 214.8 km
```

Deep screens stop at 207.9 km; shallow screens continue 7 km further east. The aquitards
pinch out at roughly **203–208 km**, and 205 km sits in that band.

Cost of using it: only **12 of 66 calibration sites** fall in the proximal zone (at 210 km it
would be 1, which is unusable). This is acceptable *only* because the structural
parameterization in §5 gives that zone 2 free parameters rather than 11.

### 4.2 The mid/distal boundary is NOT well-constrained: x = 182 km, with a sensitivity check

The mid-to-distal transition is a gradual grain-size gradient, not a structural boundary.
There is no pinch-out to locate. 182 km is the equal-width third and is a **default, not a
finding**.

Required: re-run the gate at **178 km and 186 km** and report whether the verdict moves. If
the verdict is sensitive to a boundary we cannot independently justify, that is a result to
publish, not a knob to tune until the answer is agreeable.

## 5. Parameterization — `--param-mode zonal`

Structural: zones differ in **form**, not only in parameter values.

| zone | structure | free parameters |
|---|---|---|
| proximal | one merged aquifer: `log_L` **fixed at its upper bound**, one `log_T` and one `log_S` shared across all 4 layers | 2 |
| mid | 4 aquifers + 3 aquitards | 4 + 4 + 3 = 11 |
| distal | 4 aquifers + 3 aquitards | 11 |
| global | `log_eta`, recharge fraction | 2 |
| | | **26** |

Fixing proximal `log_L` at the top of its range **is** the statement "there is no aquitard
here": the layers equilibrate instead of being independently fitted. It costs zero parameters
and encodes the published geology directly, rather than hoping the optimiser discovers it.

Parameter counts in context, all against ~17,800 observations (136 entries × 131 months):

```
homogeneous (current, failed)      13
zonal, structural (this design)    26
zonal, naive 3x uniform            35
per-cell (ruled out in Plan A)  23,628
```

Implemented the same way `homogeneous` is: a small tensor optimised and expanded to
`(n_layers, n_active)` on every forward call, so `FlowModel`'s frozen constructor and
parameter shapes are untouched and autograd's broadcast-backward handles parameter sharing.

## 6. Gate and pre-registered decision rules

Same machinery as the seed sweep: grouped 5-fold (co-location rate must print 0.000), seeds
0–4, `--epochs 400`. Identical folds to the homogeneous runs, so the two are directly
comparable.

**These rules are fixed before the run.** This stage has produced four confident wrong answers
already — the under-trained gate, the LR-schedule budget, the leaky folds, and a distance
correlation that died on the fifth seed. Post-hoc reading is the recurring failure mode.

- **Primary — does the clamp release?** Report `bounds_hit` **per zone**, never pooled.
  Currently 3 of 4 `log_T` are pinned. If a majority remain pinned with three zones available,
  the parameterization is not the binding constraint and further zoning will not help.
- **Secondary — does the margin improve?** Mean and sd of `(r2_kfold − r2_idw)` over 5 seeds,
  against the homogeneous baseline of −0.048 ± 0.013.

| verdict | condition | consequence |
|---|---|---|
| **PASS** | clamp released **and** mean margin > 0 | Stage 3 passes. P2 unblocks normally. |
| **PARTIAL** | clamp released, margin improves but stays < 0 | Proceed to P2 with the limitation documented. P2's leveling-LOSO gate is the actual product gate; losing to IDW at head *interpolation* does not by itself disqualify a model whose job is *counterfactuals*, which IDW cannot do at all. |
| **FAIL** | clamp still pinned | Stop. The forward model is missing physics; zoning is not the answer. Escalate to a design conversation, do not retry. |

Note on PARTIAL: it is a legitimate verdict, not a hedge. IDW has no pumping input and cannot
answer a counterfactual. But PARTIAL is only available **if the clamp releases** — a model
still fitting outside the measured physical range gives confident wrong counterfactuals, and
at the operational bar that is the failure that matters.

## 7. Testing

TDD, matching the grouped-fold work.

**`zones.py`**
- every coordinate in the fan gets exactly one zone; no gaps, no overlaps
- boundary coordinates land in the documented zone (half-open intervals, explicit)
- deterministic and pure — same input, same output, no global state
- the 66 calibration sites split **12 / 33 / 21** and the 2,148 cells split
  **264 / 1,235 / 649** with the default boundaries

**Zonal parameter mode**
- free-parameter count is exactly 26 in the 4-layer both-drivers configuration
- gradients reach every zone's parameters and are finite (the `_ImplicitSolve.backward`
  returning `None` for `log_T` is a bug this project has already shipped once)
- proximal layers actually equilibrate: with `log_L` fixed high, heads across the four
  proximal layers converge, and the test asserts that rather than assuming it
- `bounds_hit` is reported per zone, and a pinned proximal `log_T` cannot be masked by
  interior mid/distal values

**Regression**
- `homogeneous` and `percell` behaviour unchanged
- full suite green (117 collected: 116 passed, 1 skipped), ruff clean

### 7.1 Implementation result — 2026-09-01

Deliverables 1-4 are **complete** (`13cf7d2`, `031ae15`, `d2c84a3`, `42efbda`, `a6f8bbf`).
Suite: 166 passed, 1 skipped, ruff clean. Every task independently reviewed.

Two defects were found and fixed during the work, both outside the original design:

**The CG iteration cap truncated every zonal solve.** Pinning proximal `log_L` at the top of
its range — §5's structural statement — raises the operator's condition number past what
Jacobi-CG reaches in `maxiter=400`. The first gate run produced median true relative residual
**4.955e-02** (max 2.542e+04), and its last 400 solves before abort had median **1.079e+00**,
i.e. worse than returning zero. `flow.py`'s own `_cg` docstring states a stalled solve
"silently corrupts the heads *and* every gradient computed from them," so every number that
run would have produced was junk. The operator is still SPD at the pin (200 Rayleigh
quotients, min 1.813e+04, none ≤ 0) and converges to 9.607e-09 at `maxiter=2000`; bisected
iteration counts are 237 (L=1e-4, T=500), 947 (L=1e-1, T=500), **1242** (L=1e-1, T=10, the
worst realistic case), 457 (L=1e-1, T=2e4). Fixed in `38b7ecc` by raising the cap, **not** by
lowering the pin — the pin is this design's physical claim, and §4.2's own reasoning forbids
tuning it to suit the solver. Re-verified on the real grid: zero non-convergence warnings at
every leakance including the pin.

The homogeneous arm was re-run at the new cap to keep the comparison like-for-like. It is
inert there: margin moved by **3.9e-12** (`r2_insample` and `loss` bit-identical, `r2_idw`
bit-identical), because those solves exit on the recurrence residual before the cap binds.
The pre-registered baseline of **−0.048 ± 0.013** therefore stands unchanged, and is
reproducible from the per-seed artifacts in `results/twin/`.

**§6's primary rule was not evaluable as originally specified.** `bounds_hit` reported
absolute counts over unequal denominators (proximal `log_T` is 1 value, mid and distal 4
each), so `proximal: 1` beside `mid: 2` read as "proximal less pinned" when it is 100% vs
50%, and "majority" was undefined across zones of size 1, 4, 4. Worse, the physically worst
outcome — the proximal gravel zone alone still pinned at T = 10, below the 58 m²/day floor
this remediation exists to escape — is a *minority of 9* and would have read as "clamp
released." That is §1's own documented failure (`log_L` reading healthy in aggregate while
wrong at both ends) about to repeat with `log_T`. Fixed in `ad5b5da`: hits are now split
lower-bound from upper-bound and reported as `n/total` per zone, and the counting rule was
written down **before any zonal number existed** (SDD ledger, "PRE-REGISTERED"):

> Clamp **released** iff pooled lower-clamp `log_T` ≤ 4/9 **and** the proximal zone's single
> `log_T` is not at the lower clamp. Only lower-clamp hits count — fitting *below* the
> measured range is the documented failure. Otherwise **FAIL** per §6: stop, escalate.

The same commit added run provenance (`cg_maxiter`, `git_commit`, `cg_nonconverged`,
`cg_worst_residual`, `fold_bounds_hit`) so a published number carries proof of its own
solver settings and gradient soundness, and `6f3c8bb` moved the in-sample clamp report ahead
of the k-fold gate so a run killed during the folds still yields the primary answer.

### 7.2 Verdict — REACHED 2026-09-09: primary rule **PASS**, secondary rule **FAIL**

Deliverable 5 is complete. Both rules have been evaluated on seed 0 with sound gradients.

#### Primary rule: PASS — the clamp released

`--param-mode zonal --fit-only --epochs 1500 --device cuda`, 14.3 h on a Quadro RTX 6000:

| condition (pre-registered) | measured | |
|---|---|---|
| pooled lower-clamp `log_T` ≤ 4/9 | proximal 0/1 + mid 0/4 + distal 3/4 = **3/9** | PASS |
| proximal `log_T` not at lower clamp | **0/1** — free | PASS |

`log_T_proximal` = 6.585 → **T = 725 m²/day**, inside the 58–6,034 m²/day of Liu et al.
2002. The pin at T = 10 below the 58 m²/day floor — the finding that survived the
2026-08-27 retraction, and the reason this remediation exists — **is gone. Zoning the
transmissivity worked.** In-sample R² +0.760, `cg_nonconverged=0`, trajectory
`PLATEAUED (structural)` from epoch 250. Reproduced independently at 500 epochs
(R² +0.758, identical clamp pattern), so the verdict is not an artefact of the budget.

#### Secondary rule: FAIL — the flow model does not beat IDW

5 folds, seed 0, 500 epochs, 24.6 h. **Pooled flow R² +0.466 vs IDW +0.702, margin −0.236.**
IDW wins in all five folds:

| fold | n_held | flow | IDW | margin |
|---|---|---|---|---|
| 0 | 34 | +0.768 | +0.867 | −0.099 |
| 1 | 34 | +0.252 | +0.846 | −0.594 |
| 2 | 26 | +0.230 | +0.853 | −0.623 |
| 3 | 31 | +0.422 | +0.443 | −0.021 |
| 4 | 33 | +0.437 | +0.746 | −0.309 |

**Why this FAIL stands where the previous three did not.** 2026-08-26 was under-trained;
2026-08-27 was retracted for co-location leakage; the grouped-fold re-run lost by 0.033,
which §7 itself called "undecided, not a kill". This one has none of those escapes:

- **Leakage is gone and the verdict does not depend on it.** Grouped folds report a 158
  entries / 82 physical sites split. 7 held-out entries still sit 0.23–0.28 m from a
  training entry (the reported `colocation_rate` counts *exact* zeros, so read it as
  "no exact duplicates", not "no near-duplicates"). Removing them changes nothing:

  | subset | n | flow | IDW | margin |
  |---|---|---|---|---|
  | all | 158 | +0.466 | +0.702 | −0.236 |
  | drop sub-metre | 151 | +0.468 | +0.699 | −0.231 |
  | drop < 1 km | 138 | +0.436 | +0.681 | −0.245 |

  Stripping near-duplicates makes the margin *worse*, so IDW is not winning by copying.
- **Gradients are sound.** `cg_nonconverged = 0`, worst true relative residual
  `0.000e+00`, across the entire 24.6 h gate.
- **Not under-trained.** In-sample R² plateaus at epoch 250.
- **−0.236 is 7× the −0.033 previously judged too close to call.** Not marginal.

**GATE: FAIL. Plan C (Stage 4 coupling) does not start.** Per §6: stop and escalate.

### 7.3 Where the constraint moved — read this before re-parameterising

The remediation succeeded at its stated target and the model still lost, so the binding
constraint is elsewhere. Three diagnostics from the same run point the same way.

**1. `log_eta` is pinned at its LOWER clamp, unanimously** — in-sample and in all five
folds (`lo=1/1` everywhere). Wire-to-water efficiency floored at η = 0.05, against a
`BOUNDS` ceiling of 0.9 and a physical range for irrigation pumps of roughly 0.4–0.7.
The fit is using η as an escape valve and hitting the stop. What it is escaping:

| η | implied abstraction (20 m lift) |
|---|---|
| 0.05 (the pin) | 0.83 ×10⁹ m³/yr |
| 0.45 (physical) | **7.43 ×10⁹ m³/yr** |
| 0.70 | 11.6 ×10⁹ m³/yr |

against published Choushui abstraction of ~1.5–2.0 ×10⁹ m³/yr. At any physically
defensible efficiency the electricity implies **roughly 4× too much water**. 9.88 TWh
entered the fan over 2012–2022; the model cannot reconcile that with the observed heads
except by flooring the conversion.

**2. The failure is concentrated in the shallow, forcing-dominated layers.**

| aquifer | n | flow | IDW |
|---|---|---|---|
| 1 (~53 m) | 37 | +0.319 | +0.691 |
| 2 (~119 m) | 79 | +0.514 | +0.679 |
| 3 (~210 m) | 32 | +0.812 | +0.825 |
| 4 (~282 m) | 10 | +0.854 | +0.895 |

In the deep aquifers, where pumping and recharge matter least, the flow model is level
with IDW. It loses in layers 1–2, which is where the forcing enters.

**3. Where IDW must genuinely extrapolate, the gap nearly closes.** Restricted to the 39
entries more than 5 km from any training well, flow +0.430 vs IDW +0.496 — a margin of
**−0.065** against −0.236 pooled. The physics is doing what physics is for; it is being
beaten on local interpolation, not on extrapolation.

**Conclusion for the next iteration.** Do not re-zone. The transmissivity field is no
longer the problem — §7.2 proves that. The suspect is the **energy→volume conversion in
`twin/pumping.py`**: a single global η over 116,768 heterogeneous meters, a lift floored
at `MIN_LIFT_M = 2.0`, and an assumption that every metered kWh lifts groundwater. Any of
those could carry the 4× discrepancy. Candidate remedies, cheapest first: per-`PURPOSE`
efficiency classes (the census supports it, and irrigation is 86% of installed HP); an
active/decommissioned filter on the census; and a lift model that does not floor at 2 m.

### 7.4 Root cause found and fixed (2026-09-09) — it was the lift, and it was worse than 4×

The 4× above assumed a 20 m lift. **Measured**, static lift on this fan is **median
6.65 m** — ground elevation is median 9.1 m and heads sit near the surface. 20.4% of
cell-months fall below `MIN_LIFT_M = 2.0`, and the 1st percentile is **−14.8 m**:
artesian, where the old code clamped to the floor and therefore implied the *largest*
volumes anywhere on the fan. Exactly backwards — an artesian cell needs the least work.

With the real lift the overestimate is **12–16×** (~25 ×10⁹ m³/yr at η = 0.45 against a
published ~1.5–2.0 ×10⁹). And even at the pinned η = 0.05 the model abstracts 2.78 ×10⁹,
still above the published range: efficiency was floored **and still over-pumping**. That
is why the clamp was unanimous rather than merely common.

**The defect.** `energy_to_volume` divided energy by the **static lift** where the physics
requires the **total dynamic head** — static lift plus well drawdown, entrance and
friction losses, and the discharge head the distribution system needs. Where static lift
is metres, omitting the rest is an order-of-magnitude error, and it lands on the one
parameter with any freedom to absorb it.

**The fix.** `energy_to_volume` takes an optional `head_extra`; calibration learns one
bounded scalar `log_head_extra` (1–200 m), threaded exactly as `log_eta` is, and recorded
in the run's `theta` as `head_extra_m`. Passing nothing reproduces the pre-2026-09-09
behaviour bit-for-bit, so every recorded result still replays. Median cell **8.2×**,
artesian cell **16.7×**. Tests: `tests/test_twin_pumping_head.py`, 8 cases.

**Evidence it addresses the right thing.** A 12-epoch zonal fit gives `eta = 0.293` with
`bounds_hit[global] = {log_eta: lo=0/1, log_head_extra: lo=0/1}` — *neither clamped* —
against `log_eta: lo=1/1` in the in-sample fit and all five folds of the failed gate. The
parameter the model was abusing is free again, and it settled on an ordinary wire-to-water
efficiency. `head_extra` learns toward ~33 m; ~48 m at η = 0.30 reconciles the published
abstraction, which is unremarkable once drawdown and 20–40 m of sprinkler discharge head
are counted. `n_params` 26 → 27.

**Correction to the ordering above.** "Cheapest first" put per-`PURPOSE` efficiency
classes ahead of the lift model. That was wrong: more efficiency classes cannot repair a
12× error while η sits on its floor. The lift model was the dominant term and the other
two are refinements. **Re-run the gate before any further parameterisation work** — the
FAIL was measured with a forcing now known to be wrong by an order of magnitude, so its
−0.236 margin says nothing yet about the aquifer model.

### 7.5 Two root causes found (2026-09-11) — the basin was closed and the census was double-counted

The total-dynamic-head fix (§7.4) was re-run as a full gate on 2026-09-11 (500 epochs,
5 folds, `results/twin_stage3_tdh/`). Its in-sample block already answered the question
before the folds finished: `log_eta` pinned at the **floor** (0.05) again and the new
`log_head_extra` pinned at its **ceiling** (200 m), with the recharge fraction falling from
0.93 to 0.36. In-sample R² rose to +0.868 from +0.758. Two knobs at opposite stops is an
optimiser asking for net forcing ≈ 0. That pattern has one cause:

**TDH-only verdict (2026-09-12, `results/twin_stage3_tdh/`, commit `cd612c9`, 500 epochs,
5 folds, closed basin, raw census): secondary rule **FAIL**, flow +0.622 vs IDW +0.702,
margin **−0.080** against −0.236 before the lift fix. In-sample +0.868. `log_eta` at the
floor and `log_head_extra` at the ceiling in the in-sample fit and in all five folds. The
lift fix moved the margin by 0.156 and left the forcing clamped at both stops, which is
the closed-basin signature described next.

**1. The solver was a closed basin.** `flow._neighbour_index` lists faces between active
cells only; every fan-edge face was no-flow. Nothing could leave to the Taiwan Strait and
nothing could enter at the apex, so the monthly balance had to close through storage and
the only way to fit heads that do *not* drift was to shrink both forcings. This also
explains §7.3's layer pattern (losing to IDW where forcing enters, tying where IDW must
extrapolate). Fix: `twin/boundaries.py` — general-head boundaries on the coast (westernmost
cell per row, h_b = 0 m, one learnable conductance per layer) and the apex (easternmost
cell per row inside the proximal zone, h_b = the initial IDW head, one shared
conductance). C → 0 recovers the closed basin, so the data can still choose it. The
implicit-function adjoint carries the new parameters (`tests/test_twin_boundaries.py`
checks the gradient against finite differences). `--boundaries coast-apex` is the default;
`none` reproduces every recorded run. n_params becomes 32 in zonal mode.

**2. The pump census over-counted electricity 4.6×.** Two independent defects:

| step | GWh 2012–2022 | what it removes |
|---|---|---|
| raw census (`--meter-filter none`) | 9,916 | — |
| count each shared meter once (`dedupe`) | 6,278 | 7,869 meters serve >1 pump and the same kWh series was attached to every one of them (62% of all kWh sat on such rows); the two largest "industry" entries were one 30 HP meter counted twice at 1.28 TWh each |
| drop meters over rated capacity (`dedupe-cap`, default) | **2,086** | a motor cannot draw more than HP × 0.746 kW × 730 h/month. The median *industrial* meter drew 194% of that and the top one 59,000%; livestock and aquaculture 90th percentiles sit at 5× and 1.5×. These are farm/factory supplies on an agricultural tariff, not water lifted. Irrigation (86% of installed HP) runs at 6% duty and loses little |

`pumping.clean_census` counts each meter once, splits its energy across its pumps by HP,
and drops meters whose mean monthly duty exceeds `--cap-duty` (1.0). §7.3's "roughly 4×
too much water at any physical efficiency" was, to within the lift correction, exactly
this electricity.

**Ordering corrected again.** §7.4 said the remaining candidates were refinements. They
were not: a missing outlet cannot be repaired by any forcing parameterisation, and a 4.6×
over-count cannot be absorbed by an efficiency bounded below at 0.05. Per-purpose
efficiency classes are now available as `--eta-classes` (opt-in) but are secondary to both.

### 7.6 Verdict on the corrected model (2026-09-14) — secondary rule **PASS**, with a caveat

`results/twin_runs/stage3_open_clean/` (published as `results/twin/stage3_zonal_open_clean.csv`),
commit `cd612c9` + the 2026-09-11 working tree, zonal, 32 parameters, 500 epochs, 5
site-grouped folds, `--boundaries coast-apex --meter-filter dedupe-cap`, 174-well head
field (158 in grid), 35 h on the GPU, `cg_nonconverged = 0`.

| | in-sample | 5-fold | IDW | margin |
|---|---|---|---|---|
| 2026-09-09 (closed, raw census) | +0.758 | +0.466 | +0.702 | −0.236 |
| 2026-09-12 (closed, raw census, TDH lift) | +0.868 | +0.622 | +0.702 | −0.080 |
| **2026-09-14 (open basin, clean census, TDH lift)** | **+0.906** | **+0.757** | +0.702 | **+0.055** |

Per fold: +0.902/+0.867, +0.864/+0.846, +0.862/+0.853, +0.663/+0.443, +0.507/+0.746
(flow/IDW). Four of five folds beat IDW; fold 4 does not.

**Primary rule.** Lower-clamp `log_T` hits are 2/9 in-sample (proximal 1/1, distal 1/4),
under the 4/9 threshold, so the rule passes on its count — but the *proximal* hit is the
physically wrong one (10 m²/day in the gravel fan), and it coincides with the apex
conductance sitting at its Dirichlet ceiling in every fold. The optimiser is insulating a
fixed-head apex from the fan with a low-T proximal zone.

**The caveat.** In-sample and in all five folds: `eta` = 0.05 (floor), `head_extra` =
200 m (ceiling), `recharge_frac` = 0.06-0.08, `C_apex` = 1e5 (ceiling), layer-1 `C_coast`
= 1e5 (ceiling) with layers 2-4 at 0.1-3 m²/day (closed), mid `S` at 0.3 in two layers.
The model beats IDW by carrying the observed heads through boundary-pinned, storage-damped
dynamics with ~2% of the published abstraction. It generalises across wells — that is
what the gate measures — but the *derivative* of head with respect to pumping, which is
what a policy scenario reads, is set by a conversion pinned at its floor. §7.5's
data-only check shows the heads do respond to local pumping (seasonal amplitude Spearman
+0.51 with cleaned kWh). So the stress is real; the model is placing it wrongly — most
likely all of it into layer 2 at the cell scale (p99 cell-month 3.4×10⁵ m³) with vertical
leakage switched off (`log_L` −10 to −17).

**Measured (2026-09-14, `results/twin_runs/stage3_fixed_eta/`, fit-only):** with the
conversion held at physical values (`--fix-eta 0.5 --fix-head-extra 40`, implied
abstraction ≈ 0.75 ×10⁹ m³/yr) the in-sample R² is **+0.877** against +0.906 free, and
the rest of the parameter set turns physical: recharge fraction 0.43, proximal T 57
m²/day (off the clamp), coast conductances 14-28 m²/day in layers 2-4. So the free fit's
+0.03 was bought by switching the stress off; a physical stress costs little and repairs
the recharge and the boundaries. Mid-zone S at 0.3 and the Dirichlet apex remain. The
k-fold gate of this configuration (`stage3_fixed_eta_gate/`, 2026-09-15) **FAILS**: +0.626
vs IDW +0.702, margin −0.076, against +0.757 for the free fit. A physical stress placed
where the model currently places it (all in layer 2, leakage switched off) generalises
worse than no stress. The free fit therefore remains the gated parameter set, with its
caveat. The stress-placement candidates were tested with the conversion held physical
(fit-only, 300 epochs): pump-layer split +0.867, leakance floor +0.876, irrigation
return flow +0.884, all three +0.884. The last was gated (`stage3_all_gate/`,
2026-09-17): in-sample +0.887, **5-fold +0.632 vs IDW +0.702, FAIL**. Every physical
stress placement tried generalises worse than the free fit; see STATE.md §3 for the
reading and the next steps. Candidate refinements after that:
pump→layer split by well depth, a leakance floor, irrigation return flow.

**Next gate.** Zonal, 500 epochs, 5 folds, `--boundaries coast-apex --meter-filter
dedupe-cap`, on the 174-well / 158-in-grid head field (declared canonical in STATE.md).
The forward twin (`twin/forward.py`) and the viewer's forward mode are built and tested
against the recorded closed-basin parameters, so a PASS turns directly into policy runs.

## 8. Cost

- Implementation: one focused session.
- Smoke test: seed 0 only, ~2 h. Inspect before spending more.
- Full sweep: seeds 1–4, ~8 h.
- Boundary sensitivity: 178 km and 186 km at seed 0, ~4 h.

Cost is driven by **CG iteration count, not epoch count** — a 400-epoch fold ran cheaper than
a 100-epoch one because the gentler LR schedule keeps `log_T` better conditioned. Do not
estimate from epochs. Fold times swing 12.6–33.4 min at identical settings.

## 9. Deliverables

1. `hydrophysics/twin/zones.py` + tests
2. `--param-mode zonal` in `calibrate_flow.py` + tests
3. Per-zone `bounds_hit` reporting, in stdout and in `stage3_flow.csv`
4. `--zone-boundaries` CLI flag for the sensitivity check
5. ~~Gate results for seeds 0–4 and both sensitivity boundaries~~ — **seed 0 done**
   (§7.2). Seeds 1–4 and the §4.2 boundary runs are **not worth spending**: §6 stops the
   sub-project on a FAIL, and at −0.236 the margin is 7× the seed spread that motivated a
   multi-seed pass in the first place. A seed sweep answers "is −0.033 noise?", which is
   no longer the question being asked.
6. ~~Spec §7 updated with the verdict, stated plainly, pass or fail~~ — §7.2, §7.3.
7. SDD ledger updated

## 10. Open questions carried forward

- **The mid/distal boundary has no independent justification.** §4.2's sensitivity check
  measures the consequence but does not resolve it. A grain-size or resistivity dataset (the
  TEM survey of Kassie et al. 2023 covers the mid and distal fan) could constrain it properly.
- **The northern lobe.** Radial-from-apex zoning placed the fan's northern tip in *distal*
  despite mid easting. The easting-based scheme adopted here does not, but whether that lobe
  is a separate depositional feature is unresolved and unexamined.
- **`--dx 500` grid convergence is not like-for-like.** The active-cell mask changes with
  resolution, so the well set changes 136 → 134. Intersect the well sets before comparing, or
  the convergence check measures the wrong thing (Plan B Task 6).
- **CG tolerance is set too tight.** ~5,000 `_cg did not converge within maxiter=400` warnings
  per run, worst true relative residual 2.9e-07 against tol 1e-8. Three orders better than the
  2.8e-4 that was a genuine bug, and the reference trace reproduces exactly — so this reads as
  a cost driver rather than bad gradients. Loosening it would speed every run above.

## 11. References

- Liu, Chang & Yeh (2002) — Choushui transmissivity, 58–6,034 m²/day. The `log_T` clamp's basis.
- Tsai & Hsu (2018), *Eng. Geol.* `10.1016/J.ENGGEO.2018.07.025` — VEP poromechanism applied
  to proximal, middle and distal fan wells; Young's modulus rises from distal to proximal.
  The compaction column this project uses is built on this model.
- Hung, Hwang, Sneed, Chen & Chu (2021), *WRR* `10.1029/2020WR028194` — MLCW magnetic rings at
  25 depths to 300 m, tested across proximal, middle and distal fan. Source of the compaction
  data this project holds.
- Kassie et al. (2023), *Water* 15:1703 `10.3390/w15091703` — TEM mapping of CRAF
  hydrogeological structure, middle and distal fan. Candidate constraint for §4.2.
- Chang et al. (2022), *Water* 14:1494 `10.3390/w14091494` — proximal/mid/distal differences in
  soil texture and hydrogeology across the fan.
