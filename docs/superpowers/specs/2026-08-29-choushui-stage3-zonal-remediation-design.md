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
5. Gate results for seeds 0–4 and both sensitivity boundaries
6. Spec §7 updated with the verdict, stated plainly, pass or fail
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
