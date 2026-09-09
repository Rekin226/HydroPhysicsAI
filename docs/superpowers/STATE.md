# Project state — where to continue

**Last updated:** 2026-09-09 · Read this first if you are picking the twin up cold.

The goal, stated once so the gates below have a point:

> **A living 3D digital twin of the Choushui fan that takes a pumping scenario, runs the
> aquifer forward in time, and animates where and how fast the ground sinks —
> re-runnable on demand.**

Not a static map: it must be *animated* (built), *scenario-responsive* (partly built), and
*runnable forward past the end of the data* (not built — that needs a flow model that
passes its gate).

---

## 1. Where the chain stands

```
pumping ──▶ [flow model] ──▶ heads ──▶ [VEP column] ──▶ subsidence ──▶ [3D viewer]
            GATE FAILED               PASSES              PASSES        BUILT
            (secondary rule)          (Stage 2)      (R2 +0.242 vs      (head-space
                                                      799 benchmarks)   scenarios only)
```

| stage | status | evidence |
|---|---|---|
| Stage 1 — algebraic `Sk` on leveling | marginal | +0.040 tilt-corrected, 878 sites |
| Stage 2 — VEP compaction | **PASS** | shared-VEP LOSO +0.465 vs baseline +0.031 |
| Stage 3 primary — clamp released | **PASS** (2026-09-06) | `log_T` 3/9 ≤ 4/9, proximal free at T = 725 m²/day |
| Stage 3 secondary — margin vs IDW | **FAIL** (2026-09-09) | flow +0.466 vs IDW +0.702, margin −0.236 |
| Stage 4 — flow↔VEP coupling | built, **not started** | blocked by the gate, per remediation spec §6 |
| Stage 5 — 3D scenario twin | partial | `explorer3d.py` ships; scenario axis is head decline, not pumping |

Canonical verdict and diagnosis: `specs/2026-08-29-choushui-stage3-zonal-remediation-design.md`
§7.2 (verdict) and §7.3 (where the constraint moved).
Session narrative and corrections: `specs/2026-09-05-twin-performance-and-3d-revision.md`.

## 2. The one thing to do next

**Fix the pumping energy→volume forcing in `twin/pumping.py`. Do not re-zone.**

The zonal remediation hit its stated target — `log_T` is now physical — and the model
still lost, so the aquifer parameterisation is no longer the suspect. Three diagnostics
from the same run agree:

1. **`log_eta` pins at its LOWER clamp (η = 0.05) unanimously**, in-sample and in all five
   folds. At a physical η ≈ 0.45 the metered electricity implies **~7.4 ×10⁹ m³/yr against
   a published ~1.5–2.0 ×10⁹** — roughly 4× too much water. The fit uses efficiency as an
   escape valve and hits the stop.
2. The failure is **layer-stratified**: layer 1 (~53 m, forcing-dominated) flow +0.319 vs
   IDW +0.691; layer 4 (~282 m) +0.854 vs +0.895. It fails where the forcing enters.
3. Beyond 5 km from any training well — real extrapolation, IDW's weakest ground — the
   margin narrows to −0.065. The physics is losing on *interpolation*, not on physics.

Candidate fixes, cheapest first:

- **Per-`PURPOSE` efficiency classes.** One global η over 116,768 heterogeneous meters is
  the least defensible part of the model. `twin/scenario.py` already implements the class
  mapping (irrigation = 86% of the 457,750 installed HP inside the fan).
- **An active/decommissioned filter on the census.** Well metadata carries `DisuseDate`;
  the pump census needs an equivalent check. Dead meters inflate the forcing.
- **A lift model that does not floor at `MIN_LIFT_M = 2.0`.**

Only if the 4× discrepancy survives all three is the aquifer model the suspect again — and
then the honest move is a different forward model (a PhysicsNeMo FNO surrogate trained on
MODFLOW), not another zoning scheme.

## 3. Remaining gaps, beyond the immediate blocker

**On the critical path to the goal**

1. **Forward integration past 2022 does not exist.** Everything is hindcast. Running
   forward needs future forcing — a pumping scenario *and* a rainfall/ET assumption
   (`scenario.climatology()` is written for this, unused).
2. **Re-run cost.** "On demand" means minutes, not the 14 h a zonal fit currently takes.
   This is where **PhysicsNeMo finally earns its install** — an FNO surrogate trained on
   the calibrated solver. Installed (2.2.1) and still **not used by any twin code path**.
3. **Stage 4 coupling is built but unexercised** (`twin/coupled.py`, 9 tests). It waits on
   a flow model that passes.

**Credibility**

4. **No uncertainty anywhere.** No ensemble, no data assimilation. `docs/GPU_SERVER.md` §4
   names assimilation as layer 2 of the architecture and it does not exist. For anything
   policy-facing this is the largest gap.
5. **Modest subsidence skill** — R² +0.242, bias +3.1 cm against leveling.
6. **Head-field provenance is ambiguous.** The refetched field is 174 wells / 8.79% NaN
   against the recorded 147 / 1.1%. Stage 2 still passes on it, but no one has declared
   which is canonical.
7. **The mid/distal zone boundary has no independent justification** (remediation spec
   §10), and **`--dx 500` grid convergence is not like-for-like** (the active-cell mask
   changes the well set).
8. **`log_S` presses its upper bound** in 3 of 4 mid-zone cells (4/4 in one fold), and
   3 of 4 distal `log_T` remain at the lower clamp with one at the upper. The rule counts
   only lower-clamp hits and 3/9 clears 4/9, but the distal zone is visibly straining.

## 4. Traps that have already cost time

- **Device placement.** `calibrate_flow` had no `--device` and constructed `FlowModel`
  without one, so every flow run in this project's history — and every cost estimate drawn
  from them (the 4.3 h fit, the 12,733 s/fold, the "~144 h, we need a better GPU"
  conclusion) — silently ran on CPU. On the real problem **GPU is ~51× faster**
  (67 s/epoch vs 3,452). Fixed, and `tests/test_twin_device.py` guards it. Always confirm
  the `device:` line at startup.
- **Synthetic benchmarks mispredicted the real system by two orders of magnitude.** A
  3,224-cell synthetic grid said GPU was 1.1× CPU; the real problem is 51×. Do not size
  this solver's cost from synthetic grids.
- **Long runs need `--log-every`.** A 9 h run with no progress output cannot be triaged,
  and `ptrace_scope=1` means py-spy needs sudo to attach after the fact.
- **The WiseEnvr bearer token expires** and `demo.py` never renews it. Any fetch longer
  than ~1 h must refresh. Resume state must record *successes*, not *attempts*, or a
  restart silently skips whatever failed during an outage.
- **`results/twin/*.csv` are tracked** (published gate results); a new run overwrites
  committed files. `results/twin_stage3_*/` are ignored.

## 5. Reproducing the two Stage-3 runs

```bash
export HYDROMIND_GW_DATA=$PWD/chou-shui-data/data

# primary rule (~14 h on GPU) -- the clamp report prints before --fit-only returns
python -m hydrophysics.twin.calibrate_flow --param-mode zonal --fit-only \
  --epochs 1500 --device cuda --compile-matvec --log-every 25 \
  --polygon "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json" \
  --wells-dir AMP_V2/data/wells --stations AMP_V2/data/fan_stations.parquet \
  --pump-census AMP_V2/data/tpc_pumps.parquet --pump-kwh AMP_V2/data/pump_kwh_all.parquet \
  --rf-timeseries chou-shui-data/data/rf_timeseries.csv \
  --rf-stations chou-shui-data/data/rf_stations.csv \
  --gw-stations chou-shui-data/data/gw_stations.csv \
  --et-npz results/et/openmeteo_et0_2012_2022.npz --out results/twin_stage3_zonal

# secondary rule: drop --fit-only, add --n-folds 5 --dump-predictions (~25 h)
```

Note the paths: the CLI defaults still point at a doubled `chou-shui-data/chou-shui-data/`
prefix and at a dead scratchpad for the pump census, so **pass them explicitly**.

Data that must exist (all gitignored, ~331 MB + ~1 GB):
`chou-shui-data/data/` (restored from backup — the API cannot supply the fan polygon) and
`AMP_V2/data/{wells/,fan_stations.parquet,tpc_pumps.parquet,pump_kwh_all.parquet}`
(rebuilt from the WiseEnvr API; 26.0M kWh rows, 116,768 pumps).

## 6. Environment

torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, PhysicsNeMo 2.2.1, warp-lang 1.17.0, driver
535.230.02, Quadro RTX 6000 (Turing sm_75 — no bf16, no FP8, no FA2). conda env `hydro`.
The cu128 wheels ship sm_75 and run on driver 535 via minor-version compatibility.
Forecaster throughput was flat across the upgrade (24.81 s → 24.61 s): the upgrade was an
access fee for PhysicsNeMo, not a speedup. Baseline in `~/bench_baseline/COMPARISON.md`.
