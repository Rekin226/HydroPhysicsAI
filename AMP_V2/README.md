# AMP_V2 / AMP-G — a multi-band pumping signature, validated at network scale

**Date:** 2026-08-22 · **Status:** experiment complete, recommendation below

Tests whether the Hilbert–Gauss Transform (GAFD + Hilbert, Lin et al. 2023,
`10.3390/s23083785`) improves the Average Magnitude of Pumping of Ouédraogo, Hsu & Wang
2023 (`10.1061/JHYEFF.HEENG-5760`), and — for the first time — validates AMP against
independently measured pump electricity.

## Headline

**AMP works and is now externally validated. The HGT upgrade does not beat the original.**

## 1. Implementation

- `gafd.py` — GAFD (Eq. 1–6) + Hilbert spectrum (Eq. 7–8) = HGT. Gaussian window
  `α = 4.0728`, window rule `M = 2⌊εN/N_e⌋` with `ε = 1.8`, double-symmetrical reflection,
  FFT convolution. IMFs are the high-pass residues; the cascade continues on the
  instantaneous mean, giving high→low frequency ordering.
- `amp.py` — `amp_v1` (published: Gaussian high-pass → FFT → band-pass → envelope) and
  `amp_v2` (HGT, adaptive mode selection).

Smoke test: a two-tone signal (0.5 m at 1.0 cpd + 2.0 m at 0.05 cpd) is separated to
0.487 and 1.939 m.

## 2. Synthetic validation (`compare.py`)

Correlation with the true pumping-stress envelope, across the non-stationarity regimes that
actually occur in irrigation (the fundamental stays locked to the solar day; amplitude,
duty cycle, phase and intermittency vary):

| regime | v1 narrow (bw 0.001) | v1 wide (bw 0.05) | v2 single-mode | v2 band-sum |
|---|---|---|---|---|
| stationary | −0.019 | **0.995** | 0.986 | 0.987 |
| duty-cycle change | 0.080 | **0.990** | 0.985 | 0.986 |
| phase drift | −0.021 | **0.997** | 0.988 | 0.990 |
| intermittent | −0.025 | **0.994** | 0.450 | 0.989 |
| **mean** | 0.004 | **0.994** | 0.852 | 0.988 |

Three findings:

1. **Band-pass width has a hard floor.** Seasonal amplitude modulation puts sidebands at
   `f ± 1/365 ≈ 0.0027` cpd. A band narrower than that removes the modulation being
   measured. With 0.05 cpd, v1 is excellent.
2. **The paper's mode-selection rule (smallest `f_std`, Eq. 12) fails on intermittency**
   (0.450). It assumes one clean mode; pumping is duty-cycled, so its fundamental and 2 cpd
   harmonic sit exactly one octave apart — EMD's resolution limit — and land in different
   IMFs. Summing energy across the band instead fixes it (0.989).
3. **v2 does not beat a properly tuned v1** (0.988 vs 0.994).

## 3. Real data: 40 wells, 2012–2022 (`run_wells.py`)

4,961 well-months. v2 found a pumping mode at 40/40 wells.

- **v1 and v2 agree**: median per-well correlation **0.920** (IQR 0.898–0.966), none below
  0.5, amplitude ratio v2/v1 = 1.07. v2 is a valid reimplementation, not a different
  quantity.
- **Seasonality reproduces the 2023 paper**: both peak in **March**, trough December.
- **Layer discrimination**, and it inverts the Tuku result at network scale: layer 2
  (median 5.5–6.6 cm, 79 m depth, n=16) carries far more pumping stress than layer 1
  (1.6 cm, 42 m, n=22). Physically sensible — irrigation wells screen the deeper confined
  production aquifer; shallow layer 1 has high specific yield, so less drawdown per unit
  pumped. The three Tuku wells were local heterogeneity.
- **v2's instantaneous frequency moves** (range 0.87 cpd at all 40 wells) and is largely
  independent of amplitude (r = −0.29), but shows **no coherent seasonal signal**
  (peak-to-trough 0.17 sd). Its information content is unproven.

## 4. External validation against pump electricity (`validate_electricity.py`)

The 2023 paper had no ground truth for pumping. It exists now: `etc-tpc-etc1mon-obs` gives
monthly kWh per registered pump. Fetched **1,736,085 monthly readings from 11,973 pumps**
within 1 km of a monitoring well (25 wells with enough overlap).

**First attempt gave a null result** — Spearman +0.10 (v1) / +0.12 (v2), Wilcoxon p = 0.75.
Two diagnostics explained it:

- **Billing is bimonthly**: 50.4% of months exactly repeat the previous month. Effective
  temporal resolution is half the nominal.
- **Purpose mix**: total electricity peaks in **July**; AMP peaks in **March**. The 2023
  paper predicts this — AMP anti-correlates with rainfall and peaks in the dry season —
  while aquaculture (481 pumps), domestic (472) and livestock (250) run year-round or peak
  in summer heat.

Stratifying by purpose resolves it. Seasonal correlation with AMP:

| purpose | pumps | peak month | corr with AMP |
|---|---|---|---|
| rice, 2nd crop | 2,010 | 4 | **+0.749** |
| rice, 1st crop | 2,921 | 4 | **+0.700** |
| all irrigation | 10,183 | 4 | **+0.662** |
| dry crop | 3,258 | 5 | +0.526 |
| irrigation, other | 1,994 | 7 | +0.155 |
| livestock | 250 | 9 | +0.145 |
| domestic | 472 | 8 | −0.149 |
| aquaculture | 481 | 9 | −0.169 |

Per-well monthly Spearman improves monotonically as the index is purified (nearest 100
pumps, 1/(1+r/300) weighting):

| electricity index | v1 | v2 | v2 > 0 |
|---|---|---|---|
| all purposes | +0.165 | +0.132 | 19/25 |
| irrigation only | +0.234 | +0.207 | 23/25 |
| rice + dry crop only | **+0.303** | +0.231 | 23/25 |

**AMP is externally validated for the first time**, and it is specifically an *irrigation*
pumping sensor — it does not see aquaculture, livestock or domestic abstraction.

## 5. Recommendation

1. **Keep AMP as a first-class observable in the digital twin.** It is now validated
   against independent data, it reproduces the published seasonality, and it discriminates
   aquifer layers.
2. **Do not adopt HGT/v2 as the primary estimator.** It does not beat v1 on synthetic data
   (0.988 vs 0.994) or on real data (+0.231 vs +0.303). This contradicts §4.1 of the design
   spec, which assumed the upgrade would win; the spec should be amended.
3. **Keep from v2:** the band-sum rule as a robustness guard where pumping is intermittent,
   and v2 as an independent cross-check (0.92 agreement is a useful consistency test). The
   instantaneous-frequency channel is retained but unproven — do not build on it yet.
4. **The real upgrade is purpose-stratified electricity, not the transform.** Restricting
   to rice and dry-crop pumps nearly doubles the correlation (+0.165 → +0.303). That is the
   change that should propagate into the twin's pumping model.
5. **Consequence for the twin:** AMP constrains the *irrigation* component of abstraction
   specifically. The gap between AMP-implied stress and total-electricity-implied
   abstraction is therefore not purely "unregistered pumping" — it also contains the
   non-irrigation purpose mix, which must be modelled separately before any unregistered
   claim is made. This tightens §4 of the design spec.

## Files

```
gafd.py                  GAFD + Hilbert (HGT)
amp.py                   amp_v1 (published), amp_v2 (HGT)
run_synthetic.py         single-regime synthetic probe
compare.py               four-regime fair comparison
run_wells.py             AMP for 40 wells, 2012-2022
fetch_pump_kwh.py        pump electricity within 1 km of each well
validate_electricity.py  the external validation
data/                    cached results (gitignored via data/)
```


---

# Part 2 — AMP-G at network scale (2026-08-22)

Motivated by a correct objection: the 2023 method fixes on 1 cpd, but an unconstrained
survey of 34 wells over 11 years finds a **median of 3.9 coherent spectral lines per well**
(131 total), with the sub-daily band (<0.5 cpd, median 1.92 cm) as large as the diurnal one
(2.69 cm) and a harmonic ladder at 2–5 cpd. AMP measures one line of four.

`ampg.py` implements the generalisation: detect lines against a local noise floor (assume
nothing), complex-demodulate each (explicit per-band time/frequency trade-off), attribute
them physically, and — the useful part — invert the harmonic ladder for **duty cycle**.

## Duty cycle from harmonics

A duty-cycled pump is a rectangular pulse train, `|c_n| ∝ |sin(nπd)|/(nπ)`, so
`A2/A1 = |cos(πd)|`. The aquifer low-passes the signal and biases that estimate high, so
`fit_duty_response` fits duty and attenuation `α` jointly across the whole ladder.

Synthetic recovery (needs ≥3 harmonics): errors ≤1.0 h/day even through an aquifer
low-pass, versus +1.6 to +6.2 h for the naive A2/A1 rule. `α` tracks the aquifer cutoff
monotonically (0.20 at 4 cpd → 0.64 at 1.5 cpd), so it is a usable T/S diagnostic.

## Network scale

267 of 344 fan wells retrieved (2012–2022, hourly), **161 yield an AMP-G signature**,
71 have irrigation-electricity ground truth (≥20 rice/dry-crop pumps within 1 km), and
**40 have recoverable duty** (the ≥3-harmonic requirement is the binding limit).
Recovered duty: **9.3 h/day median**, matching the 2023 paper's Tuku observations
(M1 4 h, M2 10 h).

### Cross-well skill — ranking wells by irrigation electricity

| observable | ρ | p |
|---|---|---|
| `amp_fund` (= AMP v1) | +0.255 | 0.032 |
| `amp_total` (fund+harm) | +0.234 | 0.049 |
| duty cycle alone | +0.293 | 0.067 |
| **volume = amp × duty** | **+0.424** | **0.0065** |

Paired on the identical 40 wells: amplitude +0.286 → volume **+0.424**, difference +0.137
with bootstrap 95% CI **[−0.008, +0.323]**, P(improvement) = 0.964. Duty carries
information beyond amplitude (partial Spearman +0.232 controlling for amplitude).

**Correction to Part 1:** an earlier n=16 test showed amplitude with *zero* cross-well skill
(ρ = −0.009). At n=71 that is +0.255 (p=0.03). The zero was small-sample noise, not a
property of the method.

### What did and did not work

- **Duty-weighted volume works** — the only observable significant at p<0.01, +48% over
  amplitude on identical wells. It is also the physically right quantity: rate × duration.
- **Multi-band amplitude adds nothing** — `amp_total` is no better than `amp_fund`
  (+7%, p=0.19 per-well; slightly worse cross-well).
- **The sub-daily band actively hurts** (per-well ρ 0.202, p=0.027 against v1). Those lines
  are not irrigation rotation; more likely river stage or barometric. Hypothesis rejected.
- **α splits by layer** — 0.21 in layer 1 (38 m) vs 0.49 in layer 2 (119 m).

## Verdict

The generalisation earns its place through **one** of its components, not the one first
proposed. Multi-band amplitude is a dead end; **duty cycle is a genuinely new,
aquifer-independent observable** that single-band AMP structurally cannot produce, and it
is what makes cross-well comparison work.

**For the twin:** use `volume = amp_fund × duty` as the AMP observable where duty is
recoverable (40/161 wells today), `amp_fund` elsewhere, and carry `α` as a soft prior on
the T/S ratio. The improvement is real but not conclusively established (CI touches zero),
so it enters as an observable with an honest uncertainty, not as a headline claim.

**Not yet a standalone paper.** Duty is recoverable at only ~25% of wells, and the decisive
CI includes zero. Raising the harmonic-detection yield is the obvious next step if this is
ever pursued for its own sake.
