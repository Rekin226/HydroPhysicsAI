# Papers from the Choushui twin

One file per planned paper. Each holds the target venues, the contribution, the evidence
in the repo it rests on, and the gaps to close before submission. Pick any one up from its
file; `docs/superpowers/STATE.md` section 0 is the source of every number.

| # | file | paper | first-choice venue | status |
|---|---|---|---|---|
| 1 | [P1_differentiable_twin.md](P1_differentiable_twin.md) | A differentiable GPU digital twin of an aquifer-subsidence system, and how to test one | Geoscientific Model Development | planned (main paper) |
| 2 | [P2_gpu_throughput.md](P2_gpu_throughput.md) | Throughput of a launch-bound float64 solver on a commodity GPU | SC workshop / NVIDIA GTC | planned, needs benchmark campaign |
| 3 | [P3_fno_surrogate.md](P3_fno_surrogate.md) | A PhysicsNeMo FNO surrogate for pumping-policy ensembles | NeurIPS ML4PS workshop | planned, needs surrogate retrain |
| 4 | [P4_decision_app.md](P4_decision_app.md) | Showing scenario impact of a groundwater twin to decision makers | Environmental Modelling & Software | optional |

Suggested packaging: P1 as the journal paper; P2 and P3 merged into one shorter GPU paper
or a GTC submission if each alone is thin. Do not split one result across papers.

## Shared facts (check against STATE.md before quoting)

- Hardware: one NVIDIA Quadro RTX 6000 (Turing, sm_75, 24 GB), torch 2.11, CUDA 12.8,
  PhysicsNeMo 2.2.1. No bf16, FP8 or Flash Attention on this card.
- NVIDIA stack actually used: PyTorch CUDA, `torch.compile` / CUDA graphs for the solver
  matvec, PhysicsNeMo FNO for the surrogate, NVIDIA MPS for concurrent jobs.
  **NVIDIA Warp is not used**; do not claim it.
- Deliverable model: `results/twin_runs/stage3_datum_gate`, projection
  `results/twin_forward/datum_gate`.
- Data: WRA / WiseEnvr / Taipower records cannot be redistributed; only synthetic
  `sample_data/` ships. Every data-availability statement must say so.

## Gaps shared by every paper

1. **Re-measure the CPU vs GPU speedup.** The README's "51x" is stale (measured on an
   older solver); a 2026-09-23 check gave about 70 s/epoch on 4 CPU threads against
   36.5 s/epoch on the GPU for the current fit. No paper may quote a speedup until the
   benchmark campaign in P2 has run.
2. Freeze a release tag of the code for each submission (GMD requires a Zenodo DOI).
