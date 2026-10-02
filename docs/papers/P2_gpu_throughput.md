# P2: Throughput of a launch-bound float64 solver on a commodity GPU

**Status:** planned; blocked on the benchmark campaign. **Last updated:** 2026-10-02.

## Target venues (in order)

1. **SC workshops** (e.g. AI for Science / HPC for climate workshops at Supercomputing).
2. **NVIDIA GTC** talk or poster (lower bar, strong fit with the stack).
3. **Concurrency and Computation: Practice and Experience** (full journal version).

## Working title

Running a differentiable groundwater twin on one Turing GPU: launch-bound float64 solves,
CUDA graphs and MPS for ensemble throughput

## Contribution

- A real-problem characterisation of a small-grid (2,148 cells x 4 layers), float64,
  matrix-free CG solver: launch-bound rather than compute-bound (matvec 257 us fp64 vs
  250 us fp32), so fp64 costs little and newer tensor-core precisions do not help.
- NVIDIA MPS turns the idle GPU into throughput: 3 concurrent fits took 829 s against 791 s
  for one (2.86x), applied to k-fold folds as independent jobs
  (`calibrate_flow --only-fold` + `merge_folds`, identical to sequential).
- Lesson: synthetic-grid benchmarks mispredicted the real solver by about two orders of
  magnitude (see memory note on the solver), and a missing device argument ran every fit on
  CPU for weeks: practical guidance for scientific users.

## Evidence in the repo

- MPS runner and logs: `results/twin_runs/par*/`, `results/twin_runs/gap_queue.log`.
- Solver: `hydrophysics/twin/flow.py` (`_CG_*`, `--compile-matvec`), `calibrate_flow.py`.
- Hardware and constraints: `docs/GPU_SERVER.md`.

## Benchmark campaign (to run before writing)

- [ ] CPU baseline: 1, 4, 8, 12 threads; per-epoch time of the deliverable recipe.
- [ ] GPU: eager vs `--compile-matvec` vs explicit CUDA graph capture; CG iterations per
      step; fp64 vs fp32 matvec.
- [ ] Profile (Nsight Systems): kernel count per CG iteration, launch overhead share.
- [ ] MPS: 1, 2, 3, 4, 6 concurrent jobs; throughput and per-job latency.
- [ ] Energy per epoch (nvidia-smi power sampling) vs CPU.
- [ ] Grid scaling: dx 1000 / 500 / 250 m, to find where it stops being launch-bound.
- [ ] Ideally one run on a newer card (A100/H100, cloud) for the scaling story.

## Gaps

- The "51x" figure in the README is stale; the current speedup may be closer to 2x for the
  fit. Whatever the campaign measures is the number this paper reports.
