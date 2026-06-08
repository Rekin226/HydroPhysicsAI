# Contributing to HydroPhysicsAI

Thanks for your interest. This guide gets you from a fresh clone to a running GPU
training job, and explains the conventions that keep the benchmark honest.

## 1. Dev setup (any machine)

```bash
git clone https://github.com/Rekin226/HydroPhysicsAI.git
cd HydroPhysicsAI
python -m venv .venv && source .venv/bin/activate     # or conda/uv
pip install -e ".[dev]"          # foundation + pytest/ruff (no GPU)
```

Verify the foundation works without a GPU or any real data:

```bash
pytest -q                                 # 5 pass, model tests skip without torch
python -m hydrophysics.run_baselines      # baselines on the bundled synthetic sample
```

## 2. GPU setup (the NVIDIA Linux box)

### 2.1 PyTorch with CUDA

Install the build matching your CUDA toolkit (check `nvidia-smi` for the driver/CUDA
version, then pick the matching wheel from https://pytorch.org/get-started/locally/):

```bash
# example for CUDA 12.1 wheels
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[gpu,viz,dev]"           # adds torchdiffeq, matplotlib, geopandas
```

Sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`train.py` auto-selects `cuda > mps > cpu`; pass `--device cuda` to force it.

### 2.2 NVIDIA PhysicsNeMo (for the flagship port)

PhysicsNeMo (formerly Modulus) is the target framework for the operator port. Two
common install paths; confirm the current package name and version against the official
docs at https://docs.nvidia.com/physicsnemo/ before pinning:

```bash
# pip (into the same CUDA-enabled env)
pip install nvidia-physicsnemo

# or the maintained container (recommended for matched CUDA/driver/deps)
docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:<tag>
```

PhysicsNeMo is optional: the PyTorch models (`GlobalGRU`, `PhysicsUDE`) train on plain
CUDA without it. PhysicsNeMo is for the GPU-optimized operator port, mixed precision,
and multi-GPU. Keep it an optional dependency so the repo installs cleanly without it.

## 3. The model contract

Every model implements `hydrophysics.models.base.GroundwaterModel`:

```python
model.fit(data)                 # train on data.train_mask ONLY
pred = model.simulate(data)     # (n_wells, n_days) free-running hindcast
```

To add a model, drop a file in `hydrophysics/models/`, subclass `GroundwaterModel`, set
`name`, and register it in `hydrophysics/train.py:build_model`. It then flows through the
benchmark automatically.

### Evaluation rules (do not break these)

1. **Simulation mode only for the headline.** `simulate` must not read observed
   groundwater levels on or after the split. Use `self.initial_condition(data)` for the
   starting level. This is what makes the comparison to the gray-box fair.
2. **Same split, same metrics.** Validation starts at `Config.split_date` (2019-01-01),
   scored by KGE/NSE/RMSE in `hydrophysics.eval`. Do not retune the split per model.
3. **Report the distribution.** Median and per-well, not just the mean. A few hard wells
   dominate the mean, so always keep the per-well table and the spatial map.
4. **Forecast-mode persistence is context only.** Never compare it to simulation models.

## 4. Data policy

- **No real agency data in the repo, ever.** The Zhuoshui groundwater data is read at
  runtime via the `HYDROMIND_GW_DATA` env var (point it at a single_tankV2-style `data/`
  dir). The committed `hydrophysics/sample_data/` is synthetic and safe to publish.
- `.gitignore` blocks `data/`, `*.parquet`, and checkpoints. Keep it that way.
- Results you commit (benchmark tables, spatial maps) must be aggregate skill metrics,
  not raw level series.

## 5. Code style and checks

```bash
ruff check hydrophysics/        # lint (line length 100)
ruff format hydrophysics/       # format
pytest -q                       # tests must pass; add a test with every model/feature
```

- Match the existing style: type hints, short docstrings, no em dashes in prose.
- Keep torch imports lazy/optional so the foundation installs without a GPU.

## 6. Commits and PRs

- Small, focused commits with a clear subject line.
- A PR that changes a model should include its benchmark row (before/after) in the
  description, run on the synthetic sample at minimum.
- Branch off `main`; open a PR rather than pushing to `main` directly.

## 7. Where to start

Good first contributions:
- Implement the `PhysicsUDE` TODOs: swap the Euler loop for `torchdiffeq.odeint_adjoint`,
  add a physics-residual loss term, wire the tidal amplitude driver.
- Add a Fourier Neural Operator / DeepONet model for leave-one-well-out generalization.
- Add deep-ensemble or MC-dropout uncertainty to the simulate output.
- A small Gradio/Streamlit demo that plots observed vs simulated per well.

Questions or ideas: open an issue at
https://github.com/Rekin226/HydroPhysicsAI/issues.
