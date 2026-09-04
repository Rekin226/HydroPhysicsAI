# GPU Server — Environment, Constraints, and Stack Decisions

Reference for working on the NCU GPU VM. Written 2026-09-05.
Read this before installing anything GPU-related — several plausible choices
fail silently or waste hours on this specific hardware.

---

## 1. Hardware and environment

Connection details (addresses, account, VPN specifics) are deliberately **not** in this
file — see `GPU_SERVER.local.md`, which is gitignored. This repo is public.

| | |
|---|---|
| OS | Ubuntu 20.04.6 LTS, kernel 5.15.0-139 |
| Access | SSH key-based via institutional VPN; VS Code Remote-SSH |
| GPU | Quadro RTX 6000, 24 GB (23040 MiB) |
| Architecture | **Turing, TU102, compute capability 7.5 (sm_75)** |
| Driver | 535.230.02, CUDA 12.2 |
| Disk | 326 GB root (LVM: two disks pooled into one volume group) |
| Python | 3.11 via **Miniforge**, env `hydro` (`~/miniforge3/envs/hydro`) |

### Turing constraints — the important part

sm_75 predates several things modern GPU tooling assumes:

- **No bfloat16.** `torch.cuda.is_bf16_supported()` returns `False`.
- **No FP8.** Requires Hopper/Ada (sm_89+). Transformer Engine FP8 paths are unavailable.
- **No NVFP4.** Requires Blackwell.
- **No Flash Attention 2.** Requires sm_80+. Attention-heavy models run unaccelerated.
- **Available:** FP16 tensor cores, INT8, INT4, 1st-gen RT cores.

**Practical rule:** anything advertising bf16, FP8, NVFP4, or FA2 as a requirement
(not an option) will not work here. Prefer FFT-based (FNO/AFNO) and graph-based
(MeshGraphNet) architectures over transformer-heavy ones.

---

## 2. Stack decisions

### Use: NVIDIA PhysicsNeMo

Already declared in `pyproject.toml` as the `nemo` extra; `hydrophysics/models/ude_physicsnemo.py` exists.

Turing is **officially supported** (T4 is listed under Recommended Hardware, same sm_75).
The training harness explicitly detects missing bf16 and falls back to fp16 —
see `physicsnemo/utils/capture.py`. Transformer Engine and Flash Attention are
optional imports, not hard dependencies.

Directly relevant examples:

| Path | Why it matters |
|---|---|
| `examples/cfd/darcy_fno` | 2D Darcy flow via FNO — the steady-state groundwater equation |
| `examples/cfd/darcy_physics_informed` | Physics-guided Darcy |
| `examples/cfd/darcy_nested_fnos` | Nested/multi-GPU Darcy |
| `examples/reservoir_simulation/xmgn` | X-MeshGraphNet subsurface FV surrogate (faults, dual-porosity, fractures) |
| `examples/weather/flood_modeling/hydrographnet` | Physics-informed GNN with mass-conservation loss (surface water, but the loss pattern transfers) |

Reusable architectures shipped as plain `torch.nn.Module`: `fno`, `afno`,
`diffusion_unets`, `dit`, `meshgraphnet`, `graphcast`, `transolver`, `dpot`.

### Use: NVIDIA Warp

`warp-lang` — Python-syntax kernels JIT-compiled to CUDA, **natively differentiable**,
interops with PyTorch. Already a PhysicsNeMo dependency.

Turing sits **above** its documented minimum (PyPI wheels are built with CUDA 12.9 →
sm_52 floor; even CUDA 13 builds floor at sm_75). This is the best-supported
component of the whole stack on this hardware.

Use it if you need a custom differentiable Darcy / Richards / Biot solver.
It gives you the kernel and autodiff substrate — not a groundwater solver.
You implement the discretization and linear solve.

### Do NOT use: Isaac Sim / Isaac Lab

Rejected on two independent grounds.

**Scope.** Robotics simulator: rigid bodies, articulated systems, PhysX soft-body FEA,
RL environments. No Darcy flow, no porous media, no pore-pressure field, no Biot
coupling, no permeability tensor. PhysX soft bodies are single-phase elastic solids —
there is no path to Terzaghi/Biot consolidation without writing the physics yourself,
at which point Isaac contributes nothing.

**Hardware.** Minimum spec is RTX 4080 (Ada) — two architecture generations above
this card. NVIDIA's own guidance for older RTX GPUs is "should work, untested."

### Earth-2 / FourCastNet / CorrDiff: architectures only, no weights

All released checkpoints are atmospheric (ERA5/GFS/HRRR/GEFS). No subsurface variable
in any of them. Nothing to fine-tune.

The **CorrDiff pattern** — coarse regional field → diffusion super-resolution — does
map onto downscaling coarse head or subsidence fields. PhysicsNeMo ships the full
diffusion toolkit (`physicsnemo.diffusion`: schedulers, preconditioners, samplers,
multi-diffusion) decoupled from weather. Train from scratch on our data.

### There are no pretrained groundwater models

Verified 2026-09-05 across three sources:

- NGC catalog API: **0 hits** for `groundwater`, `subsurface`, `hydrology`, `aquifer`,
  `subsidence`, `porous`, `poroelastic`, `darcy`. (Control queries `corrdiff`,
  `fourcastnet`, `stormcast` returned hits, so the zeros are genuine.)
- Hugging Face `nvidia` org: no matching repositories.
- PhysicsNeMo repo source: zero matches for those terms.

**We train from scratch.** PhysicsNeMo's value is the framework and architectures.

---

## 3. Install constraints

### PhysicsNeMo requires a torch reinstall

PhysicsNeMo 2.2.1 (current as of 2026-08-31) requires:

- `python >=3.11,<3.15` — we have 3.11 ✓
  (the docs *System Requirements* page still says 3.10; it is stale — trust `pyproject.toml`)
- `torch >= 2.10.0`
- `warp-lang >= 1.14.0`

The env currently has torch from the **cu121** index, which is too old. Reinstall from cu128:

```bash
conda activate hydro
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "nvidia-physicsnemo[cu12]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Use `[cu12]`, not `[cu13]`.** The cu13 path needs driver ≥ 580; we are on 535.
CUDA minor-version compatibility should let cu128 wheels run on driver 535 —
verify immediately after installing.

**Known risk:** the `[cu12]` extra pulls RAPIDS (`cuml-cu12`, `pylibraft`, `nvidia-dali`)
at version 26.2+. Whether those still ship sm_75 kernels is **unverified** and is the
most likely install-time breakage. If it fails, install PhysicsNeMo without the CUDA
extra and rely on the torch wheels' bundled runtime.

### Local LLMs — DEFERRED, low priority

**Not a current workstream.** Coding assistance comes from Claude Code; the GPU
is reserved for the physics stack. Do not spend VRAM or setup time here unless
the priority is explicitly revisited.

Kept only as a reference for if that changes. What would fit 24 GB:

| Model | FP16 weights | Verdict |
|---|---|---|
| Nemotron-3-Nano-4B | ~8 GB | Best fit. Official GGUF repo exists → llama.cpp path |
| Nemotron-Nano-9B-v2 | ~17.8 GB | Fits, but ~6 GB left for KV cache. Short contexts only |
| Nemotron-Nano-12B-v2 | ~24 GB | Does not fit in fp16; needs INT8/INT4 |
| Nemotron-3-Nano-30B-A3B | ~60 GB | No. MoE activates 3B but all 30B must be resident |

**The bf16 trap.** Every Nemotron ships BF16 weights. Turing has no native bf16.
Load with explicit `dtype=torch.float16`. If you let `transformers` honor the
checkpoint's `torch_dtype: bfloat16`, PyTorch upcasts to fp32 and **doubles memory** —
the 4B becomes ~16 GB, the 9B OOMs.

Quantization: NVFP4 needs Blackwell, FP8 needs Ada — both unusable here.
**GGUF (llama.cpp) or GPTQ/AWQ INT4** are the viable paths; Turing has INT4/INT8 tensor cores.

**Unverified:** the `nemotron_h` architecture uses Mamba2 SSM layers requiring
`mamba-ssm` / `causal-conv1d` custom kernels. Whether those build on sm_75 is untested.
If they don't, fall back to GGUF.

---

## 4. Digital twin architecture

The neural surrogate is one layer of four:

1. **Forward model** — MODFLOW 6 via FloPy as the classical baseline; the neural
   operator as the fast surrogate. Keep both: the classical model is the reference
   the surrogate is validated against.
2. **Data assimilation** — keeps the twin synced to observations. EnKF or PEST++.
   This is where the GPU pays off: hundreds of ensemble members in parallel.
3. **Observation pipeline** — `data_fetch.py` for head/rainfall; InSAR for subsidence.
4. **Visualization** — Plotly (already in use via `hydrophysics.explorer`).

The GPU earns its place in layers 1 (surrogate training) and 2 (ensemble runs),
not in the classical solve.

---

## 5. Operational notes

- **Long jobs go in `tmux`**, never a bare SSH or VS Code terminal. A dropped VPN
  kills the session otherwise: `tmux new -s train`, detach `Ctrl+B` then `D`,
  reattach `tmux attach -t train`.
- **After every reboot, run `nvidia-smi`.** A kernel upgrade orphaned the driver once
  already (5.4 → 5.15 broke driver 450, requiring a full purge and reinstall of 535).
  This is the most likely failure mode to recur.
- **Do not run `do-release-upgrade`.** The 22.04 prompt at login is not worth risking
  a working GPU stack.
- **Back up.** The compute centre explicitly disclaims responsibility for data.
  Returning the GPU means a fresh VM with a new address, account, and SSH host key —
  the old server is not handed over. Commit code to GitHub; keep datasets elsewhere too.
- **The NVIDIA CUDA apt repo is disabled** (commented out in `/etc/apt/sources.list`).
  Its GPG key rotated and it was serving a corrupted `nvidia-settings` package.
  Re-enable only if you need `nvcc`, and install `cuda-keyring` first.
- **Never commit connection details.** This repo is public. Addresses, usernames,
  and host keys stay in `GPU_SERVER.local.md`.

---

## 6. Sources

- PhysicsNeMo: [PyPI](https://pypi.org/project/nvidia-physicsnemo/) ·
  [GitHub](https://github.com/NVIDIA/physicsnemo) ·
  [System requirements](https://docs.nvidia.com/physicsnemo/latest/getting-started/system_requirements.html) ·
  [Model catalog](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api_models.html)
- Isaac Sim requirements: [docs](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- Warp: [compatibility table](https://nvidia.github.io/warp/stable/user_guide/compatibility.html)
- Earth2Studio: [GitHub](https://github.com/NVIDIA/earth2studio)
- Nemotron: [NVIDIA HF org](https://huggingface.co/nvidia)
