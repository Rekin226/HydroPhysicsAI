# HydroPhysicsAI — Project Instructions

## ⚠️ Rule 1: Sanity-check before every commit. No exceptions.

**This repository is public on GitHub.** Before staging or committing anything,
always run a secrets scan and report the result:

```bash
git status --short     # know exactly what is being added
git diff --cached      # review staged content, not just filenames

# Generic secrets scan over staged files. Patterns are deliberately generic —
# do not hardcode our real addresses or prefixes into this rule.
git diff --cached --name-only | xargs -r grep -nEi \
  "([0-9]{1,3}\.){3}[0-9]{1,3}|([0-9a-f]{1,4}:){4,}[0-9a-f]{0,4}|ssh-(rsa|ed25519)|BEGIN [A-Z ]*PRIVATE KEY|SHA256:[A-Za-z0-9+/]{40,}|passwo?r?d|api[_-]?key|secret|token|bearer"
```

Expect false positives (the word "token" in ML code, for instance). Read each hit
and judge it — do not suppress the scan to make it quiet.

Never commit:

- **Server addresses** — IPv4 or IPv6, private ranges included
- **Usernames, hostnames, SSH host-key fingerprints**
- **Credentials of any kind** — passwords, API keys, tokens, private keys
- **VPN or institutional access details**
- **Real data** — `data/`, `*.parquet`, `*.npy`, checkpoints (already gitignored)

Local-only notes go in `*.local.md`, which is gitignored. `docs/GPU_SERVER.local.md`
holds the connection details; `docs/GPU_SERVER.md` is the sanitized, committable version.

A filename looking innocuous is not sufficient — check contents. If anything
sensitive is found, stop and flag it rather than committing and fixing after.

## Rule 2: Read `docs/GPU_SERVER.md` before GPU or install work

It documents the hardware constraints and the stack decisions already made,
with the reasoning. Do not deviate without flagging why.

The short version: the GPU is **Turing (sm_75)** — no bf16, no FP8, no NVFP4,
no Flash Attention 2. Anything requiring those will not work. Prefer FFT-based
(FNO/AFNO) and graph-based (MeshGraphNet) architectures.

## Rule 3: Scope — the GPU is for the physics stack

Priority is physics-informed neural operators for groundwater and subsidence:
PhysicsNeMo, PyTorch, Warp. Local LLM hosting (Nemotron etc.) is explicitly
**deprioritized** — coding assistance comes from Claude Code, not from a model
running on this VM. Do not spend GPU memory or setup effort on local LLMs
unless asked.

## Environment

- Server env: conda `hydro` (Python 3.11, Miniforge) at `~/miniforge3/envs/hydro`
- Project installed editable: `pip install -e ".[gpu,viz,explorer,dev]"`
- Tests: `pytest -q` (~11 min, 158 tests, CPU-only by design)
- Long jobs go in `tmux`, never a bare SSH or VS Code terminal
- After any server reboot, verify `nvidia-smi` before assuming the GPU works

## Code conventions

- `ruff`, line length 100
- Every module starts with `from __future__ import annotations`
- `requires-python = ">=3.10"`; CI tests 3.10, 3.11, 3.12
- Only synthetic `sample_data/` ships in the repo; real data is never committed
