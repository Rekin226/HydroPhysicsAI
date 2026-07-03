# 10-minute quickstart on the synthetic sample

This walkthrough gets from a fresh clone to a first benchmark table and hydrograph
without real groundwater data or a GPU. By default, HydroPhysicsAI reads the bundled
synthetic sample in `hydrophysics/sample_data/`, so every command below is safe to run
locally.

## 1. Install the tutorial dependencies

Create an isolated environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gpu,viz]"
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The optional extra is named `gpu` because it installs PyTorch for the neural models.
The tutorial commands force `--device cpu`, so they do not require CUDA.

## 2. Run the baseline table

Start with the deterministic baselines:

```bash
python -m hydrophysics.run_baselines
```

The first line confirms you are using the bundled sample:

```text
GWData: 4 wells x 1095 days (2017-01-01 to 2019-12-31), split 2019-01-01 [train 730d / val 365d], target coverage 100.0%
```

Then the command prints two tables:

- **Simulation mode** is the apples-to-apples hydrology benchmark. The model is started
  from an initial condition and then free-runs through the validation period.
- **Forecast mode** is a separate operational task. It can use recent observed levels at
  the forecast origin, so its scores should not be compared with simulation-mode scores.

In the table, `kge_median` is the main skill score. KGE is Kling-Gupta efficiency:
`1.0` is perfect, values near the climatology row mean little improvement over a seasonal
average, and negative values can mean the simulation is worse than a simple reference.
`rmse_median` is median error in metres, so lower is better. `nse_median` is
Nash-Sutcliffe efficiency, where `1.0` is perfect and negative values are worse than
predicting the observed mean.

## 3. Train a first UDE on CPU

Train the physics-informed UDE on the same sample. Keep the epoch count small for the
first pass:

```bash
python -m hydrophysics.train --model ude --device cpu --epochs 30 --out results/tutorial_ude
```

This repeats the dataset summary, prints the selected device, fits the model on the
training dates before `2019-01-01`, then prints a simulation-mode validation table.
The `physics_ude` row is the learned model. Compare its `kge_median` and
`rmse_median` against `climatology`, `last_value`, and the bundled `graybox_ode`
reference row.

For a publication-quality run, increase the epochs to the values used in the README and
run on CUDA when available. The small tutorial run is meant to prove the workflow and
produce a first result quickly.

## 4. Plot a hydrograph

Generate the reproducible sample figures:

```bash
python -m hydrophysics.figures --device cpu --sim-epochs 30 --fc-epochs 5 --out results/tutorial_figures
```

Open `results/tutorial_figures/simulation_hydrographs.png`. Each panel shows one
synthetic well through time:

- the dark line is the observed synthetic groundwater level
- the model line is the free-running simulation
- the shaded validation window is the period scored in the benchmark table
- the legend reports validation KGE for that well

A useful first read is whether the simulation follows the seasonal shape and recession
after rainfall without drifting away in the validation window. If the KGE is high but the
plot looks biased, check the RMSE and the observed-versus-simulated level offset.

## 5. Run it on your own data

Once the synthetic walkthrough works, point HydroPhysicsAI at your own CSV directory:

```bash
export HYDROMIND_GW_DATA=/path/to/your/gw_data
python -m hydrophysics.run_baselines
python -m hydrophysics.train --model ude --device cpu --epochs 30
```

Your directory must match the CSV contract in [`docs/DATA_FORMAT.md`](DATA_FORMAT.md).
That page explains the required groundwater, rainfall, station, and well-pairing files,
plus the optional gray-box baseline table.
