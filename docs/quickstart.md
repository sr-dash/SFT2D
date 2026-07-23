# Quick start

`sft2d` is a two-dimensional **surface flux transport (SFT)** model: it evolves
the radial magnetic field {math}`B_r(\theta,\phi,t)` on the solar (or stellar)
surface under large-scale flows, supergranular diffusion, and the emergence of
new active-region flux. SFT models have been the workhorse for understanding the
long-term evolution of the Sun's global field and the polar-field reversals that
seed the polar magnetic flux of each cycle
([Leighton 1964](https://doi.org/10.1086/148058);
[Wang, Nash & Sheeley 1989](https://doi.org/10.1126/science.245.4919.712);
[Sheeley 2005](https://doi.org/10.12942/lrsp-2005-5);
[Mackay & Yeates 2012](https://doi.org/10.12942/lrsp-2012-6);
[Jiang et al. 2014](https://doi.org/10.1007/s11214-014-0083-1)).

This package is a spherical, finite-volume 2-D generalisation of the 1-D solver
of [Yeates (2020)](https://doi.org/10.1007/s11207-020-01688-y). It assumes no
back-reaction of the field on the prescribed flows, so it is inexpensive enough
to scan the transport parameter space while still resolving the full
{math}`(\theta,\phi)` surface. The mathematical formulation and the numerical
scheme are described in the {doc}`theory manual <sft2d-theory>`; this page gets
you running.

## Installation

The package builds from `pyproject.toml` (PEP 621) with a static version, so no
git checkout is required to install.

Conda (creates the environment and installs the package editable):

```bash
conda env create -f environment.yml
conda activate sft2d
```

or plain pip from a local checkout:

```bash
pip install -e .            # runtime only
pip install -e ".[dev]"     # + pytest, build, ruff, jupyter
```

Reference data — the processed RGO active-region record, an HMI synoptic map,
and HMI polar-field / butterfly references — ship **inside** the package at
`sft2d/data/`, so the examples and the observational comparisons work from an
installed copy with no extra downloads:

```python
from sft2d.data import RGO_CSV, HMI_SYNOPTIC_FITS, load_hmi_polar_field
```

## A minimal run

```python
import sft2d as sft

grid = sft.create_grid(91, 180)                 # 91 colatitude x 180 longitude cells
mf   = sft.meridional_flow(grid, peak_speed=15.0)   # u_theta [m/s], poleward
dr   = sft.differential_rotation(grid)              # Omega(theta) [rad/s]
B0   = sft.initialize_field(grid, "dipole") * 3.0   # initial B_r [G]

B = sft.evolve(B0, grid, mf, dr, eta=2.5e8, num_days=365)   # eta in m^2/s

print("axial dipole :", sft.calculate_dm(B, grid), "G")
print("polar field  :", sft.calculate_polar_field(B, grid, pol_cap_extent_deg=20))
```

`evolve` handles the time integration internally (operator splitting, an adaptive
advective sub-cycle, and super-time-stepped diffusion — see the
{doc}`theory manual <sft2d-theory>`). Sources and data assimilation enter through
the `source=` and `assimilate=` callbacks; diagnostics through a `recorder=`
object.

## What is in the package

| module | purpose |
|---|---|
| `sft2d.src.grid` | pole-to-pole finite-volume mesh; area/flux helpers |
| `sft2d.src.operators` | conservative diffusion + TVD advection operators |
| `sft2d.src.sts` | RKL2 super-time-stepping for diffusion |
| `sft2d.src.stepper` | `evolve` — Strang-split time integration |
| `sft2d.src.transport_profiles` | meridional flow + differential rotation |
| `sft2d.src.initial_conditions` | analytic dipole / observed synoptic-map start |
| `sft2d.src.source` | analytic BMR source terms (`make_bmr`, `make_bmr_yeates`) |
| `sft2d.src.ar_driver` | `ARSource` — drive from the RGO sunspot record |
| `sft2d.src.sharp_driver` | `SHARPSource` — drive from an HMI/SHARPS BMR catalogue |
| `sft2d.analysis.analysis` | unsigned flux, axial dipole, polar field / flux |

## Verifying it works

A one-command smoke test prints the grid extent, area closure, the chosen
super-time-stepping stage count, and the conservation and polar-field
diagnostics:

```bash
python -m sft2d.example_run
```

The numerical test suite checks the properties the whole model rests on — flux
conservation, operator symmetry, analytic decay rates, TVD monotonicity, and
second-order convergence of the polar-cap diagnostics:

```bash
pytest                    # full suite
pytest -m "not slow"      # skip the multi-year driven-run tests (fast)
```

See the {doc}`theory manual <sft2d-theory>` for exactly which result each test
verifies.

## Worked examples and notebooks

Runnable scripts in `examples/` (each writes a figure):

```bash
python examples/numerical_diffusion_test.py      # limiter peak-retention
python examples/bmr_demo.py                       # flux-normalised Joy/Hale BMRs
python examples/bmr_shapes.py                     # two-Gaussian vs Yeates bipole
python examples/run_driven_cycle.py 2010 2020 30  # RGO-driven butterfly + polar field
python examples/validate_against_hmi.py 15        # cycle-24 reversal vs HMI
python examples/calibrate_sharps_vs_hmi.py <catalogue>   # SHARPS calibration check
python examples/scan_sharps_calibration.py <catalogue>   # (eta, v0, tau) scan vs HMI
```

Jupyter notebooks in `docs/notebooks/` walk through the physics interactively:

- `example-run.ipynb` — the core tour (grid, flows, integration, diagnostics);
- `rgo_driven_and_hmi.ipynb` — RGO-driven cycle vs the HMI polar field / butterfly;
- `sharps_driven_analysis.ipynb` — the data-driven SHARPS pipeline end to end;
- `polar_dispersal_experiment.ipynb` — a controlled comparison of polar
  flux-dispersal recipes.

The SHARPS catalogue used by the data-driven examples is **not** bundled (it is a
separate, GPL project); see `examples/README.md` for how to obtain it.
