# Examples

Runnable scripts demonstrating the package. Most use the reference data bundled
in `sft2d.data`, so they work from an installed copy without any extra
downloads. The exception is `run_sharps_cycle.py`, which needs an external
SHARPS catalogue (see below).

| script | what it shows |
|---|---|
| `numerical_diffusion_test.py` | Peak retention of each advection limiter over one solid-body rotation (numerical diffusion), with exact flux conservation. |
| `bmr_demo.py` | Analytic Joy/Hale BMR source; verifies flux normalisation is exact at any latitude. |
| `bmr_shapes.py` | Side-by-side of the two BMR shapes — two-Gaussian (`make_bmr`) vs the Yeates bipole (`make_bmr_yeates`) — at equal flux. |
| `run_driven_cycle.py` | RGO-driven run over a chosen window → butterfly diagram + polar-field / axial-dipole series. |
| `validate_against_hmi.py` | Cycle-24 RGO-driven run overlaid on the observed HMI polar-field (both caps, mean ± std); demonstrates the reversal. |
| `run_sharps_cycle.py` | SHARPS-catalogue-driven run (Yeates BMRs) → butterfly + polar field. Needs an external catalogue. |

```bash
python examples/numerical_diffusion_test.py
python examples/bmr_demo.py
python examples/bmr_shapes.py
python examples/run_driven_cycle.py 2010 2020 30
python examples/validate_against_hmi.py 15
python examples/run_sharps_cycle.py path/to/bmrsharps_evol.txt 2010 2023 1.0
```

Each plotting script writes a PNG to the current directory.

## Driving from a SHARPS catalogue

`run_sharps_cycle.py` drives the model from an HMI/SHARPS-derived idealized-BMR
catalogue instead of the RGO sunspot record. Each catalogue row carries the
region's **observed** unsigned flux and its **fitted** separation and tilt, so
there is no Joy's-law tilt estimate and no Hale-law polarity assumption — the
signed fitted tilt sets the orientation, and `flux_scale` stays near 1 (the flux
is measured, not inferred from spot area).

The catalogue is **not bundled** (it is GPL-3.0 and lives in a separate
project). Obtain `bmrsharps_evol.txt` from
[antyeates1983/sharps-bmrs](https://github.com/antyeates1983/sharps-bmrs) — or
generate one with that project's pipeline — and pass its path to the script.
The package only reads the published catalogue format and reconstructs each
region with `make_bmr_yeates`; it does not reimplement the SHARP download/fit
pipeline (which needs JSOC/`drms` access).
