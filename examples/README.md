# Examples

Runnable scripts demonstrating the package. All use the reference data bundled
in `sft2d.data`, so they work from an installed copy without any extra
downloads.

| script | what it shows |
|---|---|
| `numerical_diffusion_test.py` | Peak retention of each advection limiter over one solid-body rotation (numerical diffusion), with exact flux conservation. |
| `bmr_demo.py` | Analytic Joy/Hale BMR source; verifies flux normalisation is exact at any latitude. |
| `run_driven_cycle.py` | RGO-driven run over a chosen window → butterfly diagram + polar-field / axial-dipole series. |
| `validate_against_hmi.py` | Cycle-24 RGO-driven run overlaid on the observed HMI north-cap field; demonstrates the polar-field reversal. |

```bash
python examples/numerical_diffusion_test.py
python examples/bmr_demo.py
python examples/run_driven_cycle.py 2010 2020 30
python examples/validate_against_hmi.py 40
```

Each plotting script writes a PNG to the current directory.
