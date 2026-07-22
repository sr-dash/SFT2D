"""
example_run.py

Minimal end-to-end check of an SFT2D installation: a free-decay run from a
dipole initial condition, with the conservation and accuracy diagnostics that
matter.

    python -m sft2d.example_run

Everything is driven through :func:`sft2d.src.stepper.evolve`, which handles the
Strang splitting, the advective sub-cycling and the RKL2 diffusion stages for
you -- there is no hand-rolled time loop to get wrong.
"""

import time

import numpy as np

from .analysis.analysis import calculate_dm, calculate_polar_field, calculate_usflx
from .src.constants import R_SUN_M
from .src.grid import create_grid, total_flux
from .src.initial_conditions import initialize_field
from .src.stepper import evolve
from .src.transport_profiles import differential_rotation, meridional_flow


def main(n_lat=91, n_lon=180, num_days=365, eta=2.5e8, v0=15.0):
    grid = create_grid(n_lat, n_lon)
    mf = meridional_flow(grid, peak_speed=v0)
    dr = differential_rotation(grid, rotation="solar", frame="carrington")
    field = initialize_field(grid, "dipole") * 3.0

    lat = np.rad2deg(0.5 * np.pi - grid["colatitude"])
    print(f"grid {n_lat}x{n_lon}: latitude {lat.min():+.1f} .. {lat.max():+.1f} deg, "
          f"dtheta = {np.rad2deg(grid['dtheta']):.2f} deg")
    print(f"sphere area / 4*pi*R^2 = "
          f"{float(np.sum(grid['area'])) * n_lon / (4 * np.pi * R_SUN_M**2):.15f}")

    bfly, days = [], []

    class Recorder:
        def record(self, day, B):
            if day % 5 == 0:
                days.append(day)
                bfly.append(B.mean(axis=1).copy())

    f0 = total_flux(field, grid)
    u0 = calculate_usflx(field, grid)
    t0 = time.time()
    B, stats = evolve(field, grid, mf, dr, eta, num_days,
                      recorder=Recorder(), return_stats=True)
    elapsed = time.time() - t0

    print(f"\n{num_days} days in {elapsed:.1f}s ({elapsed / num_days * 365:.1f} s/yr)")
    print(f"  RKL2 stages/step          {stats['rkl2_stages']}"
          f"   (vs {stats['explicit_diffusion_steps_avoided']:.0f} explicit diffusion steps)")
    print(f"  advection subcycles/half  {stats['advection_subcycles_per_half_step']}")

    f1 = total_flux(B, grid)
    print(f"\n  net flux drift    {abs(f1 - f0) / max(abs(u0), 1e-30):.3e}  (relative to unsigned)")
    # A pure dipole has no polarity inversion line away from the equator, so
    # diffusion has almost no opposite-sign flux to cancel and the unsigned
    # total barely moves; it is the signed total that must not move at all.
    print(f"  unsigned flux     {u0:.4e} -> {calculate_usflx(B, grid):.4e} Mx")
    pn, ps = calculate_polar_field(B, grid)
    print(f"  polar field       N {pn:+.3f} G   S {ps:+.3f} G")
    print(f"  axial dipole      {calculate_dm(B, grid):+.4f} G")

    return np.array(bfly), np.array(days), grid, B


if __name__ == "__main__":
    main()
