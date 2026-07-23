"""
numerical_diffusion_test.py

Quantify the numerical diffusion of the advection scheme: advect a localised
tracer one full solid-body rotation with ZERO physical diffusivity and measure
how much of the peak survives.  First-order upwind loses most of the peak; the
TVD limiters retain far more.  The lost peak is spurious diffusion that would be
absorbed into a fitted ``eta`` -- which is why the limiter matters for
calibration.  Flux is conserved to machine precision regardless of limiter.

    python examples/numerical_diffusion_test.py
"""

import numpy as np

from sft2d.src.grid import create_grid, total_flux
from sft2d.src.operators import Advection
from sft2d.src.stepper import advect


def peak_retained(limiter, n_lat=91, n_lon=180, cfl=0.4):
    grid = create_grid(n_lat, n_lon)
    theta, phi = grid["colatitude"], grid["longitude"]
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    B0 = np.exp(-(((TH - np.pi / 2) / 0.15) ** 2 + ((PH - np.pi) / 0.15) ** 2))

    omega = 1e-6                                  # rad/s, solid body
    op = Advection(grid, np.zeros(theta.size), np.full(theta.size, omega), limiter=limiter)
    B = advect(B0, op, 2.0 * np.pi / omega, cfl=cfl)
    return B.max() / B0.max(), total_flux(B, grid) / total_flux(B0, grid) - 1.0


if __name__ == "__main__":
    print("Grid 91x180, one full rotation, eta_physical = 0\n")
    print(f"  {'limiter':<20s} {'peak retained':>14s} {'flux drift':>14s}")
    for label, lim in [
        ("1st-order upwind", "upwind1"),
        ("TVD minmod", "minmod"),
        ("TVD van Leer", "vanleer"),
        ("TVD MC", "mc"),
        ("TVD superbee", "superbee"),
    ]:
        peak, drift = peak_retained(lim)
        print(f"  {label:<20s} {peak * 100:13.1f}% {drift:14.2e}")
