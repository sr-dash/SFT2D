"""
run_driven_cycle.py

Data-driven SFT run over a chosen window, driven by the bundled RGO
active-region record, producing a butterfly diagram and polar-field /
axial-dipole time series.

    python examples/run_driven_cycle.py                 # cycle 24 (2010-2020)
    python examples/run_driven_cycle.py 1996 2009 10    # start end flux_scale

Notes
-----
* The mesh is pole-to-pole, so the butterfly reaches +/-90 deg.
* The transport operators conserve flux exactly; any net-flux drift printed at
  the end comes from the emergence source, not the solver.
* ``flux_scale`` multiplies the RGO |USFLUX|.  The tabulated values are ~10-50x
  too weak in absolute terms, so a value of order 10-60 is needed before the
  emerged trailing-polarity flux can reverse the poles (see
  ``examples/validate_against_hmi.py``).
"""

import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sft2d.analysis.analysis import calculate_dm, calculate_polar_field
from sft2d.data import RGO_CSV
from sft2d.src.ar_driver import ARSource
from sft2d.src.grid import create_grid, total_flux
from sft2d.src.initial_conditions import initialize_field
from sft2d.src.stepper import evolve
from sft2d.src.transport_profiles import differential_rotation, meridional_flow

np.seterr(all="ignore")


def main(start_year=2010, end_year=2020, flux_scale=30.0, seed_dipole=1.0,
         v0=15.0, eta=2.5e8, n_lat=91, n_lon=180, outfile=None):
    start, end = f"{start_year}-05-01", f"{end_year}-01-01"
    grid = create_grid(n_lat, n_lon)
    mf = meridional_flow(grid, peak_speed=v0)          # +v0 = poleward
    dr = differential_rotation(grid)

    src = ARSource(str(RGO_CSV), start_date=start, end_date=end, flux_scale=flux_scale)
    print(src.summary(), f"| flux_scale={flux_scale}, v0={v0:+g} m/s (poleward)")

    field = initialize_field(grid, "dipole") * seed_dipole

    bfly, dm, pfn, pfs, days, net = [], [], [], [], [], []

    class Rec:
        def record(self, day, B):
            if day % 5:
                return
            days.append(day)
            bfly.append(B.mean(axis=1).copy())
            dm.append(calculate_dm(B, grid))
            n, s = calculate_polar_field(B, grid)
            pfn.append(n); pfs.append(s)
            net.append(total_flux(B, grid))

    t0 = time.time()
    _, stats = evolve(field, grid, mf, dr, eta, src.num_days,
                      source=src, recorder=Rec(), return_stats=True)
    print(f"{src.num_days / 365.25:.1f} yr run in {time.time() - t0:.0f}s  "
          f"(RKL2 stages {stats['rkl2_stages']}, advection subcycles "
          f"{stats['advection_subcycles_per_half_step']}/half-step)")
    print(f"net flux: {net[0]:+.3e} -> {net[-1]:+.3e} Mx (source-driven, not solver drift)")

    bfly = np.array(bfly).T
    yr = np.array(days) / 365.25 + start_year + 0.33
    lat = np.rad2deg(np.pi / 2 - grid["colatitude"])

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    vmax = np.percentile(np.abs(bfly), 98) or 1.0
    pm = ax[0].pcolormesh(yr, lat, bfly, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax[0].set_ylabel("latitude [deg]")
    ax[0].set_title(f"Data-driven SFT (RGO): butterfly  <B_r>   flux_scale={flux_scale}")
    fig.colorbar(pm, ax=ax[0], label="G", pad=0.01)
    ax[1].plot(yr, pfn, "C3", label="polar N (>70)")
    ax[1].plot(yr, pfs, "C0", label="polar S (<-70)")
    ax[1].plot(yr, dm, "k", lw=1, label="axial dipole")
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].set_xlabel("year"); ax[1].set_ylabel("field [G]")
    ax[1].legend(ncol=3, fontsize=8)

    if outfile is None:
        outfile = f"driven_{start_year}_{end_year}_fs{flux_scale:g}.png"
    fig.tight_layout(); fig.savefig(outfile, dpi=140, bbox_inches="tight")
    print("saved", outfile)


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if len(a) >= 1: kw["start_year"] = int(a[0])
    if len(a) >= 2: kw["end_year"] = int(a[1])
    if len(a) >= 3: kw["flux_scale"] = float(a[2])
    main(**kw)
