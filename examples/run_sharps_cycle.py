"""
run_sharps_cycle.py

Drive the SFT model from a SHARPS-derived BMR catalogue (Yeates ``sharps-bmrs``
``bmrsharps_evol.txt`` format) instead of the RGO sunspot record, and plot the
butterfly diagram + polar-field series.

    python examples/run_sharps_cycle.py path/to/bmrsharps_evol.txt
    python examples/run_sharps_cycle.py path/to/bmrsharps_evol.txt 2010 2023 1.0

The catalogue is NOT bundled (it is GPL and lives in a separate project).  Get
it from https://github.com/antyeates1983/sharps-bmrs (file ``bmrsharps_evol.txt``,
or generate your own with that project's pipeline), then pass its path here.

Why SHARPS instead of RGO
-------------------------
Each catalogue row carries the region's *observed* unsigned flux and its
*fitted* separation and tilt, so there is no Joy's-law tilt estimate and no
Hale-law polarity assumption -- the signed fitted tilt sets the orientation.
That makes ``flux_scale`` close to 1 (the flux is measured, not inferred from
spot area as with RGO).
"""

import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sft2d.analysis.analysis import calculate_dm, calculate_polar_field
from sft2d.src.grid import create_grid
from sft2d.src.sharp_driver import SHARPSource
from sft2d.src.stepper import evolve
from sft2d.src.transport_profiles import differential_rotation, meridional_flow

np.seterr(all="ignore")


def main(catalogue, start_year=2010, end_year=2023, flux_scale=1.0,
         v0=15.0, eta=2.5e8, seed_dipole=-2.0, n_lat=91, n_lon=180):
    start, end = f"{start_year}-05-01", f"{end_year}-01-01"
    grid = create_grid(n_lat, n_lon)
    mf = meridional_flow(grid, peak_speed=v0)
    dr = differential_rotation(grid)

    src = SHARPSource(catalogue, start_date=start, end_date=end, flux_scale=flux_scale)
    print(src.summary(), f"| flux_scale={flux_scale}, v0={v0:+g} m/s")

    from sft2d.src.initial_conditions import initialize_field
    field = initialize_field(grid, "dipole") * seed_dipole

    bfly, dm, pfn, pfs, days = [], [], [], [], []

    class Rec:
        def record(self, day, B):
            if day % 5:
                return
            days.append(day); bfly.append(B.mean(axis=1).copy())
            dm.append(calculate_dm(B, grid))
            n, s = calculate_polar_field(B, grid)
            pfn.append(n); pfs.append(s)

    t0 = time.time()
    evolve(field, grid, mf, dr, eta, src.num_days, source=src, recorder=Rec())
    print(f"{src.num_days / 365.25:.1f} yr run in {time.time() - t0:.0f}s")

    bfly = np.array(bfly).T
    yr = np.array(days) / 365.25 + start_year + 0.33
    lat = np.rad2deg(np.pi / 2 - grid["colatitude"])

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    vmax = np.percentile(np.abs(bfly), 98) or 1.0
    pm = ax[0].pcolormesh(yr, lat, bfly, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax[0].set_ylabel("latitude [deg]")
    ax[0].set_title(f"SHARPS-driven SFT: butterfly  <B_r>   flux_scale={flux_scale}")
    fig.colorbar(pm, ax=ax[0], label="G", pad=0.01)
    ax[1].plot(yr, pfn, "C3", label="polar N (>70)")
    ax[1].plot(yr, pfs, "C0", label="polar S (<-70)")
    ax[1].plot(yr, dm, "k", lw=1, label="axial dipole")
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].set_xlabel("year"); ax[1].set_ylabel("field [G]")
    ax[1].legend(ncol=3, fontsize=8)

    out = f"sharps_{start_year}_{end_year}_fs{flux_scale:g}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("ERROR: provide the path to a bmrsharps_evol.txt catalogue.")
        raise SystemExit(1)
    a = sys.argv[1:]
    kw = {}
    if len(a) >= 2: kw["start_year"] = int(a[1])
    if len(a) >= 3: kw["end_year"] = int(a[2])
    if len(a) >= 4: kw["flux_scale"] = float(a[3])
    main(a[0], **kw)
