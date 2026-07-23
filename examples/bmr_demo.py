"""
bmr_demo.py

Demonstrate the analytic BMR source term: build flux-normalised bipoles in each
hemisphere (Joy tilt + Hale polarity) and verify the realised unsigned flux
matches the request to machine precision -- the normalisation uses the true
finite-volume cell areas, so it is exact at any latitude.

    python examples/bmr_demo.py
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sft2d.analysis.analysis import calculate_usflx
from sft2d.src.grid import create_grid
from sft2d.src.source import make_bmr


def main(n_lat=91, n_lon=180, outfile="bmr_demo.png"):
    grid = create_grid(n_lat, n_lon)
    lat = np.rad2deg(np.pi / 2 - grid["colatitude"])
    lon = np.rad2deg(grid["longitude"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a, la in zip(ax, [20, -20]):
        B = make_bmr(grid, la, 180, 1e22, sigma_deg=4, sep_deg=8, cycle_number=24)
        realised = calculate_usflx(B, grid)
        print(f"lat {la:+d}: requested 1.000e22 Mx, realised {realised:.4e} Mx "
              f"({100 * realised / 1e22:.3f}%)")
        vmax = np.abs(B).max()
        pm = a.pcolormesh(lon, lat, B, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        a.set_xlim(140, 220); a.set_ylim(la - 20, la + 20)
        a.set_title(f"BMR at {la} deg (cycle 24)")
        a.set_xlabel("lon"); a.set_ylabel("lat")
        fig.colorbar(pm, ax=a, label="G")

    fig.tight_layout(); fig.savefig(outfile, dpi=140, bbox_inches="tight")
    print("saved", outfile)


if __name__ == "__main__":
    main()
