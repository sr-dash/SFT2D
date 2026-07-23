"""
bmr_shapes.py

Compare the two idealized BMR shapes available in the package for the *same*
region parameters:

* ``make_bmr``        -- two separate great-circle Gaussians (one per polarity);
* ``make_bmr_yeates`` -- the single smooth antisymmetric bipole of Yeates
  (2020 / the ``sharps-bmrs`` catalogue), with a continuous polarity-inversion
  line and longitude elongation.

Both are normalised to the same total unsigned flux, so this isolates the
*shape* difference.  The Yeates form is the natural choice when driving from a
SHARPS catalogue whose separation and tilt are fitted from real magnetograms.

    python examples/bmr_shapes.py
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sft2d.analysis.analysis import calculate_usflx
from sft2d.src.grid import create_grid
from sft2d.src.source import make_bmr, make_bmr_yeates


def main(outfile="bmr_shapes.png"):
    grid = create_grid(181, 360)
    lat = np.rad2deg(np.pi / 2 - grid["colatitude"])
    lon = np.rad2deg(grid["longitude"])

    lat0, lon0, flux, sep, tilt = 20.0, 180.0, 1e22, 8.0, 20.0
    B_gauss = make_bmr(grid, lat0, lon0, flux, tilt_deg=tilt, sep_deg=sep,
                       sigma_deg=4.0, hale=False)
    B_yeates = make_bmr_yeates(grid, lat0, lon0, flux, sep_deg=sep, tilt_deg=tilt)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, B, title in ((ax[0], B_gauss, "two-Gaussian (make_bmr)"),
                        (ax[1], B_yeates, "Yeates bipole (make_bmr_yeates)")):
        vmax = np.abs(B).max()
        pm = a.pcolormesh(lon, lat, B, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        a.set_xlim(150, 210); a.set_ylim(5, 35)
        a.set_title(f"{title}\nunsigned {calculate_usflx(B, grid):.2e} Mx")
        a.set_xlabel("Carrington lon [deg]"); a.set_ylabel("lat [deg]")
        fig.colorbar(pm, ax=a, label="G")

    fig.suptitle(f"BMR at ({lat0:g}, {lon0:g}) deg, flux {flux:.0e} Mx, "
                 f"sep {sep:g} deg, tilt {tilt:g} deg", y=1.02)
    fig.tight_layout(); fig.savefig(outfile, dpi=140, bbox_inches="tight")
    print("saved", outfile)


if __name__ == "__main__":
    main()
