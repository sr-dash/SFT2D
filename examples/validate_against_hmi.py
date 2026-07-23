"""
validate_against_hmi.py

Drive the model over the HMI era (cycle 24, 2010-2023) from the bundled RGO
record and overlay the modelled north-polar-cap field on the observed HMI
CAPN2 series.  This is the end-to-end validation that the model reproduces the
observed polar-field reversal.

    python examples/validate_against_hmi.py
    python examples/validate_against_hmi.py 40      # flux_scale = 40

Key physics demonstrated
------------------------
With the correct **poleward** meridional flow (positive ``v0``) and an adequate
``flux_scale``, the trailing-polarity flux from Joy's-law-tilted regions is
carried to the poles and reverses them, matching the sign change seen by HMI.
A *negative* ``v0`` (equatorward flow) leaves the poles unchanged and never
reverses -- flow direction, not diffusivity, decides whether reversal can
happen at all.
"""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sft2d.analysis.analysis import calculate_polar_field
from sft2d.data import RGO_CSV, load_hmi_polar_field
from sft2d.src.ar_driver import ARSource
from sft2d.src.grid import create_grid
from sft2d.src.initial_conditions import initialize_field
from sft2d.src.stepper import evolve
from sft2d.src.transport_profiles import differential_rotation, meridional_flow

np.seterr(all="ignore")


def run(flux_scale=15.0, v0=15.0, seed_dipole=-2.0, cap_deg=20.0,
        start="2010-05-01", end="2023-01-01", n_lat=91, n_lon=180):
    grid = create_grid(n_lat, n_lon)
    mf = meridional_flow(grid, peak_speed=v0)
    dr = differential_rotation(grid)
    src = ARSource(str(RGO_CSV), start_date=start, end_date=end, flux_scale=flux_scale)

    # Seed the observed pre-cycle-24 polarity: the HMI north cap is NEGATIVE at
    # the 2010 start, so the seed dipole is negative in the north.  (The default
    # "dipole" profile is positive in the north, hence the negative amplitude.)
    field = initialize_field(grid, "dipole") * seed_dipole
    y0 = int(start[:4])
    yr, pn, ps = [], [], []

    class Rec:
        def record(self, day, B):
            if day % 10:
                return
            yr.append(y0 + 0.33 + day / 365.25)
            n, s = calculate_polar_field(B, grid, pol_cap_extent_deg=cap_deg)
            pn.append(n); ps.append(s)

    evolve(field, grid, mf, dr, 2.5e8, src.num_days, source=src, recorder=Rec())
    return np.array(yr), np.array(pn), np.array(ps)


def main(flux_scale=15.0):
    yr, pn_model, ps_model = run(flux_scale=flux_scale)

    # HMI reference: use the mean cap field +/- its 1-sigma spread (the SFT-1D
    # convention).  All Series share one DatetimeIndex.
    hmi = load_hmi_polar_field()
    idx = hmi["mean_north"].index
    hmi_yr = idx.year + (idx.dayofyear - 1) / 365.25
    mn, ms = hmi["mean_north"].values, hmi["mean_south"].values
    sn, ss = hmi["std_north"].values, hmi["std_south"].values

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(hmi_yr, mn - sn, mn + sn, color="C0", alpha=0.25)
    ax.fill_between(hmi_yr, ms - ss, ms + ss, color="C2", alpha=0.25)
    ax.plot(hmi_yr, mn, "C0", lw=1, label="HMI north cap")
    ax.plot(hmi_yr, ms, "C2", lw=1, label="HMI south cap")
    ax.plot(yr, pn_model, "C0", lw=2.5, ls="--", label="model N")
    ax.plot(yr, ps_model, "C2", lw=2.5, ls="--", label="model S")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("year"); ax.set_ylabel("polar field [G]")
    ax.set_title(f"Cycle 24 polar-field reversal: RGO-driven SFT vs HMI (flux_scale={flux_scale:g})")
    ax.legend(ncol=2, fontsize=8)

    rev = np.sign(pn_model[0]) != np.sign(pn_model[-1])
    print(f"model N polar field: {pn_model[0]:+.2f} -> {pn_model[-1]:+.2f} G  "
          f"({'REVERSED' if rev else 'no reversal'})")
    print(f"HMI   N (mean)     : {mn[0]:+.2f} -> {mn[-1]:+.2f} G")

    out = f"validate_hmi_fs{flux_scale:g}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    fs = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    main(fs)
