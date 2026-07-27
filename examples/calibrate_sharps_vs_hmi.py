"""
calibrate_sharps_vs_hmi.py

Check SFT calibration performance against the observed HMI data, driving the
model directly from a SHARPS-derived BMR catalogue (Yeates ``bmrsharps_evol``
format, with or without a header row).

    python examples/calibrate_sharps_vs_hmi.py [catalogue.txt]

The catalogue path is optional: it defaults to the current SHARPS database
(override with the SFT2D_SHARPS_CATALOGUE environment variable).

What it reports
---------------
1. **Axial dipole moment** -- the unambiguous, cap-independent observable, and
   the quantity the SHARPS catalogue is built around.  With ``flux_scale = 1``
   (the SHARPS flux is *measured*, not inferred) the modelled axial dipole
   should peak near the observed cycle-24 value of ~+3 to +4 G.
2. **Polar-cap field vs HMI** -- the model's mean B_r poleward of ``cap`` deg
   overlaid on the bundled HMI ``mean_north``/``mean_south`` series.  The HMI
   polar field (http://jsoc.stanford.edu/data/hmi/polarfield/) is averaged
   **poleward of +/-60 deg**, so the default ``cap_deg=30`` matches it.  The
   *timing and shape* of the reversal match closely (correlation ~0.95); the
   script also prints the single best-fit scale factor and the RMS *after*
   removing it, so any residual amplitude offset is separated from the (good)
   dynamics.

Initial condition
-----------------
By default the run starts from the **observed** HMI synoptic map bundled in
``sft2d.data`` (``hmi_CR2097.fits``, ~2010), whose polar field already matches
the observed 2010 values -- much more faithful than a scaled dipole.  Pass
``--dipole`` to fall back to the analytic seed.

Calibrated default recipe
-------------------------
The transport defaults are the calibrated "Yeates" recipe: a Schuessler-Baumann
meridional flow (``v0 = 11`` m/s), ``eta = 500`` km^2/s, and a ``tau = 10`` yr
flux decay.  Together these remove the polar flux over-concentration (near-pole
field becomes ~dipole-like) while keeping the axial dipole in the observed
3-4 G band and the reversal correlation with HMI at ~0.95.  ``examples/
scan_sharps_calibration.py`` shows the scan these values came from.

Notes
-----
* The catalogue is NOT bundled (GPL; separate project -- see examples/README.md).
* ``flux_scale`` should stay ~1 for SHARPS (the flux is measured); ``eta``,
  ``v0`` and ``tau_years`` are the transport knobs to explore.
"""

import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sft2d.analysis.analysis import calculate_dm, calculate_polar_field
from sft2d.data import load_hmi_polar_field
from sft2d.src.grid import create_grid
from sft2d.src.initial_conditions import initialize_field
from sft2d.src.sharp_driver import SHARPSource
from sft2d.src.stepper import evolve
from sft2d.src.transport_profiles import differential_rotation, meridional_flow

np.seterr(all="ignore")

# Default SHARPS catalogue (the current cycle-24/25 database).  Override with the
# SFT2D_SHARPS_CATALOGUE environment variable or a command-line path argument.
import os

DEFAULT_CATALOGUE = os.environ.get(
    "SFT2D_SHARPS_CATALOGUE",
    "/Users/sdash/NSO/Work/GIT-Projects/sharps-bmrs-db/sharps-bmrs-db/bmrsharps_evol.txt",
)


# Calibrated default recipe (Yeates sft_data style): a weaker, lower-latitude
# Schuessler-Baumann meridional flow, 2x the diffusivity, and a modest flux
# decay.  Together these remove the polar flux over-concentration (near-pole
# field ~dipole-like) while keeping the axial dipole in the observed 3-4 G band
# and improving the reversal correlation with HMI (~0.95).  See
# examples/scan_sharps_calibration.py for how these values were chosen.
FLOW_PROFILE = "schuessler-baumann"
V0 = 11.0            # m/s
ETA = 5.0e8          # m^2/s = 500 km^2/s
TAU_YEARS = 10.0     # flux-decay e-folding time
_YEAR = 365.25 * 86400.0


def main(catalogue, start="2010-05-01", end="2026-06-30", flux_scale=1.0,
         v0=V0, eta=ETA, tau_years=TAU_YEARS, flow_profile=FLOW_PROFILE,
         cap_deg=30.0, use_observed_map=True, n_lat=91, n_lon=180):
    grid = create_grid(n_lat, n_lon)
    mf = meridional_flow(grid, peak_speed=v0, profile=flow_profile)
    dr = differential_rotation(grid)
    tau_s = tau_years * _YEAR if tau_years else None
    src = SHARPSource(catalogue, start_date=start, end_date=end, flux_scale=flux_scale)
    print(src.summary(),
          f"| flux_scale={flux_scale}, flow={flow_profile}, v0={v0:g} m/s, "
          f"eta={eta / 1e6:.0f} km^2/s, tau={tau_years:g} yr")

    if use_observed_map:
        from sft2d.data import HMI_SYNOPTIC_FITS
        field = initialize_field(grid, "read", path=str(HMI_SYNOPTIC_FITS))
        print(f"initial condition: observed HMI synoptic map ({HMI_SYNOPTIC_FITS.name})")
    else:
        field = initialize_field(grid, "dipole") * (-2.0)
        print("initial condition: scaled analytic dipole (N negative)")
    y0 = int(start[:4])
    yr, pn, ps, dip = [], [], [], []

    class Rec:
        def record(self, day, B):
            if day % 10:
                return
            yr.append(y0 + 0.33 + day / 365.25)
            n, s = calculate_polar_field(B, grid, pol_cap_extent_deg=cap_deg)
            pn.append(n); ps.append(s); dip.append(calculate_dm(B, grid))

    t0 = time.time()
    evolve(field, grid, mf, dr, eta, src.num_days, source=src,
           tau_decay_s=tau_s, recorder=Rec())
    print(f"{src.num_days / 365.25:.1f} yr run in {time.time() - t0:.0f}s")

    yr = np.array(yr); pn = np.array(pn); ps = np.array(ps); dip = np.array(dip)

    hmi = load_hmi_polar_field()
    idx = hmi["mean_north"].index
    hyr = idx.year + (idx.dayofyear - 1) / 365.25
    mn, ms = hmi["mean_north"].values, hmi["mean_south"].values
    sn, ss = hmi["std_north"].values, hmi["std_south"].values

    def report(name, model, obs):
        o = np.interp(yr, hyr, obs)
        r = np.corrcoef(model, o)[0, 1]
        scale = np.sum(model * o) / np.sum(model * model)
        rms = np.sqrt(np.mean((scale * model - o) ** 2))
        print(f"  {name}: corr {r:+.3f} | best-fit scale {scale:.3f} | "
              f"RMS after scaling {rms:.2f} G")
        return r, scale

    print("\n--- axial dipole moment (unambiguous observable) ---")
    print(f"  model peak |D| = {np.abs(dip).max():.2f} G  (observed cycle-24 ~ 3-4 G)")
    print(f"  model D: {dip[0]:+.2f} -> {dip[-1]:+.2f} G")
    print("\n--- polar-cap field vs HMI (metric-dependent amplitude) ---")
    report("north", pn, mn)
    report("south", ps, ms)

    # ---- figure ----
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax[0].fill_between(hyr, mn - sn, mn + sn, color="C0", alpha=0.2)
    ax[0].fill_between(hyr, ms - ss, ms + ss, color="C2", alpha=0.2)
    ax[0].plot(hyr, mn, "C0", lw=1, label="HMI N")
    ax[0].plot(hyr, ms, "C2", lw=1, label="HMI S")
    ax[0].plot(yr, pn, "C0--", lw=2, label="model N")
    ax[0].plot(yr, ps, "C2--", lw=2, label="model S")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel(f"polar-cap field  (>{90 - cap_deg:g} deg) [G]")
    ax[0].set_title(f"SHARPS-driven SFT vs HMI  (flux_scale={flux_scale:g}, {src.n_regions} regions)")
    ax[0].legend(ncol=2, fontsize=8)

    ax[1].plot(yr, dip, "C3", lw=2, label="model axial dipole moment")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axhspan(3, 4, color="0.7", alpha=0.4, label="observed cycle-24 peak ~3-4 G")
    ax[1].axhspan(-4, -3, color="0.7", alpha=0.4)
    ax[1].set_xlabel("year"); ax[1].set_ylabel("axial dipole [G]")
    ax[1].legend(fontsize=8)

    out = "calibrate_sharps_vs_hmi.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    print("\nsaved", out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    catalogue = args[0] if args else DEFAULT_CATALOGUE
    if not os.path.exists(catalogue):
        print(__doc__)
        raise SystemExit(f"ERROR: catalogue not found: {catalogue}\n"
                         "Pass a path or set SFT2D_SHARPS_CATALOGUE.")
    main(catalogue, use_observed_map=("--dipole" not in sys.argv))
