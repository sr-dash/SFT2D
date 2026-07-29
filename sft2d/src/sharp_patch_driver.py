"""
sharp_patch_driver.py

Drive the SFT model by inserting the **observed** active-region magnetic field
directly, instead of approximating each region with an idealized bipole.

Where :class:`sft2d.src.sharp_driver.SHARPSource` reconstructs every region as a
smooth two-parameter bipole (:func:`sft2d.src.source.make_bmr_yeates`), this
driver reads the real ``B_r`` map of each region from the ``sharpNNNNN.nc`` files
of the SHARPS database and inserts it as-is.  The idealization discards the
region's internal structure (multipolar cores, asymmetric polarities, sprawling
complexes); direct insertion keeps it, at the cost of needing the per-region maps
and of the smoothing incurred when the SHARP grid is interpolated onto a coarser
model grid.

Each ``.nc`` holds a full-sphere ``br`` map on the SHARP computational grid --
uniform in ``s = sin(latitude)`` (``sdim`` x ``pdim``, typically 180 x 360) --
non-zero only over the region, at its Carrington location.  Insertion:

1. interpolate ``br`` from the ``(s, phi)`` SHARP grid onto the model grid
   (periodic in longitude);
2. remove any net flux (area-weighted balance), as for the idealized bipoles;
3. rescale the unsigned flux to the catalogue value (times ``flux_scale``) so the
   *total* inserted flux is preserved despite the interpolation, and matches what
   ``SHARPSource`` would inject for the same region.

Timing, region selection and de-duplication reuse the catalogue reader, so a
patch-driven run is set up exactly like a bipole-driven one::

    src = SHARPPatchSource("bmrsharps_evol.txt", nc_dir="<db>", start_date="2010-05-01")
    evolve(field0, grid, mf, dr, eta, src.num_days, source=src, ...)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .sharp_driver import _read_catalogue
from .source import balance_flux


class SHARPPatchSource:
    """Callable ``source(day, field, grid)`` that inserts observed AR patches.

    Parameters
    ----------
    catalogue : str or pandas.DataFrame
        A ``bmrsharps_evol`` catalogue (used for timing, region ids and flux).
    nc_dir : str or Path
        Directory holding the ``sharpNNNNN.nc`` region maps.
    start_date, end_date : str or datetime
        Simulation window (as in :class:`SHARPSource`).
    flux_scale : float
        Multiplies each region's unsigned flux (keep ~1 for measured SHARPS flux).
    good_only, dedupe_max_flux : bool
        Passed through to the catalogue reader / selection.
    rebalance : bool
        Re-zero the whole-map net flux on days that inject flux.
    var : str
        Which ``.nc`` variable to insert: ``"br"`` (real patch, default) or
        ``"br_bipole"`` (the database's own fitted bipole, for cross-checks).
    """

    def __init__(self, catalogue, nc_dir, start_date, end_date=None, flux_scale=1.0,
                 good_only=True, dedupe_max_flux=True, rebalance=False, var="br"):
        import pandas as pd

        if isinstance(catalogue, str):
            df = _read_catalogue(catalogue, dedupe_max_flux=dedupe_max_flux)
        else:
            df = catalogue.copy()
        if good_only and "good" in df:
            df = df[df["good"] == 1]
        if "sharp" not in df:
            raise ValueError("catalogue has no SHARP id column; cannot locate .nc maps")

        self.nc_dir = Path(nc_dir)
        self.flux_scale = flux_scale
        self.rebalance = rebalance
        self.var = var

        self.start = pd.Timestamp(start_date)
        self.end = pd.Timestamp(end_date) if end_date is not None else df["date"].max()
        df = df[(df["date"] >= self.start) & (df["date"] <= self.end)].reset_index(drop=True)
        self.num_days = int((self.end - self.start).days)

        self.events = {}
        n_used = 0
        for _, r in df.iterrows():
            flux = float(r["flux"]) * flux_scale
            if flux <= 0:
                continue
            day = int((r["date"] - self.start).days)
            self.events.setdefault(day, []).append((int(r["sharp"]), flux))
            n_used += 1
        self.n_regions = n_used
        self._itp_cache = None            # (grid id, s_tgt, ph_tgt) reuse

    # ------------------------------------------------------------------
    def _read_patch(self, sharp_id):
        """Return the (n_s, n_phi) sin-lat map for a SHARP, or None if missing."""
        from scipy.io import netcdf_file

        f = self.nc_dir / f"sharp{sharp_id:05d}.nc"
        if not f.exists():
            return None
        fh = netcdf_file(str(f), "r", mmap=False)
        if self.var not in fh.variables:
            fh.close()
            return None
        br = fh.variables[self.var][:].copy()
        fh.close()
        return np.asarray(br, dtype=float)

    def _to_grid(self, br_src, grid):
        """Interpolate a sin-lat SHARP map onto the model grid (periodic in phi)."""
        from scipy.interpolate import RegularGridInterpolator

        ns, nph = br_src.shape
        ds = 2.0 / ns
        s_src = np.linspace(-1 + 0.5 * ds, 1 - 0.5 * ds, ns)          # ascending
        dph = 2.0 * np.pi / nph
        ph_src = np.linspace(0.5 * dph, 2.0 * np.pi - 0.5 * dph, nph)
        # pad phi for periodicity so ARs near the 0/2pi seam interpolate cleanly
        ph_ext = np.concatenate(([ph_src[-1] - 2 * np.pi], ph_src, [ph_src[0] + 2 * np.pi]))
        br_ext = np.concatenate((br_src[:, -1:], br_src, br_src[:, :1]), axis=1)
        itp = RegularGridInterpolator((s_src, ph_ext), br_ext,
                                      bounds_error=False, fill_value=0.0)
        s_tgt = grid["cos_theta"]           # = sin(latitude) on the model grid
        ph_tgt = grid["longitude"]
        ST, PT = np.meshgrid(s_tgt, ph_tgt, indexing="ij")
        return itp(np.stack([ST.ravel(), PT.ravel()], axis=-1)).reshape(ST.shape)

    def patch_on_grid(self, sharp_id, grid, target_flux):
        """Observed AR patch on the model grid: balanced, flux-normalised.

        Returns ``None`` if the region's ``.nc`` map is unavailable.
        """
        br = self._read_patch(sharp_id)
        if br is None:
            return None
        patch = self._to_grid(br, grid)
        patch = balance_flux(patch, grid)                 # remove net flux
        us = float(np.sum(np.abs(patch) * grid["area_cm2"]))
        if us > 0 and target_flux > 0:
            patch *= target_flux / us                     # preserve total flux
        return patch

    def __call__(self, day, field, grid):
        evs = self.events.get(day)
        if not evs:
            return field
        for sharp_id, flux in evs:
            patch = self.patch_on_grid(sharp_id, grid, flux)
            if patch is not None:
                field += patch
        if self.rebalance:
            field[:] = balance_flux(field, grid)
        return field

    def summary(self):
        return (f"SHARPPatchSource: {self.n_regions} regions ({self.var}), "
                f"{self.start.date()} -> {self.end.date()} ({self.num_days} days)")
