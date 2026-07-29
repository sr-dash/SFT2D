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


# --------------------------------------------------------------- patch cache --
def build_patch_cache(catalogue, nc_dir, grid, out_path, start_date, end_date=None,
                      good_only=True, dedupe_max_flux=True, var="br", n_jobs=1,
                      verbose=True):
    """Precompute every AR patch on ``grid`` and save a compact sparse cache.

    Each region's map is interpolated, flux-balanced and normalised to **unit**
    unsigned flux, so the cache is independent of ``flux_scale``: insertion just
    multiplies by that region's target flux.

    A parameter sweep runs the same regions on the same grid for every member, so
    without a cache each member re-opens all ``sharpNNNNN.nc`` files and repeats
    identical interpolation -- ~3 GB across ~3000 file opens per member.  On a
    shared cluster filesystem with tens of concurrent workers that redundant I/O
    can dominate the run.  Build this once, then drive every member from it with
    :class:`CachedPatchSource`.

    Returns the path written.
    """
    src = SHARPPatchSource(catalogue, nc_dir=nc_dir, start_date=start_date,
                           end_date=end_date, flux_scale=1.0, good_only=good_only,
                           dedupe_max_flux=dedupe_max_flux, var=var)
    ids = sorted({sid for evs in src.events.values() for sid, _ in evs})

    def unit_patch(sharp_id):
        """Patch normalised to unit unsigned flux, as (indices, values) or None."""
        p = src.patch_on_grid(sharp_id, grid, 1.0)      # target 1 -> unit usflx
        if p is None:
            return None
        nz = np.flatnonzero(p)                          # balance_flux rescales,
        return nz.astype(np.int32), p.flat[nz]          # so zeros stay zero

    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            delayed(unit_patch)(s) for s in ids)
    else:
        out = [unit_patch(s) for s in ids]

    kept, idx_parts, val_parts, offsets = [], [], [], [0]
    missing = 0
    for sid, res in zip(ids, out):
        if res is None:
            missing += 1
            continue
        i, v = res
        kept.append(sid); idx_parts.append(i); val_parts.append(v)
        offsets.append(offsets[-1] + i.size)

    idx = (np.concatenate(idx_parts) if idx_parts
           else np.zeros(0, dtype=np.int32))
    val = (np.concatenate(val_parts) if val_parts
           else np.zeros(0, dtype=float))
    np.savez_compressed(
        out_path,
        ids=np.asarray(kept, dtype=np.int64), offsets=np.asarray(offsets, dtype=np.int64),
        idx=idx, val=val,
        shape=np.asarray([grid["n_theta"], grid["n_phi"]], dtype=np.int64),
        var=str(var),
    )
    if verbose:
        from pathlib import Path as _P
        print(f"patch cache: {len(kept)} regions ({missing} missing .nc), "
              f"{idx.size} nonzero cells, {_P(out_path).stat().st_size/1e6:.1f} MB "
              f"-> {out_path}")
    return out_path


class CachedPatchSource(SHARPPatchSource):
    """:class:`SHARPPatchSource` driven from a precomputed cache.

    Produces the same inserted field as the uncached driver, without touching the
    ``.nc`` files at run time.  The cache is grid-specific: a mismatch between the
    cache and the model grid raises immediately rather than silently inserting
    the wrong thing.
    """

    def __init__(self, cache_path, catalogue, start_date, end_date=None,
                 flux_scale=1.0, good_only=True, dedupe_max_flux=True,
                 rebalance=False):
        super().__init__(catalogue, nc_dir=Path(cache_path).parent,
                         start_date=start_date, end_date=end_date,
                         flux_scale=flux_scale, good_only=good_only,
                         dedupe_max_flux=dedupe_max_flux, rebalance=rebalance)
        z = np.load(cache_path)
        self.cache_path = str(cache_path)
        self.cache_shape = tuple(int(v) for v in z["shape"])
        self._pos = {int(s): i for i, s in enumerate(z["ids"])}
        self._off = np.asarray(z["offsets"])
        self._idx = np.asarray(z["idx"])
        self._val = np.asarray(z["val"])
        self.n_cached = len(self._pos)
        self.n_uncached = sum(1 for evs in self.events.values()
                              for sid, _ in evs if sid not in self._pos)

    def _check_grid(self, grid):
        got = (grid["n_theta"], grid["n_phi"])
        if got != self.cache_shape:
            raise ValueError(
                f"patch cache was built for a {self.cache_shape[0]}x{self.cache_shape[1]} "
                f"grid but this run uses {got[0]}x{got[1]}; rebuild the cache "
                f"(build_patch_cache) for this grid")

    def patch_on_grid(self, sharp_id, grid, target_flux):
        self._check_grid(grid)
        i = self._pos.get(int(sharp_id))
        if i is None:
            return None
        a, b = self._off[i], self._off[i + 1]
        patch = np.zeros(self.cache_shape, dtype=float)
        patch.flat[self._idx[a:b]] = self._val[a:b] * target_flux
        return patch

    def __call__(self, day, field, grid):
        evs = self.events.get(day)
        if not evs:
            return field
        self._check_grid(grid)
        flat = field.reshape(-1)
        for sharp_id, flux in evs:
            i = self._pos.get(int(sharp_id))
            if i is None:
                continue
            a, b = self._off[i], self._off[i + 1]
            # cache indices are unique within a region, so plain fancy-index
            # accumulation is correct (no np.add.at needed) and much faster
            flat[self._idx[a:b]] += self._val[a:b] * flux
        if self.rebalance:
            field[:] = balance_flux(field, grid)
        return field

    def summary(self):
        return (f"CachedPatchSource: {self.n_regions} regions "
                f"({self.n_cached} cached, {self.n_uncached} without a cached patch), "
                f"{self.cache_shape[0]}x{self.cache_shape[1]}, "
                f"{self.start.date()} -> {self.end.date()} ({self.num_days} days)")
