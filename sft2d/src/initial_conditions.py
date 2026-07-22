"""
initial_conditions.py

Initial radial-field maps on the pole-to-pole finite-volume mesh.

Functions:
    - initialize_field: axisymmetric dipole, or an observed synoptic map.
    - correct_flux_multiplicative: area-weighted flux balancing.
"""

from __future__ import annotations

import numpy as np

from .grid import polar_average


def initialize_field(grid, field_type="dipole", path=None, balance=True):
    """Build an initial radial field on ``grid``.

    Parameters
    ----------
    grid : dict
        Grid from :func:`~sft2d.src.grid.create_grid`.
    field_type : {'dipole', 'read'}
        ``'dipole'`` gives the usual ``sin(lat)|sin(lat)|^7`` axisymmetric
        profile (amplitude 1 G; scale it yourself).  ``'read'`` interpolates a
        sine-latitude synoptic FITS map onto the grid.
    path : str, optional
        FITS file for ``field_type='read'``.  Required for that mode; the old
        version hard-coded a relative path that only worked from the repo root.
    balance : bool
        Remove any net flux, area-weighted.

    Returns
    -------
    (n_theta, n_phi) ndarray
    """
    theta = grid["colatitude"]
    phi = grid["longitude"]

    if field_type == "dipole":
        lat = 0.5 * np.pi - theta
        prof = np.abs(np.sin(lat)) ** 7 * np.sin(lat)
        B_init = np.repeat(prof[:, None], phi.size, axis=1)

    elif field_type == "read":
        if path is None:
            raise ValueError("field_type='read' requires an explicit `path`")
        from astropy.io import fits
        from scipy.interpolate import RegularGridInterpolator as rgi

        hmi_br = np.asarray(fits.getdata(path), dtype=float)[::-1, :]
        nsm, npm = hmi_br.shape
        dsm = 2.0 / nsm

        # Source map is uniform in sine of latitude; leave a small gap at the
        # poles so the interpolator stays inside its domain.
        scm = np.flip(np.arccos(np.linspace(-1 + 0.05 * dsm, 1 - 0.05 * dsm, nsm)))
        pcm = np.linspace(0.0, 2.0 * np.pi, npm)

        bri = rgi((scm, pcm), hmi_br, method="linear",
                  bounds_error=False, fill_value=None)
        TH, PH = np.meshgrid(theta, phi, indexing="ij")
        # Vectorised: the old version looped over every cell in Python, which
        # cost minutes at production resolution.
        B_init = bri(np.stack([TH.ravel(), PH.ravel()], axis=-1)).reshape(TH.shape)

    else:
        raise ValueError("field_type must be 'dipole' or 'read'")

    polar_average(B_init)
    if balance:
        B_init = correct_flux_multiplicative(B_init, grid)
    return B_init


def correct_flux_multiplicative(f, grid):
    """Rescale each polarity so the map carries zero net flux.

    Weighted by the true cell areas.  The previous version explicitly assumed
    "cells have equal area", which a latitude-longitude mesh does not: it
    over-weighted the poles by up to ``1/sin(theta)``, so the correction itself
    introduced an imbalance.
    """
    w = np.broadcast_to(grid["area_cm2"], f.shape)
    ipos, ineg = f > 0, f < 0
    fluxp = float(np.sum(f[ipos] * w[ipos]))
    fluxn = float(np.abs(np.sum(f[ineg] * w[ineg])))
    if fluxp == 0.0 or fluxn == 0.0:
        return f
    fluxmn = 0.5 * (fluxp + fluxn)
    f1 = f.copy()
    f1[ipos] *= fluxmn / fluxp
    f1[ineg] *= fluxmn / fluxn
    return polar_average(f1)
