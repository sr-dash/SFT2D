"""
analysis.py

Derived quantities (unsigned flux, axial dipole moment, polar field and polar
flux) from SFT2D output.

Every integral here is now a plain area-weighted sum against ``grid['area']``,
the exact finite-volume cell area.  The previous versions had three separate
problems:

* they used ``dA = R^2 sin(theta) dtheta dphi``, the small-angle approximation
  to the cell area, which is wrong for the polar caps;
* they summed over a duplicated ``phi = 2*pi`` column that the old grid carried,
  double-counting one meridian (~0.55% at n_phi=180);
* ``calculate_polar_field`` took an unweighted ``np.mean`` over the cap, which
  weights a 0.5-deg-wide polar ring the same as an 80-deg one.

They also disagreed on the solar radius (6.98e10 cm here against 6.955e10 cm in
``source.py``), a further 0.7% in every flux.  All of them now take it from
``sft2d.src.constants``.

All functions accept either a single 2-D map ``(n_theta, n_phi)`` or a stack
``(n_time, n_theta, n_phi)``, and return a scalar or an array to match.  The old
``time_duration`` tuple argument is gone -- slice the array yourself.
"""

from __future__ import annotations

import numpy as np


def _as_stack(b):
    """Return (stack, was_2d) with stack shaped (n_time, n_theta, n_phi)."""
    b = np.asarray(b, dtype=float)
    if b.ndim == 2:
        return b[None], True
    if b.ndim == 3:
        return b, False
    raise ValueError("field must be 2-D (map) or 3-D (time, theta, phi)")


def _reduce(values, was_2d):
    return float(values[0]) if was_2d else values


def calculate_usflx(all_br_data, grid):
    """Total unsigned flux [Mx]."""
    b, was_2d = _as_stack(all_br_data)
    out = np.sum(np.abs(b) * grid["area_cm2"][None], axis=(1, 2))
    return _reduce(out, was_2d)


def calculate_net_flux(all_br_data, grid):
    """Net (signed) flux [Mx].

    Should stay at zero to round-off for a conservative run with balanced
    sources; useful as a running check that nothing is leaking.
    """
    b, was_2d = _as_stack(all_br_data)
    out = np.sum(b * grid["area_cm2"][None], axis=(1, 2))
    return _reduce(out, was_2d)


def calculate_dm(all_br_data, grid):
    """Axial dipole moment [G].

    ``(3/4pi) * integral of B_r cos(theta) dOmega``, the standard normalisation
    in which a field ``B_r = D cos(theta)`` returns ``D``.
    """
    b, was_2d = _as_stack(all_br_data)
    w = grid["area"] * grid["cos_theta"][:, None]
    sphere_area = float(np.sum(grid["area"])) * grid["n_phi"]   # = 4*pi*R^2
    out = 3.0 * np.sum(b * w[None], axis=(1, 2)) / sphere_area
    return _reduce(out, was_2d)


def calculate_polar_field(all_br_data, grid, pol_cap_extent_deg=20.0):
    """Area-weighted mean radial field in each polar cap [G].

    ``pol_cap_extent_deg`` is the cap's angular radius measured from the pole,
    so 20 means latitudes poleward of +/-70 deg.
    """
    b, was_2d = _as_stack(all_br_data)
    an, as_ = cap_areas(grid, pol_cap_extent_deg)
    bn = np.sum(b * an[None], axis=(1, 2)) / (float(np.sum(an)) * grid["n_phi"])
    bs = np.sum(b * as_[None], axis=(1, 2)) / (float(np.sum(as_)) * grid["n_phi"])
    return _reduce(bn, was_2d), _reduce(bs, was_2d)


def calculate_polar_flux(all_br_data, grid, pol_cap_extent_deg=20.0):
    """Signed magnetic flux in each polar cap [Mx].

    ``pol_cap_extent_deg`` is the cap's angular radius from the pole (20 means
    poleward of +/-70 deg latitude).
    """
    b, was_2d = _as_stack(all_br_data)
    an, as_ = cap_areas(grid, pol_cap_extent_deg, cm=True)
    fn = np.sum(b * an[None], axis=(1, 2))
    fs = np.sum(b * as_[None], axis=(1, 2))
    return _reduce(fn, was_2d), _reduce(fs, was_2d)


def cap_areas(grid, pol_cap_extent_deg, cm=False):
    """Per-cell areas of the north and south polar caps, as ``(n_theta, 1)``.

    Cells straddling the cap edge contribute only the part of themselves that
    lies inside it.  Selecting whole cells with a boolean mask instead -- what
    this module used to do -- makes the cap boundary snap to the nearest cell
    edge, so the integration region moves with resolution: a 20-deg cap on a
    4-deg mesh can come out 20% too large or too small, and a "polar flux"
    calibration target computed that way is not comparable between grids.

    Partial-cell weighting is exact for the spherical cap, so the result
    converges at O(dtheta^2) and is stable across resolutions.
    """
    from ..src.constants import R_SUN_M

    th = grid["theta_face"]
    edge = np.deg2rad(pol_cap_extent_deg)
    r2 = R_SUN_M ** 2 * (1.0e4 if cm else 1.0)

    # North: overlap of [th_j, th_{j+1}] with [0, edge].
    hi = np.minimum(th[1:], edge)
    lo = np.minimum(th[:-1], edge)
    north = r2 * grid["dphi"] * (np.cos(lo) - np.cos(hi))

    # South: overlap with [pi - edge, pi].
    lo_s = np.maximum(th[:-1], np.pi - edge)
    hi_s = np.maximum(th[1:], np.pi - edge)
    south = r2 * grid["dphi"] * (np.cos(lo_s) - np.cos(hi_s))

    return north[:, None], south[:, None]
