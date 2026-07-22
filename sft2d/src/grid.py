"""
grid.py

Pole-to-pole finite-volume mesh in spherical coordinates.

The previous version clipped ``leave_out=4`` coarse cells off each pole, so the
domain reached only +/-82 deg at 90x180 and +/-86 deg at 180x360.  Because the
clip is a fixed number of *cells*, the missing polar cap shrank with resolution:
16% of the area of the >70 deg polar cap was outside the domain at 90x180
against 4% at 180x360.  Any quantity integrated over the polar cap -- which is
exactly the calibration target -- therefore depended on the grid, and parameters
fitted at one resolution did not transfer to another.

This version carries the poles as genuine cells, so no flux leaves the domain.

Layout
------
Colatitude is a **cell-centred** mesh whose first and last points sit exactly on
the poles::

    t[0] = 0            north pole  (a spherical cap, not a ring)
    t[j] = j * dtheta
    t[nt-1] = pi        south pole

The cell faces (the "half mesh") lie midway between centres, with the two polar
cells being caps of angular radius ``dtheta/2``::

    th[0]   = 0
    th[j]   = t[j] - dtheta/2      j = 1 .. nt-1
    th[nt]  = pi

Longitude holds ``np_`` **unique** cells with no duplicated endpoint::

    p[k] = k * dphi,   dphi = 2*pi/np_

Fields are ``(nt, np_)`` arrays.  The old convention carried a duplicated
``phi = 2*pi`` column that every diagnostic silently summed twice (a 1/np_ ~
0.55% error at np_=180) and that the roll-based operators had to work around.
Periodicity is now handled entirely by ``np.roll``.

The two polar rows are physically single cells, so every longitude entry in row
0 (and row nt-1) must hold the same value.  ``polar_average`` enforces that.

Returned keys
-------------
``colatitude``, ``longitude``, ``dtheta``, ``dphi``  (as before)
``theta_face``      cell faces, length nt+1
``sin_theta``, ``cos_theta``, ``sin_theta_face``
``area``            (nt, 1) cell area [m^2], broadcasts over longitude
``area_cm2``        same in cm^2, for fluxes in Maxwell
``n_theta``, ``n_phi``
"""

from __future__ import annotations

import numpy as np

from .constants import R_SUN_M


def create_grid(n_theta, n_phi):
    """Create a pole-to-pole finite-volume mesh.

    Parameters
    ----------
    n_theta : int
        Number of colatitude cells, *including* the two polar caps.  Must be
        at least 3.  Cell centres run 0 .. pi with spacing pi/(n_theta-1).
    n_phi : int
        Number of unique longitude cells.  Spacing is 2*pi/n_phi.

    Returns
    -------
    dict
        Grid description; see the module docstring for the keys.
    """
    n_theta = int(n_theta)
    n_phi = int(n_phi)
    if n_theta < 3:
        raise ValueError("n_theta must be >= 3 (two polar caps plus interior)")
    if n_phi < 4:
        raise ValueError("n_phi must be >= 4")

    dtheta = np.pi / (n_theta - 1)
    dphi = 2.0 * np.pi / n_phi

    theta = np.arange(n_theta) * dtheta
    theta[-1] = np.pi                      # kill round-off at the south pole
    phi = np.arange(n_phi) * dphi

    # Faces: midway between centres, clamped to the poles at the ends.
    theta_face = np.empty(n_theta + 1)
    theta_face[0] = 0.0
    theta_face[-1] = np.pi
    theta_face[1:-1] = 0.5 * (theta[:-1] + theta[1:])

    # Exact cell area from the integral of sin(theta): no small-angle
    # approximation, so the areas sum to 4*pi*R^2 to machine precision.
    area = (R_SUN_M ** 2) * dphi * (np.cos(theta_face[:-1]) - np.cos(theta_face[1:]))

    return {
        "colatitude": theta,
        "longitude": phi,
        "dtheta": dtheta,
        "dphi": dphi,
        "theta_face": theta_face,
        "sin_theta": np.sin(theta),
        "cos_theta": np.cos(theta),
        "sin_theta_face": np.sin(theta_face),
        "area": area[:, None],
        "area_cm2": area[:, None] * 1.0e4,
        "n_theta": n_theta,
        "n_phi": n_phi,
    }


def polar_average(field):
    """Force the two polar rows to be longitude-independent, in place.

    Rows 0 and -1 represent single spherical caps, so they carry one value each.
    The operators produce that automatically, but sources, interpolated initial
    conditions and assimilated maps do not, so call this after modifying a field
    from outside the solver.
    """
    field[0, :] = field[0, :].mean()
    field[-1, :] = field[-1, :].mean()
    return field


def total_flux(field, grid, signed=True):
    """Area-weighted flux over the whole sphere [Mx].

    ``signed=False`` gives the total unsigned flux.
    """
    b = field if signed else np.abs(field)
    return float(np.sum(b * grid["area_cm2"]))
