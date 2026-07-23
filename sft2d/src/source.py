"""
source.py

Bipolar Magnetic Region (BMR) source term for the SFT model, implementing the
prescription documented in ``docs/sft2d-theory.md``.

A BMR is a pair of opposite-polarity Gaussian spots placed on the sphere,
separated by an angular distance ``sep``, tilted from the local east-west
direction by ``tilt`` (Joy's law), with polarity signs following Hale's law.
The pair is normalised so that its total *unsigned* flux equals a requested
value in Maxwell.

Positions are built with the 3-D tangent-basis construction from the theory
doc; the Gaussian is evaluated with the true great-circle (heliocentric) angle
between each grid point and each polarity centre, which keeps the spots
undistorted at high latitude (a plain (theta, phi) Gaussian is stretched in
longitude near the poles).

Public functions
----------------
make_bmr(...)       -> 2-D field increment for one BMR (Mx-normalised)
insert_bmr(...)     -> add one BMR to a field (optionally re-balancing flux)
joys_law_tilt(lat)  -> tilt angle from Joy's law
hale_leading_sign(lat, cycle) -> +/-1 leading polarity
"""

from __future__ import annotations

import numpy as np

from .constants import R_SUN_CM  # noqa: F401  (re-exported for compatibility)
from .grid import polar_average


# ---------------------------------------------------------------------------
# Empirical laws
# ---------------------------------------------------------------------------
def joys_law_tilt(lat_deg, coeff=0.5, form="linear"):
    """Joy's law tilt angle [deg] for a BMR at latitude ``lat_deg``.

    form='linear'  : tilt = coeff * lat                (common simple form)
    form='sqrt'    : sin(tilt) = coeff * sqrt(cos(colat))  ~ 32 deg * sqrt(cos)
    The sign follows the hemisphere (leading polarity equatorward).
    """
    lat = np.asarray(lat_deg, float)
    if form == "linear":
        tilt = coeff * lat
    elif form == "sqrt":
        lam = np.radians(np.abs(lat))
        tilt = np.degrees(np.arcsin(np.clip(coeff * np.sqrt(np.cos(np.pi/2 - lam)), -1, 1)))
        tilt = np.sign(lat) * tilt
    else:
        raise ValueError("form must be 'linear' or 'sqrt'")
    return tilt


def hale_leading_sign(lat_deg, cycle_number):
    """Sign (+/-1) of the leading (equatorward) polarity following Hale's law.

    This sets the *absolute* polarity, so it is anchored to the observed cycles,
    not left arbitrary: in **odd** cycles the northern-hemisphere leading
    polarity is **positive**, in **even** cycles it is **negative** (the south is
    opposite).  Cycle 23 (odd) -> N leading +; cycle 24 (even) -> N leading -.

    Getting this backwards inverts which polarity the meridional flow carries to
    the poles, so the modelled polar field reverses to the *wrong sign* relative
    to observations even though the transport is correct.  (The earlier version
    had the parity flipped and treated the absolute sign as arbitrary; with real
    RGO input driving a comparison against HMI it is not arbitrary.)

    The following (poleward) polarity, which actually sets the reversed polar
    field, is the opposite sign.
    """
    north = +1 if (cycle_number % 2 == 1) else -1
    return north if lat_deg >= 0 else -north


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _unit_vector(lat, lon):
    return np.array([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)])


def _polarity_centres(lat_deg, lon_deg, tilt_deg, sep_deg):
    """Return (lat,lon) of leading and following polarity centres [radians].

    ``tilt_deg`` is *signed* with the standard Joy's-law convention: positive in
    the northern hemisphere, negative in the southern, so that the leading
    polarity sits equatorward and the following polarity poleward in BOTH
    hemispheres.  The bipole axis is

        s = cos(tilt) * e_east + sin(tilt) * e_north

    with the following polarity at ``r0 + (sep/2) s`` and the leading polarity at
    ``r0 - (sep/2) s``.  No explicit hemisphere factor is needed -- the sign of
    the tilt already carries it.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg % 360.0)
    alpha = np.radians(tilt_deg)
    half = np.radians(sep_deg) / 2.0

    r0 = _unit_vector(lat, lon)
    e_phi = np.array([-np.sin(lon), np.cos(lon), 0.0])          # eastward
    e_lam = np.array([-np.sin(lat) * np.cos(lon),
                      -np.sin(lat) * np.sin(lon), np.cos(lat)])  # northward
    s = np.cos(alpha) * e_phi + np.sin(alpha) * e_lam           # bipole axis

    r_foll = r0 + half * s        # following polarity: poleward
    r_lead = r0 - half * s        # leading polarity: equatorward

    def to_latlon(r):
        r = r / np.linalg.norm(r)
        return np.arcsin(np.clip(r[2], -1, 1)), np.arctan2(r[1], r[0]) % (2*np.pi)

    return to_latlon(r_lead), to_latlon(r_foll)


def _great_circle(TH, PH, lat_c, lon_c):
    """Heliocentric angle between grid (colat TH, lon PH) and a centre."""
    latg = np.pi/2 - TH
    cosb = (np.sin(latg) * np.sin(lat_c)
            + np.cos(latg) * np.cos(lat_c) * np.cos(PH - lon_c))
    return np.arccos(np.clip(cosb, -1, 1))


# ---------------------------------------------------------------------------
# BMR field
# ---------------------------------------------------------------------------
def make_bmr(grid, lat_deg, lon_deg, flux_mx, tilt_deg=None, sep_deg=6.0,
             sigma_deg=4.0, cycle_number=24, hale=True, joy_coeff=0.5):
    """Return the 2-D radial-field increment [G] of one flux-normalised BMR.

    Parameters
    ----------
    grid : grid dict from create_grid
    lat_deg, lon_deg : BMR centre (latitude, Carrington longitude) [deg]
    flux_mx : requested total unsigned flux [Mx]
    tilt_deg : tilt angle [deg]; if None, taken from Joy's law
    sep_deg : polarity separation [deg]
    sigma_deg : Gaussian 1-sigma of each polarity [deg]
    cycle_number : sets Hale leading-polarity sign (if hale=True)
    hale : apply Hale's law; if False leading polarity is +1
    """
    theta = grid["colatitude"]
    phi = grid["longitude"]
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    if tilt_deg is None:
        tilt_deg = joys_law_tilt(lat_deg, coeff=joy_coeff)

    (lat_l, lon_l), (lat_f, lon_f) = _polarity_centres(lat_deg, lon_deg,
                                                       tilt_deg, sep_deg)
    sigma = np.radians(sigma_deg)
    b_lead = _great_circle(TH, PH, lat_l, lon_l)
    b_foll = _great_circle(TH, PH, lat_f, lon_f)

    s_lead = 1
    if hale:
        s_lead = hale_leading_sign(lat_deg, cycle_number)

    B_unit = (s_lead * np.exp(-0.5 * (b_lead / sigma) ** 2)
              - s_lead * np.exp(-0.5 * (b_foll / sigma) ** 2))

    # Flux normalisation against the true finite-volume cell areas.  The old
    # code used dA = R^2 sin(theta) dtheta dphi, which is the small-angle
    # approximation to the cell area and is wrong for the polar caps, so a BMR
    # emerging at high latitude did not carry the flux it was asked for.
    phi_unit = np.sum(np.abs(B_unit) * grid["area_cm2"])
    if phi_unit <= 0:
        return np.zeros_like(B_unit)
    scale = flux_mx / phi_unit
    B_unit *= scale
    return polar_average(B_unit)


def make_bmr_yeates(grid, lat_deg, lon_deg, flux_mx, sep_deg, tilt_deg,
                    width_frac=0.56, xi_thresh=9.0):
    """Return the 2-D radial-field increment [G] of one Yeates-style bipole.

    This is an *alternative* BMR shape to :func:`make_bmr`.  Instead of two
    separate great-circle Gaussians, it uses the single smooth antisymmetric
    bipole of Yeates (2020, and the ``sharps-bmrs`` catalogue), which has a
    continuous polarity-inversion line and is elongated in longitude.  In a
    frame rotated so the bipole sits on the equator, untilted,

        B_r = -B0 * (phi_b / w) * exp(-xi),
        xi  = (phi_b^2 + 2 * lat_b^2) / w^2,     zeroed where xi > xi_thresh,

    with ``w = width_frac * sep`` the Gaussian scale and ``(phi_b, lat_b)`` the
    longitude/latitude in the bipole frame.  The factor of 2 on ``lat_b`` makes
    the region narrower in latitude than longitude.  ``B0`` is then set by the
    requested unsigned flux.

    This is an independent implementation of the published model; it is not
    derived from the GPL ``sharps-bmrs`` source.  It is the natural shape to use
    when driving from a SHARPS catalogue (:class:`sft2d.src.sharp_driver.SHARPSource`)
    whose ``sep``/``tilt`` are fitted from real HMI magnetograms, so no Hale or
    Joy assumption is applied -- the fitted, signed ``tilt_deg`` carries the
    orientation.

    Parameters
    ----------
    grid : dict
        Grid from :func:`sft2d.src.grid.create_grid`.
    lat_deg, lon_deg : float
        Bipole centre (latitude, Carrington longitude) [deg].
    flux_mx : float
        Requested total unsigned flux [Mx].
    sep_deg : float
        Fitted polarity separation [deg] (the catalogue ``Bip-Separation``).
    tilt_deg : float
        Fitted, signed tilt [deg] (the catalogue ``Bip-Tilt``).
    width_frac : float
        Gaussian scale as a fraction of ``sep_deg`` (Yeates uses 0.56).
    xi_thresh : float
        Truncate the profile where ``xi`` exceeds this (Yeates uses 9).
    """
    theta = grid["colatitude"]
    phi = grid["longitude"]
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    lat0 = np.radians(lat_deg)
    lon0 = np.radians(lon_deg % 360.0)
    tilt0 = np.radians(tilt_deg)
    w = np.radians(width_frac * sep_deg)
    if w <= 0.0:
        return np.zeros_like(TH)

    # Grid points as unit vectors.
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)

    # Rotate into the bipole frame (bipole on the equator, untilted).  Same
    # rotation as the published model: first bring the centre to (0,0), then
    # rotate by the tilt about the local radial.
    cl, sl = np.cos(lat0), np.sin(lat0)
    co, so = np.cos(lon0), np.sin(lon0)
    ct, st = np.cos(tilt0), np.sin(tilt0)

    xb = x * cl * co + y * cl * so + z * sl
    yb = (x * (-ct * so + st * sl * co)
          + y * (ct * co + st * sl * so)
          - z * st * cl)
    zb = (x * (-st * so - ct * sl * co)
          + y * (st * co - ct * sl * so)
          + z * ct * cl)
    zb = np.clip(zb, -1.0, 1.0)

    lat_b = 0.5 * np.pi - np.arccos(zb)
    lon_b = np.arctan2(yb, xb)

    xi = (lon_b ** 2 + 2.0 * lat_b ** 2) / w ** 2
    B_unit = -(lon_b / w) * np.exp(-xi)
    B_unit[xi > xi_thresh] = 0.0

    phi_unit = np.sum(np.abs(B_unit) * grid["area_cm2"])
    if phi_unit <= 0:
        return np.zeros_like(B_unit)
    B_unit *= flux_mx / phi_unit
    # The tilted profile carries a small residual net flux on a discrete grid;
    # remove it (Yeates does the same) so many insertions do not accumulate a
    # spurious monopole.
    B_unit = balance_flux(B_unit, grid)
    return polar_average(B_unit)


def insert_bmr(field, grid, lat_deg, lon_deg, flux_mx, rebalance=False, **kw):
    """Add one BMR to ``field`` in place and return it.

    If ``rebalance`` is True, the whole map's positive/negative flux is rescaled
    to zero net flux afterwards (useful to suppress monopole drift)."""
    field += make_bmr(grid, lat_deg, lon_deg, flux_mx, **kw)
    if rebalance:
        field[:] = balance_flux(field, grid)
    return polar_average(field)


def balance_flux(f, grid):
    """Rescale each polarity so the map carries zero net flux.

    Weighted by cell area.  The previous version counted cells, which is only
    correct on an equal-area mesh; on a latitude-longitude grid it
    over-weighted the poles, so "balancing" a map actually introduced an
    imbalance of order the polar-cap area fraction.
    """
    w = np.broadcast_to(grid["area_cm2"], f.shape)
    ipos, ineg = f > 0, f < 0
    fp = float(np.sum(f[ipos] * w[ipos]))
    fn = float(np.abs(np.sum(f[ineg] * w[ineg])))
    if fp == 0.0 or fn == 0.0:
        return f
    m = 0.5 * (fp + fn)
    g = f.copy()
    g[ipos] *= m / fp
    g[ineg] *= m / fn
    return g
