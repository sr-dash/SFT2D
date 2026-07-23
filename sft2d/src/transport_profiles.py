"""
transport_profiles.py

Surface-flow profiles for the SFT model: meridional circulation and
differential rotation.

Sign convention (read this before worrying about a "wrong" sign)
----------------------------------------------------------------
Colatitude ``theta`` runs 0 at the north pole to ``pi`` at the south pole, so
the ``+theta`` direction points *southward everywhere*.  The solver advects
the radial field with the **colatitude component of the velocity**,
``u_theta`` -- that is what :class:`sft2d.src.operators.Advection` consumes and
what :func:`meridional_flow` returns.

For a normal solar **poleward** meridional flow (positive ``peak_speed``):

    * northern hemisphere: poleward = toward theta=0  => u_theta < 0
    * southern hemisphere: poleward = toward theta=pi => u_theta > 0

So ``meridional_flow`` is **negative in the north and positive in the south**
for a poleward flow.  That looks upside-down only because it is the
*colatitude* velocity.  Expressed as the more intuitive **latitude velocity**
``v_lat`` (northward positive, ``v_lat = -u_theta``) the same poleward flow is
positive in the north and negative in the south -- which is what you expect
from meridional-flow figures in the literature.  Use
:func:`meridional_flow_latitude` when you want to *plot* the flow; use
:func:`meridional_flow` to *drive the model*.

Verified: a flux blob advected with ``meridional_flow(grid, +15)`` migrates
toward the nearest pole in both hemispheres (poleward).  A **negative**
``peak_speed`` reverses this into an unphysical equatorward flow, which will
prevent trailing-polarity flux from ever reaching the poles -- do not use a
negative ``peak_speed`` expecting to "fix" a polar-reversal problem; that is a
flux-amplitude / tilt question, not a flow-direction one.
"""

import numpy as np


def meridional_flow(grid, peak_speed=15.0):
    """Poleward meridional-circulation profile as the colatitude velocity ``u_theta``.

    Parameters
    ----------
    grid : dict
        Grid from :func:`sft2d.src.grid.create_grid`.
    peak_speed : float
        Peak poleward speed [m/s].  **Positive = poleward** (the physical case).
        The profile peaks near +/-45 deg and vanishes at the equator and poles.

    Returns
    -------
    ndarray, shape (n_theta, n_phi)
        ``u_theta`` [m/s]: negative in the north, positive in the south for a
        poleward flow.  Feed this straight into
        :class:`sft2d.src.operators.Advection` / :func:`sft2d.src.stepper.evolve`.
    """
    theta = grid["colatitude"]
    n_phi = grid["longitude"].shape[0]

    # vs() is defined on latitude; the latitude here is (pi/2 - theta).  The
    # extra sign makes the result the *colatitude* velocity u_theta (poleward
    # for positive peak_speed), which is what the advection operator expects.
    v_theta_1d = vs(theta - np.pi / 2, v0=peak_speed)
    return np.tile(v_theta_1d, (n_phi, 1)).T


def meridional_flow_latitude(grid, peak_speed=15.0):
    """Same flow as :func:`meridional_flow`, but as the **latitude velocity** ``v_lat``.

    ``v_lat = -u_theta`` is northward-positive, so a poleward flow is positive
    in the north and negative in the south -- the convention used in most
    meridional-flow figures.  This is a *plotting / inspection* helper; the
    solver is driven with :func:`meridional_flow`.
    """
    return -meridional_flow(grid, peak_speed=peak_speed)


def differential_rotation(grid, rotation="solar", frame="carrington"):
    """Differential-rotation angular velocity ``Omega(theta)`` [rad/s].

    Parameters
    ----------
    grid : dict
        Grid from :func:`sft2d.src.grid.create_grid`.
    rotation : {'solar', 'rigid'}
        ``'solar'`` uses the Snodgrass-type profile; ``'rigid'`` a 2-day
        solid-body rotation (a test case).
    frame : {'carrington', 'synodic'}
        Reference frame for the solar profile.

    Returns
    -------
    ndarray, shape (n_theta, n_phi)
        ``Omega`` [rad/s].
    """
    theta = grid["colatitude"]
    n_phi = grid["longitude"].shape[0]
    Colatitude = np.tile(theta, (n_phi, 1)).T

    # degrees/day -> radians/second
    rotation_rate_fact = 2.0201e-7

    if rotation == "solar":
        if frame == "carrington":
            base = 13.38 - 360.0 / 27.2753
        elif frame == "synodic":
            base = 13.38
        else:
            raise ValueError("For solar rotation, frame must be 'carrington' or 'synodic'.")
        omega_diff = (
            base
            - 2.30 * np.cos(Colatitude) ** 2
            - 1.62 * np.cos(Colatitude) ** 4
        ) * rotation_rate_fact
    elif rotation == "rigid":
        rotation_period_days = 2.0
        omega_diff = np.full(Colatitude.shape, 360.0 / rotation_period_days)
    else:
        raise ValueError("rotation must be 'solar' or 'rigid'.")

    return omega_diff


def vs(lat, v0=1.0, p=2.33):
    """Yeates (2020, Sol. Phys.) meridional-flow shape as a function of latitude.

    ``v(lat) = Du * sin(lat) * cos(lat)^p`` with ``Du`` chosen so the peak speed
    equals ``v0``.  Positive ``v0`` gives a poleward (northward in the north)
    latitude velocity; :func:`meridional_flow` converts it to ``u_theta``.
    """
    Du = v0 * (1.0 + p) ** (0.5 * (p + 1.0)) / p ** (0.5 * p)
    return Du * np.sin(lat) * (np.cos(lat)) ** p
