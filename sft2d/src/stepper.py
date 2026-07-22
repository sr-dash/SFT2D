"""
stepper.py

Time integration for the 2-D SFT model, using Strang operator splitting.

Each step is advanced as a symmetric A-D-A sequence

    advection(dt/2) -> diffusion(dt) -> advection(dt/2)

which keeps the splitting second-order in time and, more importantly, lets the
two operators use the time step each one actually needs:

* **advection** with SSPRK(3,3) on the conservative MUSCL/TVD operator,
  sub-cycled at its own CFL (a mild restriction: the meridional limit is
  ``R dtheta / u_theta`` and the differential-rotation limit is ``dphi/Omega``,
  neither of which is singular at the poles);
* **diffusion** with RKL2 super-time-stepping, which absorbs the stiff near-pole
  longitudinal limit in ``O(sqrt)`` stages instead of ``O(n)`` explicit steps.

The previous version evaluated one combined right-hand side, so the whole run
was hostage to the diffusive polar CFL, and bought the step back with a
near-pole Fourier filter that altered the physics.  Splitting plus STS removes
both the restriction and the filter.

There is no ``apply_bcs`` any more.  The mesh is pole-to-pole and periodic in
longitude, so the only "boundary" is the pair of polar caps, which the operators
handle as ordinary finite-volume cells.  The old zero-gradient pole condition
(``B[0] = B[1]``) was itself a source of spurious flux.

Hooks
-----
``source(day, field, grid)``      BMR emergence, applied once per day.
``assimilate(day, field, grid)``  magnetogram insertion, once per day after the
                                  source.
``recorder.record(day, field)``   diagnostics.

Both hooks may modify the field in place and/or return a new one.  Anything they
return is passed through :func:`~sft2d.src.grid.polar_average` so the polar caps
stay single-valued.
"""

from __future__ import annotations

import numpy as np

from .constants import DAY_S
from .grid import polar_average
from .operators import Advection, Diffusion
from .sts import rkl2_num_stages, rkl2_step


# ---------------------------------------------------------------------------
# Advection sub-step: SSPRK(3,3)
# ---------------------------------------------------------------------------
def ssprk33_step(u0, operator, dt):
    """One SSP Runge-Kutta (3,3) step of ``du/dt = operator(u)``.

    SSP so the TVD property of the limited spatial operator survives the time
    integration: no spurious ringing across polarity inversion lines.
    """
    u1 = u0 + dt * operator(u0)
    u2 = 0.75 * u0 + 0.25 * (u1 + dt * operator(u1))
    return (u0 + 2.0 * (u2 + dt * operator(u2))) / 3.0


def advect(u, operator, dt, cfl=0.4):
    """Advance ``u`` by ``dt`` under advection, sub-cycling at the CFL limit."""
    dt_max = operator.dt_cfl(cfl)
    n = max(int(np.ceil(dt / dt_max)), 1) if np.isfinite(dt_max) else 1
    h = dt / n
    for _ in range(n):
        u = ssprk33_step(u, operator, h)
    return u


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def evolve(field0, grid, mf, dr, eta, num_days, steps_per_day=1,
           limiter="vanleer", cfl=0.4, tau_decay_s=None,
           source=None, assimilate=None, recorder=None, progress=False,
           return_stats=False):
    """Integrate the SFT equation for ``num_days`` days.

    Parameters
    ----------
    field0 : (n_theta, n_phi) ndarray
        Initial radial field [G].
    grid : dict
        Grid from :func:`~sft2d.src.grid.create_grid`.
    mf, dr : ndarray
        Meridional flow [m/s] and differential rotation [rad/s], shaped
        ``(n_theta,)`` or ``(n_theta, n_phi)``.
    eta : float or ndarray
        Supergranular diffusivity [m^2/s].
    num_days : int
        Days to integrate.
    steps_per_day : int
        Strang steps per day.  One is usually enough -- the splitting error is
        second order and both sub-integrators are stable at any step -- but
        raise it if a source injects strongly on daily timescales.
    limiter : str
        Slope limiter for the advection operator.
    cfl : float
        CFL number for the advective sub-cycling.
    tau_decay_s : float, optional
        Exponential decay timescale [s], solved analytically over each step so
        it imposes no step restriction.
    source, assimilate : callable, optional
        Daily hooks; see the module docstring.
    recorder : object, optional
        Anything with ``record(day_index, field)``.
    return_stats : bool
        If True, return ``(field, stats)`` where ``stats`` reports the chosen
        stage and sub-cycle counts.

    Returns
    -------
    ndarray
        Field after ``num_days`` days (plus stats if requested).
    """
    adv = Advection(grid, mf, dr, limiter=limiter)
    dif = Diffusion(grid, eta)

    dt = DAY_S / steps_per_day
    dt_half = 0.5 * dt
    decay = 1.0 if tau_decay_s is None else float(np.exp(-dt / tau_decay_s))

    dt_adv = adv.dt_cfl(cfl)
    dt_dif = dif.dt_explicit
    n_sub = max(int(np.ceil(dt_half / dt_adv)), 1) if np.isfinite(dt_adv) else 1
    n_stage = rkl2_num_stages(dt, dt_dif)
    stats = {
        "dt": dt,
        "dt_advection_cfl": dt_adv,
        "dt_diffusion_explicit": dt_dif,
        "advection_subcycles_per_half_step": n_sub,
        "rkl2_stages": n_stage,
        "explicit_diffusion_steps_avoided": dt / dt_dif,
    }

    B = polar_average(np.array(field0, dtype=float, copy=True))

    if recorder is not None:
        recorder.record(0, B)

    for day in range(1, num_days + 1):
        for _ in range(steps_per_day):
            B = advect(B, adv, dt_half, cfl)
            B = rkl2_step(B, dif, dt, s=n_stage)
            B = advect(B, adv, dt_half, cfl)
            if decay != 1.0:
                B *= decay
        if source is not None:
            out = source(day, B, grid)
            if out is not None:
                B = out
            polar_average(B)
        if assimilate is not None:
            out = assimilate(day, B, grid)
            if out is not None:
                B = out
            polar_average(B)
        if recorder is not None:
            recorder.record(day, B)
        if progress and day % max(1, num_days // 20) == 0:
            print(f"  day {day}/{num_days}")

    return (B, stats) if return_stats else B
