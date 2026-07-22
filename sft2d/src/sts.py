"""
sts.py

Super-time-stepping (STS) for the diffusion operator: RKL2, the second-order
Runge-Kutta-Legendre scheme of Meyer, Balsara & Aslam (2014, J. Comput. Phys.
257, 594).

Why this exists
---------------
On a uniform latitude-longitude mesh the longitudinal diffusion term carries a
forward-Euler limit ``dt < (R dphi sin(theta))^2 / eta`` that collapses towards
the poles.  The previous version of this package worked around it with a
near-pole Fourier filter (``polar_filter.py``), which relaxed the time step by
*modifying the physics* -- an approximation that then has to be defended.

RKL2 removes the restriction outright.  It takes ``s`` cheap stages and is
stable for

    dt_sts <= dt_explicit * (s^2 + s - 2) / 4

so the cost of covering a fixed interval grows like ``sqrt`` of the stiffness
ratio instead of linearly.  At 90x180 with eta = 2.5e8 m^2/s a one-day step
needs ~11 stages instead of ~30 explicit steps; at 180x360 it needs ~43 instead
of ~480.  The result is a *consistent discretisation of the real equation* --
no filtering, no polar approximation.

This mirrors HipFT, which offers the same family (``load_sts_rkl2`` /
``load_sts_rkg2``) for exactly this reason.

Usage
-----
    diff = Diffusion(grid, eta)
    B = rkl2_step(B, diff, dt)
"""

from __future__ import annotations

import numpy as np


def rkl2_num_stages(dt, dt_explicit):
    """Smallest stage count ``s`` making RKL2 stable over ``dt``.

    Inverts ``dt <= dt_explicit * (s^2 + s - 2)/4``.  The returned ``s`` is at
    least 2 (``s=1`` degenerates to forward Euler with no gain).
    """
    if dt_explicit <= 0 or not np.isfinite(dt_explicit):
        raise ValueError("dt_explicit must be positive and finite")
    ratio = dt / dt_explicit
    s = 0.5 * (-1.0 + np.sqrt(9.0 + 16.0 * ratio))
    return max(int(np.ceil(s)), 2)


def rkl2_step(u0, operator, dt, s=None, dt_explicit=None):
    """Advance ``u0`` by ``dt`` under ``du/dt = operator(u)`` using RKL2.

    Parameters
    ----------
    u0 : ndarray
        Field at the start of the step.
    operator : callable
        Linear, symmetric-negative-definite spatial operator (here
        :class:`sft2d.src.operators.Diffusion`).
    dt : float
        Step to cover [s].
    s : int, optional
        Stage count.  Derived from ``dt_explicit`` (or ``operator.dt_explicit``)
        when omitted.
    dt_explicit : float, optional
        Forward-Euler limit of ``operator``.  Defaults to
        ``operator.dt_explicit``.
    """
    if s is None:
        if dt_explicit is None:
            dt_explicit = operator.dt_explicit
        s = rkl2_num_stages(dt, dt_explicit)

    w1 = 4.0 / (s * s + s - 2.0)

    def b(j):
        if j < 2:
            return 1.0 / 3.0
        return (j * j + j - 2.0) / (2.0 * j * (j + 1.0))

    M0 = operator(u0)                      # reused by every stage
    Y0 = u0
    Yjm1 = Y0 + b(1) * w1 * dt * M0        # mu_tilde_1 = b1*w1 (b1/b0 = 1)
    Yjm2 = Y0

    for j in range(2, s + 1):
        bj, bjm1, bjm2 = b(j), b(j - 1), b(j - 2)
        mu = (2.0 * j - 1.0) / j * bj / bjm1
        nu = -(j - 1.0) / j * bj / bjm2
        mu_t = mu * w1
        gamma_t = -(1.0 - bjm1) * mu_t     # a_{j-1} = 1 - b_{j-1}

        Yj = (mu * Yjm1 + nu * Yjm2 + (1.0 - mu - nu) * Y0
              + mu_t * dt * operator(Yjm1) + gamma_t * dt * M0)
        Yjm2, Yjm1 = Yjm1, Yj

    return Yjm1
