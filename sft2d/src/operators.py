"""
operators.py

Finite-volume spatial operators for the SFT equation on the pole-to-pole mesh
built by :func:`sft2d.src.grid.create_grid`.

    dB/dt = -div( u B )  +  eta * laplacian(B)

Both operators are written in **flux form** on the cell areas returned by the
grid, so the area-weighted sum of either right-hand side is zero to machine
precision: neither can create or destroy flux.  This replaces two earlier
routines that could:

* ``diffusion.py`` discretised the Laplacian in non-conservative form, with an
  explicit ``cos/sin`` metric term that leaks flux where the metric is singular;
* ``advection_tvd.py`` was conservative in the interior but sat on a
  pole-clipped domain whose zero-gradient boundary condition (``B[0]=B[1]`` in
  the old ``apply_bcs``) overwrote whatever flux had piled up against the
  artificial wall.  Measured on a monopolar blob advected poleward, that created
  ~1.7% of spurious signed flux over 600 days, deposited directly in the polar
  cap used as the calibration target.

Both classes cache their geometric coefficients at construction.  That matters:
the super-time-stepping scheme in ``sts.py`` calls the diffusion operator tens
of times per step, and recomputing trigonometry each call dominated the runtime.

Polar cells
-----------
Rows 0 and -1 are single spherical caps of angular radius ``dtheta/2``, stored
replicated across all longitudes.  Their update is the net flux through their
one face, summed over longitude and divided by the *total* cap area -- the same
treatment HipFT uses (``diffusion_operator_cd``).  Longitudinal transport within
a cap is meaningless and is forced to zero, so the caps stay single-valued
without any filtering.

Because the poles are real cells there is no polar boundary condition to get
wrong, and -- combined with super-time-stepping for diffusion -- no need for the
near-pole Fourier filter the previous version required.
"""

from __future__ import annotations

import numpy as np

from .constants import R_SUN_M


# ---------------------------------------------------------------------------
# Slope limiters
# ---------------------------------------------------------------------------
def _minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def _maxmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.maximum(np.abs(a), np.abs(b))


def limited_slope(dm, dp, limiter="vanleer"):
    """Limited cell slope from backward/forward differences ``dm``/``dp``."""
    limiter = limiter.lower()
    if limiter in ("upwind1", "firstorder", "zero"):
        return np.zeros_like(dm)
    if limiter == "none":
        return 0.5 * (dm + dp)
    if limiter == "minmod":
        return _minmod(dm, dp)
    if limiter == "vanleer":
        denom = dm + dp
        out = np.zeros_like(dm)
        good = (dm * dp) > 0.0
        out[good] = 2.0 * dm[good] * dp[good] / denom[good]
        return out
    if limiter == "mc":
        return _minmod(0.5 * (dm + dp), _minmod(2.0 * dm, 2.0 * dp))
    if limiter == "superbee":
        return _maxmod(_minmod(dp, 2.0 * dm), _minmod(2.0 * dp, dm))
    raise ValueError(f"unknown limiter {limiter!r}")


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------
class Diffusion:
    """Conservative 5-point Laplacian, ``y = eta * lap(B)``.

    Parameters
    ----------
    grid : dict
        Grid from :func:`create_grid`.
    eta : float or ndarray
        Diffusivity [m^2/s].  A scalar, a ``(n_theta,)`` latitude profile, or a
        full ``(n_theta, n_phi)`` map.  Face values are arithmetic means of the
        two adjacent cells.
    """

    def __init__(self, grid, eta):
        self.grid = grid
        nt, npm = grid["n_theta"], grid["n_phi"]
        dth, dph = grid["dtheta"], grid["dphi"]
        sfc = grid["sin_theta_face"]                 # (nt+1,)
        sct = grid["sin_theta"]                      # (nt,)
        area = grid["area"]                          # (nt,1)

        eta = np.asarray(eta, float)
        if eta.ndim == 0:
            eta = np.full((nt, npm), float(eta))
        elif eta.ndim == 1:
            eta = np.repeat(eta[:, None], npm, axis=1)
        self.eta = eta

        # Face diffusivities.
        eta_n = np.empty((nt, npm))                  # face at th[j]   (north)
        eta_s = np.empty((nt, npm))                  # face at th[j+1] (south)
        eta_n[1:, :] = 0.5 * (eta[:-1, :] + eta[1:, :])
        eta_n[0, :] = eta[0, :]
        eta_s[:-1, :] = 0.5 * (eta[:-1, :] + eta[1:, :])
        eta_s[-1, :] = eta[-1, :]
        eta_e = 0.5 * (eta + np.roll(eta, -1, axis=1))
        eta_w = 0.5 * (eta + np.roll(eta, 1, axis=1))

        # Interior stencil coefficients (rows 1 .. nt-2 are used; the polar rows
        # are overwritten below).  Derivation in the module docstring of grid.py:
        #   dB_j/dt = eta*dphi*[ sin(th_{j+1})(B_{j+1}-B_j)
        #                       - sin(th_j)(B_j-B_{j-1}) ] / (dtheta * A_j)
        #           + eta*dtheta*(B_{k+1}+B_{k-1}-2B_k) / (sin(t_j)*dphi*A_j)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.cn = eta_n * dph * sfc[:-1, None] / (dth * area)
            self.cs = eta_s * dph * sfc[1:, None] / (dth * area)
            inv_s = np.where(sct > 0, 1.0 / np.where(sct > 0, sct, 1.0), 0.0)
            self.ce = eta_e * dth * inv_s[:, None] / (dph * area)
            self.cw = eta_w * dth * inv_s[:, None] / (dph * area)

        # Polar caps: one cell each, total area n_phi * area[row].
        cap_n = npm * float(area[0, 0])
        cap_s = npm * float(area[-1, 0])
        self._pn = float(eta_n[1, :].mean()) * dph * sfc[1] / (dth * cap_n)
        self._ps = float(eta_s[-2, :].mean()) * dph * sfc[-2] / (dth * cap_s)

        # No longitudinal transport inside a cap.
        self.ce[0, :] = self.ce[-1, :] = 0.0
        self.cw[0, :] = self.cw[-1, :] = 0.0
        self.cn[0, :] = self.cs[0, :] = 0.0
        self.cn[-1, :] = self.cs[-1, :] = 0.0

        self.cc = -(self.cn + self.cs + self.ce + self.cw)
        self.cc[0, :] = -npm * self._pn
        self.cc[-1, :] = -npm * self._ps
        self._npm = npm

    def __call__(self, B):
        y = np.empty_like(B)
        y[1:-1, :] = (
            self.cn[1:-1, :] * B[:-2, :]
            + self.cs[1:-1, :] * B[2:, :]
            + self.cw[1:-1, :] * np.roll(B, 1, axis=1)[1:-1, :]
            + self.ce[1:-1, :] * np.roll(B, -1, axis=1)[1:-1, :]
            + self.cc[1:-1, :] * B[1:-1, :]
        )
        # Polar caps: net flux through the single face, over the whole cap.
        y[0, :] = self._pn * (B[1, :].sum() - self._npm * B[0, 0])
        y[-1, :] = self._ps * (B[-2, :].sum() - self._npm * B[-1, 0])
        return y

    @property
    def dt_explicit(self):
        """Forward-Euler stability limit [s] for this operator.

        Gershgorin: the spectral radius is bounded by ``2*max|cc|``, and
        forward Euler on a symmetric negative operator is stable for
        ``dt <= 2/rho``, hence ``dt <= 1/max|cc|``.
        """
        return 1.0 / float(np.max(np.abs(self.cc)))


# ---------------------------------------------------------------------------
# Advection
# ---------------------------------------------------------------------------
class Advection:
    """Conservative MUSCL/TVD advection, ``y = -div(u B)``.

    ``meridional_flow`` is u_theta [m/s] and ``differential_rotation`` is
    Omega(theta) [rad/s], each either ``(n_theta,)`` or ``(n_theta, n_phi)``
    as produced by :mod:`sft2d.src.transport_profiles`.
    """

    def __init__(self, grid, meridional_flow, differential_rotation,
                 limiter="vanleer"):
        self.grid = grid
        self.limiter = limiter
        nt, npm = grid["n_theta"], grid["n_phi"]
        dph = grid["dphi"]
        sfc = grid["sin_theta_face"]
        area = grid["area"]

        u = np.asarray(meridional_flow, float)
        om = np.asarray(differential_rotation, float)
        if u.ndim == 1:
            u = np.repeat(u[:, None], npm, axis=1)
        if om.ndim == 1:
            om = np.repeat(om[:, None], npm, axis=1)
        self.omega = om

        # u_theta interpolated to the interior theta faces (faces 1 .. nt-1).
        # Faces 0 and nt are the poles, where sin(theta)=0 carries no flux.
        self.u_face = 0.5 * (u[:-1, :] + u[1:, :])            # (nt-1, npm)

        # Geometric factor: face length / cell area, R*dphi*sin(th) / A.
        self.gface = R_SUN_M * dph * sfc[1:-1, None]          # (nt-1, 1)
        self.inv_area = 1.0 / area                            # (nt,1)
        self.cap_inv_area = np.array([1.0 / (npm * area[0, 0]),
                                      1.0 / (npm * area[-1, 0])])
        self._npm = npm
        self._dph = dph

        # Cached CFL rates.
        out_t = np.zeros((nt, npm))
        f = np.abs(self.u_face) * self.gface
        out_t[:-1, :] += f * self.inv_area[:-1]
        out_t[1:, :] += f * self.inv_area[1:]
        # A cap's outflow rate is the sum of |flux| over its whole rim divided by
        # the whole cap area, not a single sector's.
        out_t[0, :] = np.abs(self.u_face[0, :]).sum() * self.gface[0, 0] * self.cap_inv_area[0]
        out_t[-1, :] = np.abs(self.u_face[-1, :]).sum() * self.gface[-1, 0] * self.cap_inv_area[1]
        out_p = 2.0 * np.abs(om) / dph
        out_p[0, :] = out_p[-1, :] = 0.0
        self._rate = float(np.max(out_t + out_p))

    def dt_cfl(self, cfl=0.4):
        """Largest stable advective time step [s] for the given CFL number."""
        if self._rate <= 0.0:
            return np.inf
        return cfl / self._rate

    def __call__(self, B):
        lim = self.limiter
        npm = self._npm

        # ---- theta: MUSCL reconstruction, upwinded on the face velocity ----
        dm = np.zeros_like(B)
        dp = np.zeros_like(B)
        dm[1:, :] = B[1:, :] - B[:-1, :]
        dp[:-1, :] = B[1:, :] - B[:-1, :]
        # Caps are single values: no meaningful slope inside them.
        dm[0, :] = dp[0, :] = 0.0
        dm[-1, :] = dp[-1, :] = 0.0
        st = limited_slope(dm, dp, lim)

        left = B[:-1, :] + 0.5 * st[:-1, :]
        right = B[1:, :] - 0.5 * st[1:, :]
        Bf = np.where(self.u_face >= 0.0, left, right)
        Ft = self.gface * self.u_face * Bf                    # (nt-1, npm)

        y = np.zeros_like(B)
        y[:-1, :] -= Ft * self.inv_area[:-1]                  # flux leaving south
        y[1:, :] += Ft * self.inv_area[1:]                    # flux entering north

        # Caps: their one face, summed over longitude, over the whole cap area.
        y[0, :] = -Ft[0, :].sum() * self.cap_inv_area[0]
        y[-1, :] = Ft[-1, :].sum() * self.cap_inv_area[1]

        # ---- phi: Omega dB/dphi = d(Omega B)/dphi, Omega = Omega(theta) ----
        dmp = B - np.roll(B, 1, axis=1)
        dpp = np.roll(B, -1, axis=1) - B
        sp = limited_slope(dmp, dpp, lim)
        Bfp = np.where(self.omega >= 0.0,
                       B + 0.5 * sp,
                       np.roll(B, -1, axis=1) - 0.5 * np.roll(sp, -1, axis=1))
        Hp = self.omega * Bfp
        y -= (Hp - np.roll(Hp, 1, axis=1)) / self._dph

        # Longitudinal transport inside a cap is meaningless; drop it so the
        # caps stay single-valued without any filtering.
        y[0, :] = y[0, 0]
        y[-1, :] = y[-1, 0]
        return y
