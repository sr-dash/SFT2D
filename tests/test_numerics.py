"""
Numerical correctness tests for the SFT2D finite-volume core.

These pin down the properties the whole model rests on:

* the mesh spans the full sphere and its cell areas close to 4*pi*R^2;
* both spatial operators conserve flux exactly (their area-weighted sum is
  zero), so nothing leaks at the poles;
* the diffusion operator is symmetric negative semi-definite on the reduced
  degree-of-freedom space, hence RKL2 is stable on it;
* free decay of an axisymmetric mode matches the analytic
  ``exp(-l(l+1) eta t / R^2)`` rate;
* advection is monotone (no new extrema across a polarity inversion line);
* polar-cap diagnostics converge at second order, so a value fitted at one
  resolution is meaningful at another.

Run with ``pytest tests/`` from the repository root.
"""

import numpy as np
import pytest
from numpy.polynomial import legendre

from sft2d.analysis.analysis import calculate_polar_flux, cap_areas
from sft2d.src.constants import R_SUN_M
from sft2d.src.grid import create_grid, polar_average, total_flux
from sft2d.src.operators import Advection, Diffusion
from sft2d.src.stepper import advect, evolve
from sft2d.src.sts import rkl2_num_stages
from sft2d.src.transport_profiles import differential_rotation, meridional_flow

ETA = 2.5e8


@pytest.fixture
def grid():
    return create_grid(91, 180)


def _random_field(grid, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(grid["n_theta"], grid["n_phi"]))
    return polar_average(b)


# ---------------------------------------------------------------- mesh -----
@pytest.mark.parametrize("nt,npm", [(46, 90), (91, 180), (181, 360)])
def test_mesh_spans_sphere(nt, npm):
    g = create_grid(nt, npm)
    lat = 90.0 - np.rad2deg(g["colatitude"])
    assert lat.max() == pytest.approx(90.0, abs=1e-9)
    assert lat.min() == pytest.approx(-90.0, abs=1e-9)


@pytest.mark.parametrize("nt,npm", [(46, 90), (91, 180), (181, 360)])
def test_cell_areas_close_the_sphere(nt, npm):
    g = create_grid(nt, npm)
    total = float(np.sum(g["area"])) * npm
    assert total == pytest.approx(4.0 * np.pi * R_SUN_M**2, rel=1e-14)


def test_create_grid_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        create_grid(2, 180)
    with pytest.raises(ValueError):
        create_grid(91, 3)


# ----------------------------------------------------------- conservation --
def test_diffusion_conserves_flux(grid):
    b = _random_field(grid)
    op = Diffusion(grid, ETA)
    scale = float(np.sum(np.abs(b) * grid["area"]))
    assert abs(float(np.sum(op(b) * grid["area"]))) / scale < 1e-15


def test_advection_conserves_flux(grid):
    b = _random_field(grid)
    op = Advection(grid, meridional_flow(grid, 15.0), differential_rotation(grid))
    scale = float(np.sum(np.abs(b) * grid["area"]))
    assert abs(float(np.sum(op(b) * grid["area"]))) / scale < 1e-15


def test_full_run_conserves_signed_flux(grid):
    """The case the old pole-clipped boundary condition got wrong: a
    single-signed blob advected into the pole."""
    lat = 90.0 - np.rad2deg(grid["colatitude"])
    b0 = np.repeat(np.exp(-((lat - 30.0) / 8.0) ** 2)[:, None], grid["n_phi"], axis=1)
    f0 = total_flux(b0, grid)
    bf = evolve(b0, grid, meridional_flow(grid, 15.0),
                differential_rotation(grid), ETA, num_days=600)
    assert total_flux(bf, grid) / f0 - 1.0 == pytest.approx(0.0, abs=1e-12)


# -------------------------------------------------------------- operator ---
def test_diffusion_operator_symmetric_negative_definite():
    """On the reduced DOF space (one degree of freedom per polar cap, since the
    polar rows are replicated storage for a single cell)."""
    nt, npm = 13, 12
    g = create_grid(nt, npm)
    op = Diffusion(g, ETA)
    dof = ([("N", 0)] + [(j, k) for j in range(1, nt - 1) for k in range(npm)]
           + [("S", 0)])

    def basis(d):
        e = np.zeros((nt, npm))
        if d[0] == "N":
            e[0, :] = 1.0
        elif d[0] == "S":
            e[-1, :] = 1.0
        else:
            e[d[0], d[1]] = 1.0
        return e

    def extract(y, d):
        if d[0] == "N":
            return y[0, 0]
        if d[0] == "S":
            return y[-1, 0]
        return y[d[0], d[1]]

    A = np.array([[extract(op(basis(c)), r) for c in dof] for r in dof])
    w = np.array([g["area"][0, 0] * npm if d[0] == "N"
                  else g["area"][-1, 0] * npm if d[0] == "S"
                  else g["area"][d[0], 0] for d in dof])
    As = A * w[:, None]

    assert np.max(np.abs(As - As.T)) / np.max(np.abs(As)) < 1e-12
    ev = np.linalg.eigvalsh(0.5 * (As + As.T))
    assert ev.max() < 1e-9 * abs(ev.min())


@pytest.mark.parametrize("l", [1, 2, 3])
def test_free_decay_matches_analytic_rate(l):
    """Y_l0 must decay as exp(-l(l+1) eta t / R^2) under pure diffusion."""
    g = create_grid(181, 90)
    c = np.zeros(l + 1)
    c[l] = 1.0
    b0 = np.repeat(legendre.legval(np.cos(g["colatitude"]), c)[:, None],
                   g["n_phi"], axis=1)
    days = 200
    zero = np.zeros(g["n_theta"])
    bf = evolve(b0, g, zero, zero, ETA, num_days=days)

    exact = np.exp(-l * (l + 1) * ETA * days * 86400.0 / R_SUN_M**2)
    got = bf[:, 0].dot(b0[:, 0]) / b0[:, 0].dot(b0[:, 0])
    assert got == pytest.approx(exact, rel=2e-3)


def test_advection_is_monotone_under_solid_body_rotation(grid):
    """A TVD limiter must not manufacture new extrema across a sharp polarity
    inversion line.

    The test flow is solid-body rotation in longitude, which is
    divergence-free.  Monotonicity is a statement about *that* case: the
    meridional flow converges towards the poles, so conservative transport
    legitimately amplifies B_r beyond its initial range (flux compression), and
    an over/undershoot bound would be wrong there.
    """
    phi = grid["longitude"]
    step = np.where(np.abs(phi - np.pi) < 0.6, 1.0, -1.0)
    b0 = np.repeat(step[None, :], grid["n_theta"], axis=0)
    b0 = polar_average(b0)

    omega = np.full(grid["n_theta"], 1e-6)
    op = Advection(grid, np.zeros(grid["n_theta"]), omega)
    b = advect(b0, op, 2.0 * np.pi / 1e-6)      # one full revolution

    assert b.max() <= b0.max() + 1e-10
    assert b.min() >= b0.min() - 1e-10


def test_meridional_flow_compresses_flux(grid):
    """The flip side of the test above: converging meridional flow must
    concentrate polar flux while conserving it exactly."""
    lat = 90.0 - np.rad2deg(grid["colatitude"])
    b0 = np.repeat(np.exp(-((lat - 40.0) / 10.0) ** 2)[:, None],
                   grid["n_phi"], axis=1)
    op = Advection(grid, meridional_flow(grid, 15.0),
                   np.zeros(grid["n_theta"]))
    b = advect(b0, op, 400 * 86400.0)
    assert b.max() > b0.max()                             # compressed
    assert total_flux(b, grid) / total_flux(b0, grid) == pytest.approx(1.0, abs=1e-12)


def test_upwind_is_more_diffusive_than_tvd(grid):
    """Sanity check on the limiters: first order must lose more peak."""
    theta, phi = grid["colatitude"], grid["longitude"]
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    b0 = np.exp(-(((TH - np.pi / 2) / 0.15) ** 2 + ((PH - np.pi) / 0.15) ** 2))
    omega = 1e-6
    dr = np.full(theta.size, omega)
    zero = np.zeros(theta.size)
    peaks = {}
    for lim in ("upwind1", "vanleer", "superbee"):
        op = Advection(grid, zero, dr, limiter=lim)
        peaks[lim] = advect(b0, op, 2.0 * np.pi / omega).max()
    assert peaks["upwind1"] < peaks["vanleer"] < peaks["superbee"]


# ------------------------------------------------------------------ STS ----
@pytest.mark.parametrize("ratio", [1, 10, 100, 1000])
def test_rkl2_stage_count_satisfies_stability_bound(ratio):
    s = rkl2_num_stages(ratio, 1.0)
    assert (s * s + s - 2) / 4.0 >= ratio
    if s > 2:                                   # minimal: s-1 must not suffice
        assert ((s - 1) ** 2 + (s - 1) - 2) / 4.0 < ratio


def test_rkl2_matches_many_explicit_steps():
    """RKL2 over one day must agree with brute-force forward Euler.

    The comparison uses a *resolved* field.  RKL2 is second-order accurate and
    its stability polynomial only damps grid-scale modes rather than
    reproducing their (essentially zero) exact amplitude, so a white-noise
    initial condition would differ at the grid scale by construction -- that is
    a property of every super-time-stepping scheme, not an error.
    """
    g = create_grid(46, 90)
    op = Diffusion(g, ETA)
    th, ph = np.meshgrid(g["colatitude"], g["longitude"], indexing="ij")
    b0 = polar_average(np.cos(th) + 0.3 * np.sin(th) ** 2 * np.cos(2 * ph))

    from sft2d.src.sts import rkl2_step
    dt = 86400.0
    b_sts = rkl2_step(b0, op, dt)

    n = int(np.ceil(dt / (0.5 * op.dt_explicit)))
    h = dt / n
    b_ex = b0.copy()
    for _ in range(n):
        b_ex = b_ex + h * op(b_ex)

    err = np.max(np.abs(b_sts - b_ex)) / np.max(np.abs(b0))
    assert err < 1e-4


# ---------------------------------------------------------- diagnostics ----
def test_cap_areas_are_exact(grid):
    """Partial-cell weighting must reproduce the analytic spherical-cap area."""
    for cap_deg in (10.0, 20.0, 33.7):
        an, as_ = cap_areas(grid, cap_deg)
        exact = 2.0 * np.pi * R_SUN_M**2 * (1.0 - np.cos(np.deg2rad(cap_deg)))
        assert float(np.sum(an)) * grid["n_phi"] == pytest.approx(exact, rel=1e-13)
        assert float(np.sum(as_)) * grid["n_phi"] == pytest.approx(exact, rel=1e-13)


def test_polar_flux_converges_second_order():
    """The calibration target must not depend on the grid it was computed on."""
    res = {}
    for nt in (46, 91, 181):
        g = create_grid(nt, 2 * (nt - 1))
        lat = np.deg2rad(90.0 - np.rad2deg(g["colatitude"]))
        b0 = np.repeat((np.abs(np.sin(lat)) ** 7 * np.sin(lat))[:, None],
                       g["n_phi"], axis=1)
        bf = evolve(b0, g, meridional_flow(g, 15.0), differential_rotation(g),
                    ETA, num_days=365)
        res[nt] = calculate_polar_flux(bf, g, pol_cap_extent_deg=20.0)[0]

    d1, d2 = res[91] - res[46], res[181] - res[91]
    assert np.log2(abs(d1 / d2)) == pytest.approx(2.0, abs=0.5)
    assert abs(d2 / res[181]) < 5e-3


def test_diagnostics_accept_2d_and_3d(grid):
    from sft2d.analysis.analysis import calculate_usflx
    b = _random_field(grid)
    single = calculate_usflx(b, grid)
    stack = calculate_usflx(np.stack([b, b, b]), grid)
    assert isinstance(single, float)
    assert stack.shape == (3,)
    assert stack[0] == pytest.approx(single)
