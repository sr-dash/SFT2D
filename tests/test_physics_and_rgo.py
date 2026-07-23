"""
Physics and RGO-driver tests.

These cover the parts of the model a user actually calibrates against:

* the meridional-flow **sign convention** (positive peak speed = poleward), the
  point that most often trips people up -- a negative peak speed is
  equatorward and will prevent polar reversal;
* the analytic BMR source (flux normalisation, Joy tilt, Hale polarity);
* the bundled RGO active-region driver end to end, including that an adequately
  scaled cycle reverses the polar field with the correct poleward flow and does
  not with an (unphysical) equatorward flow.

The RGO end-to-end tests run multi-year forward integrations, so they are marked
``slow``; skip them with ``pytest -m "not slow"`` while iterating.
"""

import numpy as np
import pytest

from sft2d.analysis.analysis import calculate_polar_field, calculate_usflx
from sft2d.data import RGO_CSV, load_hmi_butterfly, load_hmi_polar_field
from sft2d.src.ar_driver import ARSource, solar_cycle_number
from sft2d.src.grid import create_grid, total_flux
from sft2d.src.initial_conditions import initialize_field
from sft2d.src.operators import Advection
from sft2d.src.source import hale_leading_sign, joys_law_tilt, make_bmr
from sft2d.src.stepper import advect, evolve
from sft2d.src.transport_profiles import (
    differential_rotation,
    meridional_flow,
    meridional_flow_latitude,
)

ETA = 2.5e8


# ----------------------------------------------------- meridional-flow sign --
def test_positive_peak_speed_is_poleward():
    """A blob advected with +v0 must migrate toward the nearest pole in BOTH
    hemispheres.  This is the convention the whole calibration relies on."""
    g = create_grid(181, 4)
    lat = 90.0 - np.rad2deg(g["colatitude"])
    mf = meridional_flow(g, peak_speed=15.0)
    op = Advection(g, mf, np.zeros(g["n_theta"]))

    for lat0 in (+40.0, -40.0):
        b0 = np.repeat(np.exp(-((lat - lat0) / 8.0) ** 2)[:, None], g["n_phi"], axis=1)
        b = advect(b0, op, 500 * 86400.0)
        peak_lat = lat[np.argmax(b[:, 0])]
        assert abs(peak_lat) > abs(lat0)                 # moved poleward
        assert np.sign(peak_lat) == np.sign(lat0)        # stayed in its hemisphere


def test_negative_peak_speed_is_equatorward():
    """Guard the convention from the other side: -v0 transports equatorward."""
    g = create_grid(181, 4)
    lat = 90.0 - np.rad2deg(g["colatitude"])
    mf = meridional_flow(g, peak_speed=-15.0)
    op = Advection(g, mf, np.zeros(g["n_theta"]))
    b0 = np.repeat(np.exp(-((lat - 40.0) / 8.0) ** 2)[:, None], g["n_phi"], axis=1)
    b = advect(b0, op, 500 * 86400.0)
    assert lat[np.argmax(b[:, 0])] < 40.0                # moved toward equator


def test_u_theta_sign_matches_colatitude_convention():
    """u_theta < 0 in the north and > 0 in the south for a poleward flow."""
    g = create_grid(91, 4)
    lat = 90.0 - np.rad2deg(g["colatitude"])
    mf = meridional_flow(g, peak_speed=15.0)
    iN = np.argmin(np.abs(lat - 45)); iS = np.argmin(np.abs(lat + 45))
    assert mf[iN, 0] < 0 and mf[iS, 0] > 0
    # latitude-velocity helper is the negation (northward positive)
    vlat = meridional_flow_latitude(g, peak_speed=15.0)
    assert np.allclose(vlat, -mf)
    assert vlat[iN, 0] > 0 and vlat[iS, 0] < 0           # +N, -S (poleward)


# ------------------------------------------------------------- BMR source ---
@pytest.mark.parametrize("lat_deg", [0.0, 20.0, -35.0, 70.0, -80.0])
def test_bmr_flux_normalisation_exact(lat_deg):
    """make_bmr must realise the requested unsigned flux at any latitude."""
    g = create_grid(91, 180)
    B = make_bmr(g, lat_deg, 180.0, 1e22, sigma_deg=4, sep_deg=8, cycle_number=24)
    assert calculate_usflx(B, g) == pytest.approx(1e22, rel=1e-6)


def test_bmr_is_flux_balanced():
    """A single bipole carries no net flux."""
    g = create_grid(91, 180)
    B = make_bmr(g, 20.0, 180.0, 1e22, cycle_number=24)
    assert total_flux(B, g) == pytest.approx(0.0, abs=1e-6 * calculate_usflx(B, g))


def test_joys_law_and_hale_signs():
    """Joy tilt is signed by hemisphere; Hale flips leading polarity by cycle."""
    assert joys_law_tilt(30.0) > 0 and joys_law_tilt(-30.0) < 0
    assert hale_leading_sign(30.0, 24) == -hale_leading_sign(30.0, 25)
    assert hale_leading_sign(30.0, 24) == -hale_leading_sign(-30.0, 24)


def test_hale_absolute_polarity_matches_observed_cycles():
    """Anchored to reality: odd-cycle N leading is +, even-cycle N leading is -.
    Cycle 23 (odd) -> N leading +1; cycle 24 (even) -> N leading -1."""
    assert hale_leading_sign(30.0, 23) == +1
    assert hale_leading_sign(30.0, 24) == -1
    assert hale_leading_sign(-30.0, 24) == +1


def test_solar_cycle_number_from_date():
    assert solar_cycle_number("1998-01-01") == 23
    assert solar_cycle_number("2014-01-01") == 24


# --------------------------------------------------------- RGO driver -------
def test_arsource_loads_bundled_rgo():
    src = ARSource(str(RGO_CSV), start_date="2012-01-01", end_date="2012-12-31")
    assert src.n_regions > 0
    assert src.num_days == pytest.approx(365, abs=1)


def test_bundled_hmi_reference_available():
    hmi = load_hmi_polar_field()
    for key in ("north", "south", "mean_north", "mean_south",
                "std_north", "std_south", "time"):
        assert key in hmi
    mn = hmi["mean_north"]
    assert len(mn) > 1000
    assert mn.index[0].year == 2010
    # all series share the one datetime index
    assert (hmi["mean_south"].index == mn.index).all()


def test_bundled_hmi_butterfly_available():
    bfly, time_years, sin_lat = load_hmi_butterfly()
    assert bfly.shape[0] == sin_lat.size          # (n_lat, n_time)
    assert bfly.shape[1] == time_years.size
    assert sin_lat.min() >= -1.0 and sin_lat.max() <= 1.0
    assert 2010 <= time_years[0] < time_years[-1] <= 2025


@pytest.mark.slow
def test_driven_cycle24_reverses_north_pole_toward_observed_sign():
    """Scientific acceptance test: seeded with the observed pre-cycle-24 north
    polarity (negative, as HMI shows in 2010), an adequately-scaled cycle-24 run
    with poleward flow reverses the north pole to POSITIVE -- the same sign
    change HMI observed.  This exercises the flow (poleward) and the Hale
    absolute polarity (even cycle -> following polarity positive) together."""
    g = create_grid(91, 180)
    dr = differential_rotation(g)
    src = ARSource(str(RGO_CSV), start_date="2010-05-01", end_date="2020-01-01",
                   flux_scale=40.0)
    B0 = initialize_field(g, "dipole") * (-2.0)          # observed 2010 sign: N negative
    pn0, _ = calculate_polar_field(B0, g)
    assert pn0 < 0

    mf = meridional_flow(g, peak_speed=15.0)             # poleward
    Bf = evolve(B0, g, mf, dr, ETA, src.num_days, source=src)
    pn1, _ = calculate_polar_field(Bf, g)
    assert pn1 > 0                                        # reversed to observed sign


@pytest.mark.slow
def test_driven_cycle24_does_not_reverse_with_equatorward_flow():
    """Same emergence and seed, equatorward flow (-v0): the pole must NOT reverse
    -- isolating flow direction as the deciding factor."""
    g = create_grid(91, 180)
    dr = differential_rotation(g)
    src = ARSource(str(RGO_CSV), start_date="2010-05-01", end_date="2020-01-01",
                   flux_scale=40.0)
    B0 = initialize_field(g, "dipole") * (-2.0)
    pn0, _ = calculate_polar_field(B0, g)

    mf = meridional_flow(g, peak_speed=-15.0)            # equatorward (wrong)
    Bf = evolve(B0, g, mf, dr, ETA, src.num_days, source=src)
    pn1, _ = calculate_polar_field(Bf, g)
    assert np.sign(pn1) == np.sign(pn0)                  # no reversal
