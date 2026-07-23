"""
Tests for the Yeates-style BMR profile and the SHARPS catalogue driver.

No SHARPS data is bundled (the ``sharps-bmrs`` catalogue is GPL and lives in a
separate project); these tests synthesise a tiny catalogue in the documented
format instead.
"""

import numpy as np
import pytest

from sft2d.analysis.analysis import calculate_dm, calculate_usflx
from sft2d.src.grid import create_grid, total_flux
from sft2d.src.sharp_driver import SHARPSource, _read_catalogue
from sft2d.src.source import make_bmr_yeates
from sft2d.src.stepper import evolve
from sft2d.src.transport_profiles import differential_rotation, meridional_flow


@pytest.fixture
def grid():
    return create_grid(181, 360)


# --------------------------------------------------------- Yeates profile ---
@pytest.mark.parametrize("lat_deg", [0.0, 20.0, -35.0, 70.0])
def test_yeates_bmr_flux_normalisation_exact(grid, lat_deg):
    B = make_bmr_yeates(grid, lat_deg, 180.0, 1e22, sep_deg=8.0, tilt_deg=10.0)
    assert calculate_usflx(B, grid) == pytest.approx(1e22, rel=1e-6)


def test_yeates_bmr_is_balanced_bipole(grid):
    B = make_bmr_yeates(grid, 20.0, 180.0, 1e22, sep_deg=8.0, tilt_deg=10.0)
    # net flux zero to a tiny fraction of the unsigned flux
    assert abs(total_flux(B, grid)) < 1e-6 * calculate_usflx(B, grid)
    # genuinely bipolar: both signs present
    assert (B > 0).any() and (B < 0).any()


def test_yeates_bmr_tilt_sign_sets_dipole_sign(grid):
    """Flipping the tilt sign flips the axial-dipole contribution."""
    d_plus = calculate_dm(make_bmr_yeates(grid, 20.0, 180.0, 1e22, 8.0, +12.0), grid)
    d_minus = calculate_dm(make_bmr_yeates(grid, 20.0, 180.0, 1e22, 8.0, -12.0), grid)
    assert np.sign(d_plus) == -np.sign(d_minus)
    assert abs(d_plus) > 0


def test_yeates_zero_width_is_empty(grid):
    assert not make_bmr_yeates(grid, 0.0, 180.0, 1e22, sep_deg=0.0, tilt_deg=0.0).any()


# ------------------------------------------------------ catalogue reader ----
_CATALOGUE = """SHARPs from 2015-01-01 to 2015-03-01
-- synthetic test catalogue --
3
Grid resolution: 180 x 360
Selection criteria: ...
notes
------
SHARP\tNOAA\tCM time\t\tLatitude\tCarr-Longitude\tUnsgnd flux\tImbalance\tGood\tDipole\t\tBip-Separation\tBip-Tilt\tBip-Dipole
100\t12000\t2015-01-10\t15.0\t120.0\t2.0e22\t0.01\t1\t1e-2\t7.0\t8.0\t1e-2
101\t12001\t2015-01-25\t-18.0\t200.0\t1.5e22\t-0.02\t1\t-1e-2\t5.0\t-9.0\t-1e-2
102\t12002\t2015-02-14\t22.0\t300.0\t3.0e22\t0.03\t0\t2e-2\t6.0\t12.0\t2e-2
"""


@pytest.fixture
def catalogue_file(tmp_path):
    p = tmp_path / "bmrsharps_evol.txt"
    p.write_text(_CATALOGUE)
    return str(p)


def test_reader_parses_documented_format(catalogue_file):
    df = _read_catalogue(catalogue_file)
    assert list(df.columns) == ["date", "lat", "lon", "flux", "sep", "tilt", "good"]
    assert len(df) == 3
    assert df["flux"].iloc[0] == pytest.approx(2.0e22)
    assert df["tilt"].iloc[1] == pytest.approx(-9.0)
    assert df["date"].iloc[2].strftime("%Y-%m-%d") == "2015-02-14"


def test_sharpsource_good_only_and_min_sep(catalogue_file):
    # good_only default drops the row flagged 0
    src = SHARPSource(catalogue_file, start_date="2015-01-01")
    assert src.n_regions == 2
    # including bad rows restores it
    src_all = SHARPSource(catalogue_file, start_date="2015-01-01", good_only=False)
    assert src_all.n_regions == 3
    # min_sep filter
    src_sep = SHARPSource(catalogue_file, start_date="2015-01-01",
                          good_only=False, min_sep_deg=5.5)
    assert src_sep.n_regions == 2       # sep 7 and 6 pass, sep 5 dropped


def test_sharpsource_schedules_events_on_right_day(catalogue_file):
    src = SHARPSource(catalogue_file, start_date="2015-01-01", good_only=False)
    assert src.events[9][0]["lat"] == pytest.approx(15.0)     # 2015-01-10 is day 9
    assert src.events[9][0]["flux"] == pytest.approx(2.0e22)


def test_sharpsource_drives_and_injects_flux(grid, catalogue_file):
    mf = meridional_flow(grid, 15.0)
    dr = differential_rotation(grid)
    src = SHARPSource(catalogue_file, start_date="2015-01-01",
                      end_date="2015-03-01", good_only=False)
    B0 = np.zeros((grid["n_theta"], grid["n_phi"]))
    Bf = evolve(B0, grid, mf, dr, 2.5e8, 45, source=src)
    # flux was injected, and the net stays near zero (balanced bipoles)
    us = calculate_usflx(Bf, grid)
    assert us > 1e22
    assert abs(total_flux(Bf, grid)) < 1e-3 * us


def test_flux_scale_multiplies_injected_flux(catalogue_file):
    src1 = SHARPSource(catalogue_file, start_date="2015-01-01", good_only=False)
    src2 = SHARPSource(catalogue_file, start_date="2015-01-01", good_only=False,
                       flux_scale=3.0)
    assert src2.events[9][0]["flux"] == pytest.approx(3.0 * src1.events[9][0]["flux"])
