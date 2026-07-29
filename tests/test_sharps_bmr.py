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
    for col in ("date", "lat", "lon", "flux", "sep", "tilt", "good"):
        assert col in df.columns
    assert len(df) == 3
    assert df["flux"].iloc[0] == pytest.approx(2.0e22)
    assert df["tilt"].iloc[1] == pytest.approx(-9.0)
    assert df["date"].iloc[2].strftime("%Y-%m-%d") == "2015-02-14"


def test_reader_handles_headerless_file(tmp_path):
    """A headerless bmrsharps_evol dump (data rows only) parses by position, and
    identical duplicate rows are collapsed."""
    # 11-column canonical order: SHARP NOAA date lat lon flux imb dip sep tilt bipdip
    row = "57\t11082\t2010-06-22\t27.9\t304.3\t1.0e22\t0.05\t9.7e-3\t5.42\t27.4\t9.6e-3\n"
    (tmp_path / "h.txt").write_text(row + row + row)   # 3 identical rows
    df = _read_catalogue(str(tmp_path / "h.txt"))
    assert len(df) == 1                                # deduped
    assert df["lat"].iloc[0] == pytest.approx(27.9)
    assert df["sep"].iloc[0] == pytest.approx(5.42)
    assert df["tilt"].iloc[0] == pytest.approx(27.4)


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


# ------------------------------------------------- SHARPPatchSource ----------
def _write_nc(path, ns=180, nph=360):
    """A synthetic SHARP .nc: a localised dipole patch on the sin-lat grid."""
    from scipy.io import netcdf_file
    ds = 2.0 / ns
    s = np.linspace(-1 + 0.5 * ds, 1 - 0.5 * ds, ns)
    dph = 2 * np.pi / nph
    ph = np.linspace(0.5 * dph, 2 * np.pi - 0.5 * dph, nph)
    S, P = np.meshgrid(s, ph, indexing="ij")
    lat0, lon0 = np.deg2rad(20.0), np.deg2rad(180.0)
    r2 = (np.arcsin(S) - lat0) ** 2 + (P - lon0) ** 2
    br = (P - lon0) * np.exp(-r2 / (2 * 0.05 ** 2))     # antisymmetric bipole-ish
    br[r2 > 0.3] = 0.0
    fh = netcdf_file(str(path), "w")
    fh.createDimension("sdim", ns); fh.createDimension("pdim", nph)
    v = fh.createVariable("br", "d", ("sdim", "pdim")); v[:] = br
    fh.close()
    return br


def test_patchsource_interpolates_and_preserves_flux(tmp_path):
    import pandas as pd

    from sft2d.src.sharp_patch_driver import SHARPPatchSource
    nc_dir = tmp_path
    _write_nc(nc_dir / "sharp00042.nc")
    cat = pd.DataFrame({"date": [pd.Timestamp("2015-01-10")], "lat": [20.0],
                        "lon": [180.0], "flux": [2.0e22], "sep": [8.0],
                        "tilt": [10.0], "sharp": [42], "good": [1]})
    src = SHARPPatchSource(cat, nc_dir=nc_dir, start_date="2015-01-01")
    g = create_grid(91, 180)
    patch = src.patch_on_grid(42, g, 2.0e22)
    assert patch is not None
    # flux preserved to the requested value, and balanced (net ~ 0)
    assert calculate_usflx(patch, g) == pytest.approx(2.0e22, rel=1e-6)
    assert abs(total_flux(patch, g)) < 1e-6 * calculate_usflx(patch, g)
    assert (patch > 0).any() and (patch < 0).any()      # genuinely bipolar


def test_patchsource_missing_nc_returns_none(tmp_path):
    import pandas as pd

    from sft2d.src.sharp_patch_driver import SHARPPatchSource
    cat = pd.DataFrame({"date": [pd.Timestamp("2015-01-10")], "lat": [20.0],
                        "lon": [180.0], "flux": [1e22], "sep": [8.0],
                        "tilt": [10.0], "sharp": [999], "good": [1]})
    src = SHARPPatchSource(cat, nc_dir=tmp_path, start_date="2015-01-01")
    g = create_grid(46, 90)
    assert src.patch_on_grid(999, g, 1e22) is None      # no .nc -> None, no crash


def test_cached_patch_source_matches_uncached(tmp_path):
    """The precomputed cache must reproduce the .nc-reading driver exactly.

    The sweep drivers use the cache to avoid re-reading every region's map per
    member; if it ever stopped matching, every cached run would be subtly wrong
    with nothing to indicate it.
    """
    import pandas as pd

    from sft2d.src.sharp_patch_driver import (CachedPatchSource, SHARPPatchSource,
                                              build_patch_cache)
    _write_nc(tmp_path / "sharp00042.nc")
    _write_nc(tmp_path / "sharp00043.nc")
    cat = pd.DataFrame({
        "date": [pd.Timestamp("2015-01-10"), pd.Timestamp("2015-01-20")],
        "lat": [20.0, -15.0], "lon": [180.0, 90.0], "flux": [2.0e22, 3.0e22],
        "sep": [8.0, 6.0], "tilt": [10.0, -8.0], "sharp": [42, 43], "good": [1, 1]})
    cat_file = tmp_path / "cat.pkl"
    cat.to_pickle(cat_file)

    g = create_grid(91, 180)
    cache = tmp_path / "cache.npz"
    build_patch_cache(cat, str(tmp_path), g, str(cache), "2015-01-01", "2015-02-01",
                      verbose=False)

    raw = SHARPPatchSource(cat, nc_dir=tmp_path, start_date="2015-01-01",
                           end_date="2015-02-01")
    cch = CachedPatchSource(str(cache), cat, start_date="2015-01-01",
                            end_date="2015-02-01")
    a = np.zeros((g["n_theta"], g["n_phi"])); b = a.copy()
    for d in sorted(raw.events):
        a = raw(d, a, g)
    for d in sorted(cch.events):
        b = cch(d, b, g)
    assert np.abs(a).max() > 0
    assert np.abs(a - b).max() <= 1e-12 * np.abs(a).max()


def test_cached_patch_source_rejects_wrong_grid(tmp_path):
    """A cache built for one grid must not be silently usable on another."""
    import pandas as pd

    from sft2d.src.sharp_patch_driver import CachedPatchSource, build_patch_cache
    _write_nc(tmp_path / "sharp00042.nc")
    cat = pd.DataFrame({"date": [pd.Timestamp("2015-01-10")], "lat": [20.0],
                        "lon": [180.0], "flux": [2.0e22], "sep": [8.0],
                        "tilt": [10.0], "sharp": [42], "good": [1]})
    cache = tmp_path / "cache.npz"
    build_patch_cache(cat, str(tmp_path), create_grid(91, 180), str(cache),
                      "2015-01-01", "2015-02-01", verbose=False)
    src = CachedPatchSource(str(cache), cat, start_date="2015-01-01",
                            end_date="2015-02-01")
    with pytest.raises(ValueError, match="cache was built for"):
        src.patch_on_grid(42, create_grid(46, 90), 1e22)
