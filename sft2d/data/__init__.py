"""
sft2d.data

Bundled reference data and a small accessor so examples, tests and notebooks
work identically from a source checkout or a pip-installed copy.

Contents
--------
``sunspot_data_rgo_1901_2025.csv``  processed RGO active-region record that
                                    drives :class:`sft2d.src.ar_driver.ARSource`.
``hmi_CR2097.fits``                 one HMI synoptic map (sine-latitude) for the
                                    ``initialize_field(..., 'read')`` example.
``hmi_polar_field.p``               pandas Series of the observed HMI north polar
                                    cap field (CAPN2), 2010-2023, 12-hourly, for
                                    validating a driven run.
``hmi_bfly.p``                      observed HMI radial-field butterfly array.

Use
---
>>> from sft2d.data import get_path, load_hmi_polar_field
>>> csv = get_path("sunspot_data_rgo_1901_2025.csv")
>>> polar = load_hmi_polar_field()      # pandas Series, north cap [G]
"""

from __future__ import annotations

import pickle
from pathlib import Path

_HERE = Path(__file__).resolve().parent

RGO_CSV = _HERE / "sunspot_data_rgo_1901_2025.csv"
HMI_SYNOPTIC_FITS = _HERE / "hmi_CR2097.fits"
HMI_POLAR_FIELD = _HERE / "hmi_polar_field.p"
HMI_BUTTERFLY = _HERE / "hmi_bfly.p"


def get_path(name: str) -> Path:
    """Return the absolute path to a bundled data file, checking it exists."""
    p = _HERE / name
    if not p.exists():
        available = sorted(q.name for q in _HERE.iterdir() if q.is_file())
        raise FileNotFoundError(f"{name!r} not in sft2d.data; available: {available}")
    return p


def load_hmi_polar_field():
    """Load the observed HMI polar-field diagnostics.

    The pickle holds **seven** sequentially-pickled objects, in the same order
    used across the SFT-1D project (``n, s, mn, ms, sn, ss, t``).  Returns them
    as a dict:

    ==============  ===============================================
    key             contents
    ==============  ===============================================
    ``north``       raw north-cap field [G]        (pandas Series)
    ``south``       raw south-cap field [G]        (pandas Series)
    ``mean_north``  smoothed/mean north cap [G]    (pandas Series)
    ``mean_south``  smoothed/mean south cap [G]    (pandas Series)
    ``std_north``   1-sigma spread, north [G]      (pandas Series)
    ``std_south``   1-sigma spread, south [G]      (pandas Series)
    ``time``        datetimes, matches the Series index  (ndarray)
    ==============  ===============================================

    All Series share one 12-hourly ``DatetimeIndex`` (2010-05 .. 2023-09).  For
    a clean model/observation overlay use ``mean_north`` +/- ``std_north`` (the
    convention used in the SFT-1D comparison plots).
    """
    keys = ("north", "south", "mean_north", "mean_south",
            "std_north", "std_south", "time")
    out = {}
    with open(HMI_POLAR_FIELD, "rb") as fh:
        for k in keys:
            out[k] = pickle.load(fh)
    return out


def load_hmi_butterfly():
    """Load the observed HMI radial-field butterfly diagram.

    The pickle holds **three** sequentially-pickled objects (matching SFT-1D:
    ``bfly, timearr, latbfly``).  Returns ``(bfly, time_years, sin_lat)``:

    * ``bfly``       : ``(n_lat, n_time)`` mean B_r [G];
    * ``time_years`` : ``(n_time,)`` fractional years;
    * ``sin_lat``    : ``(n_lat,)`` latitude grid in **sine of latitude**
      (-1 .. +1).  Plot the latitude axis as ``np.rad2deg(np.arcsin(sin_lat))``.
    """
    import numpy as np

    with open(HMI_BUTTERFLY, "rb") as fh:
        bfly = pickle.load(fh)
        time_years = np.asarray(pickle.load(fh), dtype=float)
        sin_lat = np.asarray(pickle.load(fh), dtype=float)
    return bfly, time_years, sin_lat
