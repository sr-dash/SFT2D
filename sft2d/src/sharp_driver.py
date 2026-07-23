"""
sharp_driver.py

Drive the SFT model from a SHARPS-derived idealized-BMR catalogue, as an
alternative to the RGO sunspot record (:class:`sft2d.src.ar_driver.ARSource`).

Background
----------
A. Yeates's ``sharps-bmrs`` project (https://github.com/antyeates1983/sharps-bmrs,
GPL-3.0) fits an idealized bipole to each HMI/SHARP active region and tabulates
its **observed** parameters -- emergence date, latitude, Carrington longitude,
total unsigned flux, and the *fitted* separation and tilt.  Because separation
and tilt come from the actual magnetogram, driving from this catalogue needs no
Joy's-law tilt estimate and no Hale-law polarity assumption: the signed fitted
tilt already carries the orientation.  This makes it a cleaner, HMI-era
(2010-present) alternative to the RGO record, whose flux is inferred from spot
area and whose tilt/polarity are assumed.

This module reads that catalogue format and reconstructs each region with the
Yeates bipole shape :func:`sft2d.src.source.make_bmr_yeates`.  It does **not**
reimplement the SHARP download / fitting pipeline (that needs JSOC/``drms``
access and is Yeates's GPL code); it consumes the published catalogue.

Catalogue format
----------------
A whitespace/tab-separated ``bmrsharps_evol.txt`` with a multi-line header and
columns (only a subset is required)::

    SHARP  NOAA  CM-time  Latitude  Carr-Longitude  Unsgnd-flux  Imbalance
    [Good]  Dipole  Bip-Separation  Bip-Tilt  [Bip-Dipole ...]

The reader locates the header row by its ``SHARP``/``Latitude`` names, so both
the ``allsharps.txt`` (has a ``Good`` column) and ``bmrsharps_evol.txt`` layouts
work.  ``CM-time`` is the emergence/central-meridian date; latitude, longitude,
separation and tilt are in **degrees**; unsigned flux is in **Maxwell**.

Usage
-----
    src = SHARPSource("bmrsharps_evol.txt", start_date="2010-05-01")
    evolve(field0, grid, mf, dr, eta, src.num_days, source=src, ...)
"""

from __future__ import annotations

import pandas as pd

from .source import make_bmr_yeates


def _read_catalogue(path, date_col="CM time", lat_col="Latitude",
                    lon_col="Carr-Longitude", flux_col="Unsgnd flux",
                    sep_col="Bip-Separation", tilt_col="Bip-Tilt",
                    good_col="Good"):
    """Parse a Yeates SHARPS catalogue into a tidy DataFrame.

    Returns columns: date (datetime), lat, lon, flux, sep, tilt, good.
    """
    # Find the header line (the one naming the columns) and read from there.
    with open(path) as fh:
        lines = fh.readlines()
    hdr_idx = None
    for i, ln in enumerate(lines):
        cells = [c.strip() for c in ln.replace("\t", " ").split()]
        if "SHARP" in cells and "Latitude" in cells:
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError(f"{path}: could not find the column-header row")

    header = [c.strip() for c in lines[hdr_idx].rstrip("\n").split("\t") if c.strip()]
    rows = []
    for ln in lines[hdr_idx + 1:]:
        if not ln.strip():
            continue
        cells = [c.strip() for c in ln.rstrip("\n").split("\t") if c.strip() != ""]
        if len(cells) < len(header):
            continue
        rows.append(cells[:len(header)])
    df = pd.DataFrame(rows, columns=header)

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "lat": df[lat_col].astype(float),
        "lon": df[lon_col].astype(float),
        "flux": df[flux_col].astype(float).abs(),
        "sep": df[sep_col].astype(float),
        "tilt": df[tilt_col].astype(float),
    })
    out["good"] = df[good_col].astype(float).astype(int) if good_col in df else 1
    return out


class SHARPSource:
    """Callable ``source(day, field, grid)`` driven by a SHARPS BMR catalogue.

    Parameters
    ----------
    catalogue : str or pandas.DataFrame
        Path to a ``bmrsharps_evol.txt``-format file, or an already-parsed frame
        with columns ``date, lat, lon, flux, sep, tilt`` (and optionally
        ``good``).
    start_date : str or datetime
        Simulation start; day index 0.
    end_date : str or datetime, optional
        Simulation end (defaults to the last catalogue date).
    flux_scale : float
        Multiplies each region's unsigned flux (calibration knob; unlike RGO the
        SHARPS flux is measured, so this should be near 1).
    width_frac : float
        Yeates Gaussian scale as a fraction of the fitted separation (0.56).
    min_sep_deg : float
        Skip regions with a fitted separation below this (unresolved bipoles).
    good_only : bool
        Use only rows flagged good in the catalogue (if the column is present).
    rebalance : bool
        Re-zero the whole-map net flux on days that inject flux.
    """

    def __init__(self, catalogue, start_date, end_date=None, flux_scale=1.0,
                 width_frac=0.56, min_sep_deg=1.0, good_only=True,
                 rebalance=False):
        if isinstance(catalogue, str):
            df = _read_catalogue(catalogue)
        else:
            df = catalogue.copy()

        if good_only and "good" in df:
            df = df[df["good"] == 1]
        df = df[df["sep"].abs() >= min_sep_deg]

        self.start = pd.Timestamp(start_date)
        self.end = pd.Timestamp(end_date) if end_date is not None else df["date"].max()
        df = df[(df["date"] >= self.start) & (df["date"] <= self.end)].reset_index(drop=True)
        self.num_days = int((self.end - self.start).days)

        self.width_frac = width_frac
        self.rebalance = rebalance
        self.events = {}
        n_used = 0
        for _, r in df.iterrows():
            flux = float(r["flux"]) * flux_scale
            if flux <= 0:
                continue
            day = int((r["date"] - self.start).days)
            self.events.setdefault(day, []).append(
                dict(lat=float(r["lat"]), lon=float(r["lon"]), flux=flux,
                     sep=float(r["sep"]), tilt=float(r["tilt"])))
            n_used += 1
        self.n_regions = n_used

    def __call__(self, day, field, grid):
        evs = self.events.get(day)
        if not evs:
            return field
        for e in evs:
            field += make_bmr_yeates(grid, e["lat"], e["lon"], e["flux"],
                                     sep_deg=e["sep"], tilt_deg=e["tilt"],
                                     width_frac=self.width_frac)
        if self.rebalance:
            from .source import balance_flux
            field[:] = balance_flux(field, grid)
        return field

    def summary(self):
        return (f"SHARPSource: {self.n_regions} regions, "
                f"{self.start.date()} -> {self.end.date()} ({self.num_days} days)")
