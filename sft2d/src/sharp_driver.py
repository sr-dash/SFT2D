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

import numpy as np
import pandas as pd

from .source import make_bmr_yeates

# Canonical positional column order of a headerless bmrsharps_evol dump, used
# when the file has no column-header row.  The key columns (date, lat, lon,
# flux, sep, tilt) sit at these indices in Yeates's 11- and 13-column outputs.
_POS = {"sharp": 0, "date": 2, "lat": 3, "lon": 4, "flux": 5, "sep": 8, "tilt": 9}


def _read_catalogue(path, date_col="CM time", lat_col="Latitude",
                    lon_col="Carr-Longitude", flux_col="Unsgnd flux",
                    sep_col="Bip-Separation", tilt_col="Bip-Tilt",
                    good_col="Good", sharp_col="SHARP", dedupe_max_flux=True):
    """Parse a Yeates SHARPS catalogue into a tidy DataFrame.

    Handles both layouts: a file **with** a column-header row (``allsharps.txt``
    or a headed ``bmrsharps_evol.txt``), and a **headerless** dump (data rows
    only), which is parsed by position in the canonical column order.  Exact
    duplicate rows -- some pipelines list a region once per observation frame
    with identical fitted values -- are collapsed.

    Returns columns: date (datetime), lat, lon, flux, sep, tilt, good, sharp.
    """
    with open(path) as fh:
        lines = fh.readlines()

    hdr_idx = None
    for i, ln in enumerate(lines):
        cells = [c.strip() for c in ln.replace("\t", " ").split()]
        if "SHARP" in cells and "Latitude" in cells:
            hdr_idx = i
            break

    if hdr_idx is not None:
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
            "sharp": df[sharp_col] if sharp_col in df else np.arange(len(df)),
        })
        out["good"] = df[good_col].astype(float).astype(int) if good_col in df else 1
    else:
        # Headerless: parse by position in the canonical column order.
        raw = pd.read_csv(path, sep="\t", header=None, comment=None,
                          skip_blank_lines=True)
        if raw.shape[1] <= max(_POS.values()):
            raise ValueError(
                f"{path}: {raw.shape[1]} columns, too few for a headerless "
                f"bmrsharps_evol dump (need >= {max(_POS.values()) + 1}). "
                f"If this file has a header, it was not recognised.")
        out = pd.DataFrame({
            "date": pd.to_datetime(raw[_POS["date"]]),
            "lat": raw[_POS["lat"]].astype(float),
            "lon": raw[_POS["lon"]].astype(float),
            "flux": raw[_POS["flux"]].astype(float).abs(),
            "sep": raw[_POS["sep"]].astype(float),
            "tilt": raw[_POS["tilt"]].astype(float),
            "sharp": raw[_POS["sharp"]],
        })
        out["good"] = 1

    # Collapse identical duplicate rows (same region listed per frame).
    out = out.drop_duplicates()

    # One entry per active region: when a SHARP id appears in several rows
    # (different frames / fits), keep the one with the largest observed flux
    # (~area).  This mirrors the standard "maximum-area record per region"
    # preprocessing; for this catalogue the repeats are usually identical, so it
    # mostly just guards against double-counting.
    if dedupe_max_flux and "sharp" in out and out["sharp"].notna().any():
        out = out.sort_values("flux").drop_duplicates(subset="sharp", keep="last")

    # Return in chronological order (natural for a time-driven catalogue).
    return out.sort_values("date").reset_index(drop=True)


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
    dedupe_max_flux : bool
        Keep only the largest-flux entry per SHARP id (the standard
        maximum-area-record-per-region preprocessing).
    rebalance : bool
        Re-zero the whole-map net flux on days that inject flux.
    """

    def __init__(self, catalogue, start_date, end_date=None, flux_scale=1.0,
                 width_frac=0.56, min_sep_deg=1.0, good_only=True,
                 dedupe_max_flux=True, rebalance=False):
        if isinstance(catalogue, str):
            df = _read_catalogue(catalogue, dedupe_max_flux=dedupe_max_flux)
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
