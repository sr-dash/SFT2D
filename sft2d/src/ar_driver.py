"""
ar_driver.py

Drive the SFT model with an observed active-region (AR) record, turning each
recorded region into a flux-normalised BMR source (see ``source.py``).

Designed for the processed RGO sunspot table shipped with this repository
(``sunspot_data_rgo_1901_2025.csv``), whose columns are:

    PHASE  Date  Fractional_Year  GSG/NOAA  Spot_Area  Latitude  Longitude
    TILT  RADIUS  USFLUX

but any table with Date/Latitude/Longitude (+ optional TILT/RADIUS/USFLUX)
columns can be used by passing the column names.

The record is converted to a time-indexed list of BMR emergence events.  An
:class:`ARSource` instance is a callable ``source(day, field, grid)`` compatible
with :func:`sft2d.src.stepper.evolve`, so a data-driven run is simply::

    src = ARSource("sunspot_data_rgo_1901_2025.csv", start_date="1996-01-01")
    evolve(field0, grid, mf, dr, eta, src.num_days, dt, ndt, source=src, ...)

Calibration knobs
-----------------
``flux_scale``   multiplies the per-region unsigned flux (the RGO area->flux
                 conversion is uncertain, so this is a primary calibration knob).
``sep_scale``, ``sigma_scale``  map the tabulated RADIUS to bipole separation /
                 Gaussian width.  ``tilt_scale`` rescales the tabulated tilt.
"""

from __future__ import annotations

import pandas as pd

from .source import insert_bmr, joys_law_tilt

# Approximate solar-cycle start dates (minima) used only to set Hale polarity.
_CYCLE_MINIMA = {
    12: "1878-12-01", 13: "1890-03-01", 14: "1902-01-01", 15: "1913-08-01",
    16: "1923-08-01", 17: "1933-09-01", 18: "1944-02-01", 19: "1954-04-01",
    20: "1964-10-01", 21: "1976-03-01", 22: "1986-09-01", 23: "1996-08-01",
    24: "2008-12-01", 25: "2019-12-01", 26: "2030-06-01",
}


def solar_cycle_number(date) -> int:
    """Return the solar cycle number active on ``date`` (a datetime/Timestamp)."""
    d = pd.Timestamp(date)
    num = 12
    for c, start in _CYCLE_MINIMA.items():
        if d >= pd.Timestamp(start):
            num = c
    return num


class ARSource:
    def __init__(self, table, start_date, end_date=None,
                 flux_scale=1.0, tilt_scale=1.0, use_provided_tilt=True,
                 joy_coeff=0.5, sep_scale=2.0, sigma_scale=1.0,
                 default_sep_deg=6.0, default_sigma_deg=4.0,
                 hale=True, rebalance=False,
                 date_col="Date", lat_col="Latitude", lon_col="Longitude",
                 tilt_col="TILT", radius_col="RADIUS", flux_col="USFLUX",
                 sep="\t"):
        if isinstance(table, str):
            df = pd.read_csv(table, sep=sep)
        else:
            df = table.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        self.start = pd.Timestamp(start_date)
        self.end = pd.Timestamp(end_date) if end_date is not None else df[date_col].max()
        df = df[(df[date_col] >= self.start) & (df[date_col] <= self.end)]
        df = df.reset_index(drop=True)

        self.num_days = int((self.end - self.start).days)

        # build per-day event lists
        self.events = {}
        n_used = 0
        for _, row in df.iterrows():
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            flux = abs(float(row[flux_col])) * flux_scale if flux_col in df else 0.0
            if flux <= 0:
                continue
            if use_provided_tilt and tilt_col in df and not pd.isna(row[tilt_col]):
                tilt = float(row[tilt_col]) * tilt_scale
            else:
                tilt = joys_law_tilt(lat, coeff=joy_coeff)
            radius = float(row[radius_col]) if radius_col in df else 0.0
            sep = sep_scale * radius if radius > 0 else default_sep_deg
            sigma = sigma_scale * radius if radius > 0 else default_sigma_deg
            sigma = max(sigma, 1.0)
            cyc = solar_cycle_number(row[date_col]) if hale else 24
            day = int((row[date_col] - self.start).days)
            self.events.setdefault(day, []).append(
                dict(lat=lat, lon=lon, flux=flux, tilt=tilt,
                     sep=sep, sigma=sigma, cycle=cyc))
            n_used += 1
        self.n_regions = n_used
        self.hale = hale
        self.rebalance = rebalance

    def __call__(self, day, field, grid):
        for e in self.events.get(day, []):
            insert_bmr(field, grid, e["lat"], e["lon"], e["flux"],
                       tilt_deg=e["tilt"], sep_deg=e["sep"], sigma_deg=e["sigma"],
                       cycle_number=e["cycle"], hale=self.hale,
                       rebalance=False)
        if self.rebalance and day in self.events:
            from .source import balance_flux
            field[:] = balance_flux(field, grid)
        return field

    def summary(self):
        return (f"ARSource: {self.n_regions} regions, "
                f"{self.start.date()} -> {self.end.date()} ({self.num_days} days)")
