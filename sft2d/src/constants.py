"""
constants.py

Single source of truth for physical constants.

Previously the solar radius appeared as 6.955e8 (m) in the solver, 6.955e10 (cm)
in ``source.py`` and 6.98e10 (cm) in ``analysis.py``.  The last one disagrees
with the others by 0.36%, which shows up directly as a 0.7% error in every
computed flux.  Import from here instead of writing the number inline.
"""

from __future__ import annotations

#: Solar radius [m]
R_SUN_M = 6.955e8

#: Solar radius [cm] -- use for fluxes quoted in Maxwell
R_SUN_CM = R_SUN_M * 100.0

#: Seconds in a day
DAY_S = 86400.0
