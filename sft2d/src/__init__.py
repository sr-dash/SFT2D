"""
Core of the SFT model: mesh, spatial operators, time integration and sources.

Submodules
----------
- ``constants``           : solar radius and other shared constants.
- ``grid``                : pole-to-pole finite-volume mesh.
- ``operators``           : conservative diffusion and TVD advection.
- ``sts``                 : RKL2 super-time-stepping for diffusion.
- ``stepper``             : Strang-split time integration (``evolve``).
- ``transport_profiles``  : meridional flow and differential rotation.
- ``initial_conditions``  : dipole / synoptic-map initial states.
- ``source``, ``ar_driver`` : BMR emergence, driven from an observed AR record.

Removed in the finite-volume rewrite
------------------------------------
``advection``, ``diffusion``   non-conservative interior-only operators on the
                               old pole-clipped grid; superseded by ``operators``.
``advection_tvd``              conservative but tied to the clipped grid and its
                               duplicated-longitude convention.
``time_step``                  CFL helper whose phi-advection limit divided by
                               the meridional flow and used a sign-dependent
                               ``+0.001`` guard; the operators now report their
                               own limits (``Advection.dt_cfl``,
                               ``Diffusion.dt_explicit``).
``polar_filter``               near-pole Fourier filter, needed only to work
                               around the diffusive polar CFL that RKL2 now
                               handles exactly.
"""

from .constants import DAY_S, R_SUN_CM, R_SUN_M
from .grid import create_grid, polar_average, total_flux
from .operators import Advection, Diffusion, limited_slope
from .sts import rkl2_num_stages, rkl2_step
from .stepper import advect, evolve, ssprk33_step
from .initial_conditions import correct_flux_multiplicative, initialize_field
from .transport_profiles import differential_rotation, meridional_flow
from .source import balance_flux, hale_leading_sign, insert_bmr, joys_law_tilt, make_bmr
from .ar_driver import ARSource, solar_cycle_number

__all__ = [
    # constants
    "R_SUN_M",
    "R_SUN_CM",
    "DAY_S",
    # mesh
    "create_grid",
    "polar_average",
    "total_flux",
    # operators
    "Advection",
    "Diffusion",
    "limited_slope",
    # time integration
    "evolve",
    "advect",
    "ssprk33_step",
    "rkl2_step",
    "rkl2_num_stages",
    # setup
    "initialize_field",
    "correct_flux_multiplicative",
    "meridional_flow",
    "differential_rotation",
    # emergence sources
    "make_bmr",
    "insert_bmr",
    "balance_flux",
    "joys_law_tilt",
    "hale_leading_sign",
    "ARSource",
    "solar_cycle_number",
]
