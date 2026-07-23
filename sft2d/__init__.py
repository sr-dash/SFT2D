"""
sft2d: solar surface flux transport on a conservative pole-to-pole finite-volume sphere.

Subpackages
-----------
- ``src``       : mesh, spatial operators, time integration, flow profiles, sources.
- ``analysis``  : derived diagnostics (flux, dipole, polar field) and plotting.
- ``data``      : bundled RGO / HMI reference data and accessors.

Quick start
-----------
>>> import sft2d as sft
>>> grid  = sft.create_grid(91, 180)
>>> mf    = sft.meridional_flow(grid, peak_speed=15.0)     # poleward
>>> dr    = sft.differential_rotation(grid)
>>> B0    = sft.initialize_field(grid, "dipole") * 3.0
>>> B     = sft.evolve(B0, grid, mf, dr, eta=2.5e8, num_days=365)
>>> sft.calculate_polar_field(B, grid)
"""

from .analysis.analysis import (
    calculate_dm,
    calculate_net_flux,
    calculate_polar_field,
    calculate_polar_flux,
    calculate_usflx,
    cap_areas,
)
from .analysis.visualize import plot_bfly, plot_mag
from .src import (
    DAY_S,
    R_SUN_CM,
    R_SUN_M,
    Advection,
    ARSource,
    Diffusion,
    advect,
    balance_flux,
    correct_flux_multiplicative,
    create_grid,
    differential_rotation,
    evolve,
    hale_leading_sign,
    initialize_field,
    insert_bmr,
    joys_law_tilt,
    make_bmr,
    meridional_flow,
    meridional_flow_latitude,
    polar_average,
    rkl2_step,
    solar_cycle_number,
    total_flux,
)

__version__ = "0.1.0"

__all__ = [
    # constants
    "R_SUN_M",
    "R_SUN_CM",
    "DAY_S",
    # mesh
    "create_grid",
    "polar_average",
    "total_flux",
    # operators + solver
    "Advection",
    "Diffusion",
    "evolve",
    "advect",
    "rkl2_step",
    # setup
    "initialize_field",
    "correct_flux_multiplicative",
    "meridional_flow",
    "meridional_flow_latitude",
    "differential_rotation",
    # sources
    "make_bmr",
    "insert_bmr",
    "balance_flux",
    "joys_law_tilt",
    "hale_leading_sign",
    "ARSource",
    "solar_cycle_number",
    # analysis
    "calculate_usflx",
    "calculate_net_flux",
    "calculate_dm",
    "calculate_polar_field",
    "calculate_polar_flux",
    "cap_areas",
    # visualization
    "plot_bfly",
    "plot_mag",
    # version
    "__version__",
]
