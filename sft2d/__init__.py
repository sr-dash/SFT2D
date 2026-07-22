"""
sft2d: solar surface flux transport on a spherical finite-volume mesh.

Modules:
    - src: mesh, spatial operators, time integration, sources.
    - analysis: derived diagnostics and plotting.
"""

from .src import (
    ARSource,
    Advection,
    Diffusion,
    create_grid,
    differential_rotation,
    evolve,
    initialize_field,
    insert_bmr,
    make_bmr,
    meridional_flow,
    polar_average,
    solar_cycle_number,
    total_flux,
)
from .analysis.analysis import (
    calculate_dm,
    calculate_net_flux,
    calculate_polar_field,
    calculate_polar_flux,
    calculate_usflx,
)
from .analysis.visualize import plot_bfly, plot_mag

__all__ = [
    # mesh and solver
    "create_grid",
    "polar_average",
    "total_flux",
    "Advection",
    "Diffusion",
    "evolve",
    "initialize_field",
    "meridional_flow",
    "differential_rotation",
    # sources
    "make_bmr",
    "insert_bmr",
    "ARSource",
    "solar_cycle_number",
    # analysis
    "calculate_usflx",
    "calculate_net_flux",
    "calculate_dm",
    "calculate_polar_field",
    "calculate_polar_flux",
    # visualization
    "plot_bfly",
    "plot_mag",
]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sft2d")
except PackageNotFoundError:
    __version__ = "0.0.0"
