"""
sft_model: A Python package for simulating solar surface flux transport.

Modules:
    - src: Contains the core logic for the SFT model, including advection, diffusion, and grid handling.
"""

# Import core functionalities
from .src.advection import calculate_advection
from .src.diffusion import calculate_diffusion
from .src.time_step import calculate_time_step
from .src.grid import create_grid
from .src.initial_conditions import initialize_field
from .src.transport_profiles import meridional_flow, differential_rotation
from .analysis.analysis import calculate_usflx, calculate_dm, calculate_polar_field, calculate_polar_flux
from .analysis.visualize import plot_bfly, plot_mag


__all__ = [
    # Core modules
    "calculate_advection",
    "calculate_diffusion",
    "calculate_time_step",
    "create_grid",
    "initialize_field",
    "meridional_flow",
    "differential_rotation",
    # Analysis modules
    "calculate_usflx",
    "calculate_dm",
    "calculate_polar_field",
    "calculate_polar_flux",
    # Visualization modules
    "plot_bfly",
    "plot_mag"
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sft2d")
except PackageNotFoundError:
    __version__ = "0.0.0"
