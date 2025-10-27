"""
sft_model: A Python package for simulating solar surface flux transport.

Modules:
    - analysis: Contains the calculation script for derived quantities using the SFT magnetic field output. There are some sample visualization scripts as well.
"""

# Import analysis functionalities
from .analysis import calculate_usflx, calculate_dm, calculate_polar_field, calculate_polar_flux
from .visualize import plot_bfly, plot_mag


__all__ = [
    # Alanysis modules
    "calculate_usflx",
    "calculate_dm",
    "calculate_polar_field",
    "calculate_polar_flux",
    # Visualization modules
    "plot_bfly",
    "plot_mag"
]
