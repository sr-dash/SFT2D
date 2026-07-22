"""
sft_model: A Python package for simulating solar surface flux transport.

Modules:
    - analysis: Contains the calculation script for derived quantities using the SFT magnetic field output. There are some sample visualization scripts as well.
"""

# Import analysis functionalities
from .analysis import (
    calculate_dm,
    calculate_net_flux,
    calculate_polar_field,
    calculate_polar_flux,
    calculate_usflx,
)
from .visualize import plot_bfly, plot_mag


__all__ = [
    # Analysis modules
    "calculate_usflx",
    "calculate_net_flux",
    "calculate_dm",
    "calculate_polar_field",
    "calculate_polar_flux",
    # Visualization modules
    "plot_bfly",
    "plot_mag"
]
