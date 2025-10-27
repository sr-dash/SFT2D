"""
Core module of the SFT model.

This module contains the main computational components for the solar surface flux transport model, 
including advection, diffusion, grid management, time stepping, and initialization of the field.

Submodules:
    - advection: Handles the advection terms in the SFT equation.
    - diffusion: Handles the diffusion terms in the SFT equation.
    - time_step: Manages time-stepping and the evolution of the magnetic field.
    - grid: Handles grid creation and management.
    - initial_conditions: Provides utilities for setting up initial conditions.
    - transport_profiles: Contains functions for defining transport profiles like meridional flow and differential rotation.
"""

# Import core components
from .advection import calculate_advection
from .diffusion import calculate_diffusion
from .time_step import calculate_time_step
from .grid import create_grid
from .initial_conditions import initialize_field
from .transport_profiles import meridional_flow, differential_rotation

__all__ = [
    "calculate_advection",
    "calculate_diffusion",
    "calculate_time_step",
    "create_grid",
    "initialize_field",
    "meridional_flow",
    "differential_rotation"
]
