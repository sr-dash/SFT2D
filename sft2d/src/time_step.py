"""
time_step.py

This module handles the computation of the time step for the Solar Surface Flux Transport (SFT) model
based on the Courant-Friedrichs-Lewy (CFL) condition.

Functions:
    - calculate_time_step: Calculates the maximum allowable time step for the model.
"""

import numpy as np
from .transport_profiles import meridional_flow, differential_rotation


def calculate_time_step(grid, diffusivity, cfl_number=0.4):
    """
    Calculates the maximum allowable time step based on the CFL condition for advection and diffusion.

    Parameters:
        grid (dict): Dictionary containing grid information ('theta', 'phi', and their spacings).
        diffusivity: Magnetic diffusivity for SFT model in cm^2/s.
        cfl_number (float): CFL number (e.g., 0.4).

    Returns:
        float: Maximum allowable time step in seconds.
        float: Number of time steps per day.
    """
    # Read the grid information
    theta = grid['colatitude']
    phi = grid['longitude']
    delta_theta = grid['dtheta']
    delta_phi = grid['dphi']

    Colatitude, _ = np.meshgrid(theta,phi,indexing='ij')

    # Grid spacings in physical units
    solar_radius = 6.955 * 10**8  # Solar radius in meters

    # Advection velocities
    mf_ = meridional_flow(grid)
    dr_ = differential_rotation(grid)
    v_phi = dr_* solar_radius * np.sin(Colatitude)

    # Time step calculation based on CFL condition
    dt_diff_theta = np.min((solar_radius * delta_theta) ** 2 / diffusivity)
    dt_diff_phi = np.min((solar_radius * delta_phi * np.sin(theta)) ** 2 / diffusivity)
    dt_adv_theta = np.min(np.abs((solar_radius * delta_theta) / (mf_ + 0.001)))
    dt_adv_phi = np.min(np.abs((solar_radius * delta_phi * np.sin(Colatitude)) / (mf_ + 0.001)))
    dt_rot_theta = np.min((solar_radius * delta_theta) / np.abs(v_phi))
    dt_rot_phi = np.min((solar_radius * delta_phi * np.sin(Colatitude)) / np.abs(v_phi))
    dt_omega_phi = np.min(delta_phi/np.abs(dr_))

    time_step = cfl_number * np.min([dt_diff_theta, dt_diff_phi, dt_adv_theta, dt_adv_phi, dt_rot_theta, dt_rot_phi, dt_omega_phi])

    # Modify to fit exactly into one day:
    ndt = round(86400 / time_step)
    dtday = 1 / ndt
    time_step = dtday * 86400

    # Return the smaller of the two time step restrictions
    return time_step, ndt