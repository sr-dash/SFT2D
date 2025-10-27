"""
advection.py

This module handles the advection terms in the Solar Surface Flux Transport (SFT) model.
It computes the advection of the magnetic field due to differential rotation and meridional flow.

Functions:
    - calculate_advection: Computes the advection term for the SFT model.
"""

import numpy as np

def calculate_advection(field, differential_rotation, meridional_flow, grid):
    """
    Computes the advection of the magnetic field due to differential rotation and meridional flow.

    Parameters:
        field (np.ndarray): The magnetic field on the grid (2D array).
        differential_rotation (np.ndarray): Angular velocity profile (1D array).
        meridional_flow (np.ndarray): Meridional flow profile (1D array).
        grid (dict): Dictionary containing grid information ('theta', 'phi', and their spacings).

    Returns:
        np.ndarray: Updated magnetic field after applying advection.
    """
    theta = grid['colatitude']
    phi = grid['longitude']
    dtheta = grid['dtheta']
    dphi = grid['dphi']

    Colatitude, Longitude = np.meshgrid(theta,phi,indexing='ij')
    solar_radius = 6.955 * 10**8  # Solar radius in meters
    
    # Compute staggered velocities
    u_theta = meridional_flow

    # Create an array to store the updated field
    updated_field = np.zeros_like(field[1:-1, 1:-1])
    # field.copy()

    # Calculate advection terms
    # Upwind discretization for adv_theta_term
    adv_theta_term = np.zeros_like(updated_field)

    # For positive v_theta (upwind from lower latitudes)
    positive_v_theta = u_theta[1:-1, 1:-1] > 0
    adv_theta_term[positive_v_theta] = (
        (u_theta[1:-1, 1:-1][positive_v_theta] * field[1:-1, 1:-1][positive_v_theta] * np.sin(Colatitude[1:-1, 1:-1][positive_v_theta])
        - u_theta[:-2, 1:-1][positive_v_theta] * field[:-2, 1:-1][positive_v_theta] * np.sin(Colatitude[:-2, 1:-1][positive_v_theta]))
        / (solar_radius * dtheta * np.sin(Colatitude[1:-1, 1:-1][positive_v_theta]))
    )

    # For negative v_theta (upwind from higher latitudes)
    negative_v_theta = u_theta[1:-1, 1:-1] < 0
    adv_theta_term[negative_v_theta] = (
        (u_theta[2:, 1:-1][negative_v_theta] * field[2:, 1:-1][negative_v_theta] * np.sin(Colatitude[2:, 1:-1][negative_v_theta])
        - u_theta[1:-1, 1:-1][negative_v_theta] * field[1:-1, 1:-1][negative_v_theta] * np.sin(Colatitude[1:-1, 1:-1][negative_v_theta]))
        / (solar_radius * dtheta * np.sin(Colatitude[1:-1, 1:-1][negative_v_theta]))
    )

    # Upwind discretization for adv_phi_term
    adv_phi_term = np.zeros_like(updated_field)

    # For positive omega (upwind from left)
    positive_omega = differential_rotation[1:-1, 1:-1] > 0
    adv_phi_term[positive_omega] = differential_rotation[1:-1, 1:-1][positive_omega] * (
        (field[1:-1, 1:-1][positive_omega] - field[1:-1, :-2][positive_omega]) / dphi
    )

    # For negative omega (upwind from right)
    negative_omega = differential_rotation[1:-1, 1:-1] < 0
    adv_phi_term[negative_omega] = differential_rotation[1:-1, 1:-1][negative_omega] * (
        (field[1:-1, 2:][negative_omega] - field[1:-1, 1:-1][negative_omega]) / dphi
    )

    # Update field based on advection
    updated_field = adv_theta_term + adv_phi_term

    return updated_field
