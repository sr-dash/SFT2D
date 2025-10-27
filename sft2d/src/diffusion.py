"""
diffusion.py

This module handles the diffusion term in the Solar Surface Flux Transport (SFT) model.
It computes the diffusive evolution of the magnetic field on the solar surface.

Functions:
    - calculate_diffusion: Computes the diffusion term for the SFT model.
"""

import numpy as np

def calculate_diffusion(field, diffusivity, grid):
    """
    Computes the diffusion of the magnetic field using a second-order central difference scheme.

    Parameters:
        field (np.ndarray): The magnetic field on the grid (2D array).
        diffusivity (float): The diffusivity value (e.g., 250 km^2/s).
        grid (dict): Dictionary containing grid information ('theta', 'phi', and their spacings).

    Returns:
        np.ndarray: Updated magnetic field after applying diffusion.
    """

    theta = grid['colatitude']
    phi = grid['longitude']
    dtheta = grid['dtheta']
    dphi = grid['dphi']
    Colatitude, Longitude = np.meshgrid(theta,phi,indexing='ij')
    solar_radius = 6.955 * 10**8  # Solar radius in meters

    # Create an array to store the updated field
    updated_field = np.zeros_like(field[1:-1, 1:-1])
    # print(updated_field.shape)

    # Calculate diffusion terms
    d_theta_term = (diffusivity / solar_radius**2) * (
        (np.cos(Colatitude[1:-1, 1:-1]) / (2 * dtheta * np.sin(Colatitude[1:-1, 1:-1]))) *
        (field[2:, 1:-1] - field[:-2, 1:-1]) +
        (1 / dtheta**2) * (field[2:, 1:-1] + field[:-2, 1:-1] - 2 * field[1:-1, 1:-1])
    )

    d_phi_term = (diffusivity / solar_radius**2) * (
        (1 / (dphi * np.sin(Colatitude[1:-1, 1:-1]))**2) *
        (field[1:-1, 2:] + field[1:-1, :-2] - 2 * field[1:-1, 1:-1])
    )

    # Update field based on diffusion
    updated_field = d_theta_term + d_phi_term

    return updated_field
