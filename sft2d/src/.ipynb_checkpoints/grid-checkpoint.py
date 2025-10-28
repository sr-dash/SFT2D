"""
grid.py

This module handles the creation and management of the uniform grid used in the Solar Surface Flux Transport (SFT) model.

Functions:
    - create_grid: Generates a uniform grid in spherical polar coordinates (theta, phi).
"""

import numpy as np

def create_grid(n_theta, n_phi, exclude_poles=True):
    """
    Creates a uniform grid in spherical polar coordinates (theta, phi).

    Parameters:
        n_theta (int): Number of grid points in the theta (co-latitude) direction.
        n_phi (int): Number of grid points in the phi (longitude) direction.
        exclude_poles (bool): Whether to exclude 5 degrees near the poles. Default is True.

    Returns:
        dict: A dictionary containing:
            - 'theta': Array of theta values (in radians).
            - 'phi': Array of phi values (in radians).
            - 'dtheta': Spacing in the theta direction (in radians).
            - 'dphi': Spacing in the phi direction (in radians).
    """
    # Define the theta (co-latitude) range
    if exclude_poles:
        theta_min = np.deg2rad(5)  # 5 degrees from the pole
        theta_max = np.deg2rad(175)  # 5 degrees from the other pole
        delta_theta_coarse = np.pi / n_theta
        delta_phi = 2 * np.pi / n_phi
        leave_out = 4
        delta_theta = np.abs(0 + (delta_theta_coarse * leave_out) - (np.pi - (delta_theta_coarse * leave_out))) / n_theta

        # Create latitude and longitude grids
        colatitude = np.linspace(leave_out * delta_theta_coarse, np.pi - leave_out * delta_theta_coarse, n_theta + 1)
        # latitude =  colatitude - 0.5 * np.pi
        longitude = np.linspace(0, 2 * np.pi, n_phi + 1)
    else:
        theta_min = 0.0  # Include poles
        theta_max = np.pi

        # Define the phi (longitude) range
        phi_min = 0.0
        phi_max = 2 * np.pi

        # Create the theta and phi grids
        colatitude = np.linspace(theta_min, theta_max, n_theta)
        longitude = np.linspace(phi_min, phi_max, n_phi, endpoint=False)

        # Calculate grid spacings
        delta_theta = (theta_max - theta_min) / (n_theta - 1)
        delta_phi = (phi_max - phi_min) / n_phi

    # Return the grid information as a dictionary
    return {
        'colatitude': colatitude,
        'longitude': longitude,
        'dtheta': delta_theta,
        'dphi': delta_phi
    }
