"""
transport_profiles.py

This module creates transport profiles for SFT model.

Functions:
    - meridional_flow: Generates meridional flow profile.
    - differential_rotation: Generates differential rotation profile.
    - vs: Generates meridional flow profile Yeates 2020 (Sol Phy).
"""
import numpy as np

def meridional_flow(grid, peak_speed=15.0):
    """
    Creates meridional circulation profile based on the solar latitude.
    
    Parameters:
        grid (dict): Dictionary containing grid information ('theta', 'phi', and their spacings).
        peak_speed (float): Peak speed of the meridional flow in m/s.

    Returns:
        np.ndarray: Meridional flow profile.
    """

    # Read the grid information
    theta = grid['colatitude']
    phi = grid['longitude']
    
    vth_mf_1D = vs(theta-np.pi/2, v0=peak_speed)
    v_theta = np.tile(vth_mf_1D, (phi.shape[0], 1)).T

    return v_theta

def differential_rotation(grid,rotation='solar',frame='carrington'):
    """
    Creates differential rotation profile based on the solar latitude.
    
    Parameters:
        grid (dict): Dictionary containing grid information ('theta', 'phi', and their spacings).
        rotation (str): Type of rotation profile ('solar' or 'rigid').
        frame (str): Frame of reference ('carrington' or 'synodic').
        
    Returns:
        np.ndarray: Differential rotation profile.
    """
    # Read the grid information
    theta = grid['colatitude']
    phi = grid['longitude']

    Colatitude, _ = np.meshgrid(theta,phi,indexing='ij')
    # Factor to convert degrees/day to radians/second
    rotation_rate_fact = 2.0201 * 10**(-7) 
    
    if rotation == 'solar':
        if frame not in ["carrington", "synodic"]:
                raise ValueError("For solar rotation, frame must be 'carrington' or 'synodic'.")
        if frame == 'carrington':
            omega_diff = ((13.38 - 360/27.2753) - 2.30 * np.cos(Colatitude) ** 2 - 1.62 * np.cos(Colatitude) ** 4) * rotation_rate_fact
        elif frame == 'synodic':
            omega_diff = ((13.38) - 2.30 * np.cos(Colatitude) ** 2 - 1.62 * np.cos(Colatitude) ** 4) * rotation_rate_fact
        else:
            # omega_diff = ((13.38) - 2.30 * np.cos(Colatitude) ** 2 - 1.62 * np.cos(Colatitude) ** 4) * rotation_rate_fact
            raise ValueError("For solar rotation, frame must be 'carrington' or 'synodic'.")
    elif rotation == 'rigid':
        # If the star rotates as a solid body e.g., rotation period = 2 days
        rotation_period = 2.0 # days
        rotation_freq = 360.0/(rotation_period) # degrees/day
        omega_diff = (rotation_freq*np.ones(Colatitude.shape)) 

    return omega_diff

def vs(theta, v0=1, p=2.33):
    Du = v0 * (1 + p) ** (0.5 * (p + 1)) / p ** (0.5 * p)
    return Du * np.sin(theta) * (np.cos(theta)) ** p