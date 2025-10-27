"""
initial_conditions.py

This module handles the initial conditions for the Solar Surface Flux Transport (SFT) model.

Functions:
    - initialize_field: Generates a uniform grid in spherical polar coordinates (theta, phi).
"""

import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator as rgi

def initialize_field(grid, field_type='dipole'):
    """
    Creates initial condition based on user choice (Dipole/Read from fits file).

    Parameters:
        grid (dict): Dictionary containing grid information ('theta', 'phi', and their spacings).
        field_type: Choice for setting the initial field as global dipole or read data from Carrington Rotation fits file.

    Returns:
        np.ndarray: Initial magnetic field.
    """
    # Read the grid information
    theta = grid['colatitude']
    phi = grid['longitude']


    # Number of grid points in theta and phi directions
    num_theta = theta.size
    num_phi = phi.size

    # Initialize the magnetic field based on the chosen field type
    B_init = np.zeros((num_theta, num_phi))

    if field_type == 'dipole':
        dipole_strength = 1.0
        B_init = dipole_strength*np.outer(np.abs(np.sin(np.pi/2-theta))**7*(np.sin(np.pi/2-theta)),np.ones(phi.shape[0]))
    elif field_type == 'read':
        hmi_br = fits.getdata('./data/hmi_CR2097.fits')[::-1,:]
        

        # Coordinates of original map (pretend it goes only once around Sun in longitude!):
        # Latitude axis is in sine of latitude.
        nsm = np.size(hmi_br, axis=0)
        npm = np.size(hmi_br, axis=1)
        dsm = 2.0 / nsm

        # Leaving a small gap at the poles for smoother interpolation:
        scm = np.flip(np.arccos(np.linspace(-1 + 0.05 * dsm, 1 - 0.05 * dsm, nsm)))
        pcm = np.linspace(0, 2 * np.pi, npm)

        # Interpolate to the computational grid:
        bri = rgi((scm, pcm), hmi_br, method="nearest", bounds_error=False, fill_value=0)
        # B_init = np.zeros((num_theta, num_phi))

        for i in range(num_theta):
            for j in range(num_phi):
                B_init[i, j] = bri((theta[i], phi[j]))


    # Correct for flux imbalance (if any)
    B_init = correct_flux_multiplicative(B_init)

    # Return the initial magnetic field array
    return B_init

def correct_flux_multiplicative(f):
    """
        Corrects the flux balance in the map f (assumes that cells have equal area).
    """
    
    # Compute positive and negative fluxes:
    ipos = f > 0
    ineg = f < 0
    fluxp = np.abs(np.sum(f[ipos]))
    fluxn = np.abs(np.sum(f[ineg]))
    
    # Rescale both polarities to mean:
    fluxmn = 0.5*(fluxn + fluxp)
    f1 = f.copy()
    f1[ineg] *= fluxmn/fluxn
    f1[ipos] *= fluxmn/fluxp
    
    return f1
