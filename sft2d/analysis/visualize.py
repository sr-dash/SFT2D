"""
visualize.py

This module handles the visualization of SFT (2D) outputs.

Functions:
    - plot_bfly: Plots the butterfly diagram of the surface magnetic field.
    - plot_mag: Plots the magnetic field distribution.
"""

def plot_bfly(bfly_sft, grid, bmax = 10, save_path=None, show_plot=True):
    """
    Plots the butterfly diagram of the surface magnetic field.
    
    Parameters:
        bfly_sft (np.ndarray): 2D array containing the surface magnetic field data [time, latitude].
        grid (dict): Dictionary containing grid information ('theta', 'phi', and their spacings).
        save_path (str, optional): Path to save the plot. If None, the plot is displayed interactively.
        show_plot (bool, optional): If True, the plot is displayed interactively.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Read the grid information
    colatitude = grid['colatitude']
    num_days = bfly_sft.shape[0]

    # Generate the plot
    plt.figure(figsize=(7,3))
    ax1 = plt.subplot(111)
    bmax = bmax
    pm1 = ax1.pcolormesh(np.arange(num_days),np.rad2deg(np.pi/2 - colatitude),bfly_sft.T,cmap='bwr',vmax=bmax,vmin=-bmax)
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Latitude [deg]')
    ax1.set_title('Butterfly diagram of B$_r$')
    plt.colorbar(pm1,ax=ax1,label='B$_r$ [G]')
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        print(f"Plot saved to {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_mag(b_data, grid, bmax = 30, save_path=None, show_plot=True):
    """
    Plots the magnetic field strength at the solar surface.
    
    Parameters:
        b_data (np.ndarray): 2D array containing the surface magnetic field strength data [latitude, longitude].
        grid (dict): Dictionary containing grid information ('theta', 'phi', and their spacings).
        bmax (float, optional): Maximum magnetic field strength for the colorbar.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed interactively.
        show_plot (bool, optional): If True, the plot is displayed interactively.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Read the grid information
    colatitude = grid['colatitude']
    longitude = grid['longitude']

    # Generate the plot
    plt.figure(figsize=(7,3))
    ax1 = plt.subplot(111)
    pm1 = ax1.pcolormesh(np.rad2deg(longitude),np.rad2deg(np.pi/2 - colatitude),b_data,cmap='seismic',vmax = bmax, vmin = -bmax)
    ax1.set_xlabel('Longitude [deg]')
    ax1.set_ylabel('Latitude [deg]')
    ax1.set_title('Magnetic field strength at the solar surface')
    plt.colorbar(pm1,ax=ax1,label='B [G]')
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        print(f"Plot saved to {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()