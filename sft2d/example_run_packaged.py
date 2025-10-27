"""
example_run_packaged.py

This is an example python script to check your SFT (2D) installation with a few quick checks.

"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from matplotlib import rc
import matplotlib.style
#plt.ion()
## Plotting canvas properties.
params = {'legend.fontsize': 12,
          'axes.labelsize': 10,
          'axes.titlesize': 10,
          'xtick.labelsize' :10,
          'ytick.labelsize': 10,
          'grid.color': 'k',
          'grid.linestyle': ':',
          'grid.linewidth': 0.5,
          'mathtext.fontset' : 'stix',
          'mathtext.rm'      : 'DejaVu serif',
          'font.family'      : 'DejaVu serif',
          'font.serif'       : "Times New Roman", # or "Times"          
         }
matplotlib.rcParams.update(params)

from sft2d import create_grid, initialize_field, meridional_flow, differential_rotation, calculate_time_step, calculate_diffusion, calculate_advection
from sft2d import calculate_usflx, calculate_dm, calculate_polar_field, plot_bfly, plot_mag


grid_sft = create_grid(180,360)
mf_ = meridional_flow(grid_sft.copy())
dr_ = differential_rotation(grid_sft.copy(),rotation='solar',frame='carrington')
field = initialize_field(grid_sft.copy(), 'read')
diffusivity = 2.5 * 10**8 # cm^2/s
ts, ndt = calculate_time_step(grid_sft.copy(), diffusivity)

colatitude = grid_sft['colatitude']
longitude = grid_sft['longitude']
delta_theta = grid_sft['dtheta']
delta_phi = grid_sft['dphi']
solar_radius = 6.955 * 10**8

num_days = 1*365  # Example total time steps
num_theta = grid_sft['colatitude'].size
num_phi = grid_sft['longitude'].size
bfly_data = np.zeros((num_days+1,num_theta))
all_br_data = np.zeros((num_days+1,num_theta,num_phi))
all_br_data[0,:,:] = field.copy()
bfly_data[0,:] = np.mean(field,axis=1)

B_temp = field.copy()
B_temp_update = np.zeros_like(field)

# Time loop for evolution
for t in tqdm(range(1,num_days+1),desc='Simulation days: '):
    delta_t = ts
    for steps in range(ndt):
        # Calculate the diffusion term
        B_temp_diff = calculate_diffusion(B_temp, diffusivity, grid_sft.copy())
        # Calculate the advection term
        B_temp_adv = calculate_advection(B_temp, dr_, mf_, grid_sft.copy())
        
        # Update magnetic field using all terms
        B_temp_update[1:-1, 1:-1] = B_temp[1:-1, 1:-1] + delta_t * (1.0*B_temp_diff - 1.0*B_temp_adv)

        # Apply periodic boundary conditions in the phi direction
        B_temp_update[:, 0] = B_temp_update[:, -2]  # First column matches second-to-last column
        B_temp_update[:, -1] = B_temp_update[:, 1]  # Last column matches second column

        # Apply open boundary conditions in the theta direction
        B_temp_update[0, :] = B_temp_update[1, :]    # Northern boundary (pole)
        B_temp_update[-1, :] = B_temp_update[-2, :]  # Southern boundary (pole)
        B_temp = B_temp_update.copy()
    # Save the butterfly diagram
    bfly_data[t,:] = np.mean(B_temp,axis=1)
    all_br_data[t,:,:] = B_temp.copy()
    

# Calculate the unsigned magnetic flux on the solar surface, axial dipole moment and 
# polar fields near high latitudes (currently se to 55 degrees and above near both hemispheres).
usflx = calculate_usflx(all_br_data, grid_sft.copy(), [0,num_days])
dm_sft = calculate_dm(all_br_data, grid_sft.copy(), [0,num_days])
polar_n, polar_s = calculate_polar_field(all_br_data, grid_sft.copy(), [0,num_days])

# Plot the butterfly diagram 
# set save_path='path/to/save/plot.png' to save the plot as a .png file
plot_bfly(bfly_data, grid_sft.copy())

# Plot the magnetic field for a specific time step e.g., 5th day (in this case)
# set save_path='path/to/save/plot.png' to save the plot as a .png file
plot_mag(all_br_data[5,:,:], grid_sft.copy())
