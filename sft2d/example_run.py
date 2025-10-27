"""
    This is an example file to run the model with standard set of parameters.
    A small visualization routine is also provided to plot and check the outputs.

    Currently we are developing this model and more functionalities will be added in due course.
    Numerical schemes are also being tested.

    Author: Soumyaranjan Dash
    Email: dash.soumya922@gmail.com
    Date: 15th Jan 2025

"""
# Import basic packages for simulation and visualization
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import packagess from the SFT model (For local installation)
# Most likely this will also run after a simple clone of the repo. But it is recommended to install the package.
# If you find any errors, please raise an issue in the GitHub repository.

from src.grid import create_grid
from src.initial_conditions import initialize_field
from src.transport_profiles import meridional_flow, differential_rotation
from src.time_step import calculate_time_step
from src.diffusion import calculate_diffusion
from src.advection import calculate_advection
from analysis.analysis import calculate_usflx, calculate_dm, calculate_polar_field
from analysis.visualize import plot_bfly
from analysis.visualize import plot_mag


# Import packages from the SFT model (For package installation)
# from sft2d import create_grid, initialize_field, meridional_flow, differential_rotation, calculate_time_step, calculate_diffusion, calculate_advection

# Define and initialize some essential model input parameters.

## Grid creation (180x360) Uniform in latitude-longitude
grid_sft = create_grid(180,360)
## Meridional flow profile
mf_ = meridional_flow(grid_sft.copy())
## Differential rotation profile
dr_ = differential_rotation(grid_sft.copy())
## Initial magnetic field (dipole)
## There is a choice between Dipole or HMI CR map for initial condition
## For HMI map, set 'read' in place of 'dipole'.
field = initialize_field(grid_sft.copy(), 'dipole')
## Magnetic diffusivity
diffusivity = 2.5 * 10**8 # cm^2/s
## Time step calculation
ts, ndt = calculate_time_step(grid_sft.copy(), diffusivity)

## Create basic grid for visualization
colatitude = grid_sft['colatitude']
longitude = grid_sft['longitude']
delta_theta = grid_sft['dtheta']
delta_phi = grid_sft['dphi']
solar_radius = 6.955 * 10**8  # Solar radius in meters

## Visualize the flow profiles (set 1 to show the plot)
if 0:
    plt.figure(figsize=[9,3])
    ax1 = plt.subplot(121)
    ax1.plot(np.rad2deg(grid_sft['colatitude']-np.pi/2),mf_[:,180],color='k',label='Meridional flow',lw=2)
    ax1.set_xlabel('Latitude [deg]')
    ax1.set_ylabel('Velocity [m/s]')
    ax1.set_title('Meridional flow')
    ax1.set_xlim([-90,90])
    ax1.set_xticks(np.arange(-90,91,30))

    ax2 = plt.subplot(122)
    ax2.plot(np.rad2deg(grid_sft['colatitude']-np.pi/2),dr_[:,180]*np.sin(grid_sft['colatitude'])*solar_radius,c='k',label='Differential rotation',lw=2)
    ax2.set_xlabel('Latitude [deg]')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_title('Differential rotation')
    ax2.set_xlim([-90,90])
    ax2.set_xticks(np.arange(-90,91,30))

    plt.subplots_adjust(wspace=0.3)
    plt.show()

## Run the SFT evolution.

## Total number of days to run the simulation
## Time-step is rounded off to fit exactly into one day
num_days = 15*365  # Example total time [days]

## Grid properties
num_theta = grid_sft['colatitude'].size
num_phi = grid_sft['longitude'].size

## Initialize arrays to store the data
bfly_data = np.zeros((num_days+1,num_theta))
all_br_data = np.zeros((num_days+1,num_theta,num_phi))
all_br_data[0,:,:] = field.copy()
bfly_data[0,:] = np.mean(field,axis=1)

## Define temporary arrays for updating the field
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
        B_temp_update = B_temp + delta_t * (1.0*B_temp_diff - 1.0*B_temp_adv)

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

## Visualize the butterfly diagram (set 1 to run the following script)
if 0:
    plt.figure(figsize=[7,3])
    ax1 = plt.subplot(111)
    bmax = 5
    pm1 = ax1.pcolormesh(np.arange(num_days+1),np.rad2deg(grid_sft['colatitude']-np.pi/2),bfly_data.T,cmap='bwr',vmax=bmax,vmin=-bmax)
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Latitude [deg]')
    plt.colorbar(pm1)
    plt.show()

## Calculate the total unsigned flux from the magnetic field
def calc_usflx(all_br_data):
    temp_bsinth = np.zeros_like(all_br_data[0,:,:])
    usflx = np.zeros(num_days+1)
    for i in range(num_days+1):
        for i1 in range(longitude.shape[0]):
            temp_bsinth[:,i1] = np.abs(all_br_data[i,:,i1]) * np.sin(colatitude)
        usf_1d = np.sum(temp_bsinth) * delta_theta * delta_phi * (solar_radius*1e2)**2
        usflx[i] = usf_1d
    return usflx

## Calculate the total unsigned flux
usflx_diff = calc_usflx(all_br_data)

## Plot the USFLUX variation (set 1 to run the following script)
if 0:
    plt.plot(np.arange(num_days+1),usflx_diff)
    plt.xlabel('Time [days]')
    plt.ylabel('Total unsigned flux [Mx]')
    plt.title('Total unsigned flux')
    plt.show()

