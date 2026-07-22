### Surface Flux Transport (SFT) model

[![DOI](https://zenodo.org/badge/1084494967.svg)](https://doi.org/10.5281/zenodo.17459135)
[![Sphinx Documentation](https://github.com/sr-dash/SFT2D/actions/workflows/documentation.yml/badge.svg)](https://github.com/sr-dash/SFT2D/actions/workflows/documentation.yml)

Magnetic flux evolution on the solar/stellar surface is governed primarily by the large scale flows and flux cancellation.
It can be modelled by solving the magnetic induction in two dimension.
For a sun-like star where spherical geometry suits best to study the global surface flux distribution,
we can design a two dimensional SFT model with prescriptions of large-scale flow profiles and magnetic diffusivity for mimicing the
surface flux dynamics.

In this python-package, we develop a SFT model in 2D on a uniform latitude-longitude grid.

#### Numerics

The model is a **finite-volume** solver on a pole-to-pole spherical mesh. Both
spatial operators are written in flux form on the exact cell areas, so the
area-weighted total flux is conserved to machine precision — there are no
boundary conditions at the poles, because the poles are ordinary cells
(spherical caps of angular radius `dtheta/2`).

| ingredient | scheme |
|---|---|
| mesh | cell-centred, pole to pole, `n_theta` cells including both polar caps; `n_phi` unique longitude cells (periodic) |
| advection | conservative MUSCL/TVD with selectable limiter (`vanleer` default; also `minmod`, `mc`, `superbee`, `upwind1`) |
| diffusion | conservative 5-point Laplacian, symmetric negative semi-definite |
| time integration | Strang splitting (advection → diffusion → advection) |
| advection in time | SSPRK(3,3), sub-cycled at its own CFL |
| diffusion in time | **RKL2 super-time-stepping**, which absorbs the stiff near-pole longitudinal limit in `O(sqrt)` stages |

Super-time-stepping matters because the longitudinal diffusion term carries an
explicit limit `dt < (R dphi sin(theta))^2 / eta` that collapses at the poles.
RKL2 removes it without the near-pole Fourier filtering that explicit SFT codes
normally resort to, so the discretisation stays faithful to the equation being
solved. At 181×360 a one-day step takes ~62 RKL2 stages in place of ~960
explicit diffusion steps.

Verified against analytic free decay of the axisymmetric modes
(`exp(-l(l+1) eta t / R^2)`, matched to better than 0.01% for l = 1, 2, 3);
run `pytest tests/` for the full suite.

An example file is shared with the repository to test the model with basic parameter setting.

Feel free to modify and change the simulation parameters according to your requirements.

#### Installation instruction

The module is packaged to be installed with `pip`. It is recomended to create a `conda` enviornment specific to this module for testing.

1. Download a local copy (recomended for further testing/developing). If you are a collaborator you should have access to this repository.
   If you want to join us for development do let us know.

   ```
   git clone https://github.com/sr-dash/SFT2D.git
   ```

2. Create a new `conda` enviornment specific for `sft2d` package with

   ```
   conda env create -f environment.yml
   conda activate sft2d
   ```

3. Install the package with the following command.

   ```
   pip install git+https://github.com/sr-dash/SFT2D.git sft2d
   ```

   Since this is a private repository, you will need to access it with a personalized github token. You can create your own personalized token in your account developper settings.
   Let us know if you need help in this step.

After this you should be able to import sft2d module in any python enviornment or notebook.

#### Testing your installation

An example script is provided within the `sft2d/` directory:

```
python -m sft2d.example_run
```

It reports the grid extent, area closure, the chosen RKL2 stage / advection
sub-cycle counts, and the conservation and polar-field diagnostics. The
notebook `docs/notebooks/example-run.ipynb` walks through the same material
interactively.

To run the numerical test suite (mesh closure, flux conservation, operator
symmetry, analytic decay rates, TVD monotonicity, second-order convergence of
the polar-cap diagnostics):

```
pytest tests/
```

### We are currently developing the model and in the process of packaging it as a software.

For now you can just download the whole repository and run the example.py file to test the simulation.

### To-Do List

1. **Add BMR Modelling**

   - Implement the BMR (Bipolar Magnetic Region) modeling functionality.
   - Integrate BMR modeling into the Surface Flux Transport (SFT) model.
   - Provide users with an option to enable or disable BMR modeling.

2. **BMR Data Processing**
   - Process BMR properties from various sources (RGO/HMI).
   - Create standardized tables for BMR modeling input.
   - Ensure compatibility of data formats across different sources.
3. **Data assimilation**
   - Develop routines for magnetogram data assimilation with interpolation.
   - Cases for different sources HMI or any other Stellar processed data.
   - Calibrate fluxes and add as a source term to the model.
   - The `assimilate=` hook in `evolve` is already in place; what is missing is
     the callable that inserts observed B_r in a low-latitude window each
     Carrington rotation with flux balancing.
4. **Performance**
   - The operators are pure NumPy and cache their geometry, giving ~5 s per
     simulated year at 91×180 and ~45 s/yr at 181×360 on a laptop. A parameter
     scan is embarrassingly parallel across evaluations; run them in separate
     processes rather than threading the solver.

### RGO Sunspot property processing.

Observed sunspot properties are recorded by RGO and can be found [here](http://www.solarcyclescience.com/activeregions.html). To drive the SFT model, we have processed the sunspot properties.
The plot shows the butterfly diagram
![RGO Butterfly diagram](BMRs_sortedbyarea_1901-2025.png)

The sunspots are only considered at their maximum area record for any given group/noaa number.
A 13 month smoothed sunspot number is also plotted for reference.

![Sunspot time series](RGO_Sunspots_timeseries.png)

The datafile of compiled sunspot properties can be downloaded from [here](sunspot_data_rgo_1901_2025.csv).

#### Stay tuned for further updates. Contact us to collaborate on any of the to-do lists.

Contact: Soumyaranjan Dash
Email: sdash@nso.edu
