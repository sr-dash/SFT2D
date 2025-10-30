API Reference
=============

.. note::
   This section documents the main API of the `sft2d` package.

Top-level functions exposed by `sft2d/__init__.py`:

.. autosummary::
   :toctree: _autosummary
   :recursive:

   # Advection
   sft2d.calculate_advection

   # Diffusion
   sft2d.calculate_diffusion

   # Time stepping
   sft2d.calculate_time_step

   # Grid
   sft2d.create_grid

   # Initial conditions
   sft2d.initialize_field

   # Transport profiles
   sft2d.meridional_flow
   sft2d.differential_rotation

   # Analysis
   sft2d.calculate_usflx
   sft2d.calculate_dm
   sft2d.calculate_polar_field
   sft2d.calculate_polar_flux

   # Visualization
   sft2d.plot_bfly
   sft2d.plot_mag

