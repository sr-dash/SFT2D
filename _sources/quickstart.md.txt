# Quick Start

This document describes a working prototype of the Surface Flux
Transport model (SFT). The SFT was developed to provide a computationaly
inexpensive solver to explore the parameter space to better understand
the evolution of solar/stellar surface magnetic field evolution in response to
the changes in source functions and transport parameters. The model assumes no 
back-reaction on the transport profiles from the magnetic field distribution.
In its current form the SFT is numerically stable and provides a user-friendly test-bed
to understand how the properties of sun/star spots, advection profiles and
diffusivity impact the global solar/stellar surface magnetic field evolution.

The core package of the SFT model is written in Python. In the solar context the spatial resolution, 
diffusivity values and flow parameters, the code does not demand extreeme computational resources.  
Parallelization of teh computations within the script is planned for future releases. 
The model can be run in a Jupyter-Notebook and the magnetic field maps can be exported to any format as per the user.

## Acknowledgments

Original code for SFT model in one dimension was developed by [Yeates (2020)](https://doi.org/10.1007/s11207-020-01688-y) in Python. This distribution is a two dimensional version of the Surface Flux Transport model on a spherical polar coordinate system.

## System Requirements

In order to install and run the SFT 2D the following minimum system
requirements apply.

- The package requires a working Python enviornment under any operating system.

- Certain python packages are required for running and visualization of results.
  The requirements file contains the list of packages.

The code solves radial component of the magnetic induction equation in spherical polar coordinates.
It can run on a single processor with a nominal RAM memory with reasonable spatial resolution.

In addition to the above requirements, the SFT output is typically
visualized using Python. Other visualization packages may also be used,
but the output file formats should be suported by those visualization
softwares.

## A Brief Description of the SFT 2D code

The distribution in the form of the compressed tar image includes the
SFT source code. The top level directory contains the following
subdirectories:

- `docs` - the documentation directory

- `src` - all the python routines for running the code

- `data` - some example HMI data
