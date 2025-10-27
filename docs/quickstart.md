# Quick Start

This document describes a working prototype of the Surface Flux
Transport model (SFT). The SFT was developed to provide a computationaly
inexpensive solver to explore the parameter space to better understand
the evolution of solar surface magnetic field evolution in response to
the changes in source functions and transport parameters. In its current
form the SFT is numerically stable and provides a user-friendly test-bed
to understand how the properties of sunspots, advection profiles and
diffusivity impact the global solar surface magnetic field evolution.

The core of the SFT are written in Fortran 90. As the equations are
simplified to evolve in one dimensions, presently the code does not
implement parallel communications libraries. The SFT creates a single
executable which can be compiled with netCDF libraries and run on any
computer as long as the GNU compiler and netCDF libraries are properly
installed.

## Acknowledgments

Original code was developed by [Yeates (2020)](https://doi.org/10.1007/s11207-020-01688-y) in Python. This distribution is a reproduction in FORTRAN.

## System Requirements

In order to install and run the SFT 1D the following minimum system
requirements apply.

- The SFT runs only under the UNIX/Linux operating systems. This now
  includes Macintosh system 10.x because it is based on BSD UNIX. It
  does not run under any Microsoft Windows operating system.

- A GNU FORTRAN compiler must be installed.

- The file writing subroutine use netCDF output. For these the
  serial/parallel version of the netCDF library has to be installed.

As the code solves the magnetic field evolution in only one dimension,
it does not need huge computing power or higher memory load. It can run
on a single processor with a nominal RAM memory.

In addition to the above requirements, the SFT output is typically
visualized using Python. Other visualization packages may also be used,
but the output file formats should be suported by those visualization
softwares.

## A Brief Description of the SFT 1D code

The distribution in the form of the compressed tar image includes the
SFT source code. The top level directory contains the following
subdirectories:

- `docs` - the documentation directory

- `src` - all the python routines for running the code

- `data` - some example HMI data
