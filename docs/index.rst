.. Example documentation master file, created by
   sphinx-quickstart on Sat Sep 23 20:35:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. |date| date::
   Last Updated on %Y-%m-%d.
Surface Flux Transport 2D documentation
=======================================

Radiative and energetic output on our host star is primarily governed by the surface plasma and magnetic field dynamics.
Hence, it is essential for us to understand the magnetic field distribution on the solar surface. However, 
with our current observational capabilities, we can only observe half of the solar surface routinely. Even direct 
observations are not reliable near high latitudes due to the forshortening effect. In this context the unobserved 
farside of the Sun also poses increased uncertainity on the estimation of global surface magnetic flux estimates.
Such problems are more pronounced for observations of other magnetically active stellar objects due to short observating window and low resolving power.

Magnetic fields generated in the interior of a star manifests on the surface due to bouyancy subjected to large scale plasma flows on the surface.
This phenomena distributes the magnetic field globally along with possible flux cancellation processes. With the available observations and our
theoretical understanding, we can numerically model this flux transport process by solving the magnetic induction equation. 
Such modelled global distribution of the surface magnetic field is
essential for understanding and forecasting solar/stellar activity and evolution. Sun is magnetically active and this 
activity cycle lasts for about 11 years during which the magnetic field distribution on the solar
surface -- the photosphere, evolves and generates a vast range of events e.g., emergece and 
decay of sunspots, solar flares, coronal mass ejections, flow of solar wind and energetic particles
into the heliosphere etc. Surface flux transport model solves the radial component of the magnetic induction equation 
on the solar surface by using prescribed flow profiles, flux cencellation process and source terms. This code suite provides
a numerical model of SFT developed in PYTHON.

Link to the GitHub repository: `https://sr-dash.github.io/sft2d/ <https://sr-dash.github.io/sft2d/>`_.

.. note::
   The code is under active development and is not yet ready for production use. 
   Please report any issues on the GitHub repository.

|date|

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart.md
   notebooks/example-run.ipynb
   sft2d-theory.md

..
   some-feature.md
   another-feature.md
   sample0.md
   sample1.md
