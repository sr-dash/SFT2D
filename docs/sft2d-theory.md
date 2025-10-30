---
bibliography:
  - assets/references.bib
title: Surface Flux Transport Model User Manual
---

# Theoretical background

## Magnetic field evolution on the solar surface

The surface flux transport (SFT) model, which solves the radial
component of the magnetic field on the solar/stellar surface, has demonstrated
remarkable effectiveness in simulating the dynamics of the large-scale
magnetic field on the solar photosphere. The governing equation can be written
as,

```{math}
\frac{\partial B_r}{\partial t}
+ \frac{1}{R_\odot \sin\theta}
  \frac{\partial}{\partial \theta}
  \!\left( \sin\theta\, u_\theta B_r \right)
+ \Omega(\theta)
  \frac{\partial B_r}{\partial \phi}
=
\frac{\eta}{R_\odot^2 \sin\theta}
  \frac{\partial}{\partial \theta}
  \!\left( \sin\theta\, \frac{\partial B_r}{\partial \theta} \right)
+ \frac{\eta}{R_\odot^2 \sin^2\theta}
  \frac{\partial^2 B_r}{\partial \phi^2}
+ S.
```

where `\eta` is the magnetic diffusivity, {math}`\Omega (s)`
is the angular velocity in east-west direction on a sine-latitude grid,
{math}`u_\theta` is the flow profile along north-south direction on a
latitude grid and {math}`\phi` is the longitude. As the velocity profiles
involved in transporting the magnetic flux on the photosphere is a
function of latitude only. The flux from newly emerging sun/star spots are coupled to the mode via an ad-hoc source term `S`.

```{figure} flows.png
:height: 70 %
:width: 70 %
:align: "center"
:alt: Meridional flow profile

Example meridional flow profile.
```

## BMR modeling algorithm

We follow the Bipolar Magnetic Region (BMR) modeling algorithm described in [Yeates (2020)](https://doi.org/10.1007/s11207-020-01688-y) to incorporate BMRs in SFT model using observed SHARP parameters. The location of the is modelled with the positive and negative polarity positions {math}`(s_+, \phi_+)` and {math}`(s_-,\phi_-)` on the computational grid. Here {math}`s` denotes sine-latitude and {math}`\phi` denotes (Carrington) longitude. Different properties of the source functions are modeles as following,

1. Centroid of the BMR,

```{math}
    s_0 = \frac12(s_+ + s_-),\qquad \phi_0 = \frac12(\phi_+ + \phi_-)
    \label{eqn:center}
```

2. Polarity separation, which is the heliographic angle,

```{math}
    \rho = \arccos\left[s_+s_- + \sqrt{1-s_+^2}\sqrt{1 - s_-^2}\cos(\phi_+-\phi_-) \right]
    \label{eqn:separation}
```

3. The tilt angle with respect to the equator, given by,

```{math}
    \gamma = \arctan\left[\frac{\arcsin(s_+) - \arcsin(s_-)}{\sqrt{1-s_0^2}(\phi_- - \phi_+)}\right]
    \label{eqn:tilt}
```

Together with the unsigned flux, {math}`|\Phi|`, these parameters define the
BMR as following. For an untilted BMR centered at
{math}`s=\phi=0`, this functional form is defined as

```{math}
    B(s,\phi) = F(s,\phi) = -B_0\frac{\phi}{\rho}\exp\left[-\frac{\phi^2 + 2\arcsin^2(s)}{(a\rho)^2}\right],
    \label{eqn:bmr}
```

where the amplitude {math}`B_0` is scaled to match the
corrected flux of the observed region on the computational grid. To
account for the location {math}`(s_0,\phi_0)` and tilt {math}`\gamma` of a general
region, we set {math}`B(s,\phi) = F(s',\phi')`, where {math}`(s',\phi')` are
spherical coordinates in a frame where the region is centered at
{math}`s'=\phi'=0` and untilted.

```{figure} example_bmr.png
:height: 70 %
:width: 70 %
:align: "center"
:alt: example modeled BMR

Example modeled BMR.
```
