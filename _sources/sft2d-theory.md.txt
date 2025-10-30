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

where {math}`\eta` is the magnetic diffusivity, {math}`\Omega (s)`
is the angular velocity in east-west direction on a uniform latitude grid,
{math}`u_\theta` is the flow profile along north-south direction on a
latitude grid and {math}`\phi` is the longitude. The velocity profiles
involved in transporting the magnetic flux on the photosphere is a
function of latitude only. The flux from newly emerging sun/star spots are coupled to the model via an ad-hoc source term {math}`S`.

```{figure} flows.png
:height: 70 %
:width: 70 %
:align: "center"
:alt: Meridional flow profile

Example meridional flow profile.
```

## Bipolar Magnetic Region (BMR) Flux Injection — Theoretical Explanation

Here is how we model the **bipolar magnetic region (BMR) source term** in our SFT simulation.  
It describes how a pair of magnetic polarities (leading and following) is injected on the solar/stellar surface, following **Joy’s law** tilt and **Hale’s polarity law**.  
The resulting field is a **normalized radial magnetic field distribution** representing the emergence of one BMR with a specified total flux and geometric configuration.

---

### 1. Mesh and Area Elements

The surface of the Sun is represented using a **spherical grid**:

```{math}
\theta \in [0, \pi], \quad \phi \in [0, 2\pi)
```

The **surface element** on a sphere of radius \( R_\odot \) is:

```{math}
dA = R_\odot^2 \sin\theta \, d\theta \, d\phi
```

This is used to compute total flux over the spherical surface.

---

### 2. Convert Center Location

The central location of the bipole in latitude {math}`\lambda` and longitude {math}`\phi` is converted to **colatitude**:

```{math}
\theta_0 = \frac{\pi}{2} - \lambda, \quad \phi_0 = \phi \mod 2\pi
```

The corresponding unit vector on the sphere is:

```{math}
\mathbf{r}_0 =
\begin{bmatrix}
\cos\lambda \cos\phi \\
\cos\lambda \sin\phi \\
\sin\lambda
\end{bmatrix}
```

---

### 3. Local Tangent Basis

At the emergence center, a **local tangent plane** is defined using orthonormal vectors:

```{math}
\mathbf{e}_\phi = 
\begin{bmatrix} -\sin\phi \\ \cos\phi \\ 0 \end{bmatrix} , \quad
\mathbf{e}_\theta =
\begin{bmatrix} -\sin\lambda \cos\phi \\ -\sin\lambda \sin\phi \\ \cos\lambda \end{bmatrix} , \quad
\mathbf{e}_\lambda = - \mathbf{e}_\theta
```

- {math}`\mathbf{e}_\phi` → eastward direction  
- {math}`\mathbf{e}_\lambda` → northward direction  

---

### 4. Tilt and Separation

The **tilt angle** {math}`\alpha` defines the orientation of the bipole line relative to the local east-west direction:

```{math}
\mathbf{s} = \cos\alpha \, \mathbf{e}_\phi + \sin\alpha \, \mathbf{e}_\lambda
```

The **leading** and **following polarity centers** are positioned symmetrically along the separation vector:

```{math}
\mathbf{r}_\text{lead} = \mathbf{r}_0 + \frac{\Delta}{2} \mathbf{s}, \quad
\mathbf{r}_\text{foll} = \mathbf{r}_0 - \frac{\Delta}{2} \mathbf{s}
```

where {math}`\Delta` is the angular separation (in radians).

---

### 5. Convert to Spherical Coordinates

The 3D vectors are converted back to spherical coordinates:

```{math}
\theta = \arccos\left(\frac{z}{|\mathbf{r}|}\right), \quad
\phi = \arctan2(y, x) \mod 2\pi
```

---

### 6. Polarity Signs (Hale’s Law)

The polarity signs are determined according to hemisphere:

```{math}
(\text{sign}_\text{lead}, \text{sign}_\text{foll}) =
\begin{cases}
(+1, -1), & \lambda > 0 \text{ (northern hemisphere)} \\
(-1, +1), & \lambda < 0 \text{ (southern hemisphere)}
\end{cases}
```

Optionally, Hale’s law can be disabled and the leading polarity is set positive.

---

### 7. Gaussian Field Distribution

Each polarity is modeled as a **2D Gaussian** on the sphere:

```{math}
B_i(\theta, \phi) = s_i \exp\left[ -\frac{(\theta - \theta_i)^2 + \Delta\phi^2(\phi, \phi_i)}{2\sigma^2} \right]
```

where:

```{math}
\Delta\phi(\phi, \phi_i) = \min\left(|\phi - \phi_i|, 2\pi - |\phi - \phi_i|\right)
```

- {math}`\sigma` → angular width of the polarity  
- {math}`s_i = \pm 1` → polarity sign  

The **combined unscaled radial field** is:

```{math}
B_\text{unit}(\theta, \phi) = B_\text{lead}(\theta, \phi) + B_\text{foll}(\theta, \phi)
```

---

### 8. Flux Normalization

Compute the total unsigned flux:

```{math}
\Phi_\text{unit} = \sum |B_\text{unit}| \, dA
```

Scale the field to match the specified total flux \( \Phi \):

```{math}
S = \frac{\Phi}{\Phi_\text{unit}}
```

The **normalized radial field** is then:

```{math}
B_r(\theta, \phi) = S \, B_\text{unit}(\theta, \phi)
```

ensuring:

```{math}
\int |B_r(\theta, \phi)| \, dA = \Phi
```

---

### 9. Final Equation

The normalized radial magnetic field of the bipole is:

```{math}
B_r(\theta, \phi) = \frac{\Phi}{\sum |B_\text{unit}| \, dA} 
\left[
s_\text{lead} e^{- \frac{(\theta - \theta_\text{lead})^2 + \Delta\phi^2(\phi, \phi_\text{lead})}{2\sigma^2}} +
s_\text{foll} e^{- \frac{(\theta - \theta_\text{foll})^2 + \Delta\phi^2(\phi, \phi_\text{foll})}{2\sigma^2}}
\right]
```

This is the **mathematical representation of the BMR source term** used in surface flux transport simulations.

---

### 10. Summary

- Represents **idealized BMR emergence** on the solar surface.  
- Incorporates **Joy’s law tilt** and **Hale’s law polarity orientation**.  
- Normalized to ensure the **total unsigned flux** equals the specified value.  
- Provides a **smooth Gaussian representation** suitable for numerical simulations.


```{figure} example_bmr.png
:height: 70 %
:width: 70 %
:align: "center"
:alt: example modeled BMR

Example modeled BMR.
```
