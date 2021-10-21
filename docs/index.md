---
title: Glider experiments
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Optimal glider trajectories using direct collocation method and non-linear programming

[opty](https://opty.readthedocs.io/en/latest/theory.html)

## 4D model

$$
X = \begin{pmatrix} x \\  y \\ z \\ \psi \end{pmatrix}, \quad
U = \begin{pmatrix} v_a \\ \phi \end{pmatrix}
$$

$$
\begin{align}
\dot{X} &= f(X, U) \\
\dot{X} &= \begin{pmatrix} v_a \cos{\psi} + w_x \\ v_a \sin{\psi} + w_y \\ \text{polar}(v_a, \phi) + w_z \\ \frac{g}{v_a} \tan{\phi} \end{pmatrix}
\end{align}
$$


$$
\text{polar}(v_a, \phi) = -v_a(C1+\frac{C_2}{\cos{\phi}})
$$


## Thermaling
<figure>
  <img src="../plots/glider_4d_thermal_en.png" alt="" width="304" height="228">
  <img src="../plots/glider_4d_thermal_nu.png" alt="" width="304" height="228">
  <img src="../plots/glider_4d_thermal_chrono.png" alt="" width="304" height="228">
  <img src="../plots/glider_4d_thermal_3D.png" alt="" width="304" height="228">
  <figcaption>Fig1.</figcaption>
</figure>


## Slope soaring


## 2D Wind field

