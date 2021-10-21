---
title: Glider experiments
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
$$
   \newcommand{\vect}[1]{\underline{#1}}
   \newcommand{\est}[1]{\hat{#1}}
   \newcommand{\err}[1]{\tilde{#1}}
   \newcommand{\pd}[2]{\frac{\partial{#1}}{\partial{#2}}}
   \newcommand{\transp}[1]{#1^{T}}
   \newcommand{\inv}[1]{#1^{-1}}
   \newcommand{\norm}[1]{|{#1}|}
   \newcommand{\mat}[1]{\mathbf{#1}}
   \newcommand{\jac}[3]{\frac{\partial{#1}}{\partial{#2}}\bigg\vert_{#3}}
$$

# Optimal glider trajectories using direct collocation method and non-linear programming

We're using [opty](https://opty.readthedocs.io/en/latest/theory.html) to compute optimal trajectories for a glider.

## 4D model

We describe our glider by a simple dimension 4 model ([implementation)](https://github.com/poine/glider_experiments/blob/391521924e6252385f335eeeed672858713927bf/src/solve_glider_opty_4d.py#L41).

$$x, y, z$$ are the coordinates of the glider's center of gravity in an euclidian space, $$v_a$$ is its airspeed, $$\phi$$ and $$\psi$$ are respectively roll and heading angles. $$w_x, w_y, w_z$$ are the wind velocity's components.

Using $$X = \transp{\begin{pmatrix} x &  y & z & \psi \end{pmatrix}}$$ as state vector and $$ U = \transp{\begin{pmatrix} v_a & \phi \end{pmatrix}}$$ as control input, we obtain the following state space representation:

$$
\begin{align}
\dot{X} &= f(X, U) \\
\begin{pmatrix} \dot{x} \\ \dot{y} \\ \dot{z} \\ \dot{\psi}\end{pmatrix} &= \begin{pmatrix} v_a \cos{\psi} + w_x \\ v_a \sin{\psi} + w_y \\ \dot{z}_a(v_a, \phi) + w_z \\ \frac{g}{v_a} \tan{\phi} \end{pmatrix}
\end{align}
$$

Lines 1 and 2 of the dynamics are pure kinematics. Line 3 ([ref](https://arxiv.org/pdf/1802.08215.pdf)) is a polynomial model of the glider sink rate as a function of airspeed and bank angle.

$$
CL_0 = \frac{K}{v_a^2}
$$


$$
\dot{z}_a(v_a, \phi) = -v_a(\frac{CD_0}{CL_0}+\frac{B*CL_0}{\cos^2{\phi}})
$$

[implementation](https://github.com/poine/glider_experiments/blob/391521924e6252385f335eeeed672858713927bf/src/glider_opty_utils.py#L158)

The model is fitted in simulation on the trajectories of an aerodynamically correct [simulator](https://github.com/poine/pat) [(see)](https://github.com/poine/pat/blob/0fb0fbfb46e13d030742a8ec61b942eef17e2a54/src/pat3/test/fixed_wing/fit_netto_vario.py). In real life, it would have to be fitted on real flight trajectories (trimmed at different velocities and bank angles).

Line 4 is from a coordinated turn hypothesis (no slip turn).



## Thermaling

The thermal is described by a wharignton model:

$$
   w_z = w_{z0} e^{\frac{-r^2}{r_0^2}}
$$


As objective function, we tested both altitude gain ($$J = z(t_f)$$) and average altitude ($$J = \Sigma_{k=0}^n z(t_k)/n$$) ([see](https://github.com/poine/glider_experiments/blob/391521924e6252385f335eeeed672858713927bf/src/solve_glider_opty_4d.py#L26)).


<figure>
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_thermal_en.png" alt="" width="304" height="228">
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_thermal_nu.png" alt="" width="304" height="228">
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_thermal_chrono.png" alt="" width="304" height="228">
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_thermal_3D.png" alt="" width="304" height="228">
  <figcaption>Fig1. Thermaling</figcaption>
</figure>


## Slope soaring

Here we use a simple analytic model of wind blowing across a cylindrical obstacle ([see](https://github.com/poine/glider_experiments/blob/391521924e6252385f335eeeed672858713927bf/src/glider_opty_utils.py#L68)). 

The objective function can be the same as for thermaling (final or mean altitude).

<figure>
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_slope_en.png" alt="" width="304" height="228">
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_slope_nu.png" alt="" width="304" height="228">
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_slope_chrono.png" alt="" width="304" height="228">
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_slope_3D.png" alt="" width="304" height="228">
  <figcaption>Fig2. slope soaring</figcaption>
</figure>

The solver returns a convincing trajectory that is similar to the one a human pilot would perform and consisting in lines parallel to the ridge, connected by headwind turns, the lines going further away from the ridge as altitude increases.


## 2D Wind field

We can fly to a waypoint by using the average distance to the waypoint as objective function.
<figure>
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_wp0_en.png" alt="" width="304" height="228">
  <!--<img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_wp0_nu.png" alt="" width="304" height="228">-->
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_wp0_chrono.png" alt="" width="304" height="228">
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_wp0_3D.png" alt="" width="304" height="228">
  <figcaption>Fig3. going to waypoint without wind</figcaption>
</figure>

We are now able to find an optimal trajectory to a waypoint in a 2D wind gradient.
<figure>
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_wp1_en.png" alt="" width="304" height="228">
  <!--<img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_wp1_nu.png" alt="" width="304" height="228">-->
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_wp1_chrono.png" alt="" width="304" height="228">
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_wp1_3D.png" alt="" width="304" height="228">
  <figcaption>Fig4. going to waypoint in a 2D wind field</figcaption>
</figure>

## Cross country

If we add a constraint on altitude (forbiding us from hitting ground), we can find a trajectory that hunts thermals along the way.
<figure>
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_cc_en.png" alt="" width="304" height="228">
  <!--<img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_cc_nu.png" alt="" width="304" height="228">-->
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_cc_chrono.png" alt="" width="304" height="228">
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_cc_3D.png" alt="" width="304" height="228">
  <figcaption>Fig5. Cross country</figcaption>
</figure>

If we give it a weak thermal, it might decide to skip it and instead gain altitude by performing loops in the stronger one.
<figure>
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_cc1_en.png" alt="" width="304" height="228">
  <!--<img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_cc1_nu.png" alt="" width="304" height="228">-->
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_cc1_chrono.png" alt="" width="304" height="228">
  <img src="https://raw.githubusercontent.com/poine/glider_experiments/master/docs/plots/glider_4d_cc1_3D.png" alt="" width="304" height="228">
  <figcaption>Fig6. The trajectory favors strong thermals</figcaption>
</figure>


## What now?
