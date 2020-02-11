#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Optimal glider soaring trajectories

"""
import pickle

from collections import OrderedDict

import numpy as np
import sympy as sym
from opty.direct_collocation import Problem
from opty.utils import building_docs
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pat3.atmosphere as p3_atm
import pat3.vehicles.fixed_wing.guidance_ardusoaring as p3_guid
import pat3.plot_utils as p3_plu
import pdb

#target_angle = np.pi
if 0:
    duration = 20#120.0 # 10
    num_nodes = 1000#4000 # 500
else:
    duration = 100#120.0 # 10
    num_nodes = 5000#4000 # 500

    
interval_value = duration / (num_nodes - 1)
print('interval_value: {:.3f}s ({:.1f}hz)'.format(interval_value, 1./interval_value))

obj_last_z = False
obj_summ_z = True
obj_summ_z2 = False


# Symbolic equations of motion
I, m, g, d, v, t = sym.symbols('I, m, g, d, v, t')
atm_r, atm_s = sym.symbols('atm_r, atm_s')
x, y, z, phi, psi = sym.symbols('x, y, z, phi, psi', cls=sym.Function)

state_symbols = (x(t), y(t), z(t), psi(t))
constant_symbols = (I, m, g, d)
specified_symbols = (phi(t),)

# Specify the known system parameters.
par_map = OrderedDict()
par_map[g] = 9.81
par_map[v] = 8.0
#par_map[atm_r] = 50.
#par_map[atm_s] = 0.4

#def atm_zd(_x, _y, _z, _t):
#    return  par_map[atm_s]*sym.exp(-(_x**2 + _y**2)/par_map[atm_r]**2)

class AtmosphereWharingtonSym(p3_atm.AtmosphereWharington):
    def get_wind_sym(self, _x, _y, _z, _t):
        return  self.strength*sym.exp(-(_x**2 + _y**2)/self.radius**2)

atm = AtmosphereWharingtonSym(center=[0, 0, 0], radius=40, strength=0.4)


#netto_vario = p3_guid.NettoVario()
def glider_zd(va, phi):
    polar_K   =  49.5050092764    # 25.6   # Cl factor 2*m*g/(rho*S) @Units: m.m/s/s
    polar_CD0 =   0.0122440667444 #  0.027 # Zero lift drag coef
    polar_B   =   0.0192172535765 #  0.031 # Induced drag coeffient
    CL0 =  polar_K / va**2
    C1 = polar_CD0 / CL0  # constant describing expected angle to overcome zero-lift drag
    C2 = polar_B * CL0
    return -va * (C1 + C2 / sym.cos(phi)**2)


eom = sym.Matrix([x(t).diff()   - v * sym.cos(psi(t)),
                  y(t).diff()   - v * sym.sin(psi(t)),
                  z(t).diff() - atm.get_wind_sym(x(t), y(t), z(t), t) - glider_zd(par_map[v], phi(t)),
#                  z(t).diff() - atm_zd(x(t), y(t), z(t), t) - glider_zd(par_map[v], phi(t)), 
                  psi(t).diff() - g / v * sym.tan(phi(t))])



# Specify the objective function and it's gradient.


def obj(free):
    """Minimize the sum of the squares of the control torque."""
    #pdb.set_trace()
    #T = free[2 * num_nodes:]
    #return interval_value * np.sum(T**2)
    if obj_summ_z:
        _zs = free[2*num_nodes:3*num_nodes]
        return -interval_value * np.sum(_zs)
    if obj_summ_z2:
        _zs = free[2*num_nodes:3*num_nodes]
        return -interval_value * np.sum(_zs**2)
    if obj_last_z:
        _zs = free[2*num_nodes:3*num_nodes]
        return -_zs[-1]

def obj_grad(free):
    grad = np.zeros_like(free)
    #grad[2 * num_nodes:] = 2.0 * interval_value * free[2 * num_nodes:]
    if obj_summ_z:
        grad[2*num_nodes:3*num_nodes] = -1.0 * interval_value
    if obj_summ_z2:
        grad[2*num_nodes:3*num_nodes] = -2.0 * interval_value * free[2*num_nodes:3*num_nodes]
    if obj_last_z:
        grad[3*num_nodes-1] = -1
    return grad

# Specify the symbolic instance constraints, i.e. initial and end
# conditions.
instance_constraints = (x(0.0), y(0.0)-50, z(0.), psi(0.))
                        #theta(duration) - target_angle,



def compute_solution():
   
    # Create an optimization problem.
    prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, interval_value,
                   known_parameter_map=par_map,
                   instance_constraints=instance_constraints,
                   bounds={phi(t): (-np.deg2rad(30), np.deg2rad(30))})

    # https://coin-or.github.io/Ipopt/OPTIONS.html
    prob.addOption('tol', 1e-7)       # default 1e-8
    prob.addOption('max_iter', 5000)  # default 3000
    #prob.addOption('mehrotra_algorithm', 'yes')  # default 'no'
    #prob.addOption('mu_strategy', 'adaptive')  # default 'monotone'
    
    # Use a random positive initial guess.
    initial_guess = np.random.randn(prob.num_free)

    # Find the optimal solution.
    solution, info = prob.solve(initial_guess)

    return prob, solution
    

def plot_solve(prob, solution):
    prob.plot_trajectories(solution)
    #prob.plot_constraint_violations(solution)
    prob.plot_objective_value()

def plot_solution(solution, atm):
    time = np.linspace(0.0, duration, num=num_nodes)
    _x, _y = solution[:num_nodes], solution[num_nodes:2*num_nodes] 
    #fig = plt.figure()
    #ax = fig.add_subplot(111, aspect='equal')
    p3_plu.plot_slice_wind_ne(atm, n0=-100, n1=100, dn=5., e0=-100., e1=100, de=5, h0=0., t0=0.)
    plt.plot(_x, _y)
    plt.gca().axis('equal')
    plt.show()

def save_solution(filename, solution):
    print('saving {}'.format(filename))
    pickle.dump( solution, open(filename, "wb"))

def load_solution(filename):
    print('loading {}'.format(filename))
    solution = pickle.load( open(filename, "rb"))
    return solution

def main():
    if 1:
        prob, solution = compute_solution()
        save_solution('/tmp/glider_opty.pkl', solution)
        plot_solve(prob, solution)
        plot_solution(solution, atm)
    else:
        solution = load_solution('/tmp/glider_opty.pkl')
        plot_solution(solution, atm)

if __name__ == '__main__':
    main()
