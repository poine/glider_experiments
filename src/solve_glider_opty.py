#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Optimal Soaring Trajectories using Opty

https://opty.readthedocs.io/en/latest/theory.html

"""
import os, pickle

from collections import OrderedDict

import numpy as np
import sympy as sym
from opty.direct_collocation import Problem
import matplotlib.pyplot as plt

import pat3.atmosphere as p3_atm
import pat3.vehicles.fixed_wing.guidance_ardusoaring as p3_guid
import pat3.plot_utils as p3_plu
import pdb

import glider_opty_utils as go_u


# Symbolic equations of motion
I, m, g, d, v, t = sym.symbols('I, m, g, d, v, t')
#atm_r, atm_s = sym.symbols('atm_r, atm_s')
x, y, z, phi, psi = sym.symbols('x, y, z, phi, psi', cls=sym.Function)

state_symbols = (x(t), y(t), z(t), psi(t))
constant_symbols = (I, m, g, d)
specified_symbols = (phi(t),)

# Specify the known system parameters.
par_map = OrderedDict()
par_map[g] = 9.81
par_map[v] = 8.0

def atm0(): return go_u.AtmosphereWharingtonSym(center=[0, 0, 0], radius=40, strength=1.)
    
def atm1():
    centers, radiuses, strengths = ([-35, 0, 0], [35, 0, 0]), (30, 30), (1., 1.)
    return go_u.AtmosphereWharingtonArraySym(centers, radiuses, strengths)

def atm2():
    centers, radiuses, strengths = ([-50, 0, 0], [50, 0, 0]), (25, 25), (1., 1.)
    return go_u.AtmosphereWharingtonArraySym(centers, radiuses, strengths)

def atm3():
    centers, radiuses, strengths = ([-30, 10, 0], [30, 10, 0], [0, -32, 0]), (25, 25, 25), (0.4, 0.4, 0.4)
    atm = go_u.AtmosphereWharingtonArraySym(centers, radiuses, strengths)
    return atm
   
#atm = atm0()
#atm = atm1()
#atm = atm2()



def get_eom(__atm):
    return sym.Matrix([x(t).diff()   - v * sym.cos(psi(t)),
                       y(t).diff()   - v * sym.sin(psi(t)),
                       z(t).diff() - __atm.get_wind_sym(x(t), y(t), z(t), t) - go_u.glider_sink_rate(par_map[v], phi(t)),
                       psi(t).diff() - g / v * sym.tan(phi(t))])


# Specify the objective function and its gradient.

# mean altitude
def obj_sum_z(_num_nodes, _scale, free):
    return -_scale * np.sum(free[2*_num_nodes:3*_num_nodes])
def obj_grad_sum_z(_num_nodes, _scale, free):
    grad = np.zeros_like(free)
    grad[2*_num_nodes:3*_num_nodes] = -1.0 * _scale
    return grad

# mean squared altitude
def obj_sum_z2(_num_nodes, _scale, free):
    return -_scale * np.sum(free[2*_num_nodes:3*_num_nodes]**2)
def obj_grad_sum_z2(_num_nodes, _scale, free):
    grad = np.zeros_like(free)
    grad[2*_num_nodes:3*_num_nodes] = -2.0 * _scale * free[2*num_nodes:3*num_nodes]
    return grad

# final altitude
def obj_final_z(_num_nodes, _scale, free):
    return -_scale * free[3*_num_nodes-1]
def obj_grad_final_z(_num_nodes, _scale, free):
    grad = np.zeros_like(free)
    grad[3*_num_nodes-1] = -_scale
    return grad


# cross country
def obj_cc(_num_nodes, _scale, free):
    return -_scale * np.sum(free[0*_num_nodes:1*_num_nodes])
def obj_grad_cc(_num_nodes, _scale, free):
    grad = np.zeros_like(free)
    grad[0*_num_nodes:1*_num_nodes] = -1.0 * _scale
    return grad



class Planner:
    def __init__(self,
                 _obj_fun=obj_final_z, _obj_grad=obj_grad_final_z,
                 _atm=None,
                 _min_bank=-np.deg2rad(45.), _max_bank=np.deg2rad(45.),
                 x0=-25, y0=0, z0=1, psi0=0):
        self.duration  =   40
        self.num_nodes = 2000 # time discretization
        self.interval_value = self.duration / (self.num_nodes - 1)
        print('solver: interval_value: {:.3f}s ({:.1f}hz)'.format(self.interval_value, 1./self.interval_value))
        self.atm = _atm if _atm is not None else atm0()

        self._slice_x   = slice(0*self.num_nodes, 1*self.num_nodes, 1)
        self._slice_y   = slice(1*self.num_nodes, 2*self.num_nodes, 1)
        self._slice_z   = slice(2*self.num_nodes, 3*self.num_nodes, 1)
        self._slice_psi = slice(3*self.num_nodes, 4*self.num_nodes, 1)
        self._slice_phi = slice(4*self.num_nodes, 5*self.num_nodes, 1)


        # Specify the symbolic instance constraints, i.e. initial and end conditions.
        instance_constraints = (x(0.0)-x0, y(0.0)-y0, z(0.)-z0, psi(0.)-psi0)
        #theta(duration) - target_angle,
        bounds = {phi(t): (_min_bank, _max_bank)}
        #bounds = {phi(t): (-0.1, _max_bank)}
        # Create an optimization problem.
        self.prob = Problem(lambda _free: _obj_fun(self.num_nodes, self.interval_value, _free),
                            lambda _free: _obj_grad(self.num_nodes, self.interval_value, _free),
                            #eom,
                            get_eom(self.atm),
                            state_symbols, self.num_nodes, self.interval_value,
                            known_parameter_map=par_map,
                            instance_constraints=instance_constraints,
                            bounds=bounds,
                            parallel=False)

    def configure(self, tol=1e-8, max_iter=3000):
        # https://coin-or.github.io/Ipopt/OPTIONS.html
        self.prob.addOption('tol', tol)            # default 1e-8
        self.prob.addOption('max_iter', max_iter)  # default 3000

    def run(self):

        # Use a random positive initial guess.
        initial_guess = np.random.randn(self.prob.num_free)
        # Find the optimal solution.
        self.solution, info = self.prob.solve(initial_guess)
        self.interpret_solution()
        
    def interpret_solution(self):
        self.sol_time = np.linspace(0.0, self.duration, num=self.num_nodes)
        self.sol_x = self.solution[self._slice_x]
        self.sol_y = self.solution[self._slice_y]
        self.sol_z = self.solution[self._slice_z]
        self.sol_psi = self.solution[self._slice_psi]
        self.sol_phi = self.solution[self._slice_phi]
        self.sol_v = par_map[v]*np.ones(self.num_nodes)
        
    def save_solution(self, filename):
        print('saving {}'.format(filename))
        pickle.dump(self.solution, open(filename, "wb"))

    def load_solution(self, filename):
        print('loading {}'.format(filename))
        self.solution = pickle.load(open(filename, "rb"))
        self.interpret_solution()

    def run_or_load(self, filename, force_run=False):
        if force_run or not os.path.exists(filename):
            self.run()
            self.save_solution(filename)
        else:
            self.load_solution(filename)
        


def plot_solve(prob, solution):
    prob.plot_trajectories(solution)
    #prob.plot_constraint_violations(solution)
    prob.plot_objective_value()


def plot_run(planner, figure=None, ax=None):
    fig = figure if figure is not None else plt.figure()
    ax = ax if ax is not None else fig.add_subplot(111)
    planner.prob.plot_objective_value()
    return ax, figure

def main(force_recompute=False, filename='/tmp/glider_opty.pkl'):
    _atm = go_u.AtmosphereWharingtonSym(center=[0, 0, 0], radius=40, strength=1.)
    _atm.center = np.array([-20, 0, 0])
    _atm.strength = 2.
    _atm.radius = 20.
    _p = Planner(_atm=_atm, x0=-50)#obj_cc, obj_grad_cc)#_min_bank=np.deg2rad(-10))
    _p.configure(tol=1e-8, max_iter=200)
    _p.run_or_load(filename, force_recompute)
    plot_run(_p)
    alt_final, alt_mean = _p.sol_z[-1], np.mean(_p.sol_z)
    txt = 'alt: final {:.1f} m, mean {:.1f}'.format(alt_final, alt_mean)
    go_u.plot_solution_chronogram(_p)
    go_u.plot_solution_2D_en(_p, title=txt)
    go_u.plot_solution_2D_nu(_p, title=txt)
    go_u.plot_solution_3D(_p, title=txt)
    plt.show()
    
if __name__ == '__main__':
    main(force_recompute=True)
    #atm2()
