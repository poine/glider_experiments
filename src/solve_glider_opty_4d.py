#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Optimal Soaring Trajectories using Opty

https://opty.readthedocs.io/en/latest/theory.html


Dimension4 (kinematic) model

"""
import sys, numpy as np, sympy as sym
import matplotlib.pyplot as plt
import collections
import pdb

import opty.direct_collocation
import glider_opty_utils as go_u
import pat3.plot_utils as p3_plu

#
# Objective functions
#
# final altitude
def obj_final_z(free, _p):
    return -_p.obj_scale * free[_p._slice_u][-1]
def obj_grad_final_z(free, _p):
    grad = np.zeros_like(free)
    grad[_p._slice_u][-1] = -_p.obj_scale
    return grad
# mean altitude
def obj_sum_z(free, _p):
    return -_p.obj_scale * np.sum(free[_p._slice_u])/_p.num_nodes
def obj_grad_sum_z(free, _p):
    grad = np.zeros_like(free)
    grad[_p._slice_u] = -_p.obj_scale/_p.num_nodes
    return grad


class Glider:
    sv_x, sv_y, sv_z, sv_psi = range(4)
    def __init__(self):
        # symbols
        self._st = sym.symbols('t')
        self._se, self._sn, self._su, self._sv, self._sphi, self._spsi = sym.symbols('e, n, u, v, phi, psi', cls=sym.Function)
        self._state_symbols = (self._se(self._st), self._sn(self._st), self._su(self._st), self._spsi(self._st))
        self._input_symbols = (self._sv, self._sphi)

    def get_eom(self, atm, g=9.81):
        wn, we, wd = atm.get_wind_ned_sym2(self._sn(self._st), self._se(self._st), -self._su(self._st), self._st)
        eq1 = self._se(self._st).diff() - self._sv(self._st) * sym.cos(self._spsi(self._st)) - we
        eq2 = self._sn(self._st).diff() - self._sv(self._st) * sym.sin(self._spsi(self._st)) - wn
        eq3 = self._su(self._st).diff()\
              +wd +go_u.glider_sink_rate(self._sv(self._st), self._sphi(self._st))
        eq4 = self._spsi(self._st).diff() - g / self._sv(self._st) * sym.tan(self._sphi(self._st))
        return sym.Matrix([eq1, eq2, eq3, eq4])
    
        
class Planner:
    def __init__(self,
                 _obj_fun=obj_final_z, _obj_grad=obj_grad_final_z,
                 _atm=None,
                 _phi_constraint = (-np.deg2rad(60.), np.deg2rad(60.)), #_min_bank=-np.deg2rad(60.), _max_bank=np.deg2rad(60.),
                 _v_constraint = (7., 20.),                             # velocity constraint
                 _u_constraint = None,
                 _n_constraint = (-100, 100),
                 _e_constraint = (-100, 100),
                 e0=-25, n0=0, u0=1, psi0=0,
                 duration=40., hz=50., obj_scale=1.):
        self.obj_scale = obj_scale           # objective function scaling
        self.num_nodes = int(duration*hz)+1
        self.interval_value = 1./hz
        self.duration  = (self.num_nodes-1)*self.interval_value#duration
        print('solver: interval_value: {:.3f}s ({:.1f}hz)'.format(self.interval_value, 1./self.interval_value))
        self.atm = _atm if _atm is not None else go_u.AtmosphereWharingtonSym()

        self.glider = _g = Glider()

        self._slice_e, self._slice_n, self._slice_u, self._slice_psi, self._slice_phi, self._slice_v = \
            [slice(_i*self.num_nodes, (_i+1)*self.num_nodes, 1) for _i in range(6)]
        
        self._e_constraint = _e_constraint
        self._n_constraint = _n_constraint
        self._u_constraint = _u_constraint
        
        # Specify the known system parameters.
        self._par_map = collections.OrderedDict()
        #self._par_map[g] = 9.81
        #self._par_map[v] = 8.0

        #scale=100.#-0.1#-1.#-10.
        
        # Specify the symbolic instance constraints, i.e. initial and end conditions.
        self._instance_constraints = (_g._se(0.)-e0, _g._sn(0.)-n0, _g._su(0.)-u0, _g._spsi(0.)-psi0)

        self._bounds = {_g._sphi(_g._st): _phi_constraint,
                        _g._sv(_g._st): _v_constraint,
                        _g._se(_g._st): _e_constraint,
                        _g._sn(_g._st): _n_constraint,
        }
        if self._u_constraint is not None: self._bounds[_g._su(_g._st)] = _u_constraint
        self.prob =  opty.direct_collocation.Problem(lambda _free: _obj_fun(_free, self),
                                                     lambda _free: _obj_grad(_free, self),
                                                     _g.get_eom(self.atm),
                                                     _g._state_symbols,
                                                     self.num_nodes,
                                                     self.interval_value,
                                                     known_parameter_map=self._par_map,
                                                     instance_constraints=self._instance_constraints,
                                                     bounds=self._bounds,
                                                     parallel=False)
        

    def run(self, initial_guess=None):
        # Use a random positive initial guess.
        initial_guess = np.random.randn(self.prob.num_free) if initial_guess is None else initial_guess
        # Find the optimal solution.
        self.solution, info = self.prob.solve(initial_guess)   
        self.interpret_solution()

    def interpret_solution(self):
        self.sol_time = np.linspace(0.0, self.duration, num=self.num_nodes)
        self.sol_e   = self.solution[self._slice_e]
        self.sol_n   = self.solution[self._slice_n]
        self.sol_u   = self.solution[self._slice_u]
        self.sol_psi = self.solution[self._slice_psi]
        self.sol_phi = self.solution[self._slice_phi]
        self.sol_v   = self.solution[self._slice_v]

    def configure(self, tol=1e-8, max_iter=3000):
        # https://coin-or.github.io/Ipopt/OPTIONS.html
        self.prob.addOption('tol', tol)            # default 1e-8
        self.prob.addOption('max_iter', max_iter)  # default 3000

    def save_solution(self, filename):
        np.savez(filename, sol_time=self.sol_time, sol_e=self.sol_e, sol_n=self.sol_n, sol_u=self.sol_u,
                 sol_psi=self.sol_psi, sol_phi=self.sol_phi, sol_v=self.sol_v)
        print('saved {}'.format(filename))

    def load_solution(self, filename):
        _data =  np.load(filename)
        labels = ['sol_time', 'sol_e', 'sol_n', 'sol_u', 'sol_psi', 'sol_phi', 'sol_v']
        self.sol_time, self.sol_e, self.sol_n, self.sol_u, self.sol_psi, self.sol_phi, self.sol_v = [_data[k] for k in labels]
        print('loaded {}'.format(filename))

        
def compute_or_load(atm, _p, force_recompute=False, filename='/tmp/glider_opty_4d.pkl', tol=1e-5, max_iter=1500, initial_guess=None):
    if force_recompute:
        _p.configure(tol, max_iter)
        _p.run(initial_guess)
        _p.prob.plot_objective_value()
        #_p.prob.plot_trajectories(_p.solution)
        #_p.prob.plot_constraint_violations(_p.solution)
        _p.save_solution(filename)
    else:
        _p.load_solution(filename)
        
def main(force_recompute=False, filename='/tmp/glider_opty.pkl'):
    #atm = go_u.AtmosphereWharingtonSym(radius=40., strength=-1)
    #atm = go_u.AtmosphereCalmSym()
    #atm = go_u.atm2() # broken?
    atm = go_u.AtmosphereRidgeSym()
    _p = Planner( #_obj_fun=obj_final_z, _obj_grad=obj_grad_final_z,
                  _obj_fun=obj_sum_z, _obj_grad=obj_grad_sum_z,
                  _atm=atm, x0=10, y0=0, z0=25, psi0=np.pi,
                  duration=50, hz=50.)

    compute_or_load(atm, _p, force_recompute, filename, tol=1e-5, max_iter=1500, initial_guess=None)
    go_u.plot_solution_chronogram(_p)
    go_u.plot_solution_2D_en(_p)
    go_u.plot_solution_2D_nu(_p, n0=-40, n1=50, dn=5., e0=0., h0=0., h1=70, dh=2.5)
    go_u.plot_solution_3D(_p)
    plt.show()
    
if __name__ == '__main__':
    main(force_recompute='-force' in sys.argv)
