#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Optimal Soaring Trajectories using Opty

https://opty.readthedocs.io/en/latest/theory.html

"""
import numpy as np, sympy as sym
import matplotlib.pyplot as plt
import collections

import opty.direct_collocation
import glider_opty_utils as go_u

# final altitude
def obj_final_z(_num_nodes, _scale, free):
    return -_scale * free[3*_num_nodes-1]
def obj_grad_final_z(_num_nodes, _scale, free):
    grad = np.zeros_like(free)
    grad[3*_num_nodes-1] = -_scale
    return grad

class Planner:
    def __init__(self,
                 _obj_fun=obj_final_z, _obj_grad=obj_grad_final_z,
                 _atm=None,
                 _min_bank=-np.deg2rad(60.), _max_bank=np.deg2rad(60.),
                 _min_v=7., _max_v=20.,
                 x0=-25, y0=0, z0=1, psi0=0):
        self.duration  =   40
        self.num_nodes = 2000 # time discretization
        self.interval_value = self.duration / (self.num_nodes - 1)
        print('solver: interval_value: {:.3f}s ({:.1f}hz)'.format(self.interval_value, 1./self.interval_value))
        self.atm = _atm if _atm is not None else go_u. AtmosphereWharingtonSym()
 
        # symbols
        self._st = sym.symbols('t')
        self._sx, self._sy, self._sz, self._sv, self._sphi, self._spsi = sym.symbols('x, y, z, v, phi, psi', cls=sym.Function)
        self._state_symbols = (self._sx(self._st), self._sy(self._st), self._sz(self._st), self._spsi(self._st))
        self._input_symbols = (self._sv, self._sphi)

        self._slice_x   = slice(0*self.num_nodes, 1*self.num_nodes, 1)
        self._slice_y   = slice(1*self.num_nodes, 2*self.num_nodes, 1)
        self._slice_z   = slice(2*self.num_nodes, 3*self.num_nodes, 1)
        self._slice_psi = slice(3*self.num_nodes, 4*self.num_nodes, 1)
        self._slice_phi = slice(4*self.num_nodes, 5*self.num_nodes, 1)
        self._slice_v   = slice(5*self.num_nodes, 6*self.num_nodes, 1)
        
        # Specify the known system parameters.
        self._par_map = collections.OrderedDict()
        #self._par_map[g] = 9.81
        #self._par_map[v] = 8.0
        
        # Specify the symbolic instance constraints, i.e. initial and end conditions.
        self._instance_constraints = (self._sx(0.)-x0, self._sy(0.)-y0, self._sz(0.)-z0, self._spsi(0.)-psi0)

        self._bounds = {self._sphi(self._st): (_min_bank, _max_bank),
                        self._sv(self._st): (_min_v, _max_v)        }


        # Create an optimization problem.
        self.prob =  opty.direct_collocation.Problem(lambda _free: _obj_fun(self.num_nodes, 1., _free),
                                                     lambda _free: _obj_grad(self.num_nodes, 1., _free),
                                                     self.get_eom(),
                                                     self._state_symbols,
                                                     self.num_nodes,
                                                     self.interval_value,
                                                     known_parameter_map=self._par_map,
                                                     instance_constraints=self._instance_constraints,
                                                     bounds=self._bounds,
                                                     parallel=False)


    def get_eom(self, g=9.81):
        eq1 = self._sx(self._st).diff() - self._sv(self._st) * sym.cos(self._spsi(self._st))
        eq2 = self._sy(self._st).diff() - self._sv(self._st) * sym.sin(self._spsi(self._st))
        eq3 = self._sz(self._st).diff() - self.atm.get_wind_ned_sym(self._sx(self._st), self._sy(self._st), self._sz(self._st), self._st)\
                                        - go_u.glider_sink_rate(self._sv(self._st), self._sphi(self._st))
        eq4 = self._spsi(self._st).diff() - g / self._sv(self._st) * sym.tan(self._sphi(self._st))
        return sym.Matrix([eq1, eq2, eq3, eq4])

    def run(self):
        # Use a random positive initial guess.
        initial_guess = np.random.randn(self.prob.num_free)
        # Find the optimal solution.
        self.solution, info = self.prob.solve(initial_guess)   
        self.interpret_solution()

    def interpret_solution(self):
        self.sol_time = np.linspace(0.0, self.duration, num=self.num_nodes)
        self.sol_x   = self.solution[self._slice_x]
        self.sol_y   = self.solution[self._slice_y]
        self.sol_z   = self.solution[self._slice_z]
        self.sol_psi = self.solution[self._slice_psi]
        self.sol_phi = self.solution[self._slice_phi]
        self.sol_v   = self.solution[self._slice_v]

    def configure(self, tol=1e-8, max_iter=3000):
        # https://coin-or.github.io/Ipopt/OPTIONS.html
        self.prob.addOption('tol', tol)            # default 1e-8
        self.prob.addOption('max_iter', max_iter)  # default 3000
        
def main(force_recompute=False, filename='/tmp/glider2_opty.pkl'):
    atm = go_u. AtmosphereWharingtonSym(center=[0, 0, 0], radius=40, strength=1.)
    _p = Planner(_atm=atm, x0=-100, y0=-100, z0=5)
    _p.configure(tol=1e-7, max_iter=2000)
    _p.run()
    go_u.plot_solution_chronogram(_p)
    go_u.plot_solution_2D_en(_p)
    go_u.plot_solution_2D_nu(_p)
    go_u.plot_solution_3D(_p)
    plt.show()
    
if __name__ == '__main__':
    main(force_recompute=True)
