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

# final altitude
def obj_final_z(free, _p):
    return _p.obj_scale * free[_p._slice_z][-1]
def obj_grad_final_z(free, _p):
    grad = np.zeros_like(free)
    grad[_p._slice_z][-1] = _p.obj_scale
    return grad
# mean altitude
def obj_sum_z(free, _p):
    return _p.obj_scale * np.sum(free[_p._slice_z])/_p.num_nodes
def obj_grad_sum_z(free, _p):
    grad = np.zeros_like(free)
    grad[_p._slice_z] = _p.obj_scale/_p.num_nodes
    return grad


class Glider:
    sv_x, sv_y, sv_z, sv_psi = range(4)
    def __init__(self):
        # symbols
        self._st = sym.symbols('t')
        self._sx, self._sy, self._sz, self._sv, self._sphi, self._spsi = sym.symbols('x, y, z, v, phi, psi', cls=sym.Function)
        self._state_symbols = (self._sx(self._st), self._sy(self._st), self._sz(self._st), self._spsi(self._st))
        self._input_symbols = (self._sv, self._sphi)

    def get_eom(self, atm, g=9.81):
        wn, we, wd = atm.get_wind_ned_sym2(self._sx(self._st), self._sy(self._st), self._sz(self._st), self._st)
        eq1 = self._sx(self._st).diff() - self._sv(self._st) * sym.cos(self._spsi(self._st)) -wn
        eq2 = self._sy(self._st).diff() - self._sv(self._st) * sym.sin(self._spsi(self._st)) -we
        eq3 = self._sz(self._st).diff()\
              -wd -go_u.glider_sink_rate(self._sv(self._st), self._sphi(self._st))
        eq4 = self._spsi(self._st).diff() - g / self._sv(self._st) * sym.tan(self._sphi(self._st))
        return sym.Matrix([eq1, eq2, eq3, eq4])
    
        
class Planner:
    def __init__(self,
                 _obj_fun=obj_final_z, _obj_grad=obj_grad_final_z,
                 _atm=None,
                 _phi_constraint = (-np.deg2rad(60.), np.deg2rad(60.)), #_min_bank=-np.deg2rad(60.), _max_bank=np.deg2rad(60.),
                 _v_constraint = (7., 20.),                             # velocity constraint
                 _n_constraint = (-100, 100),
                 _e_constraint = (-100, 100),
                 x0=-25, y0=0, z0=1, psi0=0,
                 duration=40., hz=50., obj_scale=1.):
        self.obj_scale = obj_scale
        self.num_nodes = int(duration*hz)+1
        self.interval_value = 1./hz
        self.duration  = (self.num_nodes-1)*self.interval_value#duration
        print('solver: interval_value: {:.3f}s ({:.1f}hz)'.format(self.interval_value, 1./self.interval_value))
        self.atm = _atm if _atm is not None else go_u.AtmosphereWharingtonSym()

        self.glider = _g = Glider()

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

        #scale=100.#-0.1#-1.#-10.
        
        # Specify the symbolic instance constraints, i.e. initial and end conditions.
        self._instance_constraints = (_g._sx(0.)-x0, _g._sy(0.)-y0, _g._sz(0.)-z0, _g._spsi(0.)-psi0)

        self._bounds = {_g._sphi(_g._st): _phi_constraint, #(_min_bank, _max_bank),
                        _g._sv(_g._st): _v_constraint, #(_min_v, _max_v),
                        _g._sx(_g._st): _n_constraint, #(-50, 10),
                        _g._sy(_g._st): _e_constraint, #(-35, 35)
        }
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

    def save_solution(self, filename):
        np.savez(filename, sol_time=self.sol_time, sol_x=self.sol_x, sol_y=self.sol_y, sol_z=self.sol_z,
                 sol_psi=self.sol_psi, sol_phi=self.sol_phi, sol_v=self.sol_v)
        print('saved {}'.format(filename))

    def load_solution(self, filename):
        _data =  np.load(filename)
        labels = ['sol_time', 'sol_x', 'sol_y', 'sol_z', 'sol_psi', 'sol_phi', 'sol_v']
        self.sol_time, self.sol_x, self.sol_y, self.sol_z, self.sol_psi, self.sol_phi, self.sol_v = [_data[k] for k in labels]
        print('loaded {}'.format(filename))

        
def check_atm(_p):
    n0, n1, dn, e0, h0, h1, dh =-100, 100, 5., 0., 0., 30., 2.
    #figure, ax = None, None
    figure, axes = plt.subplots(2, 1)
    figure, ax = p3_plu.plot_slice_wind_nu(_p.atm,
                                           n0=n0, n1=n1, dn=dn, e0=e0, h0=h0, h1=h1, dh=dh, zdir=-1.,
                                           show_quiver=False, show_color_bar=False, title="North Up slice",
                                           figure=figure, ax=axes[0])
    
    nlist, dlist = np.arange(n0, n1, dn), -np.arange(h0, h1, dh)
    _n, _d = np.meshgrid(nlist, dlist)
    wn, wd = np.meshgrid(nlist, dlist)
    for _in in range(wn.shape[0]):
        for _id in range(wn.shape[1]):
            pos_ned = [_n[_in, _id], e0, _d[_in, _id]]
            #wn[_in, _id], _, wd[_in, _id] = _p.atm.get_wind_ned(pos_ned, t=0)
            wd[_in, _id] = _p.atm.get_wind_ned_sym(_n[_in, _id], e0, _d[_in, _id], _t=0.)
            #pdb.set_trace()
            
    cp = axes[1].contourf(_n, -_d, -wd, alpha=0.4)
    p3_plu.decorate(axes[1], title="north up slice, symbolic", xlab='north in m', ylab='h in m (positive up)')
    plt.show()
        
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
        
def main(force_recompute=False, filename='/tmp/glider2_opty.pkl'):
    #atm = go_u.AtmosphereWharingtonSym(radius=40., strength=-1)
    #atm = go_u.AtmosphereCalmSym()
    #atm = go_u.atm2()
    atm = go_u.AtmosphereRidgeSym()
    _p = Planner( #_obj_fun=obj_final_z, _obj_grad=obj_grad_final_z,
                  _obj_fun=obj_sum_z, _obj_grad=obj_grad_sum_z,
                  #_atm=atm, x0=20, y0=0, z0=-35, psi0=np.pi,
                  _atm=atm, x0=10, y0=0, z0=-25, psi0=np.pi,
                  duration=50, hz=50.)

    #check_atm(_p)
    filename = '/tmp/glider_slope_soaring_opty1.npz'
    compute_or_load(atm, _p, force_recompute, filename, tol=1e-5, max_iter=1500, initial_guess=None)
    go_u.plot_solution_chronogram(_p)
    go_u.plot_solution_2D_en(_p)
    go_u.plot_solution_2D_nu(_p, n0=-40, n1=50, dn=5., e0=0., h0=0., h1=70, dh=2.5)
    go_u.plot_solution_3D(_p)
    plt.show()
    
if __name__ == '__main__':
    main(force_recompute='-force' in sys.argv)
