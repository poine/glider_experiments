#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Going to write better equations of motion and solve a ground circular trajectory in a constant wind field


Optimal Soaring Trajectories using Opty

https://opty.readthedocs.io/en/latest/theory.html

"""
import sys, numpy as np, sympy as sym
import matplotlib.pyplot as plt
import collections
import pdb

import opty.direct_collocation
import glider_opty_utils as go_u
import pat3.plot_utils as p3_plu

# final altitude
def obj_final_z(_num_nodes, _scale, free):
    return _scale * free[3*_num_nodes-1]
def obj_grad_final_z(_num_nodes, _scale, free):
    grad = np.zeros_like(free)
    grad[3*_num_nodes-1] = _scale
    return grad

class Glider:
    sv_x, sv_y, sv_z, sv_xd, sv_yd, sv_zd, sv_psi = range(7)
    iv_phi, iv_theta, iv_thrust = range(3)
    def __init__(self):
        self._st = sym.symbols('t')
        self._sx, self._sy, self._sz, self._sxd, self._syd, self._szd , self._spsi = sym.symbols('x, y, z, xd, yd, zd, psi', cls=sym.Function)
        self._sphi, self._stheta, self._sthrust = sym.symbols('phi, theta, thrust', cls=sym.Function)
        self._state_symbols = (self._sx(self._st), self._sy(self._st), self._sz(self._st),
                               self._sxd(self._st), self._syd(self._st), self._szd(self._st), self._spsi(self._st))
        #self._input_symbols = (self._sv, self._sphi)
        

    def get_eom(self, atm, g=9.81):
        eq1 = self._sx(self._st).diff() - self._sxd(self._st)
        eq2 = self._sy(self._st).diff() - self._syd(self._st)
        eq3 = self._sz(self._st).diff() - self._szd(self._st)
        fx,fy,fz = self.get_forces(atm)
        eq4 = self._sxd(self._st).diff() - fx
        eq5 = self._syd(self._st).diff() - fy
        eq6 = self._szd(self._st).diff() - fz
        eq7 = self._spsi(self._st).diff() - g / 10. * sym.tan(self._sphi(self._st))
        return sym.Matrix([eq1, eq2, eq3, eq4, eq5, eq6, eq7])

    def get_forces(self, atm):
        pos_ned = np.array([self._sxd(self._st), self._sxd(self._st), self._sxd(self._st)])
        vel_ned = np.array([self._sxd(self._st), self._sxd(self._st), self._sxd(self._st)])
        va_ned = vel_ned - atm.get_wind_ned_sym2(self._sx(self._st), self._sy(self._st), self._sz(self._st), self._st)
        return [0, 0, 0]

    def get_aero_forces_body(self):
        CL, CY, CD = self.get_aero_coefs()
        Pdyn, Sref = 0, 0
        #return Pdyn*P.Sref*np.dot(p3_fr.R_aero_to_body(alpha, beta), [-CD, CY, -CL])
        return [0, 0, 0]
    
    def get_aero_coefs(self): 
        CL, CY, CD = 0, 0, 0
        return CL, CY, CD
    
class Planner:
    def __init__(self,
                 _obj_fun=obj_final_z, _obj_grad=obj_grad_final_z,
                 _atm=None,
                 _min_bank=-np.deg2rad(60.), _max_bank=np.deg2rad(60.),
                 _min_v=7., _max_v=20.,
                 x0=-25, y0=0, z0=1, psi0=0,
                 xd0=12., yd0=0., zd0=0.,
                 duration=40., hz=50.):
        self.duration  = duration
        self.num_nodes = int(duration*hz)+1
        self.interval_value = 1./hz
        print('solver: interval_value: {:.3f}s ({:.1f}hz)'.format(self.interval_value, 1./self.interval_value))
        self.atm = _atm if _atm is not None else go_u.AtmosphereWharingtonSym()
        self.glider = _g = Glider()

        self._slice_x   = slice(0*self.num_nodes, 1*self.num_nodes, 1)
        self._slice_y   = slice(1*self.num_nodes, 2*self.num_nodes, 1)
        self._slice_z   = slice(2*self.num_nodes, 3*self.num_nodes, 1)
        self._slice_xd  = slice(3*self.num_nodes, 4*self.num_nodes, 1)
        self._slice_yd  = slice(4*self.num_nodes, 5*self.num_nodes, 1)
        self._slice_zd  = slice(5*self.num_nodes, 6*self.num_nodes, 1)
        self._slice_psi = slice(6*self.num_nodes, 7*self.num_nodes, 1)
        self._slice_phi = slice(7*self.num_nodes, 8*self.num_nodes, 1)
        
        # Specify the known system parameters.
        self._par_map = collections.OrderedDict()
        scale=100.
        self._instance_constraints = (_g._sx(0.)-x0, _g._sy(0.)-y0, _g._sz(0.)-z0,
                                      _g._sxd(0.)-xd0, _g._syd(0.)-yd0, _g._szd(0.)-zd0,
                                      _g._spsi(0.)-psi0)
        self._bounds = {_g._sphi(_g._st): (_min_bank, _max_bank),
#                        _g._sv(_g._st): (_min_v, _max_v),
                        _g._sx(_g._st): (-50, 10),
                        _g._sy(_g._st): (-25, 25)
                       }
        self.prob =  opty.direct_collocation.Problem(lambda _free: _obj_fun(self.num_nodes, scale, _free),
                                                     lambda _free: _obj_grad(self.num_nodes, scale, _free),
                                                     _g.get_eom(self.atm),
                                                     _g._state_symbols,
                                                     self.num_nodes,
                                                     self.interval_value,
                                                     known_parameter_map=self._par_map,
                                                     instance_constraints=self._instance_constraints,
                                                     bounds=self._bounds,
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
        self.sol_x   = self.solution[self._slice_x]
        self.sol_y   = self.solution[self._slice_y]
        self.sol_z   = self.solution[self._slice_z]
        self.sol_xd  = self.solution[self._slice_xd]
        self.sol_yd  = self.solution[self._slice_yd]
        self.sol_zd  = self.solution[self._slice_zd]
        self.sol_psi = self.solution[self._slice_psi]
        self.sol_phi = self.solution[self._slice_phi]

    def save_solution(self, filename):
        np.savez(filename, sol_time=self.sol_time, sol_x=self.sol_x, sol_y=self.sol_y, sol_z=self.sol_z,
                 sol_xd=self.sol_xd, sol_yd=self.sol_yd, sol_zd=self.sol_zd,
                 sol_psi=self.sol_psi, sol_phi=self.sol_phi)
        print('saved {}'.format(filename))

    def load_solution(self, filename):
        _data =  np.load(filename)
        labels = ['sol_time', 'sol_x', 'sol_y', 'sol_z', 'sol_xd', 'sol_yd', 'sol_zd', 'sol_psi', 'sol_phi']
        self.sol_time, self.sol_x, self.sol_y, self.sol_z, self.sol_xd, self.sol_yd, self.sol_zd, self.sol_psi, self.sol_phi = [_data[k] for k in labels]
        print('loaded {}'.format(filename))
        
def main(force_recompute=False, filename='/tmp/glider3_opty.pkl'):
    atm = go_u.AtmosphereCalmSym()
    _p = Planner( _obj_fun=obj_final_z, _obj_grad=obj_grad_final_z,
                  #_obj_fun=obj_sum_z, _obj_grad=obj_grad_sum_z,
                  #_atm=atm, x0=20, y0=0, z0=-35, psi0=np.pi,
                  _atm=atm, x0=10, y0=0, z0=-25, psi0=np.pi,
                  duration=120, hz=50.)
    filename = '/tmp/glider_better_opty.npz'
    if force_recompute:
        _p.configure(tol=1e-5, max_iter=200)
        _p.run()
        _p.prob.plot_objective_value()
        _p.prob.plot_trajectories(_p.solution)
        #_p.prob.plot_constraint_violations(_p.solution)
        _p.save_solution(filename)
    else:
        _p.load_solution(filename)

    go_u.plot_solution_2D_en(_p)
    go_u.plot_solution_2D_nu(_p, n0=-40, n1=50, dn=5., e0=0., h0=0., h1=70, dh=2.5)
    go_u.plot_solution_3D(_p)
    plt.show()
    
if __name__ == '__main__':
    main(force_recompute='-force' in sys.argv)
