#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Optimal Soaring Trajectories using Opty
https://opty.readthedocs.io/en/latest/theory.html

Dimension4 (kinematic) model, trip

"""
import sys, numpy as np, sympy as sym, matplotlib.pyplot as plt
import pdb
import glider_opty_utils as go_u
import solve_glider_opty_4d as sgo4d

#
# objective function for wp
#
# for now, means distance to waypoint
# I'm finding my way in a horizontal wind field
nwp, ewp = 25, 150
def obj_wp(free, _p):
    dists_to_wp = np.linalg.norm(np.vstack((free[_p._slice_e], free[_p._slice_n])).T - [ewp, nwp], axis=1)
    return _p.obj_scale * np.sum(dists_to_wp) / _p.num_nodes

def obj_grad_wp(free, _p):
    grad = np.zeros_like(free)
    disps_to_wp = np.vstack((free[_p._slice_e], free[_p._slice_n])).T - [ewp, nwp]
    dists_to_wp = np.linalg.norm(disps_to_wp, axis=1)
    grad[_p._slice_e] = _p.obj_scale/_p.num_nodes*disps_to_wp[:,0]/dists_to_wp
    grad[_p._slice_n] = _p.obj_scale/_p.num_nodes*disps_to_wp[:,1]/dists_to_wp
    return grad

def test_trip(force_recompute=False, filename='/home/poine/tmp/glider_opty_4d_wp{}.npz', exp='0'):
    if   exp == '0': atm = go_u.AtmosphereCalmSym()
    elif exp == '1': atm = go_u.AtmosphereHorizFieldSym(-5.)
    else: atm = go_u.AtmosphereHorizFieldSym(-7.5)
    _p = sgo4d.Planner( #_obj_fun=sgo4d.obj_sum_z, _obj_grad=sgo4d.obj_grad_sum_z,
                        #_obj_fun=sgo4d.obj_final_z, _obj_grad=sgo4d.obj_grad_final_z,
                        _obj_fun=obj_wp, _obj_grad=obj_grad_wp,
                        #_obj_fun=obj_cc, _obj_grad=obj_grad_cc,
                        _atm=atm, e0=-50, n0=0, u0=25, psi0=0,
                        _v_constraint = (7., 12.),
                        _e_constraint = (-75, 175),
                        _n_constraint = (-50, 100),
                        #_u_constraint = (   25, 100),
                        duration=20, hz=50.)

    sgo4d.compute_or_load(atm, _p, force_recompute, filename.format(exp), tol=1e-5, max_iter=2000, initial_guess=None)

    dists_to_wp = np.linalg.norm(np.vstack((_p.sol_e, _p.sol_n)).T - [ewp, nwp], axis=1)
    arrival_idx = np.argmax(dists_to_wp<20)
    
    go_u.plot_solution_chronogram(_p, max_idx=arrival_idx); plt.savefig(f'plots/glider_4d_wp{exp}_chrono.png')
    go_u.plot_solution_2D_en(_p, max_idx=arrival_idx); plt.savefig(f'plots/glider_4d_wp{exp}_en.png')
    #go_u.plot_solution_2D_nu(_p, n0=-40, n1=50, dn=5., e0=0., h0=0., h1=70, dh=2.5)
    go_u.plot_solution_3D(_p, max_idx=arrival_idx); plt.savefig(f'plots/glider_4d_wp{exp}_3D.png')
    plt.show()




# adding final/mean altitude for cross country using thermals
ccg_n, ccg_e = 75, 250
cc_falt_c = 0.1
cc_malt_c = 5.#0.1 # tradeoff between go to waypoint and maximize altitude
def obj_cc(free, _p):
    dists_to_wp = np.linalg.norm(np.vstack((free[_p._slice_e], free[_p._slice_n])).T - [ccg_e, ccg_n], axis=1)
    cost1 = np.sum(dists_to_wp) / _p.num_nodes                 # average dist to waypoint
    #cost2 = -cc_malt_c*np.sum(free[_p._slice_u])/_p.num_nodes  # average altitude
    #cost2 = -cc_falt_c*free[_p._slice_u][-1]                   # final altitude
    cost2 = 0.
    return _p.obj_scale * (cost1+cost2)

def obj_grad_cc(free, _p):
    grad = np.zeros_like(free)
    disps_to_wp = np.vstack((free[_p._slice_e], free[_p._slice_n])).T - [ccg_e, ccg_n]
    dists_to_wp = np.linalg.norm(disps_to_wp, axis=1)
    grad[_p._slice_e] = _p.obj_scale/_p.num_nodes*disps_to_wp[:,0]/dists_to_wp
    grad[_p._slice_n] = _p.obj_scale/_p.num_nodes*disps_to_wp[:,1]/dists_to_wp
    #grad[_p._slice_u] = -_p.obj_scale*cc_malt_c/_p.num_nodes    # average altitude
    #grad[_p._slice_u][-1] = -cc_falt_c*_p.obj_scale             # final altitude
    return grad


def test_cc(force_recompute=False, filename='/home/poine/tmp/glider_opty_4d_cc1.npz'):
     atm = go_u.atm4()
     _p = sgo4d.Planner(_obj_fun=obj_cc, _obj_grad=obj_grad_cc,
                        _atm=atm, e0=-100, n0=0, u0=25, psi0=0,
                        _v_constraint = (7., 15.),
                        _e_constraint = (-150, 300),
                        _n_constraint = (-100, 100),
                        _u_constraint = (   23, 100),
                        duration=60, hz=20.)
     #sgo4d.compute_or_load(atm, _p, force_recompute, filename, tol=1e-5, max_iter=2000, initial_guess=None)
     sgo4d.compute_or_load(atm, _p, force_recompute, filename, tol=1e-4, max_iter=1500, initial_guess=None)
     go_u.plot_solution_chronogram(_p); plt.savefig('plots/glider_4d_cc1_chrono.png')
     go_u.plot_solution_2D_en(_p); plt.savefig('plots/glider_4d_cc1_en.png')
     go_u.plot_solution_3D(_p); plt.savefig('plots/glider_4d_cc1_3D.png')
     plt.show()

    
if __name__ == '__main__':
    test_trip(force_recompute='-force' in sys.argv, exp='2')
    #test_cc(force_recompute='-force' in sys.argv)

