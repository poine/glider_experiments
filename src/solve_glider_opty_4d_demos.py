#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Optimal Soaring Trajectories using Opty
https://opty.readthedocs.io/en/latest/theory.html

Dimension4 (kinematic) model, demos

"""
import sys, numpy as np, sympy as sym, matplotlib.pyplot as plt
import pdb
import glider_opty_utils as go_u
import solve_glider_opty_4d as sgo4d

def test_thermal(force_recompute=False, filename='/home/poine/tmp/glider_opty_4d_thermal_{}.npz', exp_id='2'):
    if exp_id == '0':
        atm = go_u.AtmosphereWharingtonSym(radius=40., strength=-1)
    elif exp_id == '1':
        atm = go_u.AtmosphereWharingtonOval(radius=60., strength=-0.8)
    elif exp_id == '2':
        atm = go_u.atm2()
    #atm = go_u.AtmosphereArrayTest()

    _p = sgo4d.Planner( _obj_fun=sgo4d.obj_sum_z, _obj_grad=sgo4d.obj_grad_sum_z,
                        #_obj_fun=sgo4d.obj_final_z, _obj_grad=sgo4d.obj_grad_final_z,
                        _atm=atm, e0=-100, n0=0, u0=25, psi0=0,
                        _v_constraint = (7., 15.),
                        _e_constraint = (-120, 120),
                        _n_constraint = (-120, 120),
                        #_u_constraint = (   25, 100),
                        duration=40, hz=50., obj_scale=1.)

    #sgo4d.compute_or_load(atm, _p, force_recompute, filename, tol=1e-5, max_iter=4000, initial_guess=None)
    sgo4d.compute_or_load(atm, _p, force_recompute, filename.format(exp_id), tol=1e-4, max_iter=4000, initial_guess=None)
    print(f'last altitude {_p.sol_u[-1]} m')
    go_u.plot_solution_chronogram(_p); plt.savefig('plots/glider_4d_thermal_chrono.png')
    go_u.plot_solution_2D_en(_p); plt.savefig('plots/glider_4d_thermal_en.png')
    go_u.plot_solution_2D_nu(_p, n0=-40, n1=50, dn=5., e0=0., h0=0., h1=70, dh=2.5); plt.savefig('plots/glider_4d_thermal_nu.png')
    go_u.plot_solution_2D_eu(_p, contour_wz=True, e0=-40, e1=50, de=5., n0=0., h0=0., h1=70, dh=2.5); plt.savefig('plots/glider_4d_thermal_eu.png')
    go_u.plot_solution_3D(_p); plt.savefig('plots/glider_4d_thermal_3D.png')
    plt.show()

def test_slope(force_recompute=False, filename='/home/poine/tmp/glider_opty_4d_slope.npz'):
    atm = go_u.AtmosphereRidgeSym()
    _p = sgo4d.Planner( #_obj_fun=sgo4d.obj_sum_z, _obj_grad=sgo4d.obj_grad_sum_z,
                        _obj_fun=sgo4d.obj_final_z, _obj_grad=sgo4d.obj_grad_final_z,
                        _atm=atm, e0=0, n0=0, u0=25, psi0=-np.pi/6,
                        _v_constraint = (7., 15.),
                        _n_constraint = (-100, 100),
                        _e_constraint = (-100, 100),
                        duration=50, hz=50.)
    sgo4d.compute_or_load(atm, _p, force_recompute, filename, tol=1e-4, max_iter=3000, initial_guess=None)
    #sgo4d.compute_or_load(atm, _p, force_recompute, filename, tol=1e-5, max_iter=2000, initial_guess=None)
    print(f'last altitude {_p.sol_u[-1]} m')
    go_u.plot_solution_chronogram(_p); plt.savefig('plots/glider_4d_slope_chrono.png')
    go_u.plot_solution_2D_en(_p, contour_wz=True) ; plt.savefig('plots/glider_4d_slope_en.png')#, n0=-40, n1=50, dn=5., e0=0., h0=0., h1=70, dh=2.5)
    go_u.plot_solution_2D_nu(_p, contour_wz=True, n0=-40, n1=50, dn=5., e0=0., h0=0., h1=70, dh=2.5); plt.savefig('plots/glider_4d_slope_nu.png')
    go_u.plot_solution_2D_eu(_p, contour_wz=True, e0=-40, e1=50, de=5., n0=0., h0=0., h1=70, dh=2.5); plt.savefig('plots/glider_4d_slope_eu.png')
    go_u.plot_solution_3D(_p); plt.savefig('plots/glider_4d_slope_3D.png') 
    plt.show()

if __name__ == '__main__':
    test_thermal(force_recompute='-force' in sys.argv, exp_id='2')
    #test_slope(force_recompute='-force' in sys.argv)
    
