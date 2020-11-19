#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, numpy as np, matplotlib.pyplot as plt
import pdb
import pat3.atmosphere as p3_atm
import solve_glider_opty_4d as go4, glider_opty_utils as go_u


_cache_dir = '/tmp/glider_optim'
def cache_filename(prefix, ident): return os.path.join(_cache_dir, prefix, ident)

def compare_nb_iter(max_iters = [10, 100, 200, 1000], filename_prefix='/tmp/glider_opty_cmp_iter/',
                    force_recompute=False):
    _obj, _obj_grad = solve_glider_opty.obj_final_z, solve_glider_opty.obj_grad_final_z
    _ps = []
    for _max_iter in max_iters:
        _p = solve_glider_opty.Planner(_obj, _obj_grad)
        _ps.append(_p)
        _p.configure(tol=1e-8, max_iter=_max_iter)
        filename = filename_prefix+'{}.pkl'.format(_max_iter)
        if force_recompute or not os.path.exists(filename):
            _p.run()
            solve_glider_opty.plot_run(_p)
            _p.save_solution(filename)
        else:
            _p.load_solution(filename)
        alt_final, alt_mean = _p.zs[-1], np.mean(_p.zs)
        txt = 'iteration {}: alt final {:.1f} m, mean {:.1f}'.format(_max_iter, alt_final, alt_mean)
        solve_glider_opty.plot_solution_3D(_p, title=txt)
    plt.show()
    

def compare_obj(_max_iter=200):
    max_iters = 100
    _obj, _obj_grad = solve_glider_opty.obj_final_z, solve_glider_opty.obj_grad_final_z
    _p = solve_glider_opty.Planner(_obj, _obj_grad)
    _p.configure(tol=1e-8, max_iter=_max_iter)
    _p.run()
    solve_glider_opty.plot_run(_p)
    solve_glider_opty.plot_solution_3D(_p, title='iteration {}'.format(_max_iter))
    plt.show()
    
def compare_wharington_params(force_run=False, cache_prefix='wharington_params'):
    fig, axes = plt.subplots(2, 4)
    for _i, _r in enumerate([10, 25, 50, 100]):
        atm = solve_glider_opty.AtmosphereWharingtonSym(center=[0, 0, 0], radius=_r, strength=1.)
        _p = solve_glider_opty.Planner(_atm=atm)
        _p.configure(tol=1e-5, max_iter=10000)
        _p.run_or_load(cache_filename(cache_prefix, 'r_{}.pkl'.format(_r)), force_run)
        alt_final, alt_mean = _p.zs[-1], np.mean(_p.zs)
        txt = 'radius {} m, final alt {:.1f} m'.format(_r, _p.zs[-1])
        solve_glider_opty.plot_solution_2D_en(_p, fig, axes[0,_i], title=txt)
        solve_glider_opty.plot_solution_2D_nu(_p, fig, axes[1,_i])
    fig.suptitle('Wharington_Params')
    plt.show()

#
# mean altitude
#
def obj_sum_z(free, _p):
    return _p.obj_scale * np.sum(free[_p._slice_z])/_p.num_nodes
def obj_grad_sum_z(free, _p):
    grad = np.zeros_like(free)
    grad[_p._slice_z] = _p.obj_scale/_p.num_nodes
    return grad
#
# circle trajectory
#
def obj_circle(free, _p, R=50):
    _x = free[_p._slice_x]
    _y = free[_p._slice_y]
    _r2s = _x**2+_y**2
    _es = _r2s - R**2
    #pdb.set_trace()
    return _p.obj_scale * np.sum(_es)/_p.num_nodes
def obj_grad_circle(free, _p):
    grad = np.zeros_like(free)
    grad[_p._slice_x] = _p.obj_scale/_p.num_nodes*2*free[_p._slice_x]
    grad[_p._slice_y] = _p.obj_scale/_p.num_nodes*2*free[_p._slice_y]
    return grad
def initial_guess_circle(_p, R=50, v=15.):
    ts = np.linspace(0.0, _p.duration, num=_p.num_nodes)
    #_p.num_nodes*_p.interval_value
    alphas = ts*v/R
    initial_guess = np.zeros(_p.prob.num_free)
    initial_guess[_p._slice_x] = R*np.cos(alphas)
    initial_guess[_p._slice_y] = R*np.sin(alphas)
    initial_guess[_p._slice_psi] = alphas
    initial_guess[_p._slice_phi] = R*np.sin(alphas)
    return initial_guess
#
# min average absolute roll angle (aka don't bank)
#
def obj_min_bank(free, _p):
    return _p.obj_scale * np.sum(np.abs(free[_p._slice_phi]))/_p.num_nodes
def obj_grad_min_bank(free, _p):
    grad = np.zeros_like(free)
    grad[_p._slice_phi] = _p.obj_scale/_p.num_nodes*np.sign(free[_p._slice_phi])
    return grad



    
def test_cst_wind(force_recompute=False, filename='/tmp/glider_opty_4d_cst_wind.npz'):
    atm = go_u.AtmosphereCstWindSym([4, 0, 0])
    #obj_f, obj_grad = obj_circle, obj_grad_circle
    #obj_f, obj_grad = obj_min_bank, obj_grad_min_bank
    obj_f, obj_grad = obj_sum_z, obj_grad_sum_z
    _p = go4.Planner( _obj_fun=obj_f, _obj_grad=obj_grad,
                      _atm=atm,
                      _n_constraint = (-40, 40),
                      _e_constraint = (-40, 40),
                      x0=10, y0=0, z0=-25, psi0=np.pi,
                      duration=30, hz=50., obj_scale=100.)
    initial_guess = initial_guess_circle(_p)
    go4.compute_or_load(atm, _p, force_recompute, filename, tol=1e-5, max_iter=500, initial_guess=initial_guess)
    go_u.plot_solution_chronogram(_p)
    go_u.plot_solution_2D_en(_p)
    go_u.plot_solution_2D_nu(_p, n0=-40, n1=50, dn=5., e0=0., h0=0., h1=70, dh=2.5)
    go_u.plot_solution_3D(_p)


def plot_all(_p, n0, n1, dn, e0, e1, de, h0, h1, dh):
    go_u.plot_solution_chronogram(_p)
    go_u.plot_solution_2D_en(_p)
    go_u.plot_solution_2D_nu(_p, n0=n0, n1=n1, dn=dn, e0=e0, h0=h0, h1=h1, dh=dh)
    go_u.plot_solution_3D(_p)

def test_thermal(force_recompute=False, filename='/tmp/glider_opty_4d_thermal.npz'):
    atm = go_u.AtmosphereWharingtonSym(radius=40., strength=-1, cst=[4, 0, 0])
    _p = go4.Planner( _obj_fun=go4.obj_final_z, _obj_grad=go4.obj_grad_final_z,
                       _atm=atm,
                      _n_constraint = (-40, 40),
                      _e_constraint = (-40, 40),
                      
                      x0=40, y0=0, z0=-1, psi0=np.pi,
                       duration=50, hz=50., obj_scale=1.)
    go4.compute_or_load(atm, _p, force_recompute, filename, tol=1e-5, max_iter=1000)
    plot_all(_p,  n0=-40, n1=50, dn=5., e0=-40, e1=40, de=5, h0=0., h1=70, dh=2.5)

def test_thermal_dual(force_recompute=False, filename='/tmp/glider_opty_4d_thermal_dual.npz'):
    atm = go_u.atm2()
    _p = go4.Planner( _obj_fun=go4.obj_final_z, _obj_grad=go4.obj_grad_final_z,
                       _atm=atm,
                      _phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.)),
                      _v_constraint = (8., 20.),
                      _n_constraint = (-40, 40),
                      _e_constraint = (-40, 40),
                      x0=40, y0=0, z0=-1, psi0=np.pi,
                      duration=50, hz=50., obj_scale=1.)
    go4.compute_or_load(atm, _p, force_recompute, filename, tol=1e-5, max_iter=1000)
    plot_all(_p,  n0=-40, n1=50, dn=5., e0=-40, e1=40, de=5, h0=0., h1=70, dh=2.5)
    
def test_slope(force_recompute=False, filename='/tmp/glider_opty_4d_slope.npz'):
    atm = go_u.AtmosphereRidgeSym(winf=4.)
    _p = go4.Planner( _obj_fun=go4.obj_sum_z, _obj_grad=go4.obj_grad_sum_z,
                      _atm=atm,
                      _n_constraint = (-50, 10),
                      _e_constraint = (-50, 50),
                      x0=10, y0=0, z0=-25, psi0=np.pi,
                      duration=60, hz=50., obj_scale=1.)
    go4.compute_or_load(atm, _p, force_recompute, filename, tol=1e-5, max_iter=1500)
    plot_all(_p,  n0=-40, n1=50, dn=5., e0=-40, e1=40, de=5, h0=0., h1=70, dh=2.5)
    
def main(force_recompute=False):
    #compare_nb_iter(force_recompute=False)
    #compare_obj()
    #compare_wharington_params(force_run=False)
    #test_cst_wind(force_recompute)
    #test_thermal(force_recompute)
    #test_thermal_dual(force_recompute)
    test_slope(force_recompute)
    plt.show()
    
if __name__ == '__main__':
    main('-force' in sys.argv)
