#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, matplotlib.pyplot as plt

import solve_glider_opty


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
    
def main():
    #compare_nb_iter(force_recompute=False)
    #compare_obj()
    compare_wharington_params(force_run=False)

    
if __name__ == '__main__':
    main()
