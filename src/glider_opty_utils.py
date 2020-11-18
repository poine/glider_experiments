#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np, sympy as sym
import matplotlib.pyplot as plt

import pat3.atmosphere as p3_atm
import pat3.plot_utils as p3_plu

#
# Atmosphere
#
class AtmosphereCalmSym(p3_atm.AtmosphereCalm):
    def get_wind_ned_sym(self, _x, _y, _z, _t):
        return 0.
    def get_wind_ned_sym2(self, _x, _y, _z, _t):
        return [0., 0., 0.]

class AtmosphereWharingtonSym(p3_atm.AtmosphereWharington):
    def get_wind_ned_sym(self, _x, _y, _z, _t):
        return  self.strength*sym.exp(-((_x-self.center[0])**2 + (_y-self.center[1])**2)/self.radius**2)

class AtmosphereWharingtonArraySym(p3_atm.AtmosphereWharingtonArray):
    def __init__(self, centers, radiuses, strengths):
        p3_atm.AtmosphereWharingtonArray.__init__(self, centers, radiuses, strengths)
        self._thermals = [AtmosphereWharingtonSym(_c, _r, _s) for _c, _r, _s in zip(centers, radiuses, strengths)]
        
    def get_wind_ned_sym(self, _x, _y, _z, _t):
        return np.sum([_th.get_wind_ned_sym(_x, _y, _z, _t) for _th in self._thermals])


class AtmosphereRidgeSym(p3_atm.AtmosphereRidge):
    def get_wind_ned_sym(self, _x, _y, _z, _t):
        pos_ned = np.array([_x, _y, _z])
        dpos = pos_ned - self.c
        #dpos[1] = 0 # cylinder axis is y
        r2 = dpos[0]**2 + dpos[2]**2 
        #r = sym.sqrt(r2)#np.linalg.norm(dpos)
        eta = -sym.atan2(dpos[2], dpos[0])
        ceta, seta = sym.cos(eta), sym.sin(eta)
        R2ovr2 = self.R2/r2
        #wx, wz = 0, 0
        #wz = 2*self.winf*R2ovr2*ceta*seta
        #wz = 2*self.winf*ceta*seta*sym.Piecewise((R2ovr2, r >= self.R), (1., True))
        #wz = sym.Piecewise((2*self.winf*ceta*seta*R2ovr2, r >= self.R), (2*self.winf*ceta*seta, True))
        wz = sym.Piecewise((2*self.winf*ceta*seta*R2ovr2, r2 >= self.R2), (0., True))
        return wz

        
    
def atm2():
    centers, radiuses, strengths = ([-50, 0, 0], [50, 0, 0]), (25, 25), (-1., -1.)
    return AtmosphereWharingtonArraySym(centers, radiuses, strengths)
    
#
# Glider sink rate model
#
def glider_sink_rate(va, phi):
    #            pat cularis      # ardusoaring
    polar_K   =  49.5050092764    # 25.6   # Cl factor 2*m*g/(rho*S) @Units: m.m/s/s
    polar_CD0 =   0.0122440667444 #  0.027 # Zero lift drag coef
    polar_B   =   0.0192172535765 #  0.031 # Induced drag coeffient
    CL0 =  polar_K / va**2
    C1 = polar_CD0 / CL0  # constant describing expected angle to overcome zero-lift drag
    C2 = polar_B * CL0
    #return -va * (C1 + C2 / sym.cos(phi)**2)  # z up
    return va * (C1 + C2 / sym.cos(phi)**2)  # z down





# Chronograms
def plot_solution_chronogram(planner):
    fig, axes = plt.subplots(6, 1)
    axes[0].plot(planner.sol_time, planner.sol_x)
    p3_plu.decorate(axes[0], 'N', 't in s', 'x in m')
    axes[1].plot(planner.sol_time, planner.sol_y)
    p3_plu.decorate(axes[1], 'E', 't in s', 'y in m')
    axes[2].plot(planner.sol_time, -planner.sol_z)
    p3_plu.decorate(axes[2], 'h', 't in s' , 'h in m')
    axes[3].plot(planner.sol_time, np.rad2deg(planner.sol_psi))
    p3_plu.decorate(axes[3], 'psi', 't in s' , 'psi in deg')
    axes[4].plot(planner.sol_time, planner.sol_v)
    p3_plu.decorate(axes[4], 'v', 't in s' , 'v in m/s')
    axes[5].plot(planner.sol_time, np.rad2deg(planner.sol_phi))
    p3_plu.decorate(axes[5], 'phi', 't in s', 'phi in deg')
    


# 2D horizontal trajectory
def plot_solution_2D_en(planner, figure=None, ax=None, title=None): # east north (up)
    figure, ax = p3_plu.plot_slice_wind_ne(planner.atm,
                                           n0=-100, n1=100, dn=5., e0=-100., e1=100, de=5, h0=20., t0=0.,
                                           show_color_bar=True,
                                           figure=figure, ax=ax)
    ax.plot(planner.sol_x, planner.sol_y)
    ax.axis('equal')
    if title is not None: ax.set_title(title)
    return figure, ax


# 2D vertical trajectory
def plot_solution_2D_nu(planner, figure=None, ax=None, title=None,             # north (east) up
                        n0=-100, n1=100, dn=5., e0=0., h0=0., h1=30, dh=2.):
    figure, ax = p3_plu.plot_slice_wind_nu(planner.atm,
                                           n0=n0, n1=n1, dn=dn, e0=e0, h0=h0, h1=h1, dh=dh, zdir=-1.,
                                           show_quiver=True, show_color_bar=True, title="North Up slice",
                                           figure=figure, ax=ax)
    ax.plot(planner.sol_x, -planner.sol_z)  # ENU??
    ax.axis('equal')
    if title is not None: ax.set_title(title)
    return figure, ax

# 3D trajectory
def plot_solution_3D(planner, figure=None, ax=None, title=""):
    fig = figure if figure is not None else plt.figure()
    ax = ax if ax is not None else fig.add_subplot(111, projection='3d')
    ax.plot(planner.sol_x, planner.sol_y, planner.sol_z, color='b', label='aircraft trajectory')
    ax.set_xlabel('N axis')
    ax.set_ylabel('E axis')
    ax.set_zlabel('D axis')
    ax.set_title(title)
    p3_plu.set_3D_axes_equal()
    return figure, ax
