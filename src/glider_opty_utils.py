#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np, sympy as sym
import matplotlib.pyplot as plt

import pat3.atmosphere as p3_atm
import pat3.plot_utils as p3_plu

import pdb

#
# Atmosphere
#
class AtmosphereCalmSym(p3_atm.AtmosphereCalm):
    # def get_wind_ned_sym(self, _x, _y, _z, _t):
    #     return 0.
    def get_wind_ned_sym2(self, _x, _y, _z, _t):
        return [0., 0., 0.]

class AtmosphereCstWindSym(p3_atm.AtmosphereCstWind):
    # def get_wind_ned_sym(self, _x, _y, _z, _t):
    #     return self.v[2]
    def get_wind_ned_sym2(self, _x, _y, _z, _t):
        return self.v

class AtmosphereWharingtonSym(p3_atm.AtmosphereWharington):
    # def get_wind_ned_sym(self, _x, _y, _z, _t):
    #     return  self.strength*sym.exp(-((_x-self.center[0])**2 + (_y-self.center[1])**2)/self.radius**2)
    def get_wind_ned_sym2(self, _x, _y, _z, _t):
        wd = self.strength*sym.exp(-((_x-self.center[0])**2 + (_y-self.center[1])**2)/self.radius**2)
        ret = sym.Array([self.cst[0], self.cst[1], self.cst[2]+wd])
        #ret = sym.Array(self.cst)+[0, 0, wd] # nope
        return ret

class AtmosphereWharingtonArraySym(p3_atm.AtmosphereWharingtonArray):
    def __init__(self, centers, radiuses, strengths):
        p3_atm.AtmosphereWharingtonArray.__init__(self, centers, radiuses, strengths)
        self._thermals = [AtmosphereWharingtonSym(_c, _r, _s) for _c, _r, _s in zip(centers, radiuses, strengths)]
        
    # def get_wind_ned_sym(self, _x, _y, _z, _t):
    #     return np.sum([_th.get_wind_ned_sym(_x, _y, _z, _t) for _th in self._thermals])

    def get_wind_ned_sym2(self, _x, _y, _z, _t):
        ret =  sym.Array([0, 0, 0])
        for _th in self._thermals:
            ret += _th.get_wind_ned_sym2(_x, _y, _z, _t)
        #_foo = [_th.get_wind_ned_sym2(_x, _y, _z, _t) for _th in self._thermals]
        #pdb.set_trace()
        return ret#np.sum(_foo)

class AtmosphereArray(p3_atm.AtmosphereArray):
    def get_wind_ned_sym2(self, _x, _y, _z, _t):
        ret =  sym.Array([0, 0, 0])
        for atm in self.atms:
            ret += atm.get_wind_ned_sym2(_x, _y, _z, _t)
        return ret


class AtmosphereArrayTest(AtmosphereArray):
    def __init__(self):
        a1 = AtmosphereHorizFieldSym()
        #a1 = AtmosphereWharingtonSym([0, 50, 0], 40, -1.)
        a2 = AtmosphereWharingtonSym([75, -75, 0], 40, -1.)
        a3 = AtmosphereWharingtonSym([-75, -75, 0], 40, -1.)
        a4 = AtmosphereWharingtonSym([0, 75, 0], 40, -1.)
        AtmosphereArray.__init__(self, [a1, a2, a3, a4])
        
class AtmosphereRidgeSym(p3_atm.AtmosphereRidge):
    # def get_wind_ned_sym(self, _x, _y, _z, _t):
    #     pos_ned = np.array([_x, _y, _z])
    #     dpos = pos_ned - self.c
    #     #dpos[1] = 0 # cylinder axis is y
    #     r2 = dpos[0]**2 + dpos[2]**2 
    #     #r = sym.sqrt(r2)#np.linalg.norm(dpos)
    #     eta = -sym.atan2(dpos[2], dpos[0])
    #     ceta, seta = sym.cos(eta), sym.sin(eta)
    #     R2ovr2 = self.R2/r2
    #     #wx, wz = 0, 0
    #     #wz = 2*self.winf*R2ovr2*ceta*seta
    #     #wz = 2*self.winf*ceta*seta*sym.Piecewise((R2ovr2, r >= self.R), (1., True))
    #     #wz = sym.Piecewise((2*self.winf*ceta*seta*R2ovr2, r >= self.R), (2*self.winf*ceta*seta, True))
    #     wz = sym.Piecewise((2*self.winf*ceta*seta*R2ovr2, r2 >= self.R2), (0., True))
    #     return wz

    def get_wind_ned_sym2(self, _x, _y, _z, _t):
        pos_ned = np.array([_x, _y, _z])
        dpos = pos_ned - self.c
        r2 = dpos[0]**2 + dpos[2]**2 
        eta = -sym.atan2(dpos[2], dpos[0])
        ceta, seta = sym.cos(eta), sym.sin(eta)
        R2ovr2 = self.R2/r2
        wx = self.winf*(1-R2ovr2*(ceta*ceta-seta*seta))
        wz = sym.Piecewise((2*self.winf*ceta*seta*R2ovr2, r2 >= self.R2), (0., True))
        return [wx, 0., wz]
    
def atm2():
    centers, radiuses, strengths = ([0, -75, 0], [50, 75, 0]), (25, 25), (-0.8, -1.5)
    return AtmosphereWharingtonArraySym(centers, radiuses, strengths)

def atm3():
    centers, radiuses, strengths = ([25, -75, 0], [75, 100, 0], [0, 0, 0]), (25, 25, 25), (-0.6, -1.5, -0.9)
    return AtmosphereWharingtonArraySym(centers, radiuses, strengths)
def atm4():
    centers, radiuses, strengths = ([25, -75, 0], [75, 250, 0], [0, 0, 0], [75, 100, 0]), (25, 25, 25, 20), (-0.6, -1.4, -0.8, -0.6) # last was 1.9
    return AtmosphereWharingtonArraySym(centers, radiuses, strengths)



class AtmosphereHorizFieldSym():

    def __init__(self, s=1.):
        self.s = s
    
    def get_wind_ned_sym2(self, _n, _e, _d, _t):
        return sym.Array([0, self.s*sym.cos(_n/25.), 0]) 

    def get_wind_ned(self, pos_ned, t):
        return [0, self.s*np.cos(pos_ned[0]/25.), 0]

class AtmosphereHorizFieldSym2():

    def __init__(self, s=1.):
        self.s = s
    
    def get_wind_ned_sym2(self, _n, _e, _d, _t):
        return [0, self.s*sym.cos(_n/25.), 0] 

    def get_wind_ned(self, pos_ned, t):
        return [0, self.s*np.cos(pos_ned[0]/25.), 0]



    
# oval
class AtmosphereWharingtonOval:
    def __init__(self, center=None, radius=50, strength=-2, cst=[0, 0, 0]):
        self.center = np.asarray(center) if center is not None else np.array([0, 0, 0])
        self.radius = radius
        self.strength = strength
        self.foo=3.
        self.cst = cst
        self.r2 = self.radius**2
    
    def get_wind_ned(self, pos_ned, t): 
        dpos = pos_ned - self.center
        r2 = dpos[0]**2+self.foo*dpos[1]**2
        wz = self.strength*np.exp(-r2/self.r2)
        return np.array([0, 0, wz])+self.cst
        
    def get_wind_ned_sym2(self, _n, _e, _d, _t):
        wd = self.strength*sym.exp(-((_n-self.center[0])**2 + self.foo*(_e-self.center[1])**2)/self.radius**2)
        ret = sym.Array([self.cst[0], self.cst[1], self.cst[2]+wd])
        return ret
    
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
    fig, axes = plt.subplots(3, 2)
    axes[0,0].plot(planner.sol_time, planner.sol_e)
    p3_plu.decorate(axes[0,0], 'E', 't in s', 'e in m')
    axes[1,0].plot(planner.sol_time, planner.sol_n)
    p3_plu.decorate(axes[1,0], 'N', 't in s', 'n in m')
    axes[2,0].plot(planner.sol_time, planner.sol_u)
    p3_plu.decorate(axes[2,0], 'U', 't in s' , 'h in m')
    axes[0,1].plot(planner.sol_time, np.rad2deg(planner.sol_psi))
    p3_plu.decorate(axes[0,1], 'psi', 't in s' , 'psi in deg')
    axes[1,1].plot(planner.sol_time, planner.sol_v)
    p3_plu.decorate(axes[1,1], 'v', 't in s' , 'v in m/s')
    axes[2,1].plot(planner.sol_time, np.rad2deg(planner.sol_phi))
    p3_plu.decorate(axes[2,1], 'phi', 't in s', 'phi in deg')
    


# 2D horizontal trajectory
def plot_solution_2D_en(planner, figure=None, ax=None, title=None,  # east north (up)
                        show_quiver=True, contour_wz=False,):
    figure, ax = p3_plu.plot_slice_wind_ne(planner.atm,
                                           n0=planner._n_constraint[0], n1=planner._n_constraint[1], dn=5.,
                                           e0=planner._e_constraint[0], e1=planner._e_constraint[1], de=5, h0=20., t0=0.,
                                           show_quiver=show_quiver, contour_wz=contour_wz, show_color_bar=True, title=title,
                                           figure=figure, ax=ax)
    ax.plot(planner.sol_e, planner.sol_n)
    ax.axis('equal')
    return figure, ax


# 2D vertical trajectory
def plot_solution_2D_nu(planner, figure=None, ax=None, title=None,             # north (east) up
                        show_quiver=True, contour_wz=False,
                        n0=-100, n1=100, dn=5., e0=0., h0=0., h1=30, dh=2.):
    figure, ax = p3_plu.plot_slice_wind_nu(planner.atm,
                                           n0=planner._n_constraint[0], n1=planner._n_constraint[1], dn=dn,
                                           e0=e0,
                                           h0=h0, h1=h1, dh=dh, zdir=-1.,
                                           show_quiver=show_quiver, contour_wz=contour_wz, show_color_bar=True, title=title,
                                           figure=figure, ax=ax)
    ax.plot(planner.sol_n, planner.sol_u)  
    ax.axis('equal')
    return figure, ax

def plot_solution_2D_eu(planner, figure=None, ax=None, title=None,             # north (east) up
                        show_quiver=True, contour_wz=False,
                        e0=-100, e1=100, de=5., n0=0., h0=0., h1=30, dh=2.):
    figure, ax = p3_plu.plot_slice_wind_eu(planner.atm,
                                           e0=planner._n_constraint[0], e1=planner._n_constraint[1], de=de,
                                           n0=n0, h0=h0, h1=h1, dh=dh, zdir=-1.,
                                           show_quiver=show_quiver, contour_wz=contour_wz, show_color_bar=True, title=title,
                                           figure=figure, ax=ax)
    ax.plot(planner.sol_e, planner.sol_u)  # ENU??
    ax.axis('equal')
    return figure, ax


# 3D trajectory
def plot_solution_3D(planner, figure=None, ax=None, title=""):
    fig = figure if figure is not None else plt.figure()
    ax = ax if ax is not None else fig.add_subplot(111, projection='3d')
    ax.plot(planner.sol_n, planner.sol_e, planner.sol_u, color='b', label='aircraft trajectory')
    ax.set_xlabel('N axis')
    ax.set_ylabel('E axis')
    ax.set_zlabel('U axis')
    ax.set_title(title)
    p3_plu.set_3D_axes_equal()
    return figure, ax
