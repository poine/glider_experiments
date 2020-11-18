#! /usr/bin/env python
import sys, os, math, numpy as np
import logging
import pdb

import matplotlib.pyplot as plt

import pat3.atmosphere as p3_atm
import pat3.frames as p3_fr
import pat3.plot_utils as p3_pu
import pat3.vehicles.fixed_wing.legacy_6dof as p1_fw_dyn
import pat3.vehicles.fixed_wing.guidance as p3_guid


def main():
    # Read aircraft trajectory and control variables from a file
    ctl_logger = p3_guid.GuidancePurePursuitLogger()
    time, X, U = ctl_logger.load('/tmp/pat_glider_ds.npz')
    # This is needed :(
    atm = p3_atm.AtmosphereShearX(wind1=7.0, wind2=-1.0, xlayer=60.0, zlayer=40.0)
    # Plot aircraft trajectory
    p1_fw_dyn.plot_trajectory_ae(time, X, U, window_title='chronogram', atm=atm)
    # Convert aircraft trajectory to euclidian/euler state vector
    Xee = np.array([p3_fr.SixDOFAeroEuler.to_six_dof_euclidian_euler(_X, atm, _t) for _X, _t in zip(X, time)])
    # Compute energy
    inertial_vel_ned = Xee[:,p3_fr.SixDOFEuclidianEuler.sv_slice_vel]
    inertial_vel_norm = np.linalg.norm(inertial_vel_ned, axis=1)
    mass, g = 1., 9.81  # we can get that from the cularis dynamic model if needed, as well as inertia
    kinetic_energy = 0.5*mass*inertial_vel_norm**2
    pos_ned = Xee[:,p3_fr.SixDOFEuclidianEuler.sv_slice_pos]
    potential_energy = mass*g*-pos_ned[:,2]
    #pdb.set_trace()

    plt.figure()
    plt.plot(time, kinetic_energy, label='kinetic energy')
    plt.plot(time, potential_energy, label='potential energy')
    plt.plot(time, potential_energy+kinetic_energy, label='total energy')
    p3_pu.decorate(plt.gca(), title='Energy', xlab='time in s', ylab='E in joules', legend=True)
    plt.show()

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=500)
    main()
