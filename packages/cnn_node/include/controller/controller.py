#!/usr/bin/env python
import math
import time
import numpy as np
import time
import rospy
import os


class lane_controller:
    def __init__(self):
        # Init Params which are independent of controller
        self.v_des = 0.22
        self.v_bar = 0.22

        self.phi_last = None

        #Timing
        self.time_update_pose = None
        self.time_update_pose_last = None

        # Setup parameters
        self.velocity_to_m_per_s = 1.53
        self.omega_to_rad_per_s = 4.75
        self.setParams()
        print('initialized lane_controller')

    def setParams(self):
        # Init controller params
        self.cross_track_err = 0
        self.heading_err = 0

        self.cross_track_integral = 0
        self.heading_integral = 0

        self.cross_track_differential = 0
        self.heading_differential = 0

        # Init controller cutoffs
        self.cross_track_integral_top_cutoff = 1.5
        self.cross_track_integral_bottom_cutoff = -1.5

        self.heading_integral_top_cutoff = 0.5
        self.heading_integral_bottom_cutoff = -0.5

        # Init last previous values
        self.cross_track_err_last = None
        self.heading_err_last = None
        self.dt = None

        # init kinematics cutoffs
        self.omega_max = 8
        self.omega_min = -8

        # other init stuff we don't know about
        self.pose_initialized = False
        self.pose_msg_dict = dict()
        self.v_ref_possible = dict()
        self.main_pose_source = None
        self.active = True
        self.sleepMaintenance = False


    def updatePose(self, d, phi):
        if self.time_update_pose is None:
            self.time_update_pose = time.time()
            self.dt = 0
        else:
            self.time_update_pose_last = self.time_update_pose
            self.time_update_pose = time.time()
            self.dt = self.time_update_pose - self.time_update_pose_last

        # Set controller gains
        self.k_d = -2
        self.k_theta = -3.2

        self.k_Id = -0.9
        self.k_Iphi = -0.6

        self.k_Dd = -0
        self.k_Dphi= -0.12

        # Offsets compensating learning or optimize position on lane
        self.d_offset = 0.01

        self.v = 0.16

        # if change in angle (phi) is large don't use distance for control
        if self.phi_last is not None and self.dt is not 0:
            if np.abs((phi - self.phi_last)/self.dt) > 0.00065:
                self.k_d = 0

        # phi dependence on last estimate. If phi negative increase value 
        # to avoid leaving the road
        if self.phi_last is not None:
            phi = (phi*3 + self.phi_last)/4
            if phi < 0:
                phi = phi * 1.23

        # if distance negative increase value. Avoid leaving the road
        if d<0:
            d = d * 1.1

        # Upper threshhold for phi. Improves stability
        if np.abs(phi) > 0.36:
            if np.sign(phi) == np.sign(1):
                phi = 0.36
            else:
                phi = -0.36

        # Calc errors
        self.cross_track_err = d - self.d_offset
        self.heading_err = phi

        if self.cross_track_err < 0:
            self.cross_track_err=self.cross_track_err*3

        if self.dt is not 0:
            # Apply Integral
            self.cross_track_integral += self.cross_track_err * self.dt
            self.heading_integral += self.heading_err * self.dt

            # Apply Differential
            self.cross_track_differential = (self.cross_track_err - self.cross_track_err_last)/self.dt
            self.heading_differential = (self.heading_err - self.heading_err_last)/self.dt

        # Check integrals
        if self.cross_track_integral > self.cross_track_integral_top_cutoff:
            self.cross_track_integral = self.cross_track_integral_top_cutoff
        if self.cross_track_integral < self.cross_track_integral_bottom_cutoff:
            self.cross_track_integral = self.cross_track_integral_bottom_cutoff

        if self.heading_integral > self.heading_integral_top_cutoff:
            self.heading_integral = self.heading_integral_top_cutoff
        if self.heading_integral < self.heading_integral_bottom_cutoff:
            self.heading_integral = self.heading_integral_bottom_cutoff

        if self.cross_track_err_last is not None:
            if np.sign(self.cross_track_err) != np.sign(self.cross_track_err_last):  # sign of error changed => error passed zero
                self.cross_track_integral = 0
            if np.sign(self.heading_err) != np.sign(self.heading_err_last):  # sign of error changed => error passed zero
                self.heading_integral = 0

        omega = 0
        # Apply Controller to kinematics
        omega += (self.k_d * (self.v_des / self.v_bar) * self.cross_track_err) + (self.k_theta * (self.v_des / self.v_bar) * self.heading_err)
        omega += (self.k_Id * (self.v_des / self.v_bar) * self.cross_track_integral) + (self.k_Iphi * (self.v_des / self.v_bar) * self.heading_integral)
        omega += (self.k_Dd * (self.v_des / self.v_bar) * self.cross_track_differential) + (self.k_Dphi * (self.v_des / self.v_bar) * self.heading_differential)
        

        # apply magic conversion factors
        v = self.v * self.velocity_to_m_per_s
        omega = omega * self.omega_to_rad_per_s

        # check if kinematic constraints are ok
        if omega > self.omega_max:
            omega = self.omega_max
        if omega < self.omega_min:
            omega = self.omega_min

        # write actual params as pervious params
        self.cross_track_err_last = self.cross_track_err
        self.heading_err_last = self.heading_err
        self.phi_last = phi

        return v, omega
