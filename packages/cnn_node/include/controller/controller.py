#!/usr/bin/env python
import math
import time
import numpy as np
import time
import rospy
import os


class SteeringToWheelVelWrapper:
    """
    Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format

    """

    def __init__(self,

        gain=1.0,
        trim=0.0,
        radius=0.0318,
        k=27.0,
        limit=1.0):

        self.vehicle = os.environ['VEHICLE_NAME']

        try:        
            self.gain = float(os.environ['gain'])
            self.trim = rospy.get_param("/"+self.vehicle+"/kinematics_node/trim")
            self.radius = rospy.get_param("/"+self.vehicle+"/kinematics_node/radius")
            self.k = rospy.get_param("/"+self.vehicle+"/kinematics_node/k")
            self.limit = rospy.get_param("/"+self.vehicle+"/kinematics_node/limit")
        except KeyError:
            print("No ROS params. Use default values")
            self.gain = gain
            self.trim = trim
            self.radius = radius
            self.k = k
            self.limit = limit


        print('initialized wrapper')

    def action(self, action):
        vel, angle = action

        # Distance between the wheels
        baseline = 0.1

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels



class lane_controller:
    def __init__(self):
        # Init Params which are independent of controller
        self.lane_reading = None
        self.pub_counter = 0
        self.fsm_state = None
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
        # Subscriptions

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

        # update those controller params every iteration
        self.k_d = -2
        self.k_theta = -3.2

        self.k_Id = -0.8
        self.k_Iphi = -0.6

        self.k_Dd = -0
        self.k_Dphi= -0.12

        # Offsets compensating learning or optimize position on lane
        self.d_offset = 0.01

        self.v = 0.17

        #print("latency:",self.dt)

        if self.phi_last is not None and self.dt is not 0:
            if np.abs((phi - self.phi_last)/self.dt) > 0.00065:
                self.k_d = 0

        if self.phi_last is not None:
            phi = (phi*3 + self.phi_last)/4
            if phi < 0:
                phi = phi * 1.23

        if d<0:
            d = d * 1.1

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


        if not self.fsm_state == "SAFE_JOYSTICK_CONTROL":
            omega = 0
            # Apply Controller to kinematics
            omega += (self.k_d * (self.v_des / self.v_bar) * self.cross_track_err) + (self.k_theta * (self.v_des / self.v_bar) * self.heading_err)
            omega += (self.k_Id * (self.v_des / self.v_bar) * self.cross_track_integral) + (self.k_Iphi * (self.v_des / self.v_bar) * self.heading_integral)
            omega += (self.k_Dd * (self.v_des / self.v_bar) * self.cross_track_differential) + (self.k_Dphi * (self.v_des / self.v_bar) * self.heading_differential)

        # print("Crosstrack_Error: ",self.cross_track_err)
        # print("Heading_Error: ",self.heading_err)
        # print("Crosstrack_Int: ",self.cross_track_integral)
        # print("Heading_Int: ",self.heading_integral)
        # print("Crosstrack_Diff: ",self.cross_track_differential)
        # print("Heading_Diff: ",self.heading_differential)


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
