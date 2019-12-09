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
        trim=0.1,
        radius=0.0318,
        k=27.0,
        limit=1.0):

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = float(os.environ['gain'])

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
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
        k_r_inv = (self.gain *(1+self.trim)) / k_r
        k_l_inv = (self.gain *(1-self.trim)) / k_l

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
        self.last_ms = None
        self.dt = 0
        self.pub_counter = 0
        self.fsm_state = None
        self.v_des = 0.22
        self.v_bar = 0.22

        # TODO-TAL weird double initialisation
        self.velocity_to_m_per_s = 0.67
        self.omega_to_rad_per_s = 0.45 * 2 * math.pi

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
        self.cross_track_integral_top_cutoff = 0.3
        self.cross_track_integral_bottom_cutoff = -0.3

        self.heading_integral_top_cutoff = 1.2
        self.heading_integral_bottom_cutoff = -1.2

        # Init last previous values
        self.cross_track_err_last = 0
        self.heading_err_last = 0


        # init kinematics cutoffs
        self.omega_max = 20
        self.omega_min = -20

        # other init stuff we don't know about
        self.pose_initialized = False
        self.pose_msg_dict = dict()
        self.v_ref_possible = dict()
        self.main_pose_source = None
        self.active = True
        self.sleepMaintenance = False


    def updatePose(self, d, phi, image_timestamp=0):
        # Calculating the delay image processing took
        # timestamp_now = rospy.Time.now()
        # image_delay_stamp = timestamp_now - image_timestamp

        # delay from taking the image until now in seconds
        # image_delay = image_delay_stamp.secs + image_delay_stamp.nsecs / 1e9
        # print(image_delay)

        # Calculating the delay image processing took
        timestamp_now = rospy.Time.now()
        image_delay_stamp = timestamp_now - image_timestamp
        image_delay = image_delay_stamp.secs + image_delay_stamp.nsecs / 1e9
        self.image_delay = image_delay
        print(self.image_delay)

        # update those controller params every iteration
        self.k_d = 0
        self.k_theta = 0

        self.k_Id = 0
        self.k_Iphi = 0


        self.k_Dd = 0
        self.k_Dphi= 0

        # Offsets compensating learning or optimize position on lane
        self.d_offset = 0.06

        # Params which are update by OS input
        self.k_d = float(os.environ['kPd'])
        self.k_theta = float(os.environ['kPp'])
        self.k_Id = float(os.environ['kId'])
        self.k_Iphi = float(os.environ['kIp'])
        self.k_Dd = float(os.environ['kDd'])
        self.k_Dphi = float(os.environ['kDp'])
        self.v_des = float(os.environ['v_des'])

        # if self.image_delay > 0.2:
        #     self.v_des = 0.05
        # else:
        #     self.v_des = self.v_des


        # Calc errors
        self.cross_track_err = d - self.d_offset
        self.heading_err = phi

        currentMillisec = int(round(time.time() * 1000))

        if self.last_ms is not None:
            self.dt = (currentMillisec - self.last_ms) / 1000.0

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


        # if abs(self.cross_track_err) <= 0.011:  # TODO: replace '<= 0.011' by '< delta_d' (but delta_d might need to be sent by the lane_filter_node.py or even lane_filter.py)
        #     self.cross_track_integral = 0
        # if abs(self.heading_err) <= 0.051:  # TODO: replace '<= 0.051' by '< delta_phi' (but delta_phi might need to be sent by the lane_filter_node.py or even lane_filter.py)
        #     self.heading_integral = 0
        if np.sign(self.cross_track_err) != np.sign(self.cross_track_err_last):  # sign of error changed => error passed zero
            self.cross_track_integral = 0
        if np.sign(self.heading_err) != np.sign(self.heading_err_last):  # sign of error changed => error passed zero
            self.heading_integral = 0


        if not self.fsm_state == "SAFE_JOYSTICK_CONTROL":
            omega = 0
            # Apply Controller to kinematics
            # P-Controller
            omega += (self.k_d * (self.v_des / self.v_bar) * self.cross_track_err) + (self.k_theta * (self.v_des / self.v_bar) * self.heading_err)
            omega += (self.k_Id * (self.v_des / self.v_bar) * self.cross_track_integral) + (self.k_Iphi * (self.v_des / self.v_bar) * self.heading_integral)
            omega += (self.k_Dd * (self.v_des / self.v_bar) * self.cross_track_differential) + (self.k_Dphi * (self.v_des / self.v_bar) * self.heading_differential)


        # apply magic conversion factors
        v = (self.v_des / self.v_bar) * self.velocity_to_m_per_s
        omega = omega * self.omega_to_rad_per_s

        # check if kinematic constraints are ok
        if omega > self.omega_max:
            omega = self.omega_max
        if omega < self.omega_min:
            omega = self.omega_min

        # write actual params as pervious params
        self.cross_track_err_last = self.cross_track_err
        self.heading_err_last = self.heading_err
        self.last_ms = currentMillisec


        return v, omega
