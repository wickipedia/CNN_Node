#!/usr/bin/env python
import math
import time
import numpy as np
import time


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

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

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
        self.lane_reading = None
        self.last_ms = None
        self.pub_counter = 0

        # TODO-TAL weird double initialisation
        self.velocity_to_m_per_s = 0.67
        self.omega_to_rad_per_s = 0.45 * 2 * math.pi

        # Setup parameters
        self.velocity_to_m_per_s = 1.53
        self.omega_to_rad_per_s = 4.75
        self.setGains()
        print('initialized lane_controller')
        # Subscriptions




    def setGains(self):
        self.v_bar_gain_ref = 0.5
        v_bar_fallback = 0.25  # nominal speed, 0.25m/s
        self.v_max = 1
        k_theta_fallback = -2.0
        k_d_fallback = - (k_theta_fallback ** 2) / (4.0 * self.v_bar_gain_ref)
        theta_thres_fallback = math.pi / 6.0
        d_thres_fallback = math.fabs(k_theta_fallback / k_d_fallback) * theta_thres_fallback
        d_offset_fallback = 0.0

        k_theta_fallback = k_theta_fallback
        k_d_fallback = k_d_fallback

        k_Id_fallback = 2.5
        k_Iphi_fallback = 1.25

        self.fsm_state = None
        self.cross_track_err = 0
        self.heading_err = 0
        self.cross_track_integral = 0
        self.heading_integral = 0
        self.cross_track_integral_top_cutoff = 0.3
        self.cross_track_integral_bottom_cutoff = -0.3
        self.heading_integral_top_cutoff = 1.2
        self.heading_integral_bottom_cutoff = -1.2
        # -1.2
        self.time_start_curve = 0
        self.cross_track_err_last = 0
        self.heading_err_last = 0

        self.use_feedforward_part_fallback = False


        self.omega_max = 999.0  # considering radius limitation and actuator limits   # to make sure the limit is not hit before the message is received

        self.use_radius_limit_fallback = True

        self.pose_initialized = False
        self.pose_msg_dict = dict()
        self.v_ref_possible = dict()
        self.main_pose_source = None

        self.active = True

        self.sleepMaintenance = False


    def updatePose(self, d, phi):

        prev_cross_track_err = self.cross_track_err
        prev_heading_err = self.heading_err

        self.d_offset = 0.0
        self.v_bar = 0.22
        #self.k_d = -3.5
        self.k_d = -4.5
        self.k_theta = -2
        self.d_thres = 0.2615
        self.theta_thres = 0.523
        self.d_offset = 0.0
        self.k_Id = 2
        self.k_Iphi = 0
        self.use_feedforward_part = False
        self.omega_ff = 0
        self.omega_max = 999
        self.omega_min = -999
        self.use_radius_limit = True
        self.min_rad = 0.06
        self.d_ref = 0
        self.phi_ref = 0
        self.object_detected = 0

        self.cross_track_err = d - self.d_offset

        self.heading_err = phi


        v = 0.15

        if math.fabs(self.cross_track_err) > self.d_thres:
            self.cross_track_err = self.cross_track_err / math.fabs(self.cross_track_err) * self.d_thres

        currentMillis = int(round(time.time() * 1000))

        if self.last_ms is not None:
            dt = (currentMillis - self.last_ms) / 1000.0
            self.cross_track_integral += self.cross_track_err * dt
            self.heading_integral += self.heading_err * dt

        if self.cross_track_integral > self.cross_track_integral_top_cutoff:
            self.cross_track_integral = self.cross_track_integral_top_cutoff
        if self.cross_track_integral < self.cross_track_integral_bottom_cutoff:
            self.cross_track_integral = self.cross_track_integral_bottom_cutoff

        if self.heading_integral > self.heading_integral_top_cutoff:
            self.heading_integral = self.heading_integral_top_cutoff
        if self.heading_integral < self.heading_integral_bottom_cutoff:
            self.heading_integral = self.heading_integral_bottom_cutoff

        if abs(
                self.cross_track_err) <= 0.011:  # TODO: replace '<= 0.011' by '< delta_d' (but delta_d might need to be sent by the lane_filter_node.py or even lane_filter.py)
            self.cross_track_integral = 0
        if abs(
                self.heading_err) <= 0.051:  # TODO: replace '<= 0.051' by '< delta_phi' (but delta_phi might need to be sent by the lane_filter_node.py or even lane_filter.py)
            self.heading_integral = 0
        if np.sign(self.cross_track_err) != np.sign(prev_cross_track_err):  # sign of error changed => error passed zero
            self.cross_track_integral = 0
        if np.sign(self.heading_err) != np.sign(prev_heading_err):  # sign of error changed => error passed zero
            self.heading_integral = 0



        omega_feedforward = v * 0.2
        if self.main_pose_source == "lane_filter" and not self.use_feedforward_part:
            omega_feedforward = 0

        # Scale the parameters linear such that their real value is at 0.22m/s TODO do this nice that  * (0.22/self.v_bar)
        omega = self.k_d * (0.15 / self.v_bar) * self.cross_track_err + self.k_theta * (
                    0.15 / self.v_bar) * self.heading_err
        omega += (omega_feedforward)

        # check if nominal omega satisfies min radius, otherwise constrain it to minimal radius
        if math.fabs(omega) > v / self.min_rad:
            if self.last_ms is not None:
                self.cross_track_integral -= self.cross_track_err * dt
                self.heading_integral -= self.heading_err * dt
            omega = math.copysign(v / self.min_rad, omega)

        Kdh = 4
        Kdp = 0

        errorDiffCross = self.cross_track_err - self.cross_track_err_last
        errorDiffHead = self.heading_err - self.heading_err_last

        if not self.fsm_state == "SAFE_JOYSTICK_CONTROL":
            # apply integral correction (these should not affect radius, hence checked afterwards)
            omega -= self.k_Id * (0.15 / self.v_bar) * self.cross_track_integral
            omega -= self.k_Iphi * (0.15 / self.v_bar) * self.heading_integral
            omega += (Kdh * errorDiffCross + Kdp * errorDiffHead)
        if v == 0:
            omega = 0
        else:
            # check if velocity is large enough such that car can actually execute desired omega
            if v - 0.5 * math.fabs(omega) * 0.1 < 0.065:
                v = 0.065 + 0.5 * math.fabs(omega) * 0.1

        # apply magic conversion factors
        v = v * self.velocity_to_m_per_s
        omega = omega * self.omega_to_rad_per_s

        if omega > self.omega_max: omega = self.omega_max
        if omega < self.omega_min: omega = self.omega_min
        omega += self.omega_ff
        self.cross_track_err_last = self.cross_track_err
        self.heading_err_last = self.heading_err
        return v, omega
