import math
import numpy as np


class SteeringToWheelVelWrapper():
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




class Controller():

    def __init__(self, k_P, k_I, u_sat, k_t, c1, c2):

        # C matrix of LTI system
        self.c1 = c1
        self.c2 = c2

        # Gains for controller
        self.k_P = k_P
        self.k_I = k_I

        # Saturation of motors [rad/s]
        self.u_sat = u_sat

        # Feedback gain for anti-windup
        self.k_t = k_t

        # Variable for integral
        self.C_I = 0

    # Inputs:   d_est   Estimation of distance from lane center (positve when
    #                   offset to the left of driving direction) [m]
    #           phi_est Estimation of angle of bot (positive when angle to the
    #                   left of driving direction) [rad]
    #           d_ref   Reference of d (for lane following, d_ref = 0) [m]
    #           v_ref   Reference of velocity [m/s]
    #           t_delay Delay it took from taking image up to now [s]
    #           dt_last Time it took from last processing to current [s]

    # Output:   v_out       velocity of Duckiebot [m/s]
    #           omega_out   angular velocity of Duckiebot [rad/s]

    def getControlOutput(self, d_est, phi_est, d_ref, phi_ref, v_ref, t_delay, dt_last):

        # Calculate the output y
        ref =   (self.c1 * d_ref + self.c2 * phi_ref)
        y =     (self.c1 * d_est + self.c2 * phi_est)
        err = ref - y

        # PI-Controller
        C_P = self.k_P * err
        omega = C_P + self.C_I

        # Calculate new value of integral while considering the anti-windup
        self.C_I = self.C_I + dt_last * ( self.k_I * err + self.k_t * ( self.sat(omega) - omega ) )

        # Declaring return values
        omega_out = omega
        v_out = v_ref
        return (v_out, omega_out)





    # Defining the saturation function of the motors
    def sat(self, u):
        if u > self.u_sat:
            return self.u_sat
        if u < -self.u_sat:
            return -self.u_sat
        return u


    def updateParams(self, k_P, k_I, u_sat, k_t, c1, c2):
        self.k_P = k_P
        self.k_I = k_I
        self.u_sat = u_sat
        self.k_t = k_t
        self.c1 = c1
        self.c2 = c2