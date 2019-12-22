#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import rospy
import yaml
from torchvision import models, transforms
import torch
import math

from cv_bridge import CvBridge
from PIL import Image
import sys
from duckietown import DTROS
from sensor_msgs.msg import CompressedImage, Temperature
from duckietown_msgs.msg import WheelsCmdStamped, LanePose, Twist2DStamped, BoolStamped
from controller.controller import lane_controller
#from CNN_Model.CNN_Model import OurCNN
from dt_cnn.model import model_dist, model_angle


class CNN_Node(DTROS):
    def __init__(self, node_name, DEBUG=False):
        # Initialize the DTROS parent class
        super(CNN_Node, self).__init__(node_name=node_name)
        self.vehicle = os.environ['VEHICLE_NAME']

        # set debug mode
        self.debug = DEBUG
        rospy.set_param('/' + self.vehicle + '/camera_node/exposure_mode', 'auto')

        # initialize cnn models
        path_to_home = os.path.dirname(os.path.abspath(__file__))
        loc_dist = path_to_home + "/../models/CNN_1575930672.3374033_lr0.05_bs16_epo150_Model_final_crop062_SQD_Huber_w_statedict"
        loc_theta = path_to_home + "/../models/CNN_1575756253.5257602_lr0.04_bs16_epo150_Model_finaltheta_w_statedict"
        self.model_d = model_dist(as_gray=True,use_convcoord=False)
        self.model_d.load_state_dict(torch.load(loc_dist))
        self.model_th = model_angle(as_gray=True, use_convcoord=False)
        self.model_th.load_state_dict(torch.load(loc_theta)) 
        self.model_d.eval()
        self.model_d.float()
        self.model_th.float()
        self.model_th.eval()

        # initialize controller
        self.pidController = lane_controller()
        self.pidController.setParams()


        self.time_image_rec = None

        # set flags
        self.onShutdown_trigger = False
        self.stop_pub_pose = True
        
        # initalize subscribers and publishers
        topicSub_image = '/' + self.vehicle + '/camera_node/image/compressed'
        self.subscriber(topicSub_image, CompressedImage, self.compute_pose)

        topicPub_WheelCmd = '/' + self.vehicle + '/wheels_driver_node/wheels_cmd'
        self.pub_wheels_cmd = self.publisher(topicPub_WheelCmd, WheelsCmdStamped, queue_size=1)
        self.msg_wheels_cmd = WheelsCmdStamped()
        
        topicPub_CarCmd = "/" + self.vehicle + "/cnn_node/car_cmd"
        self.pub_car_cmd = self.publisher(topicPub_CarCmd, Twist2DStamped, queue_size=1)

        # pub_joy_override and pub_cnn_toggle use to change states
        topicPub_joy_override = "/" + self.vehicle + "/joy_mapper_node/joystick_override"
        self.pub_joy_override = self.publisher(topicPub_joy_override, BoolStamped, queue_size=5)

        topicPub_cnn_toggle = "/" + self.vehicle + "/joy_mapper_node/cnn_lane_toggle"
        self.pub_cnn_toggle = self.publisher(topicPub_cnn_toggle, BoolStamped, queue_size=5)
        
        self.log("CNN Node Initialized")
        rospy.on_shutdown(self.onShutdown)


    def compute_pose(self, frame):
        '''
        Callback function for /camera_node/image/compressed
        Input: compressed image
        Output: velocity and angular velocity
        '''


        # don't compute pose if on shutdown or in different state (stop_pub_pose)
        if self.onShutdown_trigger or self.stop_pub_pose:
            return

        cv_image = CvBridge().compressed_imgmsg_to_cv2(frame, desired_encoding="passthrough")
        # Convert CV image to PIL image for pytorch
        im_pil = Image.fromarray(cv_image)
        # Transform image for CNN input
        X_d = self.model_d.transform(im_pil).unsqueeze(1)
        X_th = self.model_th.transform(im_pil).unsqueeze(1)

        self.time_image_rec = frame.header.stamp

        # Compute Pose with CNN
        state = [0,0]
        state[0] = self.model_d(X_d).detach().numpy()[0][0]
        state[1] = self.model_th(X_th).detach().numpy()[0][1]

        if self.debug:
            print("states (distance, angle)", state)

        # Compute velocity (v) and angular velocity (omega) with PID controller
        v, omega = self.pidController.updatePose(state[0], state[1])

        # Publish velocity and angular velocity for kinematics node
        car_cmd_msg = Twist2DStamped()
        car_cmd_msg.header.stamp = rospy.get_rostime()
        car_cmd_msg.v = v
        car_cmd_msg.omega = omega
        self.pub_car_cmd.publish(car_cmd_msg)

        if self.debug:
            print("Car Commands [v [m/s], omega [degr/sec]]",v,omega)


    def onShutdown(self):
        """Shutdown procedure.

        Publishes a zero velocity command at shutdown."""

        self.onShutdown_trigger = True
        rospy.sleep(1)

        self.msg_wheels_cmd = WheelsCmdStamped()
        self.msg_wheels_cmd.header.stamp = rospy.get_rostime()
        self.msg_wheels_cmd.vel_left = 0.0
        self.msg_wheels_cmd.vel_right = 0.0
        for g in range(0,50):
            self.pub_wheels_cmd.publish(self.msg_wheels_cmd)

        self.log("Close CNN_Node")

    def change_state(self):
        '''
        Changes state from NORMAL_JOYSTICK_CONTROL to CNN_LANE_FOLLOWING and vice versa
        '''
        while not self.onShutdown_trigger:
            cmd = None
            cmd = raw_input("Input your cmd ([a]: start cnn lane following. \
                [s] stop cnn lane following)")
            if cmd == 'a':
                # Change state from NORMAL_JOYSTICK_CONTROL to CNN_LANE_FOLLOWING
                self.log("Change state to CNN_LANE_FOLLOWING")
                self.stop_pub_pose = False
                cmd_msg = BoolStamped()
                cmd_msg.data = False
                self.pub_joy_override.publish(cmd_msg)
                cmd_msg.data = True
                self.pub_cnn_toggle.publish(cmd_msg)

            elif cmd == 's':
                # Change state from CNN_LANE_FOLLOWING to NORMAL_JOYSTICK_CONTROL
                self.log("Change state to NORMAL_JOYSTICK_CONTROL")
                self.stop_pub_pose = True
                # Wait until last pose computed
                rospy.sleep(1)

                # Publish zero wheel commands
                self.msg_wheels_cmd = WheelsCmdStamped()
                self.msg_wheels_cmd.header.stamp = rospy.get_rostime()
                self.msg_wheels_cmd.vel_left = 0.0
                self.msg_wheels_cmd.vel_right = 0.0
                for g in range(0,20):
                    self.pub_wheels_cmd.publish(self.msg_wheels_cmd)
                
                # Trigger state transition 
                cmd_msg = BoolStamped()
                cmd_msg.data = True
                self.pub_joy_override.publish(cmd_msg)
                cmd_msg.data = False
                self.pub_cnn_toggle.publish(cmd_msg)

        print("Exit change mode")

if __name__ == '__main__':
    # Initialize the node
    try:
        DEBUG = os.environ['DEBUG']
    except KeyError:
        DEBUG = False

    camera_node = CNN_Node(node_name='cnn_node', DEBUG=DEBUG)
    # Change FSM
    camera_node.change_state()
