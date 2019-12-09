#!/usr/bin/env python

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
from duckietown_msgs.msg import WheelsCmdStamped, LanePose, Twist2DStamped
from controller import SteeringToWheelVelWrapper, lane_controller
#from CNN_Model.CNN_Model import OurCNN
from CNN_Model import OurCNN


class TransCropHorizon(object):
    """Crop the Horizon.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, crop_value, set_black=False):
        assert isinstance(set_black, (bool))
        self.set_black = set_black

        if crop_value >= 0 and crop_value < 1:
            self.crop_value = crop_value
        else:
            print('One or more Arg is out of range!')

    def __call__(self, image):
        crop_value = self.crop_value
        set_black = self.set_black
        image_heiht = image.size[1]
        crop_pixels_from_top = int(round(image_heiht*crop_value,0))

        # convert from PIL to np
        image = np.array(image)

        if set_black==True:
            image[:][0:crop_pixels_from_top-1][:] = np.zeros_like(image[:][0:crop_pixels_from_top-1][:])
        else:
            image = image[:][crop_pixels_from_top:-1][:]

        # plt.figure()
        # plt.imshow(image)
        # plt.show()  # display it

        # convert again to PIL
        image = Image.fromarray(image)

        return image


class CNN_Node(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(CNN_Node, self).__init__(node_name=node_name)
        self.vehicle = os.environ['VEHICLE_NAME']

        topic = '/' + self.vehicle + '/camera_node/image/compressed'
        print(topic)
        topicPub = '/'+self.vehicle+'/'+"LanePose"

        path_to_home = os.path.dirname(os.path.abspath(__file__))
        self.msg_wheels_cmd = WheelsCmdStamped()
        loc_d = path_to_home + "/CNN_1575282886.6939018_lr0.05_bs16_epo200_Model_final"
        loc_theta = path_to_home + "/CNN_1575756253.5257602_lr0.04_bs16_epo150_Model_finaltheta"
        rospy.set_param("".join(['/', self.vehicle, '/camera_node/exposure_mode']), 'auto')
        # change resolution camera
        #rospy.set_param('/' + self.vehicle + '/camera_node/res_w', 80)
        #rospy.set_param('/' + self.vehicle + '/camera_node/res_h', 60)

        print("Init Publisher")
        # self.LanePosePub = self.publisher(topicPub, LanePose, queue_size=10)
        # self.msgLanePose = LanePose()
        # self.veh_name = rospy.get_namespace().strip("/")
        topicName = "".join(['/', self.vehicle, '/wheels_driver_node/wheels_cmd'])
        print(topicName)
        self.pub_wheels_cmd = self.publisher(topicName, WheelsCmdStamped, queue_size=1)
        self.car_cmd_topic = "/" + self.vehicle + "/lane_controller_node/car_cmd"
        self.pub_car_cmd = self.publisher(self.car_cmd_topic, Twist2DStamped, queue_size=1)

        print("Initialized")
        self.model_d = torch.load(loc_d, map_location=torch.device('cpu'))
        self.model_th = torch.load(loc_theta, map_location=torch.device('cpu'))
        self.angleSpeedConvertsion = SteeringToWheelVelWrapper()
        # self.pidController = Controller(0.5,0.5,1,1,1,1)

        self.pidController = lane_controller()
        self.pidController.setParams()
        image_res = 64

        self.transforms = transforms.Compose([
            transforms.Resize(image_res),
            TransCropHorizon(0.5, set_black=False),
            # transforms.RandomCrop(, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
            transforms.Grayscale(num_output_channels=1),
            # TransConvCoord(),
            # ToCustomTensor(),
            transforms.ToTensor(),
            # transforms.Normalize(mean = [0.3,0.5,0.5],std = [0.21,0.5,0.5])
            ])

        self.model_d.eval()
        self.model_d.float()

        self.model_th.eval()
        self.model_th.float()


        self.time_image_rec = None
        self.time_image_rec_prev = None
        self.time_prop = False
        # numpy array with the command history for delay
        self.cmd_prev = [0, 0]
        self.state_prev = None
        self.onShutdown_trigger = False
        self.kalman_update_trigger = False
        self.trigger_car_cmd = False
        self.subscriber(topic, CompressedImage, self.compute_pose)

        rospy.on_shutdown(self.onShutdown)



        # Model class must be defined somewhere
        #model.load_state_dict(torch.load('/code/catkin_ws/src/pytorch_test/packages/modelNode/model/lane_navigation.h5'))

        #model = torch.load('/code/catkin_ws/src/pytorch_test/packages/modelNode/model/conv_net_model.ckpt')



    # def compute_action(self, observation):
        #if observation.shape != self.preprocessor.transposed_shape:
        #    observation = self.preprocessor.preprocess(observation)

        # return v, omega

        # arrayReturn = np.array([v, omega])
        # vel = self.angleSpeedConvertsion.action(arrayReturn)
        # return vel.astype(float)

    def compute_pose(self, frame):
        if self.onShutdown_trigger:
            return

        self.kalman_update_trigger = rospy.get_param("~kalman")
        self.time_prop = rospy.get_param("~time_prop")
        self.trigger_car_cmd = rospy.get_param("~car_cmd")

        cv_image = CvBridge().compressed_imgmsg_to_cv2(frame, desired_encoding="passthrough")
        im_pil = Image.fromarray(cv_image)
        img_t = self.transforms(im_pil)

        X = img_t.unsqueeze(1)

        self.time_image_rec = frame.header.stamp
        if self.time_image_rec_prev is None:
            self.time_image_rec_prev = self.time_image_rec

        state = [0,0]
        state[0] = self.model_d(X).detach().numpy()[0][0]
        state[1] = self.model_th(X).detach().numpy()[0][1]

        # print(state)
        
        if self.state_prev is None:
            self.state_prev = state

        if self.kalman_update_trigger:
            dt = self.time_image_rec - self.time_image_rec_prev
            state_est = self.time_propagation(self.state_prev, dt.to_sec())
            state = self.kalman_update(state, state_est)
            print('kalman',state)


        if self.time_prop:
            dt = rospy.get_rostime().to_sec() - self.time_image_rec.to_sec()
            state = self.time_propagation(state, dt)
            print('time_prop', state)

        # print(state)
        v, omega = self.pidController.updatePose(state[0], state[1])

        if self.trigger_car_cmd:
            car_cmd_msg = Twist2DStamped()
            car_cmd_msg.header.stamp = rospy.get_rostime()
            car_cmd_msg.v = v
            car_cmd_msg.omega = omega
            self.cmd_prev = [v, omega]
            self.pub_car_cmd.publish(car_cmd_msg)

        print((self.time_image_rec.to_sec() - self.time_image_rec_prev.to_sec()))
        self.time_image_rec_prev = self.time_image_rec
        self.state_prev = state

        # Put the wheel commands in a message and publish
        # Record the time the command was given to the wheels_driver
        # self.msg_wheels_cmd.header.stamp = rospy.get_rostime()
        arrayReturn = np.array([v, omega])
        vel = self.angleSpeedConvertsion.action(arrayReturn)
        pwm_left, pwm_right = vel.astype(float)

        self.msg_wheels_cmd.vel_left = pwm_left
        self.msg_wheels_cmd.vel_right = pwm_right
        self.pub_wheels_cmd.publish(self.msg_wheels_cmd)
        #self.msgLanePose.d = out.detach().numpy()[0][0]
        #self.msgLanePose.d_ref = 0
        #self.msgLanePose.phi = out.detach().numpy()[0][1]*3.14159
        #self.msgLanePose.phi_ref = 0
        #print(self.msgLanePose.d, self.msgLanePose.phi)

        #self.LanePosePub.publish(self.msgLanePose)


    def check_time_delay(self):
        r = rospy.Rate(20)
        if self.time_image_rec is None:
            return

        if (self.time_image_rec.to_sec() - rospy.get_rostime().to_sec()) > 0.5:
            car_cmd_msg = Twist2DStamped()
            car_cmd_msg.v = 0
            car_cmd_msg.omega = 0
            self.pub_car_cmd.publish(car_cmd_msg)
            self.cmd_prev = [car_cmd_msg.v, car_cmd_msg.omega]
        r.sleep()

    def time_propagation(self, state, dt):
        # print(dt)
        state[0] = state[0] + math.sin(state[1])*dt*self.cmd_prev[0]
        state[1] = state[1] + dt*self.cmd_prev[1]/360
        print(state)
        return state

    def kalman_update(self, state, state_est):
        y = np.array(state)
        x_prev = np.array(state_est)
        K = np.array([[0.7320508075688774,0],[0,0.8541019662496846]])
        state = x_prev + K.dot(y - x_prev)
        return [state[0], state[1]]


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

        self.log("Peace Out")

if __name__ == '__main__':
    # Initialize the node
    camera_node = CNN_Node(node_name='cnn_node')
    # camera_node.check_time_delay()
    # Keep it spinning to keep the node alive
    rospy.spin()