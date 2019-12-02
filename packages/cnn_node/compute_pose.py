#!/usr/bin/env python

import cv2
import numpy as np
import os
import rospy
import yaml
from torchvision import models, transforms
import torch

from cv_bridge import CvBridge
from PIL import Image
import sys
from duckietown import DTROS
from sensor_msgs.msg import CompressedImage, Temperature
from duckietown_msgs.msg import WheelsCmdStamped, LanePose
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
        self.vehicle = 'queenmary2'

        topic = '/' + self.vehicle + '/camera_node/image/compressed'
        print(topic)
        topicPub = '/'+self.vehicle+'/'+"LanePose"

        path_to_home = os.path.dirname(os.path.abspath(__file__))
        self.msg_wheels_cmd = WheelsCmdStamped()
        loc = path_to_home + "/CNN_1574936479.7700994_lr0.05_bs16_epo100_Model_final"
        rospy.set_param("".join(['/', self.vehicle, '/camera_node/exposure_mode']), 'off')
        # change resolution camera
        #rospy.set_param('/' + self.vehicle + '/camera_node/res_w', 80)
        #rospy.set_param('/' + self.vehicle + '/camera_node/res_h', 60)

        self.subscriber(topic, CompressedImage, self.compute_pose)

        print("Init Publisher")
        # self.LanePosePub = self.publisher(topicPub, LanePose, queue_size=10)
        # self.msgLanePose = LanePose()
        # self.veh_name = rospy.get_namespace().strip("/")
        topicName = "".join(['/', self.vehicle, '/wheels_driver_node/wheels_cmd'])
        print(topicName)
        self.pub_wheels_cmd = self.publisher(topicName, WheelsCmdStamped, queue_size=1)

        # self.TempPub = self.publisher("/queenmary2/temp", Temperature, queue_size=1)
        # self.msgTemp = Temperature()

        print("Initialized")
        self.model = torch.load(loc, map_location=torch.device('cpu'))
        self.angleSpeedConvertsion = SteeringToWheelVelWrapper()
        # self.pidController = Controller(0.5,0.5,1,1,1,1)

        self.pidController = lane_controller()
        self.pidController.setGains()
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

        self.model.eval()
        self.model.float()


        # Model class must be defined somewhere
        #model.load_state_dict(torch.load('/code/catkin_ws/src/pytorch_test/packages/modelNode/model/lane_navigation.h5'))

        #model = torch.load('/code/catkin_ws/src/pytorch_test/packages/modelNode/model/conv_net_model.ckpt')



    def compute_action(self, observation):
        #if observation.shape != self.preprocessor.transposed_shape:
        #    observation = self.preprocessor.preprocess(observation)
        action = self.model(observation)
        print(action)
        action = action.detach().numpy()[0]
        v, omega = self.pidController.updatePose(action[0], action[1])


        arrayReturn = np.array([v, omega])
        vel = self.angleSpeedConvertsion.action(arrayReturn)
        return vel.astype(float)

    def compute_pose(self, frame):
        cv_image = CvBridge().compressed_imgmsg_to_cv2(frame, desired_encoding="passthrough")
        im_pil = Image.fromarray(cv_image)
        img_t = self.transforms(im_pil)
        X = img_t.unsqueeze(1)
        pwm_left, pwm_right = self.compute_action(X)
        # Put the wheel commands in a message and publish
        # Record the time the command was given to the wheels_driver
        self.msg_wheels_cmd.header.stamp = rospy.get_rostime()
        self.msg_wheels_cmd.vel_left = pwm_left
        self.msg_wheels_cmd.vel_right = pwm_right
        self.pub_wheels_cmd.publish(self.msg_wheels_cmd)
        #self.msgLanePose.d = out.detach().numpy()[0][0]
        #self.msgLanePose.d_ref = 0
        #self.msgLanePose.phi = out.detach().numpy()[0][1]*3.14159
        #self.msgLanePose.phi_ref = 0
        #print(self.msgLanePose.d, self.msgLanePose.phi)
        #self.msgTemp.variance = 0

        #self.LanePosePub.publish(self.msgLanePose)
        #self.TempPub.publish(self.msgTemp)

    def onShutdown(self):
        """Shutdown procedure.

        Publishes a zero velocity command at shutdown."""

        super(CNN_Node, self).onShutdown()
        # MAKE SURE THAT THE LAST WHEEL COMMAND YOU PUBLISH IS ZERO,
        # OTHERWISE YOUR DUCKIEBOT WILL CONTINUE MOVING AFTER
        # THE NODE IS STOPPED

        # PUT YOUR CODE HERE
        #self.driver.setWheelsSpeed(left=0.0, right=0.0)

        # Put the wheel commands in a message and publish
        # Record the time the command was given to the wheels_driver
        self.msg_wheels_cmd = WheelsCmdStamped()

        self.msg_wheels_cmd.header.stamp = rospy.get_rostime()
        self.msg_wheels_cmd.vel_left = 0
        self.msg_wheels_cmd.vel_right = 0
        rospy.sleep(1)
        self.pub_wheels_cmd.publish(self.msg_wheels_cmd)
        print('published')
        self.log("Wheel commands published")

if __name__ == '__main__':
    # Initialize the node
    camera_node = CNN_Node(node_name='cnn_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
