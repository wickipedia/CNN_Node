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
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, LanePose
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


class CNN_Node():
    def __init__(self):

        # Initialize the DTROS parent class
        #self.veh_name = rospy.get_namespace().strip("/")
        # Use the kinematics calibration for the gain and trim
        self.vehicle = 'autobot29'
        #rospy.set_param('/' + self.vehicle + '/camera_node/res_w', 227) # 640
        #rospy.set_param('/' + self.vehicle + '/camera_node/res_h', 227) # 480
        topic = '/' + self.vehicle + '/imageSparse/compressed'
        topicPub = '/'+self.vehicle+'/'+"LanePose"
        #model = models.resnet50(pretrained=True)
        #self.model = OurCNN()
        path_to_home = os.path.dirname(os.path.abspath(__file__))
        print(path_to_home)

        loc = path_to_home + "/CNN_1575287035.6950421_lr0.05_bs16_epo400_Model_final"
        self.model = torch.load(loc, map_location=torch.device('cpu'))

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

        rospy.init_node("cnn_node", anonymous=False)
        rospy.Subscriber(topic, CompressedImage, self.compute_pose, queue_size=1)

        print("Init Publisher")
        self.LanePosePub = rospy.Publisher(topicPub, LanePose, queue_size=10)
        self.msgLanePose = LanePose()

        print("Initialized")

        # Model class must be defined somewhere
        #model.load_state_dict(torch.load('/code/catkin_ws/src/pytorch_test/packages/modelNode/model/lane_navigation.h5'))

        #model = torch.load('/code/catkin_ws/src/pytorch_test/packages/modelNode/model/conv_net_model.ckpt')

    def compute_pose(self, frame):
        cv_image = CvBridge().compressed_imgmsg_to_cv2(frame, desired_encoding="passthrough")
        im_pil = Image.fromarray(cv_image)
        img_t = self.transforms(im_pil)
        X = img_t.unsqueeze(1)
        out = self.model(X)
        
        self.msgLanePose.d = out.detach().numpy()[0][0]
        self.msgLanePose.d_ref = 0
        self.msgLanePose.phi = out.detach().numpy()[0][1]
        self.msgLanePose.phi_ref = 0 
        print(self.msgLanePose.d, self.msgLanePose.phi)
        self.LanePosePub.publish(self.msgLanePose)


if __name__ == '__main__':
    # Initialize the node
    camera_node = CNN_Node()
    # Keep it spinning to keep the node alive
    rospy.spin()