#!/usr/bin/env python

import cv2
import numpy as np
import os
import rospy
import yaml
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from cv_bridge import CvBridge
from PIL import Image
import sys
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped

class OurCNN(nn.Module):
    def __init__(self, as_gray=True, use_convcoord=True):
        super(OurCNN, self).__init__()

        # Handle dimensions
        if as_gray:
            self.input_channels = 1
        else:
            self.input_channels = 3

        if use_convcoord:
            self.input_channels += 2

        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64))

        self.drop_out_lin1 = nn.Dropout(0.4)
        self.lin1 = nn.Linear(1152 ,512)
        self.drop_out_lin2 = nn.Dropout(0.2)
        self.lin2 = nn.Linear(512, 256)
        self.drop_out_lin3 = nn.Dropout(0.1)
        self.lin3 = nn.Linear(256, 2)


    def forward(self, x):
        # print('dim of input')
        # print(x.size())
        # print(x[0][2][0])
        out = self.layer1(x)
        # print('dim after L1')
        # print(out.size())
        out = self.layer2(out)
        # print('dim after L2')
        # print(out.size())
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        # print('dim after reshape')
        # print(out.size())
        out = self.drop_out_lin1(out)
        out = F.relu(self.lin1(out))
        out = self.drop_out_lin2(out)
        out = F.relu(self.lin2(out))
        out = self.drop_out_lin3(out)
        out = F.tanh(self.lin3(out))

        return out


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
        self.vehicle = 'autobot04'
        #rospy.set_param('/' + self.vehicle + '/camera_node/res_w', 227) # 640
        #rospy.set_param('/' + self.vehicle + '/camera_node/res_h', 227) # 480
        topic = '/' + self.vehicle + '/imageSparse/compressed'
        #model = models.resnet50(pretrained=True)
        self.model = OurCNN()
        #path_to_home = os.environ['HOME']
        loc = "/media/elias/Samsung_T5/recordings_proj_lf_ml/savedModels/CNN_1574818091.5095592_lr0.1_bs16_epo120_Model"
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
        self.model.double()

        rospy.Subscriber(topic, CompressedImage, self.compute_pose, queue_size=1)

        print("Initialized")

        # Model class must be defined somewhere
        #model.load_state_dict(torch.load('/code/catkin_ws/src/pytorch_test/packages/modelNode/model/lane_navigation.h5'))




        #model = torch.load('/code/catkin_ws/src/pytorch_test/packages/modelNode/model/conv_net_model.ckpt')

    def compute_pose(self, frame):
        print('got frame')
        cv_image = CvBridge().compressed_imgmsg_to_cv2(frame, desired_encoding="passthrough")
        im_pil = Image.fromarray(cv_image)
        img_t = self.transform(im_pil)
        X = img_t.unsqueeze(1)
        out = model(X)
        print(out)

        return out


if __name__ == '__main__':
    # Initialize the node
    camera_node = CNN_Node()
    # Keep it spinning to keep the node alive
    rospy.spin()