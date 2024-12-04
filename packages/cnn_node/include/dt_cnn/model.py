#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Model to compute distance
class model_dist(nn.Module):
    def __init__(self, as_gray=True, use_convcoord=True):
        super(model_dist, self).__init__()

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
        self.lin1 = nn.Linear(576 ,512)
        self.drop_out_lin2 = nn.Dropout(0.2)
        self.lin2 = nn.Linear(512, 256)
        self.drop_out_lin3 = nn.Dropout(0.1)
        self.lin3 = nn.Linear(256, 2)

        image_res = 64

        self.transform = transforms.Compose([
            transforms.Resize(image_res),
            TransCropHorizon(0.62, set_black=False),
            transforms.Grayscale(num_output_channels=1),
            ToCustomTensor(False),
            ])


    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out_lin1(out)
        out = F.relu(self.lin1(out))
        out = self.drop_out_lin2(out)
        out = F.relu(self.lin2(out))
        out = self.drop_out_lin3(out)
        out = F.tanh(self.lin3(out))

        return out

# Model to compute heading
class model_angle(nn.Module):
    def __init__(self, as_gray=True, use_convcoord=True):
        super(model_angle, self).__init__()

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

        image_res = 64

        self.transform = transforms.Compose([
            transforms.Resize(image_res),
            TransCropHorizon(0.5, set_black=False),
            transforms.Grayscale(num_output_channels=1),
            ToCustomTensor(False),
            ])


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out_lin1(out)
        out = F.relu(self.lin1(out))
        out = self.drop_out_lin2(out)
        out = F.relu(self.lin2(out))
        out = self.drop_out_lin3(out)
        out = F.tanh(self.lin3(out))

        return out


class ToCustomTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, use_convcoord):
        self.use_convcoord = use_convcoord


    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        # handle numpy arrays
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            if self.use_convcoord:
                pic[:,:,0] = pic[:,:,0]/255
            else:
                pic = pic/255

            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

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

        # convert again to PIL
        image = Image.fromarray(image)

        return image
