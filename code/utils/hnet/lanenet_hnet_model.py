#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-21 上午11:38
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_hnet_model.py
# @IDE: PyCharm Community Edition
"""
LaneNet中的HNet模型
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils.hnet.lanenet_hnet_loss import hnet_loss, hnet_transformation

import numpy as np
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            bias=bias)
        # Initialize batch normalization
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply
        self.out_activation = activation()

    def forward(self, x):
        x = self.conv2d(x)


        # Apply batch normalization
        out = self.batch_norm(x)

        return self.out_activation(out)

class LaneNetHNet(nn.Module):
    """
    实现lanenet中的hnet模型
    """
    def __init__(self, input_size=(3,64,128)):
        """
        :param phase:
        """
        super().__init__()


        self.features = nn.Sequential(
            ConvBlock(3, 16),
        ConvBlock(16, 16),
        nn.MaxPool2d(2, stride=2),
        ConvBlock(16, 32),
        ConvBlock(32, 32),
        nn.MaxPool2d(2, stride=2),
        ConvBlock(32, 64),
        ConvBlock(64, 64),
        nn.MaxPool2d(2, stride=2)
        )

        # self.flat_fts = self.get_flat_fts(input_size, self.features)
        # print(self.flat_fts)
        self.backend = nn.Sequential(
            nn.Linear(3072, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 6),
        )

        self.pre_H = torch.tensor(
            [-2.04835137e-01, -3.09995252e+00, 7.99098762e+01, -2.94687413e+00, 7.06836681e+01, -4.67392998e-02],
            dtype=torch.float32, requires_grad=False).cuda()

    def get_flat_fts(self, in_size, fts):
        f = fts(Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))


    def forward(self, x, gt_pts):
        # print(x.size())
        x = self.features(x)
        x = x.view(-1, 3072)
        x = self.backend(x)

        # print(self.flat_fts)
        res = {
            "transformation_coefficient": x,
            "pre_loss": torch.mean(torch.norm((x - self.pre_H) / self.pre_H)),
            "loss": hnet_loss(gt_pts=gt_pts, transformation_coefficient=x)
        }
        return res

    def generate_matrix(self, x, gt_label_pts):
        # print(x.size())
        x = self.features(x)
        x = x.view(-1, 3072)
        x = self.backend(x)

        x = x[0]

        return hnet_transformation(gt_label_pts, x)


if __name__ == '__main__':
    model = LaneNetHNet()
    model.eval()
    x = model.forward(torch.ones(1, 3, 256, 512))
    print(x)
