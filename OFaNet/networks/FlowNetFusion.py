import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .submodules import *
'Parameter count = 581,226'

class padding(nn.Module):

    def __init__(self):
        super(padding, self).__init__()
        self.wpad = nn.ReplicationPad2d((0, -1, 0, 0))
        self.hpad = nn.ReplicationPad2d((0, 0, 0, -1))

    def forward(self, input, targetsize):
        if input.size()[2] != targetsize[2]:
            input = self.hpad(input)
        if input.size()[3] != targetsize[3]:
            input = self.wpad(input)
        return input

class FlowNetFusion(nn.Module):
    def __init__(self,args, batchNorm=True):
        super(FlowNetFusion,self).__init__()

        self.batchNorm = batchNorm
        self.pad = padding()
        self.conv0   = conv(self.batchNorm,  11,   64)
        self.conv1   = conv(self.batchNorm,  64,   64, stride=2)
        self.conv1_1 = conv(self.batchNorm,  64,   128)
        self.conv2   = conv(self.batchNorm,  128,  128, stride=2)
        self.conv2_1 = conv(self.batchNorm,  128,  128)

        self.deconv1 = deconv(128,32)
        self.deconv0 = deconv(162,16)

        self.inter_conv1 = i_conv(self.batchNorm,  162,   32)
        self.inter_conv0 = i_conv(self.batchNorm,  82,   16)

        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)

        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        flow2       = self.predict_flow2(out_conv2)
        flow2_up    = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)

        flow2_up = self.pad(flow2_up, out_conv1.size())
        out_deconv1 = self.pad(out_deconv1, out_conv1.size())
        concat1 = torch.cat((out_conv1,out_deconv1,flow2_up),1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1       = self.predict_flow1(out_interconv1)
        flow1_up    = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)

        flow1_up = self.pad(flow1_up, out_conv0.size())
        out_deconv0 = self.pad(out_deconv0, out_conv0.size())
        concat0 = torch.cat((out_conv0,out_deconv0,flow1_up),1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0       = self.predict_flow0(out_interconv0)

        return flow0

