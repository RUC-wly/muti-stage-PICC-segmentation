import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np

nonlinearity = partial(F.relu, inplace=True)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32):
        super(UNet, self).__init__()

        self.conv1 = DoubleConv(1, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(base_ch*4, base_ch*8)


        self.conv_S1 = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=(1, 1), padding=1, bias=False),
            nn.Conv2d(1, 1, 3, stride=(1, 1), padding=1, bias=False)
        )
        self.conv_S2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, stride=(1, 1), padding=1, bias=False),
            nn.Conv2d(base_ch, base_ch, 3, stride=(1, 1), padding=1, bias=False)
        )
        self.conv_S3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=(1, 1), padding=1, bias=False),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=(1, 1), padding=1, bias=False)
        )

        self.pam = PAM_Module(base_ch * 8)

        self.up7 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv7 = DoubleConv(base_ch*8, base_ch*4)
        self.up8 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv8 = DoubleConv(base_ch*4, base_ch*2)
        self.up9 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv9 = nn.Conv2d(base_ch*2, out_ch, 1)

    def forward(self, x):
        # x1:原图 x2:mask
        x1, x2 = x.split([1, 1], dim=1)

        # ASW1
        x2 = self.conv_S1(x2)
        x2 = self.conv_S1(x2)
        x2_out = nn.Sigmoid()(x2)

        x1 = x1.mul(x2_out)

        # layer1
        c1 = self.conv1(x1)
        p1 = self.pool1(c1)

        # layer2
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        # layer3
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        # layer4
        c4 = self.conv4(p3)

        c4 = self.pam(c4)

        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        out = nn.Sigmoid()(c9)


        return out, x2_out


# from thop import profile
# from thop import clever_format
#
# input= torch.randn(1,2,512,512)
# model = UNet(2,1)
# flops,params = profile(model, inputs=(input,))
# print(flops,params)
#
# flops,params = clever_format([flops,params])
# print(flops,params)