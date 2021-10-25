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


class DoubleUPConv(nn.Module):
    def __init__(self, in_ch):
        super(DoubleUPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3,
                              padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x



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
            nn.Conv2d(1, 1, 3, stride=(1, 1), padding=1, bias=False),
            nn.Conv2d(1, 1, 3, stride=(1, 1), padding=1, bias=False)
        )
        self.conv_S3 = nn.Sequential(
            nn.Conv2d(1,1, 3, stride=(1, 1), padding=1, bias=False),
            nn.Conv2d(1, 1, 3, stride=(1, 1), padding=1, bias=False)
        )

        self.pam = PAM_Module(base_ch * 8)

        self.up4 = DUC(base_ch * 8, base_ch * 16)
        self.decoder4 = DoubleUPConv(base_ch * 8)

        self.up3 = DUC(base_ch * 8, base_ch * 8)
        self.decoder3 = DoubleUPConv(base_ch * 4)

        self.up2 = DUC(base_ch * 4, base_ch * 4)
        self.decoder2 = DoubleUPConv(base_ch * 2)

        self.conv9 = nn.Conv2d(base_ch * 2, out_ch, 1)

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

        # ASW2
        # x2_1 = self.conv1(x2)
        x2_1 = self.pool1(x2)
        x2_1 = self.conv_S2(x2_1)
        x2_1 = self.conv_S2(x2_1)
        x2_1_out = nn.Sigmoid()(x2_1)

        p1 = p1.mul(x2_1_out)
        # p1 = p1 + p1_

        # layer2
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        # layer3
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        # layer4
        c4 = self.conv4(p3)

        c5 = self.pam(c4)

        up_4 = self.up4(c5)
        up_4_ = torch.cat([up_4, c3], dim=1)
        up_4_ = self.decoder4(up_4_)

        up_3 = self.up3(up_4_)
        up_3_ = torch.cat([up_3, c2], dim=1)
        up_3_ = self.decoder3(up_3_)

        up_2 = self.up2(up_3_)
        up_2_ = torch.cat([up_2, c1], dim=1)
        up_2_ = self.decoder2(up_2_)

        out = self.conv9(up_2_)

        out = nn.Sigmoid()(out)

        return out, x2_out, x2_1_out


# import os
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# model = UNet(2, 1).to(device)
# x = np.random.random((1, 2, 512, 512))
# x = torch.from_numpy(x).to(device).float()
# y, y1 = model(x)
# print(y)

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