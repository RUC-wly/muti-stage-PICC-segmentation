
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


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32):
        super(UNet, self).__init__()

        self.conv1 = DoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(base_ch*4, base_ch*8)

        self.up7 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv7 = DoubleConv(base_ch*8, base_ch*4)
        self.up8 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv8 = DoubleConv(base_ch*4, base_ch*2)
        self.up9 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv9 = nn.Conv2d(base_ch*2, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

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

        return out

# #
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNet(2, 1).to(device)
# x = np.random.random((1, 2, 512, 512))
# x = torch.from_numpy(x).to(device).float()
# y = model(x)
# print(y)
#

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from torchstat import stat
# model = UNet(2, 1)
# stat(model, (2, 512,512))
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