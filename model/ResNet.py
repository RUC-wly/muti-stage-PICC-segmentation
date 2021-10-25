import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import numpy as np


nonlinearity = partial(F.relu, inplace=True)

class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1,shortcut=None):
        super(ResBlock, self).__init__()

        self.Res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch)

        )

        self.right = shortcut

    def forward(self, x):
        x1 = x if self.right is None else self.right(x)
        x2 = self.Res(x)
        out = x1 + x2
        return F.relu(out)

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

class Res_Net_(nn.Module):
    def __init__(self, in_ch=1, num_classes=1, base_ch=32, num_channels=3):
        super(Res_Net_, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)

        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(inchannel=base_ch, outchannel=base_ch, bloch_num=3)
        self.layer2 = self._make_layer(inchannel=base_ch, outchannel=base_ch*2, bloch_num=4, stride=2)
        self.layer3 = self._make_layer(inchannel=base_ch*2, outchannel=base_ch*4, bloch_num=6, stride=2)
        self.layer4 = self._make_layer(inchannel=base_ch*4, outchannel=base_ch*8, bloch_num=3, stride=2)

        self.up7 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv7 = DoubleConv(base_ch*8, base_ch*4)
        self.up8 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv8 = DoubleConv(base_ch*4, base_ch*2)
        self.up9 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv9 = DoubleConv(base_ch*2, base_ch)
        self.up10 = nn.ConvTranspose2d(base_ch, base_ch//2, 2, stride=2)
        self.up11 = nn.ConvTranspose2d(base_ch//2, base_ch//4, 2, stride=2)

        self.conv10 = nn.Conv2d(base_ch//4, num_classes, 1)

    def _make_layer(self, inchannel, outchannel, bloch_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, bloch_num):
            layers.append(ResBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)  # 1,1,512,512 ---->1,64,128,128
        e1 = self.layer1(x)    # 1,64,128,128 --->1,64,128,128
        e2 = self.layer2(e1)   # 1,64,128,128 --->1,128,64,64
        e3 = self.layer3(e2)   # 1,128,64,64----->1,256,32,32

        e4 = self.layer4(e3)   # 1,256,32,32---->1,512,16,16


        # Decoder
        up_7 = self.up7(e4)
        merge7 = torch.cat([up_7, e3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, e2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, e1], dim=1)
        c9 = self.conv9(merge9)
        up_10 = self.up10(c9)
        up_11 = self.up11(up_10)
        c10 = self.conv10(up_11)

        out = nn.Sigmoid()(c10)
        return out


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Res_Net_(1, 1).to(device)
# x = np.random.random((1, 1, 512, 512))
# x = torch.from_numpy(x).to(device).float()
# y = model(x)
# print(y)
#
# from thop import profile
# from thop import clever_format
#
# input= torch.randn(1,2,512,512)
# model = Res_Net_(2,1)
# flops,params = profile(model, inputs=(input,))
# print(flops,params)
#
# flops,params = clever_format([flops,params])
# print(flops,params)