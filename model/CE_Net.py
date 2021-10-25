import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

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

class CE_Net_(nn.Module):
    def __init__(self, in_ch=1,num_classes=1, num_channels=3):
        super(CE_Net_, self).__init__()

        filters = [64, 128, 256, 512]

        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)

        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(inchannel=64, outchannel=64, bloch_num=3)
        self.layer2 = self._make_layer(inchannel=64, outchannel=128, bloch_num=4, stride=2)
        self.layer3 = self._make_layer(inchannel=128, outchannel=256, bloch_num=6, stride=2)
        self.layer4 = self._make_layer(inchannel=256, outchannel=512, bloch_num=3, stride=2)

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

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

        # Center
        e4 = self.dblock(e4)   # 1, 516, 16, 16
        e4 = self.spp(e4)      # 1, 516, 16, 16

        # Decoder
        d4 = self.decoder4(e4) + e3  # 1, 516, 16, 16--->1, 256, 32, 32
        d3 = self.decoder3(d4) + e2  # 1, 256, 32, 32--->1, 128, 64, 64
        d2 = self.decoder2(d3) + e1  # 1, 128, 64, 64--->1, 64, 128, 128
        d1 = self.decoder1(d2)       # 1, 64, 128, 128-->1, 64, 256, 256

        out = self.finaldeconv1(d1)  # 1, 64, 256, 256-->1, 32, 512, 512
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)   # 1, 32, 512, 512--->1, 1, 512, 512

        return F.sigmoid(out)

# from thop import profile
# from thop import clever_format
#
# input= torch.randn(1,2,512,512)
# model = CE_Net_(2,1)
# flops,params = profile(model, inputs=(input,))
# print(flops,params)
#
# flops,params = clever_format([flops,params])
# print(flops,params)
