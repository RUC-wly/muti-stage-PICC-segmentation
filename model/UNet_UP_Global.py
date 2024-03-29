
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


class PSPNet_PyramidPool(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPNet_PyramidPool, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


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

        self.ppm = PSPNet_PyramidPool(base_ch*8, 256, (1, 2, 3, 6))

        self.up7 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.conv7 = DoubleConv(base_ch*8, base_ch*4)
        self.up8 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.conv8 = DoubleConv(base_ch*4, base_ch*2)
        self.up9 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.conv9 = DoubleConv(base_ch*2, base_ch)
        self.conv10 = nn.Sequential(
            nn.Upsample(1024),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            # nn.Conv2d(base_ch, base_ch, 1),
            # nn.BatchNorm2d(base_ch),
            # nn.ReLU(inplace=True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, 1)
        )


    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        c4 = self.ppm(c4)

        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)
        c11 = self.conv11(c10)
        out = nn.Sigmoid()(c11)

        return out

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNet(1, 1).to(device)
# x = np.random.random((1, 1, 512, 512))
# x = torch.from_numpy(x).to(device).float()
# y = model(x)
# print(y.shape)
