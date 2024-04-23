import torch
import torch.nn as nn
from model_components import *


class BaseUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BaseUnet, self).__init__()
        
        # Max pooling used in Unet descent
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encode layers on Unet descent
        self.encode1 = ConvBlock(in_ch=in_ch, out_ch=64, kernel_size=3)
        self.encode2 = ConvBlock(in_ch=64, out_ch=128, kernel_size=3)
        self.encode3 = ConvBlock(in_ch=128, out_ch=256, kernel_size=3)
        self.encode4 = ConvBlock(in_ch=256, out_ch=512, kernel_size=3)

        # Bridge Layer
        self.bridge = ConvBlock(in_ch=512, out_ch=1024, kernel_size=3)

        # Decode layers on Unet ascent
        self.upconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.decode1 = ConvBlock(in_ch=1024, out_ch=512, kernel_size=3)

        self.upconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.decode2 = ConvBlock(in_ch=512, out_ch=256, kernel_size=3)

        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.decode3 = ConvBlock(in_ch=256, out_ch=128, kernel_size=3)

        self.upconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.decode4 = ConvBlock(in_ch=128, out_ch=64, kernel_size=3)

        # Output 1x1 Convolution
        self.outconv = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        skip1 = self.maxpool(self.encode1(x))
        skip2 = self.maxpool(self.encode2(skip1))
        skip3 = self.maxpool(self.encode3(skip2))
        skip4 = self.maxpool(self.encode4(skip3))

        out = self.bridge(skip4)

        out = self.decode1(torch.cat([out, skip4]))
        out = self.decode2(torch.cat([out, skip3]))
        out = self.decode3(torch.cat([out, skip2]))
        out = self.decode4(torch.cat([out, skip1]))

        return self.outconv(out)