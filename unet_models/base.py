import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop

# Convolution Block with 2 Convolution and ReLU Activations
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                               kernel_size=kernel_size, stride=stride)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, 
                               kernel_size=kernel_size, stride=stride)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        return self.relu2(self.conv2(out))

# Decode (Expanding) block that integrates skip connection concatenation
class DecodeBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecodeBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2)
        self.convblock = ConvBlock(in_ch=in_ch, out_ch=out_ch, kernel_size=3)

    def forward(self, x, skip):
        out = self.upconv(x)
        cropped_skip = center_crop(skip, output_size=out.shape[-2:])

        return self.convblock(torch.cat([out, cropped_skip], 1))


# Original U-net model
class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        
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
        self.decode1 = DecodeBlock(in_ch=1024, out_ch=512)
        self.decode2 = DecodeBlock(in_ch=512, out_ch=256)
        self.decode3 = DecodeBlock(in_ch=256, out_ch=128)
        self.decode4 = DecodeBlock(in_ch=128, out_ch=64)

        # Output 1x1 Convolution
        self.outconv = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        skip1 = self.encode1(x)
        skip2 = self.encode2(self.maxpool(skip1))
        skip3 = self.encode3(self.maxpool(skip2))
        skip4 = self.encode4(self.maxpool(skip3))

        out = self.bridge(self.maxpool(skip4))

        out = self.decode1(out, skip4)
        out = self.decode2(out, skip3)
        out = self.decode3(out, skip2)
        out = self.decode4(out, skip1)

        return self.outconv(out)