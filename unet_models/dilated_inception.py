import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop

# Dilated Inception Module: 3 dilated conv branches (l=1, 2, 3)
class DilatedInceptionModule(nn.Module):
    def __init__(self, in_ch, branch_out_ch):
        super(DilatedInceptionModule, self).__init__()
        self.relu = nn.ReLU()

        # 1-dilated branch
        self.l1_conv1 = nn.Conv2d(in_ch, branch_out_ch, kernel_size=1)
        self.l1_conv3 = nn.Conv2d(branch_out_ch, branch_out_ch, kernel_size=3, stride=1, padding=1,
                                  dilation=1)

        # 2-dilated branch
        self.l2_conv1 = nn.Conv2d(in_ch, branch_out_ch, kernel_size=1)
        self.l2_conv3 = nn.Conv2d(branch_out_ch, branch_out_ch, kernel_size=3, stride=1, padding=2,
                                  dilation=2)

        # 3-dilated branch
        self.l3_conv1 = nn.Conv2d(in_ch, branch_out_ch, kernel_size=1)
        self.l3_conv3 = nn.Conv2d(branch_out_ch, branch_out_ch, kernel_size=3, stride=1, padding=3,
                                  dilation=3)
        
        # Batchnorm
        self.batchnorm = nn.BatchNorm2d(3*branch_out_ch)

    def forward(self, x):
        l1 = self.relu(self.l1_conv3(self.relu(self.l1_conv1(x))))
        l2 = self.relu(self.l2_conv3(self.relu(self.l2_conv1(x))))
        l3 = self.relu(self.l3_conv3(self.relu(self.l3_conv1(x))))
        
        return self.batchnorm(torch.cat([l1, l2, l3], 1))


# Decode (Expanding) block for DIU-net that integrates skip connection concatenation
class DIDecodeBlock(nn.Module):
    def __init__(self, in_ch, branch_out_ch):
        super(DIDecodeBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch, kernel_size=2, stride=2)
        self.inception = DilatedInceptionModule(in_ch=in_ch*2, branch_out_ch=branch_out_ch)

    def forward(self, x, skip):
        out = self.upconv(x)
        cropped_skip = center_crop(skip, out.shape[-2:])

        return self.inception(torch.cat([out, cropped_skip], 1))


# DIU-net (Dilated Inception U-net) model 
class DIUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DIUnet, self).__init__()

        # Max pooling used in Unet descent
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encode layers on Unet descent
        self.encode1 = DilatedInceptionModule(in_ch, 32)
        self.encode2 = DilatedInceptionModule(3*32, 64)
        self.encode3 = DilatedInceptionModule(3*64, 128)
        self.encode4 = DilatedInceptionModule(3*128, 256)
        
        # Bridge layers at bottom of Unet
        self.bridge1 = DilatedInceptionModule(3*256, 512)
        self.bridge2 = DilatedInceptionModule(3*512, 256)

        # Decode layers on Unet ascent
        self.decode1 = DIDecodeBlock(3*256, 128)
        self.decode2 = DIDecodeBlock(3*128, 64)
        self.decode3 = DIDecodeBlock(3*64, 32)

        # Output up-Convolution and 1x1 Convolution
        self.out_upconv = nn.ConvTranspose2d(in_channels=3*32, out_channels=3*32, kernel_size=2, stride=2)
        self.outconv = nn.Conv2d(in_channels=6*32, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        skip1 = self.encode1(x)
        skip2 = self.encode2(self.maxpool(skip1))
        skip3 = self.encode3(self.maxpool(skip2))
        skip4 = self.encode4(self.maxpool(skip3))

        out = self.bridge1(self.maxpool(skip4))
        out = self.bridge2(out)

        out = self.decode1(out, skip4)
        out = self.decode2(out, skip3)
        out = self.decode3(out, skip2)

        out = self.out_upconv(out)
        cropped_skip1 = center_crop(skip1, out.shape[-2:])
        return self.outconv(torch.cat([out, cropped_skip1], 1))