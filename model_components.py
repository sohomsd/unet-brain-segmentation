import torch
import torch.nn as nn

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