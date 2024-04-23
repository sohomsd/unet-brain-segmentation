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
# Inception Module: 3 conv and 1 pool
class InceptionModule(nn.Module):
    def __init__(self, in_channels, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pool_proj):
        super(InceptionModule, self).__init__()
        # 1x1 conv branch
        self.conv1x1 = nn.Conv2d(in_channels, f_1x1, kernel_size=1)

        # 3x3 conv branch
        self.conv3x3_reduce = nn.Conv2d(in_channels, f_3x3_r, kernel_size=1)
        self.conv3x3 = nn.Conv2d(f_3x3_r, f_3x3, kernel_size=3, padding=1)

        # 5x5 conv branch
        self.conv5x5_reduce = nn.Conv2d(in_channels, f_5x5_r, kernel_size=1)
        self.conv5x5 = nn.Conv2d(f_5x5_r, f_5x5, kernel_size=5, padding=2)

        # Pooling branch
        self.pool_proj = nn.Conv2d(in_channels, f_pool_proj, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        branch1x1 = self.conv1x1(x)

        branch3x3 = self.conv3x3_reduce(x)
        branch3x3 = self.conv3x3(branch3x3)

        branch5x5 = self.conv5x5_reduce(x)
        branch5x5 = self.conv5x5(branch5x5)

        branch_pool = self.pool(x)
        branch_pool = self.pool_proj(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)