from collections import OrderedDict

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, padding=None, dilation=1, *,
                 bias=False, batch_norm=True, momentum=0.1, activation=None, drop_rate=None):

        super().__init__()
        layers = OrderedDict()
        padding = padding or (dilation if dilation > 1 else k_size//2)

        if padding:
            layers['pad'] = nn.ReflectionPad2d(padding)
        layers['conv'] = nn.Conv2d(in_ch, out_ch, k_size, stride, dilation=dilation, bias=bias)

        if batch_norm:
            layers['bn'] = nn.BatchNorm2d(out_ch, momentum=momentum)
        if activation:
            layers['act'] = activation(inplace=True)
        if drop_rate:
            layers['drop'] = nn.Dropout2d(drop_rate, inplace=True)

        self.layers = nn.Sequential(layers)

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=None, dilation=1, activation=nn.ReLU):
        super().__init__()

        self.block1 = ConvBlock(in_ch, out_ch, 3, stride, padding, dilation, activation=activation)
        self.block2 = ConvBlock(out_ch, out_ch, 3, 1, padding, dilation)
        self.downsample = None if stride == 1 and in_ch == out_ch else ConvBlock(in_ch, out_ch, 1, stride)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out += self.downsample(x) if self.downsample else x
        return out


class SimpleConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        return self.nonlin(self.conv(x))


class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""
    def __init__(self, in_channels, out_channels, use_refl=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1) if use_refl else nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        return self.conv(self.pad(x))
