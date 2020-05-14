from argparse import ArgumentParser
from dataclasses import dataclass
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ops
from networks.base_model import BaseModel
from networks.building_blocks import ConvBlock, ResidualBlock

ACTIVATIONS = {'relu': nn.ReLU, 'elu': nn.ELU}


@dataclass(eq=False)
class FeatNet(BaseModel):
    n_dims: int = 3
    spp_branches: list = None
    activation: str = 'relu'
    im_pad: int = None
    norm: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.n_dims:
            raise TypeError("__init__ missing 1 required argument: 'n_dims'")
        self.spp_branches = self.spp_branches or [32, 16, 8, 4]
        self.in_channels = 32

        conv_block = partial(ConvBlock, activation=ACTIVATIONS[self.activation])
        residual_block = partial(ResidualBlock, activation=ACTIVATIONS[self.activation])

        self.first_conv = nn.Sequential(OrderedDict([
            ('block1', conv_block(3, 32, 3, stride=2)),
            ('block2', conv_block(32, 32, 3)),
            ('block3', conv_block(32, 32, 3))
        ]))

        self.layer1 = self._make_layer(residual_block, 32, 3)
        self.layer2 = self._make_layer(residual_block, 64, 16, stride=2)
        self.layer3 = self._make_layer(residual_block, 128, 3)
        self.layer4 = self._make_layer(residual_block, 128, 3, dilation=2)

        self.spp_module = nn.ModuleList(self._make_pooling_branch(p, conv_block) for p in self.spp_branches)

        self.last_conv = nn.Sequential(OrderedDict([
            ('block', conv_block(320, 128, 3)),
            ('conv', nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False)),
        ]))

        self.feature_conv = nn.Conv2d(32, self.n_dims, kernel_size=1, padding=0, stride=1, bias=False)

    @staticmethod
    def add_parser_args(parser: ArgumentParser) -> None:
        parser.add_argument('--n-dims', default=3, type=int, help='Number of feature dimensions')
        parser.add_argument('--spp-branches', default=None, type=int, nargs='*', help='SPP pooling sizes')
        parser.add_argument('--im-pad', default=None, type=int, help='Input image padding size')
        parser.add_argument('--activation', default='relu', choices=list(ACTIVATIONS), help='Conv activation')
        parser.add_argument('--no-norm', action='store_false', dest='norm', help='Remove L2 normalization')

    def forward(self, images: torch.tensor) -> torch.tensor:
        x = F.pad(images, [self.im_pad]*4, mode='reflect') if self.im_pad else images

        # Encoder
        output = self.first_conv(x)
        skip = output
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        # SPP Outputs
        output_spp = (ops.upsample_like(branch(output_skip), output_raw) for branch in self.spp_module)

        # Decoder
        encoder_features = (output_raw, output_skip) + tuple(output_spp)
        output_features = torch.cat(encoder_features, 1)

        output_features = ops.upsample_like(output_features, skip)
        output_features = self.last_conv(output_features).add_(skip)

        output_features = ops.upsample_like(output_features, x)
        output_features = self.feature_conv(output_features)

        if self.im_pad:
            output_features = output_features[..., self.im_pad:-self.im_pad, self.im_pad:-self.im_pad]

        return F.normalize(output_features) if self.norm else output_features

    def _make_layer(self, block_type, out_channels, blocks, stride=1, padding=None, dilation=1):
        layers = OrderedDict([('block1', block_type(self.in_channels, out_channels, stride, padding, dilation))])
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers[f'block{i+1}'] = block_type(self.in_channels, out_channels, 1, padding, dilation)
        return nn.Sequential(layers)

    @staticmethod
    def _make_pooling_branch(kernel_size, conv_block):
        return nn.Sequential(OrderedDict([
            ('pool', nn.AvgPool2d(kernel_size, stride=kernel_size)),
            ('block', conv_block(128, 32, 1))
        ]))
