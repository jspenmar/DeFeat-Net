from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from networks.base_model import BaseModel
from networks.building_blocks import SimpleConvBlock, Conv3x3
from utils import ops


@dataclass(eq=False)
class DepthNet(BaseModel):
    num_layers: int
    preres: bool = True
    scales: list = range(4)
    use_skips: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Number of layers
        self.enc_ch = np.array([64, 64, 128, 256, 512])
        self.dec_ch = np.array([16, 32, 64, 128, 256])
        if self.num_layers > 34:
            self.enc_ch[1:] *= 4

        # Build multiscale decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            in_ch = self.enc_ch[-1] if i == 4 else self.dec_ch[i + 1]
            self.convs[('upconv', i, 0)] = SimpleConvBlock(in_ch, self.dec_ch[i])

            in_ch = self.dec_ch[i] + (self.enc_ch[i - 1] if self.use_skips and i > 0 else 0)
            self.convs[('upconv', i, 1)] = SimpleConvBlock(in_ch, self.dec_ch[i])

        # Upsample convs for each scale
        for s in self.scales:
            self.convs[('dispconv', s)] = Conv3x3(self.dec_ch[s], 1)

        self.encoder = getattr(models, f'resnet{self.num_layers}')(self.preres)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--num-layers', default=18, choices=[18, 34, 50, 101, 152], type=int,
                            help='ResNet encoder layers')
        parser.add_argument('--preres', action='store_true', help='Pretrained ResNet encoder')
        parser.add_argument('--scales', default=range(4), nargs='*', type=int)
        parser.add_argument('--no-skips', action='store_false', dest='use_skips', help='Disable skip connections')

    def forward(self, images):
        features, outputs = [], {}

        # Encoder
        x = (images - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        # Decoder
        x = features[-1]
        for i in range(4, -1, -1):
            x = ops.upsample(self.convs[('upconv', i, 0)](x), factor=2, bilinear=False)

            if self.use_skips and i > 0:
                x = torch.cat((x, features[i-1]), dim=1)

            x = self.convs[('upconv', i, 1)](x)

            if i in self.scales:
                outputs[('disp', i)] = torch.sigmoid(self.convs[('dispconv', i)](x))

        return outputs
