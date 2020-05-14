from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils import model_zoo

from networks.base_model import BaseModel


def resnet_multiimage(num_layers, pretrained=False, num_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], 'Can only run with 18 or 50 layer resnet'
    blocks = [2]*4 if num_layers == 18 else [3, 4, 6, 3]
    block_type = models.resnet.BasicBlock if num_layers == 18 else models.resnet.Bottleneck
    model = ResNetMultiImage(block_type, blocks, num_images=num_images)

    if pretrained:
        state_dict = model_zoo.load_url(models.resnet.model_urls[f'resnet{num_layers}'])
        state_dict['conv1.weight'] = state_dict['conv1.weight'].repeat(1, num_images, 1, 1) / num_images
        model.load_state_dict(state_dict)

    return model


class ResNetMultiImage(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_images=1):
        super().__init__(block, layers)
        self.inplanes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_images*3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


@dataclass(eq=False)
class PoseNet(BaseModel):
    num_layers: int
    preres: bool

    def __post_init__(self):
        super().__post_init__()
        self.enc_ch = np.array([64, 64, 128, 256, 512])
        if self.num_layers > 34:
            self.enc_ch[1:] *= 4

        self.convs = OrderedDict([
            ('squeeze', nn.Conv2d(self.enc_ch[-1], 256, 1)),
            (('pose', 0), nn.Conv2d(256, 256, 3, 1, 1)),
            (('pose', 1), nn.Conv2d(256, 256, 3, 1, 1)),
            (('pose', 2), nn.Conv2d(256, 6*2, 1))
        ])

        self.relu = nn.ReLU()

        self.encoder = resnet_multiimage(self.num_layers, self.preres, 2)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--num-layers', default=18, choices=[18, 50], type=int, help='ResNet encoder layers')
        parser.add_argument('--preres', action='store_true', help='Pretrained ResNet encoder')

    def forward(self, source_images, target_images):
        # Encoder
        x = torch.cat((source_images, target_images), dim=1)
        x = (x - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        features = self.encoder.layer4(x)

        # Decoder
        out = self.relu(self.convs['squeeze'](features))
        for i in range(3):
            out = self.convs[('pose', i)](out)
            if i != 2:
                out = self.relu(out)

        out = 0.01 * out.mean(3).mean(2).view(-1, 2, 1, 6)
        axisangles, translation = out[..., :3], out[..., 3:]

        return axisangles, translation
