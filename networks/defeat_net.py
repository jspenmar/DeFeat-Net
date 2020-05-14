from dataclasses import dataclass

import torch

from networks.base_model import BaseModel
from networks.depth_net import DepthNet
from networks.pose_net import PoseNet
from networks.feat_net import FeatNet


@dataclass(eq=False)
class DeFeatNet(BaseModel):
    num_layers: int
    preres: bool
    scales: list = range(4)
    use_skips: bool = True

    n_dims: int = 3
    spp_branches: list = None
    activation: str = 'relu'
    im_pad: int = None
    norm: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.depth_net = DepthNet(self.num_layers, self.preres, self.scales, self.use_skips)
        self.pose_net = PoseNet(self.num_layers, self.preres)
        self.feat_net = FeatNet(self.n_dims, self.spp_branches, self.activation, self.im_pad, self.norm)

    @staticmethod
    def add_parser_args(parser):
        DepthNet.add_parser_args(parser)
        FeatNet.add_parser_args(parser)

    def forward(self, target_frames, support_frames, support_idxs):
        """
        :param target_frames: Frame we want to predict depth for
        :param support_frames: Previous and/or following frames
        :param support_idxs: Index wrt original frames
        :return:
        """
        # Process target frame
        target_disps = self.depth_net(target_frames)
        target_features = self.feat_net(target_frames)

        # Process support frames
        support_features = self.feat_net(torch.cat(support_frames, dim=0)).chunk(len(support_frames), dim=0)
        poses = []
        for idx, sf in zip(support_idxs, support_frames):
            # Predict forward transform, i.e. (-1 -> 0), (0 -> 1)
            inp = (sf, target_frames) if idx > 0 else (target_frames, sf)
            poses.append(self.pose_net(*inp))

        return target_disps, target_features, support_features, poses
