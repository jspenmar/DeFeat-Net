import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.ssim import SSIM
from losses.smoothness_loss import SmoothnessLoss
from losses.reprojection import Depth2Grid
from losses.hierarchical_context_aggregation import HierarchicalContextAggregationLoss as HCALoss
from utils import ops
from utils.transforms import disp2depth, params2tform


class DeFeatLoss(nn.Module):
    def __init__(self, n_corrs=15_000, hca_dict=None, use_features=True, use_scales=True):
        super().__init__()
        self.n_corrs = n_corrs
        self.use_features = use_features
        self.use_scales = use_scales
        self.use_automask = True

        self.smooth_weight = 1e-3

        self.min_depth = 0.1
        self.max_depth = 100

        self.ssim = SSIM(ssim_weight=0.85)
        self.smoothness = SmoothnessLoss()
        self.d2g = Depth2Grid()
        self.hca_loss = HCALoss(**(hca_dict or {}))

    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--n-corrs', default=15_000, type=int)
        parser.add_argument('--no-features', action='store_false', dest='use_features')
        parser.add_argument('--no-scales', action='store_false', dest='use_scales')
        HCALoss.add_parser_args(parser)

    def forward(self, pred_disp, pred_feats, support_feats, pred_poses, frames, support_frames, gt_depths, k, idxs):
        total_loss = torch.tensor(0.0, device=frames.device)
        n_scales = len(pred_disp) if self.use_scales else 1
        size = pred_disp[('disp', 0)].shape[-2:]

        for scale in range(n_scales):
            factor = 2 ** scale

            # Downsample frames (smoothness) and upsample disparity
            scale_disp = pred_disp[('disp', scale)]
            scale_frames = ops.downsample(frames, factor=factor)
            disp = scale_disp if scale == 0 else F.interpolate(scale_disp, size, mode='bilinear', align_corners=False)

            # Generate warping functions from predictions
            warp_fns, pix_coords = self.get_warp_functions(disp, pred_poses, k, idxs)

            # Photometric based errors
            l_ssim, _, automask = self.compute_ssim(frames, support_frames, warp_fns)

            # Feature based errors
            l_ssim_feat = torch.tensor(0, device=l_ssim.device)
            l_feat = torch.tensor(0, device=l_ssim.device)
            if self.use_features and scale == 0:
                l_ssim_feat = self.compute_ssim(pred_feats, support_feats, warp_fns, index=automask)[0]

                correspondences = self.get_correspondences(pix_coords, automask, self.n_corrs)
                l_feat = [self.hca_loss(torch.cat((pred_feats, sf), dim=2), corr)
                          for sf, corr in zip(support_feats, correspondences)]
                l_feat = sum(l_feat)

            # Enforce smoothness unless there's an edge in the original images
            norm_disp = scale_disp / (scale_disp.mean(2, True).mean(3, True) + 1e-7)
            l_smooth = self.smoothness(norm_disp, scale_frames) / factor

            # Final loss
            loss = l_ssim + l_ssim_feat + l_feat + l_smooth*self.smooth_weight
            total_loss += loss

        total_loss /= n_scales
        return total_loss

    def get_warp_functions(self, disparity, poses, k, idxs):
        depth = disp2depth(disparity,  min_depth=self.min_depth, max_depth=self.max_depth)[-1]
        inv_depth = ops.channel_mean(1/depth, keepdim=True).squeeze(1)

        transforms = [params2tform(aa[:, 0], t[:, 0]*inv_depth, invert=idx < 0) for (aa, t), idx in zip(poses, idxs)]
        warp_fns, pix_coords = list(zip(*[self.d2g(depth, tform, k) for tform in transforms]))
        return warp_fns, pix_coords

    @staticmethod
    def get_correspondences(pix_coords, automask, max_n=None):
        # Round coordinates
        pix_coords = [coords.round().short() for coords in pix_coords]

        # Filter based on automasking
        n_images = len(pix_coords)
        corrmask = ops.make_one_hot((automask-n_images).clamp_min(0), n=n_images)
        corrmasks = [m.squeeze(1).bool() for m in corrmask.split(split_size=1, dim=1)]

        # Filter based on in-bound
        h, w = automask.shape[-2:]
        for i, coords in enumerate(pix_coords):
            in_x = (coords[..., 0] > 0) & (coords[..., 0] < w)
            in_y = (coords[..., 1] > 0) & (coords[..., 1] < h)
            corrmasks[i] &= in_x & in_y

        # Identity coordinates
        yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w))
        id_coords = torch.stack((xv, yv), dim=-1).short().to(corrmask.device)

        # Reshape all as column vectors
        pix_coords = [coords.view(-1, h*w, 2) for coords in pix_coords]
        id_coords = id_coords.view(-1, 2)
        corrmasks = [mask.view(-1, h*w) for mask in corrmasks]

        # Filter and add original coordinates (frame 0) -- (x1, y1, x2, y2)
        correspondences = [[torch.cat((id_coords[m], c[m]), dim=-1) for m, c in zip(mask, coords)]
                           for mask, coords in zip(corrmasks, pix_coords)]

        # Randomize selected correspondences
        sizes = [[c.shape[0] for c in corr] for corr in correspondences]
        n_corrs = [min((size+[max_n] if max_n else size)) for size in sizes]
        idxs = [[torch.randperm(s)[:n] for s in size] for n, size in zip(n_corrs, sizes)]

        correspondences = [torch.stack([c[i] for i, c in zip(idx, corrs)], dim=0)
                           for idx, corrs in zip(idxs, correspondences)]

        return correspondences

    def compute_ssim(self, target, sources, warp_fns, index=None):
        # Warp source frames
        warped_sources = [fn(source) for fn, source in zip(warp_fns, sources)]
        ssim = torch.cat([self.ssim(ws, target) for ws in warped_sources], dim=1)

        # Compute SSIM wrt original support frames (automask static pixels)
        if self.use_automask:
            identity_ssim = torch.cat([self.ssim(source, target) for source in sources], dim=1)
            identity_ssim += torch.randn_like(identity_ssim) * 1e-5  # Break ties
            ssim = torch.cat((identity_ssim, ssim), dim=1)

        index = ssim.argmin(dim=1, keepdim=True) if index is None else index
        l_ssim = ssim.gather(dim=1, index=index).mean()
        return l_ssim, warped_sources, index
