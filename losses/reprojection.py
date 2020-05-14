from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class Depth2Grid(nn.Module):
    def __init__(self):
        super().__init__()
        self.backproject_depth = BackprojectDepth()
        self.project_3d = Project3D()

    def forward(self, depth, tform, k):
        k, inv_k = k

        cam_points = self.backproject_depth(depth, inv_k)
        pix_coords, raw_pix_coords = self.project_3d(cam_points, k, tform, depth.shape[-2:])

        warp_fn = partial(F.grid_sample, grid=pix_coords, padding_mode='border')
        return warp_fn, raw_pix_coords


class Project3D(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, points, k, tform, size):
        h, w = size
        p = torch.matmul(k, tform)[:, :3, :]
        cam_points = torch.matmul(p, points)

        raw_pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        raw_pix_coords = raw_pix_coords.view(-1, 2, h, w).permute(0, 2, 3, 1)

        pix_coords = raw_pix_coords.clone()
        pix_coords[..., 0] /= w - 1
        pix_coords[..., 1] /= h - 1
        pix_coords = (pix_coords - 0.5) * 2

        return pix_coords, raw_pix_coords


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud"""
    def __init__(self):
        super().__init__()

    def forward(self, depth, inv_k):
        b, _, h, w = depth.shape

        ys, xs = torch.meshgrid([torch.arange(h), torch.arange(w)])
        id_coords = torch.stack((xs, ys), dim=0).float().to(depth.device)

        ones = torch.ones(b, 1, h*w, device=depth.device)
        pix_coords = torch.stack((id_coords[0].view(-1), id_coords[1].view(-1)), dim=0)[None]
        pix_coords = torch.cat((pix_coords.repeat(b, 1, 1), ones), dim=1)

        cam_points = torch.matmul(inv_k[:, :3, :3], pix_coords)
        cam_points = depth.view(b, 1, -1) * cam_points
        cam_points = torch.cat((cam_points, ones), dim=1)
        return cam_points
