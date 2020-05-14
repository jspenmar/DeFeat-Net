from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageGradients(nn.Module):
    def __init__(self, sobel=True, as_magnitude=True):
        super().__init__()
        self.sobel = sobel
        self.as_magnitude = as_magnitude

        if self.sobel:
            self.dx_filter = nn.Parameter(torch.tensor([[[[0, 1., 0], [0, -2., 0], [0, 1., 0]]]]), requires_grad=False)
            self.dy_filter = nn.Parameter(torch.tensor([[[[0, 0, 0], [1., -2., 1.], [0, 0, 0]]]]), requires_grad=False)

            self.dx_edges = partial(F.conv2d, weight=self.dx_filter, padding=0)
            self.dy_edges = partial(F.conv2d, weight=self.dy_filter, padding=0)

            self.pad = nn.ReflectionPad2d(1)

    def forward(self, tensor):
        (b, c, h, w), device = tensor.shape, tensor.device

        if self.sobel:
            tensor = self.pad(tensor)
            dx = abs(self.dx_edges(tensor.view(-1, 1, h+2, w+2)).view(b, c, h, w))
            dy = abs(self.dy_edges(tensor.view(-1, 1, h+2, w+2)).view(b, c, h, w))
        else:
            dx = abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:]).mean(dim=1, keepdim=True)
            dy = abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :]).mean(dim=1, keepdim=True)

            dx = torch.cat((dx, torch.zeros(b, 1, h, 1, device=device)), dim=-1)
            dy = torch.cat((dy, torch.zeros(b, 1, 1, w, device=device)), dim=-2)

        return (dx**2 + dy**2)**0.5 if self.as_magnitude else (dx, dy)


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_fn = ImageGradients(sobel=False, as_magnitude=False)

    def forward(self, depth, image):
        depth_dx, depth_dy = self.grad_fn(depth)
        image_dx, image_dy = self.grad_fn(image)

        depth_dx *= torch.exp(-image_dx)
        depth_dy *= torch.exp(-image_dy)

        return depth_dx.mean() + depth_dy.mean()
