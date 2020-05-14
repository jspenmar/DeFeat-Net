import torch
import torch.nn as nn


class SSIM(nn.Module):
    """Layer to compute the weighted SSIM and L1 loss between a pair of images"""
    def __init__(self, ssim_weight=0.85):
        super().__init__()
        self.a = ssim_weight
        self.b = 1 - ssim_weight

        self.pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, target):
        l1_loss = torch.abs(target - pred).mean(1, keepdim=True)

        pred, target = self.refl(pred), self.refl(target)
        mu_pred, mu_target = self.pool(pred), self.pool(target)

        sigma_pred = self.pool(pred**2) - mu_pred**2
        sigma_target = self.pool(target**2) - mu_target**2
        sigma_pt = self.pool(pred*target) - mu_pred*mu_target

        ssim_n = (2 * mu_pred * mu_target + self.C1) * (2 * sigma_pt + self.C2)
        ssim_d = (mu_pred**2 + mu_target**2 + self.C1) * (sigma_pred + sigma_target + self.C2)

        sim = torch.clamp((1 - ssim_n/ssim_d) / 2, min=0, max=1)
        sim = sim.mean(1, keepdim=True)

        loss = self.a*sim + self.b*l1_loss
        return loss
