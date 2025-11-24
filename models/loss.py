import torch
from torch import nn
from models.triplet_loss import HardTripletLoss
import torch.nn.functional as F
import numpy as np


# 各种分布特征聚合
class LossFun(nn.Module):
    def __init__(self, alpha, margin):
        super(LossFun, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.alpha = alpha
        self.alpha_recon = 1.0

    def forward(self, pred, label, feat, feat2, recon_loss, args):
        # feat (b, n, c), x (b, t, c)
        if feat is not None:
            device = feat.device
            b, n, c = feat.shape
            flat_feat = feat.reshape(-1, c)  # (bn, c)
            la = torch.arange(n, device=device).repeat(b)

            t_loss = self.triplet_loss(flat_feat, la)
            if feat2 is not None:
                device = feat2.device
                b, n, c = feat2.shape
                flat_feat2 = feat2.reshape(-1, c)  # (bn, c)
                la2 = torch.arange(n, device=device).repeat(b)

                t_loss += self.triplet_loss(flat_feat2, la2)
        else:
            self.alpha = 0
            t_loss = 0
        mse_loss = 10. * self.mse_loss(pred, label)
        return mse_loss + self.alpha * t_loss + self.alpha_recon * recon_loss
