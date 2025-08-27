import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class DilatedConvBlock(nn.Module):
    """膨胀卷积块"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=4, dilation=4)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        return x + residual if x.shape == residual.shape else x

class ClassPriorGenerator(nn.Module):
    def __init__(self, num_classes=6, geo_in_dim=1, proto_dim=64, stat_dim=1, hidden_dim=32):
        super().__init__()

        self.geo_cnn = nn.Sequential(
            nn.Conv2d(geo_in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )

        self.class_prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))
        self.proto_proj = nn.Sequential(
            nn.Conv2d(geo_in_dim, proto_dim, kernel_size=1),
            nn.ReLU()
        )

        self.register_buffer("running_mean", torch.zeros(num_classes, stat_dim))
        self.register_buffer("running_var", torch.ones(num_classes, stat_dim))
        self.momentum = 0.01

        self.weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))

    def update_statistics(self, class_ids, geo_feats):
        with torch.no_grad():
            B, _, H, W = geo_feats.shape
            for c in range(self.running_mean.shape[0]):
                mask = (class_ids == c).float().unsqueeze(1)
                if mask.sum() > 0:
                    feat_c = geo_feats * mask
                    mean_c = feat_c.sum() / mask.sum()
                    var_c = ((feat_c - mean_c) ** 2).sum() / mask.sum()
                    self.running_mean[c] = (1 - self.momentum) * self.running_mean[c] + self.momentum * mean_c
                    self.running_var[c] = (1 - self.momentum) * self.running_var[c] + self.momentum * var_c

    def forward(self, geo_feats: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # 训练时更新统计信息
        if labels is not None:
            self.update_statistics(labels, geo_feats)

        B, _, H, W = geo_feats.shape
        num_classes = self.class_prototypes.shape[0]

        geo_logits = self.geo_cnn(geo_feats)
        geo_probs = F.softmax(geo_logits, dim=1)

        feat_proj = self.proto_proj(geo_feats)
        feat_flat = feat_proj.flatten(2).permute(0, 2, 1)
        proto = F.normalize(self.class_prototypes, dim=1)
        feat_norm = F.normalize(feat_flat, dim=2)
        sim = torch.matmul(feat_norm, proto.t())
        proto_map = sim.permute(0, 2, 1).reshape(B, num_classes, H, W)
        proto_probs = F.softmax(proto_map, dim=1)

        geo_flat = geo_feats.flatten(2).permute(0, 2, 1)
        mean = self.running_mean.view(1, 1, num_classes)
        var = self.running_var.view(1, 1, num_classes) + 1e-6
        stat_probs = torch.exp(-0.5 * (geo_flat - mean) ** 2 / var) / torch.sqrt(2 * torch.pi * var)
        stat_map = stat_probs.permute(0, 2, 1).reshape(B, num_classes, H, W)
        stat_probs = F.softmax(stat_map, dim=1)

        w = F.softmax(self.weights, dim=0)
        fused = w[0] * geo_probs + w[1] * proto_probs + w[2] * stat_probs
        return fused


class GeoAdapter2(nn.Module):
    def __init__(self, dim_R=768, dim_X=16, hidden_dim=64, stage_size=128, num_classes=6):
        super().__init__()
        self.size = stage_size
        self.down = nn.Conv2d(dim_R, hidden_dim, kernel_size=1)
        self.DCB = DilatedConvBlock(in_ch=hidden_dim, out_ch=hidden_dim)

        self.cond_encoder = nn.Sequential(
            nn.Conv2d(dim_X, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # Class Prior Encoder
        self.class_proj = nn.Sequential(
            nn.Conv2d(num_classes, hidden_dim, kernel_size=1),
            nn.ReLU()
        )

        self.film_conv = nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=1)

        self.act = nn.ReLU()
        self.up = nn.Conv2d(hidden_dim, dim_R, kernel_size=1)
        self.gate = nn.Parameter(torch.tensor(0.5))

        self.class_prior = ClassPriorGenerator(num_classes=num_classes, geo_in_dim=1)

    def forward(self,rgb, x, class_prior):
        rgb = rgb.permute(0, 3, 1, 2)

        if self.size > 16:
            rgb_up = F.interpolate(rgb, size=(self.size, self.size), mode='bilinear', align_corners=False)
            x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
            z = self.down(rgb_up)
        else:
            x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
            z = self.down(rgb)

        z = self.DCB(z)

        cond_feat = self.cond_encoder(x)
        class_prior = F.interpolate(class_prior, size=(self.size, self.size), mode='bilinear', align_corners=False)
        class_prior = self.class_proj(class_prior)
        cond_fused = cond_feat + class_prior
        gamma_beta = self.film_conv(cond_fused)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        z = gamma * z + beta
        z = self.act(z)
        z = F.interpolate(z, size=(16, 16), mode='bilinear', align_corners=False)
        z = self.up(z)

        out = rgb + self.gate * z
        out = out.permute(0, 2, 3, 1)
        return out

