#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass


# ============================================================
# PhysNoise-GS: 异方差 NLL 损失（方案 4.1 节）
# ============================================================

def heteroscedastic_nll(pred_raw: torch.Tensor,
                        target_raw: torch.Tensor,
                        a: float,
                        b: float,
                        clip_min: float = 1e-6) -> torch.Tensor:
    """
    异方差高斯负对数似然损失。

    σ²(x̂) = a·x̂ + b   （信号相关方差，物理噪声模型）
    L = (x̂ - D)² / (2σ²) + 0.5·log(σ²)

    注意：σ² 中的 x̂ 使用 .detach()（stop-gradient），
    防止模型靠放大方差来降低损失。

    Args:
        pred_raw:   模型预测的 RAW 值（由线性辐射经增益转换得到），[C, H, W]
        target_raw: 真实 RAW 观测（已归一化），[C, H, W]
        a:          shot noise 系数（标定值或经验值）
        b:          read noise 方差（标定值或经验值）
        clip_min:   方差下界（数值稳定性）
    Returns:
        标量损失
    """
    sigma2 = (a * pred_raw.detach() + b).clamp(min=clip_min)
    loss = (pred_raw - target_raw) ** 2 / (2.0 * sigma2) \
           + 0.5 * torch.log(sigma2)
    return loss.mean()


def snr_weighted_l1(pred: torch.Tensor,
                    target: torch.Tensor,
                    a: float,
                    b: float,
                    clip_min: float = 1e-6) -> torch.Tensor:
    """
    SNR 加权 L1 损失（NLL 的简化版，用于 warm-up 阶段）。
    暗区（低 SNR）自动降低权重，亮区（高 SNR）权重高。

    weight(x̂) = 1 / σ(x̂) = 1 / sqrt(a·x̂ + b)
    """
    sigma2 = (a * pred.detach() + b).clamp(min=clip_min)
    weight = 1.0 / sigma2.sqrt()
    return (weight * torch.abs(pred - target)).mean()

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
