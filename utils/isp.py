#
# PhysNoise-GS: Differentiable ISP (Image Signal Processing Pipeline)
# 实现方案第3节的可微 ISP：线性 RAW → sRGB
#   步骤：WB（白平衡）→ CCM（色彩矩阵）→ Tonemap（色调映射）
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableISP(nn.Module):
    """
    可微 ISP 模块：将线性 RAW 辐射值转换为 sRGB 图像。

    流程：
        linear_raw --[WB]--> --[CCM]--> --[Tonemap]--> sRGB

    参数说明：
      wb_gains (list[3] or None):  白平衡增益 [R, G, B]，None 则设为可学习
      ccm (3×3 list or None):      色彩矩阵，None 则初始化为单位矩阵（可学习）
      learnable_wb (bool):         白平衡是否可学习（有标定数据时设 False）
      learnable_ccm (bool):        CCM 是否可学习
      n_tone_knots (int):          色调映射曲线控制点数（越多越灵活）
    """

    def __init__(self,
                 wb_gains=None,
                 ccm=None,
                 learnable_wb: bool = True,
                 learnable_ccm: bool = True,
                 n_tone_knots: int = 64):
        super().__init__()

        # --- 白平衡（逐通道乘性增益）---
        if wb_gains is not None:
            wb_init = torch.tensor(wb_gains, dtype=torch.float32)
        else:
            wb_init = torch.ones(3, dtype=torch.float32)

        if learnable_wb:
            self.wb_log = nn.Parameter(torch.log(wb_init))   # log 空间，保证正值
        else:
            self.register_buffer('wb_log', torch.log(wb_init))

        # --- 色彩矩阵（3×3 线性变换）---
        if ccm is not None:
            ccm_init = torch.tensor(ccm, dtype=torch.float32).reshape(3, 3)
        else:
            ccm_init = torch.eye(3, dtype=torch.float32)

        if learnable_ccm:
            self.ccm = nn.Parameter(ccm_init)
        else:
            self.register_buffer('ccm', ccm_init)

        # --- 色调映射曲线（单调样条，用累积 softplus 保证单调）---
        # 控制点初始化为均匀分布（近似线性映射）
        self.n_tone_knots = n_tone_knots
        knots_init = torch.ones(n_tone_knots) / n_tone_knots
        self.tone_knots = nn.Parameter(knots_init)

    @property
    def wb_gains(self) -> torch.Tensor:
        """白平衡增益（保证正值）"""
        return torch.exp(self.wb_log)                         # [3]

    def _build_tone_curve(self) -> torch.Tensor:
        """
        构建单调色调映射曲线（控制点值）。
        使用 softplus + cumsum 保证严格单调递增，归一化到 [0, 1]。
        返回: [n_tone_knots] 的曲线 y 值，对应等间距的 x 输入
        """
        deltas = F.softplus(self.tone_knots) + 1e-6          # 正增量
        curve = torch.cumsum(deltas, dim=0)
        curve = curve / curve[-1]                             # 归一化到 [0,1]
        return curve

    def apply_tonemap(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入 x（已 clamp 到 [0,1]）应用色调映射曲线。
        使用线性插值查表。

        x: 任意形状，值域 [0, 1]
        返回: 与 x 同形状的映射结果
        """
        curve = self._build_tone_curve()                      # [K]
        K = self.n_tone_knots
        # 构建 x 坐标（等间距，从 0 到 1）
        x_coords = torch.linspace(0.0, 1.0, K, device=x.device, dtype=x.dtype)
        curve = curve.to(x.device)

        # 将 x 映射到索引空间，做线性插值
        x_flat = x.reshape(-1).clamp(0.0, 1.0)
        # 找到每个 x 落在哪个区间
        idx_float = x_flat * (K - 1)
        idx_low = idx_float.long().clamp(0, K - 2)
        idx_high = (idx_low + 1).clamp(0, K - 1)
        frac = idx_float - idx_low.float()

        y_low = curve[idx_low]
        y_high = curve[idx_high]
        y = y_low + frac * (y_high - y_low)

        return y.reshape(x.shape)

    def forward(self, linear_raw: torch.Tensor) -> torch.Tensor:
        """
        线性 RAW → sRGB

        Args:
            linear_raw: [C, H, W] 或 [B, C, H, W]，线性辐射值（非负）
        Returns:
            srgb: 同形状，值域 [0, 1]
        """
        squeeze = False
        if linear_raw.dim() == 3:
            linear_raw = linear_raw.unsqueeze(0)              # [1, C, H, W]
            squeeze = True

        B, C, H, W = linear_raw.shape

        # 1. 白平衡：逐通道乘
        wb = self.wb_gains.to(linear_raw.device)              # [3]
        x = linear_raw * wb.view(1, 3, 1, 1)                 # [B, 3, H, W]

        # 2. CCM：3×3 矩阵变换
        # [B, 3, H, W] → [B, H, W, 3] → matmul → [B, H, W, 3] → [B, 3, H, W]
        ccm = self.ccm.to(linear_raw.device)                  # [3, 3]
        x = x.permute(0, 2, 3, 1)                            # [B, H, W, 3]
        x = torch.matmul(x, ccm.T)                           # [B, H, W, 3]
        x = x.permute(0, 3, 1, 2)                            # [B, 3, H, W]
        x = x.clamp(0.0, 1.0)

        # 3. 色调映射（逐像素，通道独立）
        x = self.apply_tonemap(x)

        if squeeze:
            x = x.squeeze(0)                                  # [C, H, W]
        return x

    def get_params_for_optimizer(self, lr_wb=1e-3, lr_ccm=1e-4, lr_tone=1e-3):
        """返回分组参数，便于给不同模块设置不同学习率"""
        params = []
        if isinstance(self.wb_log, nn.Parameter):
            params.append({'params': [self.wb_log], 'lr': lr_wb, 'name': 'isp_wb'})
        if isinstance(self.ccm, nn.Parameter):
            params.append({'params': [self.ccm], 'lr': lr_ccm, 'name': 'isp_ccm'})
        params.append({'params': [self.tone_knots], 'lr': lr_tone, 'name': 'isp_tone'})
        return params
