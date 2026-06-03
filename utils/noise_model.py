#
# PhysNoise-GS: CMOS Physical Noise Model
# 实现方案 2.2 节的异方差高斯噪声模型
# σ²(x̂) = a·x̂ + b
#   a: 信号相关项系数（shot noise），通过平场标定得到
#   b: 信号无关项方差（read noise + dark + quant），通过暗场标定得到
#

import torch
import torch.nn as nn
import numpy as np


class CMOSNoiseModel:
    """
    CMOS 传感器复合噪声模型（异方差高斯近似）。

    物理来源（对应方案表1）：
      - 光子散粒噪声 (shot):  方差 ∝ 信号强度
      - 读出噪声 (read):      固定方差 b 的一部分
      - 暗电流散粒 (dark):    固定方差 b 的一部分
      - 量化噪声 (quant):     固定方差 b 的一部分

    合并近似：D ~ N(x̂, σ²),  σ²(x̂) = a·x̂ + b

    参数说明：
      a (float):           shot noise 系数（离线标定，或设为可学习）
      b (float):           信号无关噪声方差（离线标定，或设为可学习）
      gain (float):        整体增益 = ISO/100 * exposure_time（用于辐射→RAW的转换）
      black_level (float): 传感器黑电平（减去后 RAW 值应≥0）
    """

    def __init__(self, a: float = 0.01, b: float = 0.001,
                 gain: float = 1.0, black_level: float = 0.0):
        self.a = a
        self.b = b
        self.gain = gain
        self.black_level = black_level

    @classmethod
    def from_iso_exposure(cls, a: float, b: float,
                          iso: int, exposure_time: float,
                          black_level: float = 0.0):
        """从 ISO + 曝光时间构造增益"""
        gain = (iso / 100.0) * exposure_time
        return cls(a=a, b=b, gain=gain, black_level=black_level)

    def linear_to_raw(self, linear_radiance: torch.Tensor) -> torch.Tensor:
        """
        线性辐射值 → 期望 RAW DN 值（前向模型）
        linear_radiance: [C, H, W] 或 [H, W, C]，范围 [0, ∞)
        返回与输入同形状的期望 RAW 值
        """
        return self.gain * linear_radiance + self.black_level

    def noise_variance(self, expected_raw: torch.Tensor) -> torch.Tensor:
        """
        逐像素噪声方差：σ²(x̂) = a·x̂ + b
        expected_raw: 期望 RAW 值（已经 detach，stop-gradient）
        返回同形状的方差张量，clamp 防止数值问题
        """
        return (self.a * expected_raw + self.b).clamp(min=1e-6)

    def nll_loss(self, pred_linear: torch.Tensor,
                 target_raw: torch.Tensor) -> torch.Tensor:
        """
        异方差高斯负对数似然损失（方案公式 4.1）：

            L_NLL = Σ [ (D - x̂)² / (2σ²) + 0.5·log(σ²) ]

        注意：σ²(x̂) 中的 x̂ 使用 stop-gradient（.detach()），
        防止模型靠放大方差来降低损失。

        Args:
            pred_linear:  渲染出的线性辐射 [C, H, W]，范围 [0, ∞)
            target_raw:   真实 RAW 观测（已减黑电平并归一化）[C, H, W]
        Returns:
            标量损失值
        """
        pred_raw = self.linear_to_raw(pred_linear)                  # 线性辐射 → 期望RAW
        sigma2 = self.noise_variance(pred_raw.detach())             # stop-gradient 方差
        nll = (pred_raw - target_raw) ** 2 / (2.0 * sigma2) \
              + 0.5 * torch.log(sigma2)
        return nll.mean()

    def snr_map(self, pred_linear: torch.Tensor) -> torch.Tensor:
        """
        逐像素信噪比估计：SNR = x̂ / σ(x̂)
        用于不确定性感知致密化（方案第5节）

        Args:
            pred_linear: 渲染出的线性辐射 [C, H, W]
        Returns:
            SNR map，同形状，值越小表示该像素越暗/越不可靠
        """
        pred_raw = self.linear_to_raw(pred_linear.detach())
        sigma2 = self.noise_variance(pred_raw)
        snr = pred_raw / (sigma2.sqrt() + 1e-8)
        return snr


class LearnableCMOSNoiseModel(nn.Module):
    """
    可学习版本的噪声模型：当无法离线标定时，
    将 (a, b) 作为可训练参数，在训练中联合优化。

    使用 softplus 激活保证 a, b > 0（物理约束）。
    """

    def __init__(self, a_init: float = 0.01, b_init: float = 0.001,
                 gain: float = 1.0, black_level: float = 0.0):
        super().__init__()
        # 用 log 空间参数化，保证正值
        self._log_a = nn.Parameter(torch.tensor(float(np.log(a_init))))
        self._log_b = nn.Parameter(torch.tensor(float(np.log(b_init))))
        self.gain = gain
        self.black_level = black_level

    @property
    def a(self) -> torch.Tensor:
        return torch.exp(self._log_a)

    @property
    def b(self) -> torch.Tensor:
        return torch.exp(self._log_b)

    def linear_to_raw(self, linear_radiance: torch.Tensor) -> torch.Tensor:
        return self.gain * linear_radiance + self.black_level

    def noise_variance(self, expected_raw: torch.Tensor) -> torch.Tensor:
        return (self.a * expected_raw + self.b).clamp(min=1e-6)

    def nll_loss(self, pred_linear: torch.Tensor,
                 target_raw: torch.Tensor) -> torch.Tensor:
        pred_raw = self.linear_to_raw(pred_linear)
        sigma2 = self.noise_variance(pred_raw.detach())
        nll = (pred_raw - target_raw) ** 2 / (2.0 * sigma2) \
              + 0.5 * torch.log(sigma2)
        return nll.mean()

    def snr_map(self, pred_linear: torch.Tensor) -> torch.Tensor:
        pred_raw = self.linear_to_raw(pred_linear.detach())
        sigma2 = self.noise_variance(pred_raw)
        return pred_raw / (sigma2.sqrt() + 1e-8)

    def forward(self, pred_linear: torch.Tensor,
                target_raw: torch.Tensor) -> torch.Tensor:
        """直接调用即返回 NLL loss"""
        return self.nll_loss(pred_linear, target_raw)
