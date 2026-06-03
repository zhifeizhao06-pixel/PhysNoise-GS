#
# PhysNoise-GS: 离线噪声标定工具
# 方案第5节 - 通过平场图(flat field)和暗场图(dark frame)标定 (a, b, K, black_level)
#
# 使用方法:
#   python noise_calibration/calibrate.py \
#       --flat_dir path/to/flat_frames \
#       --dark_dir path/to/dark_frames \
#       --iso 1600 \
#       --output noise_params.json
#

import os
import json
import argparse
import numpy as np

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False
    print("[警告] 未安装 rawpy，将只支持 .npy 格式的 RAW 数据")


def load_raw_image(path: str) -> np.ndarray:
    """
    加载 RAW 图像，返回 float32 的 numpy 数组（已减去黑电平）
    支持 .npy（直接的数值数组）和 .dng/.arw/.cr2 等相机 RAW 格式
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path).astype(np.float32)
    elif HAS_RAWPY:
        with rawpy.imread(path) as raw:
            # 读取原始 Bayer 数据，不做任何 ISP 处理
            bayer = raw.raw_image_visible.astype(np.float32)
            # 减去黑电平
            black = np.array(raw.black_level_per_channel, dtype=np.float32).mean()
            bayer = bayer - black
            return bayer
    else:
        raise ValueError(f"不支持的文件格式: {ext}，请安装 rawpy 或使用 .npy 格式")


def estimate_local_mean_variance(img1: np.ndarray, img2: np.ndarray,
                                  patch_size: int = 64) -> tuple:
    """
    用两张相同场景的 RAW 图估计局部均值和方差。

    原理：
        diff = img1 - img2  →  var(diff) = var(img1) + var(img2) = 2·σ²
        mean = (img1 + img2) / 2  →  近似真实信号均值

    Args:
        img1, img2: 同一场景的两张 RAW 图（float32）
        patch_size: 采样 patch 的大小

    Returns:
        means:     每个 patch 的均值列表
        variances: 每个 patch 的方差列表（单张图的方差）
    """
    h, w = img1.shape[:2]
    means, variances = [], []

    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            p1 = img1[y:y+patch_size, x:x+patch_size].ravel()
            p2 = img2[y:y+patch_size, x:x+patch_size].ravel()

            mean = (p1 + p2).mean() / 2.0
            diff = p1.astype(np.float64) - p2.astype(np.float64)
            var = np.var(diff) / 2.0            # 单张图的方差

            # 过滤掉饱和或极暗的 patch（不可靠）
            if mean > 10 and mean < 0.95 * p1.max():
                means.append(mean)
                variances.append(var)

    return np.array(means), np.array(variances)


def calibrate_noise(flat_dir: str, dark_dir: str = None,
                    iso: int = 1600, patch_size: int = 64) -> dict:
    """
    主标定函数。

    Args:
        flat_dir:  平场图文件夹（多张，相同ISO，均匀照明）
        dark_dir:  暗场图文件夹（多张，镜头盖上，同ISO同曝光），可选
        iso:       ISO 值（仅用于记录）
        patch_size: 采样 patch 大小

    Returns:
        dict 包含 'a', 'b', 'iso' 等标定结果
    """
    # 1. 加载平场图
    flat_paths = sorted([
        os.path.join(flat_dir, f) for f in os.listdir(flat_dir)
        if not f.startswith('.')
    ])
    if len(flat_paths) < 2:
        raise ValueError(f"平场图至少需要 2 张，当前只有 {len(flat_paths)} 张")

    print(f"[标定] 加载 {len(flat_paths)} 张平场图...")
    flat_images = [load_raw_image(p) for p in flat_paths]

    # 2. 用平场图的两两配对估计 (mean, variance)
    all_means, all_variances = [], []
    for i in range(0, len(flat_images) - 1, 2):
        m, v = estimate_local_mean_variance(
            flat_images[i], flat_images[i+1], patch_size)
        all_means.append(m)
        all_variances.append(v)

    means = np.concatenate(all_means)
    variances = np.concatenate(all_variances)

    # 3. 线性拟合：variance = a * mean + b
    print(f"[标定] 用 {len(means)} 个 patch 进行线性拟合...")
    coeffs = np.polyfit(means, variances, 1)
    a = float(coeffs[0])   # shot noise 系数（斜率）
    b = float(coeffs[1])   # 读出噪声方差（截距）
    b = max(b, 1e-6)       # 物理约束：b > 0

    # 4. 估计黑电平（从暗场图）
    black_level = 0.0
    if dark_dir and os.path.isdir(dark_dir):
        dark_paths = sorted([
            os.path.join(dark_dir, f) for f in os.listdir(dark_dir)
            if not f.startswith('.')
        ])
        print(f"[标定] 加载 {len(dark_paths)} 张暗场图...")
        dark_images = [load_raw_image(p) for p in dark_paths]
        black_level = float(np.mean([img.mean() for img in dark_images]))
        print(f"[标定] 估计黑电平: {black_level:.2f}")

    result = {
        'a': a,
        'b': b,
        'iso': iso,
        'black_level': black_level,
        'n_patches': len(means),
        'mean_range': [float(means.min()), float(means.max())],
    }

    print(f"\n[标定结果] ISO={iso}")
    print(f"  a (shot noise 系数) = {a:.6f}")
    print(f"  b (read noise 方差) = {b:.6f}")
    print(f"  black_level         = {black_level:.2f}")
    print(f"  拟合 patch 数量      = {len(means)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="PhysNoise-GS 噪声参数标定工具")
    parser.add_argument('--flat_dir', type=str, required=True,
                        help='平场图文件夹路径（均匀照明的多张 RAW 图）')
    parser.add_argument('--dark_dir', type=str, default=None,
                        help='暗场图文件夹路径（可选，用于估计黑电平）')
    parser.add_argument('--iso', type=int, default=1600,
                        help='拍摄 ISO 值（默认 1600）')
    parser.add_argument('--patch_size', type=int, default=64,
                        help='采样 patch 大小（默认 64）')
    parser.add_argument('--output', type=str, default='noise_params.json',
                        help='输出 JSON 文件路径（默认 noise_params.json）')
    args = parser.parse_args()

    result = calibrate_noise(
        flat_dir=args.flat_dir,
        dark_dir=args.dark_dir,
        iso=args.iso,
        patch_size=args.patch_size,
    )

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[完成] 标定结果已保存到: {args.output}")
    print("在 train.py 中使用: --noise_a {:.6f} --noise_b {:.6f}".format(
        result['a'], result['b']))


if __name__ == '__main__':
    main()
