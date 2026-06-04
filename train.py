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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, heteroscedastic_nll, snr_weighted_l1
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# PhysNoise-GS 新增导入
from utils.noise_model import CMOSNoiseModel, LearnableCMOSNoiseModel
from utils.isp import DifferentiableISP

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # ================================================================
    # PhysNoise-GS: 初始化噪声模型和可微 ISP
    # ================================================================
    if opt.use_raw_nll:
        if opt.learnable_noise:
            noise_model = LearnableCMOSNoiseModel(
                a_init=opt.noise_a,
                b_init=opt.noise_b,
                gain=opt.sensor_gain,
                black_level=opt.black_level,
            ).cuda()
            print(f"[PhysNoise-GS] 使用可学习噪声模型 (a_init={opt.noise_a}, b_init={opt.noise_b})")
        else:
            noise_model = CMOSNoiseModel(
                a=opt.noise_a,
                b=opt.noise_b,
                gain=opt.sensor_gain,
                black_level=opt.black_level,
            )
            print(f"[PhysNoise-GS] 使用标定噪声模型 (a={opt.noise_a}, b={opt.noise_b})")

        # 可微 ISP（用于感知损失，lambda_perc > 0 时启用）
        isp_model = None
        if opt.lambda_perc > 0:
            isp_model = DifferentiableISP(learnable_wb=True, learnable_ccm=True).cuda()
            print(f"[PhysNoise-GS] 启用可微 ISP（lambda_perc={opt.lambda_perc}）")
    else:
        noise_model = None
        isp_model = None
        print("[PhysNoise-GS] 使用标准 sRGB L1+SSIM 损失（原版模式）")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # ================================================================
    # PhysNoise-GS: 诊断统计变量（用于判断改进是否有效）
    # ================================================================
    diag = {
        # 损失对比
        "ema_nll_loss":        0.0,   # NLL 损失（PhysNoise 模式）
        "ema_l1_srgb":         0.0,   # sRGB L1（两种模式都记录，用于横向对比）
        "ema_ssim":            0.0,   # SSIM 值
        # 线性域健康度（判断逆 gamma 是否正确）
        "ema_linear_pred_mean": 0.0,  # 渲染线性辐射均值（应在合理范围）
        "ema_linear_gt_mean":   0.0,  # GT 逆 gamma 后均值（应与上面接近）
        "ema_linear_ratio":     0.0,  # pred/gt 比值（接近1说明尺度对齐）
        # 噪声模型健康度
        "ema_sigma2_mean":     0.0,   # 平均噪声方差（判断 a/b 参数合理性）
        "ema_snr_mean":        0.0,   # 平均 SNR（低于1说明噪声模型过强）
        # 暗区 vs 亮区分析（核心：NLL 是否真的对暗区降权了）
        "ema_dark_l1":         0.0,   # 暗区（GT<0.1）的 L1
        "ema_bright_l1":       0.0,   # 亮区（GT>0.5）的 L1
        "ema_dark_weight":     0.0,   # 暗区的 NLL 权重均值（越小越好）
        "ema_bright_weight":   0.0,   # 亮区的 NLL 权重均值
        # 高斯数量趋势（判断 floater 是否受控）
        "num_gaussians":       0,
    }
    DIAG_ALPHA = 0.01   # EMA 平滑系数（比 loss 的 0.4 更平滑，看趋势用）

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # PhysNoise-GS: NLL 模式下用 softplus 线性辐射渲染
        _use_linear = (opt.use_raw_nll and noise_model is not None)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                            use_linear_radiance=_use_linear)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # ================================================================
        # PhysNoise-GS: 损失计算分支
        # ================================================================
        gt_image = viewpoint_cam.original_image.cuda()

        if opt.use_raw_nll and noise_model is not None:
            # --- RAW 域物理噪声似然损失 ---
            # render_linear: 线性辐射输出（非负，未 clamp 到1）
            linear_image = render_pkg["render_linear"]
            if viewpoint_cam.alpha_mask is not None:
                linear_image = linear_image * viewpoint_cam.alpha_mask.cuda()

            # PNG/JPG 是 sRGB（经过 gamma 压缩），需要逆 gamma 还原到线性域
            # sRGB 标准近似：linear ≈ srgb^2.2
            # 两侧保持一致：渲染输出是线性辐射，GT 也转到线性域再比较
            gt_linear = gt_image.clamp(min=1e-6) ** 2.2

            # warm-up 阶段用 SNR 加权 L1（更稳定），之后切换到完整 NLL
            if iteration < opt.nll_warmup_iter:
                pred_raw = noise_model.linear_to_raw(linear_image)
                gt_raw   = noise_model.linear_to_raw(gt_linear)
                loss = snr_weighted_l1(pred_raw, gt_raw,
                                       a=opt.noise_a, b=opt.noise_b)
            else:
                loss = noise_model.nll_loss(linear_image, gt_linear,
                                            max_weight=opt.nll_max_weight)

            # sRGB 感知监督：渲染线性值经 gamma 压缩后与原始 GT 比较
            # 这样 SSIM 在感知上一致的 sRGB 域计算，补偿纯 NLL 的感知不足
            srgb_pred = linear_image.clamp(min=0) ** (1.0 / 2.2)   # 简单 gamma 压缩
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(srgb_pred.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(srgb_pred, gt_image)
            loss = loss + opt.lambda_dssim * (1.0 - ssim_value)

            # 可选：可微 ISP 感知损失（lambda_perc > 0 时额外叠加）
            if isp_model is not None and opt.lambda_perc > 0:
                srgb_isp_pred = isp_model(linear_image)
                srgb_isp_gt   = isp_model(gt_linear)
                if FUSED_SSIM_AVAILABLE:
                    ssim_perc = fused_ssim(srgb_isp_pred.unsqueeze(0), srgb_isp_gt.unsqueeze(0))
                else:
                    ssim_perc = ssim(srgb_isp_pred, srgb_isp_gt)
                loss = loss + opt.lambda_perc * (1.0 - ssim_perc)

            Ll1 = loss  # 用于日志记录

            # ---- 诊断统计（PhysNoise 模式）----
            with torch.no_grad():
                # 1. 线性域尺度检查：pred 和 gt 均值是否对齐
                pred_mean = linear_image.mean().item()
                gt_mean   = gt_linear.mean().item()
                ratio     = pred_mean / (gt_mean + 1e-8)
                diag["ema_linear_pred_mean"] = (1-DIAG_ALPHA)*diag["ema_linear_pred_mean"] + DIAG_ALPHA*pred_mean
                diag["ema_linear_gt_mean"]   = (1-DIAG_ALPHA)*diag["ema_linear_gt_mean"]   + DIAG_ALPHA*gt_mean
                diag["ema_linear_ratio"]     = (1-DIAG_ALPHA)*diag["ema_linear_ratio"]     + DIAG_ALPHA*ratio

                # 2. 噪声模型健康度：方差和 SNR 是否合理
                pred_raw_d = noise_model.linear_to_raw(linear_image)
                sigma2     = noise_model.noise_variance(pred_raw_d)
                snr        = pred_raw_d / (sigma2.sqrt() + 1e-8)
                diag["ema_sigma2_mean"] = (1-DIAG_ALPHA)*diag["ema_sigma2_mean"] + DIAG_ALPHA*sigma2.mean().item()
                diag["ema_snr_mean"]    = (1-DIAG_ALPHA)*diag["ema_snr_mean"]    + DIAG_ALPHA*snr.mean().item()

                # 3. 暗区 vs 亮区 L1 分析（在 sRGB 域判断）
                dark_mask   = gt_image < 0.1    # 暗区像素（sRGB）
                bright_mask = gt_image > 0.5    # 亮区像素
                srgb_pred_d = linear_image.clamp(min=0) ** (1.0/2.2)
                if dark_mask.sum() > 0:
                    dark_l1   = torch.abs(srgb_pred_d[dark_mask]   - gt_image[dark_mask]).mean().item()
                    diag["ema_dark_l1"] = (1-DIAG_ALPHA)*diag["ema_dark_l1"] + DIAG_ALPHA*dark_l1
                    # NLL 在暗区的实际权重 = 1/(2σ²)，越小说明暗区被越弱惩罚
                    dark_weight = (1.0 / (2.0 * sigma2[dark_mask])).mean().item()
                    diag["ema_dark_weight"] = (1-DIAG_ALPHA)*diag["ema_dark_weight"] + DIAG_ALPHA*dark_weight
                if bright_mask.sum() > 0:
                    bright_l1 = torch.abs(srgb_pred_d[bright_mask] - gt_image[bright_mask]).mean().item()
                    diag["ema_bright_l1"] = (1-DIAG_ALPHA)*diag["ema_bright_l1"] + DIAG_ALPHA*bright_l1
                    bright_weight = (1.0 / (2.0 * sigma2[bright_mask])).mean().item()
                    diag["ema_bright_weight"] = (1-DIAG_ALPHA)*diag["ema_bright_weight"] + DIAG_ALPHA*bright_weight

                # 4. sRGB L1（与 baseline 可比的指标）
                srgb_l1 = torch.abs(srgb_pred_d - gt_image).mean().item()
                diag["ema_l1_srgb"] = (1-DIAG_ALPHA)*diag["ema_l1_srgb"] + DIAG_ALPHA*srgb_l1
                diag["ema_ssim"]    = (1-DIAG_ALPHA)*diag["ema_ssim"]     + DIAG_ALPHA*ssim_value.item()
                diag["ema_nll_loss"]= (1-DIAG_ALPHA)*diag["ema_nll_loss"] + DIAG_ALPHA*loss.item()
                diag["num_gaussians"] = gaussians.get_xyz.shape[0]

        else:
            # --- 原版 sRGB L1 + SSIM 损失 ---
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # ---- 诊断统计（baseline 模式）----
            with torch.no_grad():
                diag["ema_l1_srgb"] = (1-DIAG_ALPHA)*diag["ema_l1_srgb"] + DIAG_ALPHA*Ll1.item()
                diag["ema_ssim"]    = (1-DIAG_ALPHA)*diag["ema_ssim"]     + DIAG_ALPHA*ssim_value.item()
                dark_mask   = gt_image < 0.1
                bright_mask = gt_image > 0.5
                if dark_mask.sum() > 0:
                    diag["ema_dark_l1"]   = (1-DIAG_ALPHA)*diag["ema_dark_l1"]   + DIAG_ALPHA*torch.abs(image[dark_mask]-gt_image[dark_mask]).mean().item()
                if bright_mask.sum() > 0:
                    diag["ema_bright_l1"] = (1-DIAG_ALPHA)*diag["ema_bright_l1"] + DIAG_ALPHA*torch.abs(image[bright_mask]-gt_image[bright_mask]).mean().item()
                diag["num_gaussians"] = gaussians.get_xyz.shape[0]

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                if opt.use_raw_nll:
                    progress_bar.set_postfix({
                        "Loss":    f"{ema_loss_for_log:.5f}",
                        "L1_rgb":  f"{diag['ema_l1_srgb']:.4f}",
                        "SNR":     f"{diag['ema_snr_mean']:.2f}",
                        "#GS":     f"{diag['num_gaussians']//1000}k",
                    })
                else:
                    progress_bar.set_postfix({
                        "Loss":    f"{ema_loss_for_log:.5f}",
                        "L1_rgb":  f"{diag['ema_l1_srgb']:.4f}",
                        "#GS":     f"{diag['num_gaussians']//1000}k",
                    })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # ---- PhysNoise-GS: 诊断信息写入 TensorBoard ----
            if tb_writer and opt.use_raw_nll:
                # 损失对比
                tb_writer.add_scalar('diag/nll_loss',         diag['ema_nll_loss'],        iteration)
                tb_writer.add_scalar('diag/l1_srgb',          diag['ema_l1_srgb'],         iteration)
                tb_writer.add_scalar('diag/ssim',             diag['ema_ssim'],            iteration)
                # 线性域健康度（关键：判断逆gamma是否正确）
                tb_writer.add_scalar('diag/linear_pred_mean', diag['ema_linear_pred_mean'],iteration)
                tb_writer.add_scalar('diag/linear_gt_mean',   diag['ema_linear_gt_mean'],  iteration)
                tb_writer.add_scalar('diag/linear_ratio',     diag['ema_linear_ratio'],    iteration)
                # 噪声模型健康度
                tb_writer.add_scalar('diag/sigma2_mean',      diag['ema_sigma2_mean'],     iteration)
                tb_writer.add_scalar('diag/snr_mean',         diag['ema_snr_mean'],        iteration)
                # 暗区 vs 亮区（核心验证指标）
                tb_writer.add_scalar('diag/dark_l1',          diag['ema_dark_l1'],         iteration)
                tb_writer.add_scalar('diag/bright_l1',        diag['ema_bright_l1'],       iteration)
                tb_writer.add_scalar('diag/dark_weight',      diag['ema_dark_weight'],     iteration)
                tb_writer.add_scalar('diag/bright_weight',    diag['ema_bright_weight'],   iteration)
                tb_writer.add_scalar('diag/num_gaussians',    diag['num_gaussians'],       iteration)

            # ---- 每500步打印一次诊断报告（控制台）----
            if opt.use_raw_nll and iteration % 500 == 0 and iteration > 0:
                print(f"\n{'='*60}")
                print(f"[PhysNoise 诊断] iter={iteration}")
                print(f"  [损失]    NLL={diag['ema_nll_loss']:.5f}  L1_sRGB={diag['ema_l1_srgb']:.5f}  SSIM={diag['ema_ssim']:.4f}")
                print(f"  [线性域]  pred_mean={diag['ema_linear_pred_mean']:.4f}  gt_mean={diag['ema_linear_gt_mean']:.4f}  ratio={diag['ema_linear_ratio']:.3f}")
                print(f"            {'✓ 尺度对齐' if 0.5<diag['ema_linear_ratio']<2.0 else '✗ 尺度偏差过大，检查逆gamma或sensor_gain参数'}")
                print(f"  [噪声]    σ²={diag['ema_sigma2_mean']:.6f}  SNR={diag['ema_snr_mean']:.2f}")
                # 检测 b 是否主导 σ²（b主导时 NLL 退化为 L2）
                b_ratio = opt.noise_b / (diag['ema_sigma2_mean'] + 1e-12)
                if b_ratio > 0.8:
                    print(f"            ✗ b={opt.noise_b}主导σ²({b_ratio*100:.0f}%)，NLL退化为L2！")
                    suggested_b = diag['ema_linear_gt_mean'] * 0.0001
                    print(f"            → 建议: --noise_b {suggested_b:.2e}  --noise_a 0.2")
                elif diag['ema_snr_mean'] < 1.0:
                    print(f"            ✗ SNR<1，noise_a/b过大，暗区被完全忽略")
                else:
                    print(f"            ✓ SNR合理，噪声模型参数匹配")
                print(f"  [暗/亮区] dark_L1={diag['ema_dark_l1']:.5f}  bright_L1={diag['ema_bright_l1']:.5f}")
                print(f"  [NLL权重] dark={diag['ema_dark_weight']:.4f}  bright={diag['ema_bright_weight']:.4f}")
                weight_ratio = diag['ema_bright_weight'] / (diag['ema_dark_weight'] + 1e-8)
                print(f"            bright/dark权重比={weight_ratio:.2f}  {'✓ 亮区权重更高（正确）' if weight_ratio>1.5 else '✗ 暗亮区权重接近，物理加权未生效'}")
                print(f"  [高斯数]  {diag['num_gaussians']:,} 个")
                print(f"{'='*60}")

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # PhysNoise-GS: 计算 SNR map 用于不确定性感知致密化
                    snr_map = None
                    if opt.use_raw_nll and noise_model is not None:
                        with torch.no_grad():
                            linear_image = render_pkg["render_linear"]
                            snr_map = noise_model.snr_map(linear_image)  # [C, H, W]

                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005,
                        scene.cameras_extent, size_threshold, radii,
                        snr_map=snr_map,
                        snr_threshold=opt.snr_threshold,
                    )
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
