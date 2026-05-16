import os
import time
import copy
import json
import pickle
import psutil
import swanlab
import warnings
import numpy as np
import properscoring
from datetime import datetime
from functools import partial

import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

from physicsnemo.utils.diffusion import stochastic_sampler
from physicsnemo.utils.diffusion import deterministic_sampler

warnings.filterwarnings("ignore", category=UserWarning)

# 自定义模块
import dnnlib


def validate_diffusion(
    model,
    loader,
    device,
    dataset,
    seed_batch_size,
    rank_batches,
    sampler_fn,
    img_shape,
    img_out_channels,
    dist,
):
    from generate_diffusion import generate_fn

    time_start = datetime.now()
    model.eval()
    truths_list = []
    preds_list = []
    with torch.inference_mode():
        for i, (batch_y, batch_x) in enumerate(loader):
            batch_x = batch_x.to(device)  # img_lr: [1, C_in, H, W]
            batch_y = batch_y.to(device)  # img_clean: [1, C_out, H, W]

            # 扩展低分辨率图像以生成多个样本
            img_lr = batch_x.expand(seed_batch_size, -1, -1, -1)

            pred = generate_fn(
                model=model,
                sampler_fn=sampler_fn,
                img_shape=img_shape,
                img_out_channels=img_out_channels,
                rank_batches=rank_batches,
                img_lr=img_lr,
                dist=dist,
            )

            if dist.get_rank() != 0:
                continue  # 仅主进程处理结果

            # 逆归一化
            truth_phy = dataset.denormalize_output_tensor(batch_y)  # [1, C_out, H, W]
            pred_phy = dataset.denormalize_output_tensor(pred)  # [num_ensembles, C_out, H, W]

            # pred_phy 添加维度
            pred_phy = pred_phy.unsqueeze(0)  # [1, num_ensembles, C_out, H, W]

            truths_list.append(truth_phy.cpu().numpy())
            preds_list.append(pred_phy.cpu().numpy())

            if (i + 1) % 10 == 0:
                elapsed = (datetime.now() - time_start).total_seconds()
                time_start = datetime.now()
                remaining = (elapsed / 10) * (len(loader) - (i + 1))  # 预计剩余时间 s
                print(
                    f"[{datetime.now():%Y-%m-%d %H:%M:%S}][INFO] 进度 {i + 1}/{len(loader)}, 剩余: {remaining/60:.2f} 分钟"
                )

    if dist.get_rank() != 0:
        return  # 仅主进程继续

    # 合并所有批次
    truths_arr = np.concatenate(truths_list, axis=0)  # [N, C_out, H, W]
    preds_arr = np.concatenate(preds_list, axis=0)  # [N, num_ensembles, C_out, H, W]

    return truths_arr, preds_arr


def validate_residual_diffusion(
    model_res,
    loader,
    device,
    dataset,
    seed_batch_size,
    rank_batches,
    sampler_fn,
    img_shape,
    img_out_channels,
    dist,
    loss_kwargs,
):
    from generate_residual_diffusion import generate_fn

    cfg = dnnlib.EasyDict()
    cfg.generation = dnnlib.EasyDict(
        seed_batch_size=seed_batch_size,
        hr_mean_conditioning=loss_kwargs.get("hr_mean_conditioning", False),
    )

    model_reg = loss_kwargs.regression_net

    model_reg.eval()
    model_res.eval()

    time_start = datetime.now()
    truths_list = []
    preds_list = []

    with torch.inference_mode():
        for i, (batch_y, batch_x) in enumerate(loader):
            batch_x = batch_x.to(device)  # [1, C_in, H, W]
            batch_y = batch_y.to(device)  # [1, C_out, H, W]

            pred = generate_fn(
                model_reg=model_reg,
                model_res=model_res,
                sampler_fn=sampler_fn,
                img_shape=img_shape,
                img_out_channels=img_out_channels,
                rank_batches=rank_batches,
                img_lr=batch_x,
                dist=dist,
                cfg=cfg,
            )

            if dist.get_rank() != 0:
                continue  # 仅主进程处理结果

            # 逆归一化
            truth_phy = dataset.denormalize_output_tensor(batch_y)  # [1, C_out, H, W]
            pred_phy = dataset.denormalize_output_tensor(pred)  # [num_ensembles, C_out, H, W]

            # pred_phy 添加维度
            pred_phy = pred_phy.unsqueeze(0)  # [1, num_ensembles, C_out, H, W]

            truths_list.append(truth_phy.cpu().numpy())
            preds_list.append(pred_phy.cpu().numpy())

            if (i + 1) % 10 == 0:
                elapsed = (datetime.now() - time_start).total_seconds()
                time_start = datetime.now()
                remaining = (elapsed / 10) * (len(loader) - (i + 1))  # 预计剩余时间 s
                print(
                    f"[{datetime.now():%Y-%m-%d %H:%M:%S}][INFO] 进度 {i + 1}/{len(loader)}, 剩余: {remaining/60:.2f} 分钟"
                )

    if dist.get_rank() != 0:
        return  # 仅主进程继续

    # 合并所有批次
    truths_arr = np.concatenate(truths_list, axis=0)  # [N, C_out, H, W]
    preds_arr = np.concatenate(preds_list, axis=0)  # [N, num_ensembles, C_out, H, W]

    return truths_arr, preds_arr


def validate(
    model,
    val_ds,
    loss_kwargs,
    generation_kwargs,
    device,
    step,
    dist,
    num_workers=1,
):
    """
    使用训练好的模型进行生成，并评估生成结果。
    """

    _wrapped_classes = ["training.loss.EDMLossSR", "training.loss.ResidualLoss"]
    loss_class_name = loss_kwargs.class_name  # 损失函数类型
    assert loss_class_name in _wrapped_classes, f"Unsupported loss_class_name: {loss_class_name}"

    sampler_type = generation_kwargs.sampler_type
    num_ensembles = generation_kwargs.num_ensembles
    max_samples = generation_kwargs.max_samples
    rain_threshold = generation_kwargs.get("threshold", 0.01)

    img_shape = val_ds.image_shape()  # (x, y)
    img_out_channels = len(val_ds.output_channels())

    # 2. 划分批次 FIXME: 这里假设每个批次的种子数量等于 num_ensembles
    seed_batch_size = num_ensembles  # 每个批次的种子数量
    seeds = list(np.arange(num_ensembles))
    num_batches = ((len(seeds) - 1) // (seed_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # 样本采样
    total_samples = len(val_ds)
    if max_samples < total_samples:
        indices = np.linspace(0, total_samples - 1, max_samples, dtype=int)
    else:
        indices = np.arange(total_samples)

    subset = Subset(val_ds, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=num_workers)

    # 采样函数
    sigma_min, sigma_max = 0.002, 80  # EDM论文推荐值
    if sampler_type == "deterministic":
        sampler_fn = partial(
            deterministic_sampler,
            num_steps=9,
            solver="euler",  # [euler, heun]
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    elif sampler_type == "stochastic":
        sampler_fn = partial(
            stochastic_sampler,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")

    #######################
    # 生成样本
    #######################

    # 根据 regression_checkpoint_path 判断是 diffusion 还是 residual_diffusion
    if loss_class_name == "training.loss.EDMLossSR":
        truths_arr, preds_arr = validate_diffusion(
            model=model,
            loader=loader,
            device=device,
            dataset=val_ds,
            seed_batch_size=seed_batch_size,
            rank_batches=rank_batches,
            sampler_fn=sampler_fn,
            img_shape=img_shape,
            img_out_channels=img_out_channels,
            dist=dist,
        )
    elif loss_class_name == "training.loss.ResidualLoss":
        truths_arr, preds_arr = validate_residual_diffusion(
            model_res=model,
            loader=loader,
            device=device,
            dataset=val_ds,
            seed_batch_size=seed_batch_size,
            rank_batches=rank_batches,
            sampler_fn=sampler_fn,
            img_shape=img_shape,
            img_out_channels=img_out_channels,
            dist=dist,
            loss_kwargs=loss_kwargs,
        )

    #######################
    # 计算评估指标
    #######################

    # truths_arr:  [N, C_out, H, W]
    # preds_arr:  [N, num_ensembles, C_out, H, W]

    preds_mean = preds_arr.mean(axis=1)  # 集合平均 [N, C, H, W]

    # ---- MAE / RMSE（对 N,C,H,W 全局平均）----
    diff = preds_mean - truths_arr  # [N, C, H, W]
    val_mae = float(np.mean(np.abs(diff)))
    val_rmse = float(np.sqrt(np.mean(diff**2)))

    # ---- CRPS ----
    val_crps = properscoring.crps_ensemble(truths_arr, preds_arr, axis=1).mean()

    # ---- SSIM（逐样本逐通道平均）----
    ssim_vals = []
    N, C, H, W = truths_arr.shape
    for n in range(N):
        for c in range(C):
            p = preds_mean[n, c]
            t = truths_arr[n, c]
            data_range = float(t.max() - t.min())
            if data_range == 0:
                data_range = 1.0
            ssim_vals.append(ssim(p, t, data_range=data_range))
    val_ssim = float(np.mean(ssim_vals)) if len(ssim_vals) > 0 else 0.0

    # ---- POD / TNR / FAR / CSI ----
    rain_channel = 0  # 降水通道
    p_rain = preds_mean[:, rain_channel]  # [N, H, W]
    t_rain = truths_arr[:, rain_channel]  # [N, H, W]

    # 二值化
    p_bin = p_rain >= rain_threshold
    t_bin = t_rain >= rain_threshold

    # 计算混淆矩阵
    hits = int(np.logical_and(p_bin, t_bin).sum())
    misses = int(np.logical_and(~p_bin, t_bin).sum())
    fa = int(np.logical_and(p_bin, ~t_bin).sum())
    cn = int(np.logical_and(~p_bin, ~t_bin).sum())

    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far = fa / (hits + fa) if (hits + fa) > 0 else 0.0
    tnr = cn / (cn + fa) if (cn + fa) > 0 else 0.0
    csi = hits / (hits + misses + fa) if (hits + misses + fa) > 0 else 0.0

    # 写入 SwanLab
    swanlab.log(
        {
            "val/CRPS": val_crps,
            "val/MAE": val_mae,
            "val/RMSE": val_rmse,
            "val/SSIM": val_ssim,
            "val/CSI": csi,
            "val/POD": pod,
            "val/FAR": far,
            "val/TNR": tnr,
        },
        step=step,
    )
