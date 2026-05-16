import os
import re
import json
import time
import copy
import hydra
import pickle
import swanlab
import warnings
import itertools
import numpy as np
import setproctitle
import xarray as xr
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf


import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from physicsnemo.launch.logging import PythonLogger

# 自定义模块
import dnnlib
from dataset import DATASETS
from torch_utils.soap import SOAP
from training.loss import RegressionLoss
from models import CorrDiffRegressionUNet
from torch_utils.occupy_gpu_memory import occupy_available_gpu_memory
from torch_utils.mem_usage import get_grouped_mem_usage



def generate_and_evaluate(model, val_ds, device, cfg, step, logger):
    """
    使用训练好的模型进行生成，并评估生成结果。
    """

    logger.info("Starting inference...")

    img_in_channels = len(val_ds.input_channels())
    img_out_channels = len(val_ds.output_channels())
    img_shape = val_ds.image_shape()  # [H, W]
    max_samples = cfg.generation.get("max_samples", 1000000)
    rain_threshold = cfg.generation.get("threshold", 0.01)
    num_workers = cfg.workers

    latents_shape = (1, img_out_channels, img_shape[0], img_shape[1])  # (1, C, H, W)
    zero_input = torch.zeros(latents_shape, device=device)

    #######################
    # 生成样本
    #######################
    # 样本采样
    total_samples = len(val_ds)
    if max_samples < total_samples:
        indices = np.linspace(0, total_samples - 1, max_samples, dtype=int)
    else:
        indices = np.arange(total_samples)
    subset = Subset(val_ds, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=num_workers)

    model.eval()
    truths_list = []
    preds_list = []
    with torch.inference_mode():
        for i, (batch_y, batch_x) in enumerate(loader):
            batch_x = batch_x.to(device)  # img_lr: [1, C_in, H, W]
            batch_y = batch_y.to(device)  # img_clean: [1, C_out, H, W]

            pred = model(x=zero_input, img_lr=batch_x)  # [1, C_out, H, W]

            # 逆归一化
            truth_phy = val_ds.denormalize_output_tensor(batch_y)
            pred_phy = val_ds.denormalize_output_tensor(pred)

            truths_list.append(truth_phy.cpu().numpy())
            preds_list.append(pred_phy.cpu().numpy())

    # 合并所有批次
    truths_arr = np.concatenate(truths_list, axis=0)  # [N, C_out, H, W]
    preds_arr = np.concatenate(preds_list, axis=0)  # [N, C_out, H, W]

    # # 添加预测的集合 ensemble 维度: [N, 1, C_out, H, W]
    # preds_arr = np.expand_dims(preds_arr, axis=1)

    #######################
    # 计算评估指标
    #######################
    # ---- MAE / RMSE（对 N,C,H,W 全局平均）----
    diff = preds_arr - truths_arr  # [N, C, H, W]
    val_mae = float(np.mean(np.abs(diff)))
    val_rmse = float(np.sqrt(np.mean(diff**2)))

    # ---- SSIM ----
    ssim_vals = []
    N, C, H, W = truths_arr.shape
    for n in range(N):
        for c in range(C):
            p = preds_arr[n, c]
            t = truths_arr[n, c]
            data_range = float(t.max() - t.min())
            if data_range == 0:
                data_range = 1.0
            ssim_vals.append(ssim(p, t, data_range=data_range))
    val_ssim = float(np.mean(ssim_vals)) if len(ssim_vals) > 0 else 0.0

    # ---- POD / TNR / FAR / CSI ----
    rain_channel = 0  # 降水通道
    p_rain = preds_arr[:, rain_channel]  # [N, H, W]
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
            # "val/CRPS": val_crps,
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


def instantiate_dataset(cfg_group, dataset_type):
    """把 OmegaConf 转 dict，去掉 type 后动态实例化 Dataset"""
    kwargs = OmegaConf.to_container(cfg_group, resolve=True)
    kwargs.pop("type", None)  # 去除 type 字段
    return dataset_type(**kwargs)


@hydra.main(version_base=None, config_path="conf/regression")
def main(cfg: DictConfig):
    """
    降尺度回归模型训练脚本。
    """
    # ---------------- 配置验证区域 ----------
    dataset_cls = DATASETS[cfg.dataset.type]
    assert cfg.batch >= cfg.batch_gpu, "batch must be >= batch_gpu"
    assert cfg.batch % cfg.batch_gpu == 0, "batch must be divisible by batch_gpu"
    # ---------------- 配置验证区域 ----------

    # 初始化 logger
    logger = PythonLogger("regression_train")  # General python logger

    # 设置进程启动方式为 spawn
    torch.multiprocessing.set_start_method("spawn")

    # 数据集名称 part1-US-Rocky
    dataset_name = os.path.basename(os.path.dirname(cfg.dataset.input_path))

    swanlab.init(
        project="Benchmark-NIPS26",
        experiment_name=f"Regression-{dataset_name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        logdir="outputs/swanlog",
    )
    # print("[INFO] config:")
    # print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################
    #  1. 数据集
    ########################

    # 占用显存
    if OCCUPY_GPU_MEMORY := True:
        dummy_tensor = occupy_available_gpu_memory(reserve_gb=5)
        logger.info("GPU memory reserved.")

    dataset_cls = DATASETS[cfg.dataset.type]
    val_cfg = OmegaConf.merge(cfg.dataset, cfg.get("validation", {}))
    train_ds = instantiate_dataset(cfg.dataset, dataset_cls)
    val_ds = instantiate_dataset(val_cfg, dataset_cls) if ("validation" in cfg) else None

    # 释放显存
    if OCCUPY_GPU_MEMORY:
        del dummy_tensor
        torch.cuda.empty_cache()
        logger.info("GPU memory released.")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_gpu,
        shuffle=True,
        num_workers=cfg.workers,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.batch_gpu,
            shuffle=False,
            num_workers=cfg.workers,
        )
        if val_ds is not None
        else None
    )

    # train_iterator = itertools.cycle(train_loader) # 无限迭代器 会爆内存
    train_iterator = iter(train_loader)

    ####################################
    #  2. 模型 & 损失函数 & 优化器
    ####################################
    # 网络架构。
    in_channels = len(train_ds.input_channels())
    out_channels = len(train_ds.output_channels())
    img_resolution = train_ds.image_shape()  # [H, W]

    interface_args = dict(
        img_in_channels=in_channels,
        img_out_channels=out_channels,
        img_resolution=img_resolution,
    )

    if cfg.arch == "ddpmpp":
        model_args = dict(
            model_type="PhysicsnemoSongUNet",
            embedding_type="zero",
            encoder_type="standard",
            decoder_type="standard",
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    elif cfg.arch == "ncsnpp":
        model_args = dict(
            model_type="PhysicsnemoSongUNet",
            embedding_type="zero",
            encoder_type="residual",
            decoder_type="standard",
            channel_mult_noise=2,
            resample_filter=[1, 3, 3, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    elif cfg.arch == "adm":
        model_args = dict(
            model_type="DhariwalUNet",
            model_channels=192,
            channel_mult=[1, 2, 3, 4],
        )
    else:
        raise ValueError(f"--arch: unknown architecture '{cfg.arch}'")

    # 网络选项。
    if cfg.cbase is not None:
        model_args["model_channels"] = cfg.cbase
    if cfg.cres is not None:
        model_args["channel_mult"] = cfg.cres
    if cfg.augment:
        raise NotImplementedError("数据增强尚未实现")
    model_args.update(dropout=cfg.dropout, use_fp16=cfg.fp16)

    # 实例化模型, 损失函数, 优化器, EMA 模型
    loss_fn = RegressionLoss()
    model = CorrDiffRegressionUNet(**interface_args, **model_args).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)  # EMA 模型

    # 优化器参数
    # optimizer_kwargs = dict(lr=cfg.lr, betas=[0.9, 0.999], eps=1e-8, fused=True)
    optimizer_kwargs = dict(**cfg.optim)
    optimizer_class_name = optimizer_kwargs.pop("class_name", None)  # 去除 class_name
    if optimizer_class_name == "torch.optim.Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), **optimizer_kwargs)
    elif optimizer_class_name == "torch_utils.soap.SOAP":
        optimizer = SOAP(params=model.parameters(), **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown optimizer class name: {optimizer_class_name}")

    ####################################
    #  3. 创建输出目录, 保存配置文件
    ####################################
    # 创建输出目录。
    logger.info("Creating output directory...")
    desc = f"Regression-{cfg.dataset.type:s}-{dataset_name:s}-{cfg.arch:s}-{datetime.now():%Y%m%d_%H%M%S}"

    prev_run_dirs = []
    if os.path.isdir(cfg.outdir):
        prev_run_dirs = [x for x in os.listdir(cfg.outdir) if os.path.isdir(os.path.join(cfg.outdir, x))]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    checkpoint_dir = os.path.join(cfg.outdir, f"{cur_run_id:05d}-{desc}")
    assert not os.path.exists(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.file_logging(f"{checkpoint_dir}/train.log")

    # 更新 SwanLab 配置
    swanlab.config.dataset_name = dataset_name
    swanlab.config.model_args = model_args
    swanlab.config.interface_args = interface_args
    swanlab.config.optimizer_kwargs = optimizer_kwargs
    swanlab.config.checkpoint_dir = checkpoint_dir

    # 保存config文件到输出目录
    cfg_save_path = os.path.join(checkpoint_dir, "config.json")
    with open(cfg_save_path, "w") as f:
        config_dict = dict(swanlab.config)
        json.dump(config_dict, f, indent=2)

    ####################################
    #  4. 训练 & 验证
    ####################################

    # 训练参数
    total_kimg = max(int(cfg.duration * 1000), 1)  # 训练时长（以千张图像计）
    batch_size = cfg.batch  # 单次训练迭代的总批大小。
    batch_gpu = cfg.batch_gpu  # 每块 GPU 的批大小上限
    num_accumulation_rounds = max(1, batch_size // batch_gpu)  # 梯度累积步数

    ema_halflife_kimg = cfg.ema  # 模型权重 EMA 的半衰期（千张图像）。
    ema_rampup_ratio = 0.05  # EMA 预热系数，None 表示不预热。
    lr_rampup_kimg = 10000  # 学习率预热时长。

    kimg_per_tick = cfg.tick  # 进度打印间隔（千张图像）。
    snapshot_ticks = cfg.snap  # 网络快照保存频率，None 表示禁用。
    state_dump_ticks = cfg.dump  # 训练状态转储频率，None 表示禁用。

    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    start_time = time.time()
    maintenance_time = 0.0

    logger.info(f"Starting training...")
    logger.info(f"Total duration: {total_kimg} kimg")
    logger.info(f"Batch size: {batch_size} (累积步数: {num_accumulation_rounds})")

    while True:
        # 记录每个 total_batch 的 loss
        loss_sum = torch.tensor(0.0, device=device)
        count = torch.tensor(0.0, device=device)

        # 累积梯度前清零
        optimizer.zero_grad()
        model.train()
        for round_idx in range(num_accumulation_rounds):
            # 获取数据, Dataset returns (hr, lr)
            try:
                img_clean, img_lr = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                img_clean, img_lr = next(train_iterator)

            img_clean = img_clean.to(device)
            img_lr = img_lr.to(device)

            # 前向传播, 损失计算, 反向传播
            loss = loss_fn(model, img_clean, img_lr)  # [B, C_hr, H, W]
            (loss.mean() / num_accumulation_rounds).backward()  # 梯度累积
            # (loss.sum() / batch_size).backward()  # 这种方式是上面的 C_hr*H*W 倍

            loss_sum += loss.sum().detach()
            count += loss.numel()  # 元素个数 [B, H, W] -> B*H*W

            # print(f"[TEST] round {round_idx+1}/{num_accumulation_rounds}, loss: {loss.mean().item():.6f}")

        # Update weights.
        # 更新权重。
        for g in optimizer.param_groups:
            # 学习率预热
            g["lr"] = optimizer_kwargs["lr"] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in model.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad)

        optimizer.step()

        # Update EMA.
        # 更新 EMA: Exponential Moving Average（指数移动平均）
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        with torch.no_grad():
            for p_ema, p_net in zip(ema.parameters(), model.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        # 每个 tick 执行一次维护任务。
        train_loss = loss_sum / count
        swanlab.log({"train/loss": train_loss.item()}, step=int(cur_nimg))

        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        ##################
        # 完成一个 tick
        ##################

        # 打印状态
        tick_end_time = time.time()
        logger.info(
            f"Tick {cur_tick:4d}: "
            f"{cur_nimg / 1000:8.1f} kimg, "
            f"{tick_end_time - start_time:8.1f} sec total, "
            f"{tick_end_time - tick_start_time:6.1f} sec tick, "
            f"{maintenance_time:6.1f} sec maintenance, "
            f"loss: {train_loss.item():.6f}"
        )

        # 验证 validation, 使用和保存网络快照相同的频率
        if (val_loader is not None) and (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            ema.eval()  # 评估模式
            val_loss_sum = torch.tensor(0.0, device=device)
            val_count = torch.tensor(0.0, device=device)

            with torch.no_grad():
                for img_clean, img_lr in val_loader:
                    img_clean = img_clean.to(device)
                    img_lr = img_lr.to(device)

                    vloss = loss_fn(ema, img_clean, img_lr)  # [B, C_hr, H, W]
                    val_loss_sum += vloss.sum()
                    val_count += vloss.numel()

            val_loss = (val_loss_sum / val_count).item()
            swanlab.log({"val/loss": val_loss}, step=int(cur_nimg))
            logger.info(f"val/loss: {val_loss:.6f}")

            # 生成 & 评估
            generate_and_evaluate(ema, val_ds, device, cfg, step=int(cur_nimg), logger=logger)

        # Save network snapshot.
        # 保存网络快照。
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(
                ema=copy.deepcopy(ema).eval().requires_grad_(False).cpu(),
                loss_fn=copy.deepcopy(loss_fn),
                model_args=model_args,
                interface_args=interface_args,
                optimizer_kwargs=optimizer_kwargs,
            )
            snapshot_pkl = os.path.join(checkpoint_dir, f"network-snapshot-{cur_nimg//1000:06d}.pkl")
            with open(snapshot_pkl, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Saved snapshot: {snapshot_pkl}")
            del data  # 释放内存

        # Save full dump of the training state.
        # 保存完整的训练状态。
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0:
            state_path = os.path.join(checkpoint_dir, f"training-state-{cur_nimg//1000:06d}.pt")
            torch.save(
                dict(
                    model=model.state_dict(),
                    ema=ema.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    cur_nimg=cur_nimg,
                    cur_tick=cur_tick,
                ),
                state_path,
            )
            logger.info(f"Saved training state: {state_path}")

        # Update state.
        # 更新状态。
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
    # 完成训练
    logger.info("Done.")


if __name__ == "__main__":
    # 设置全局seed
    torch.manual_seed(42)
    np.random.seed(42)
    main()

"""
# 正式训练脚本。
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part1_US_Rocky.yaml
"""
