import os
import hydra
import pickle
import warnings
import numpy as np
import setproctitle
import xarray as xr
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from physicsnemo.launch.logging import PythonLogger
from physicsnemo.utils.corrdiff.utils import regression_step

# 自定义模块
from dataset import DATASETS
from models import UNetBlock, CorrDiffRegressionUNet, PhysicsnemoSongUNet

warnings.filterwarnings("ignore", category=UserWarning)


def instantiate_dataset(cfg_group, dataset_type):
    """把 OmegaConf 转 dict，去掉 type 后动态实例化 Dataset"""
    kwargs = OmegaConf.to_container(cfg_group, resolve=True)
    kwargs.pop("type", None)  # 去除 type 字段
    return dataset_type(**kwargs)


@hydra.main(version_base=None, config_path="conf/regression")
def main(cfg: DictConfig):
    """
    降尺度回归模型推理脚本。
    """
    # ---------------- 配置验证区域 ----------
    dataset_cls = DATASETS[cfg.dataset.type]
    # ---------------- 配置验证区域 ----------
    # 初始化 logger
    os.makedirs(f"logs/{Path(__file__).stem}", exist_ok=True)  # 确保日志目录存在
    logger = PythonLogger("generate_regression")  # General python logger
    logger.file_logging(f"logs/{Path(__file__).stem}/{datetime.now().strftime('%y%m%d_%H%M%S')}.log")

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 验证 checkpoint 存在性
    ckpt_path = cfg.generation.checkpoint_dir
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}")
        raise FileNotFoundError(f"[{datetime.now():%Y-%m-%d %H:%M:%S}][Error] Checkpoint not found at {ckpt_path}")

    # 验证 checkpoint 以 pkl 结尾
    if not ckpt_path.endswith(".pkl"):
        logger.error(f"Checkpoint file must have a .pkl extension, got {ckpt_path}")
        raise ValueError(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}][Error] Checkpoint file must have a .pkl extension, got {ckpt_path}"
        )

    # 1. 数据集
    logger.info(f"初始化数据集...")
    dataset_cls = DATASETS[cfg.dataset.type]
    dataset = instantiate_dataset(cfg.dataset, dataset_cls)

    img_shape = dataset.image_shape()  # (x, y)
    img_in_channels = len(dataset.input_channels())
    img_out_channels = len(dataset.output_channels())

    # 2. 样本采样
    total_samples = len(dataset)
    max_samples = cfg.generation.get("max_samples", total_samples)

    if max_samples < total_samples:
        # 获取均匀采样的索引
        indices = np.linspace(0, total_samples - 1, max_samples, dtype=int)
        logger.info(f"Subsampling {max_samples} samples from {total_samples} total samples.")
    else:
        indices = np.arange(total_samples)
        logger.info(f"Using all {total_samples} samples.")

    # batch_size=1 每次只取1个样本进行生成
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=cfg.generation.num_workers)

    # 5. 模型, 加载 checkpoint
    logger.info(f"Loading model...")
    logger.info(f"Loading model from checkpoint={ckpt_path}")
    with open(ckpt_path, "rb") as f:
        model = pickle.load(f)["ema"]
    model.eval().to(device)
    logger.info(f"Loaded model type: {type(model)}")

    if use_torch_compile := cfg.generation.get("use_torch_compile", False):
        logger.info(f"Compiling model with torch.compile...")
        model = torch.compile(model)

    # 4. 推理, 生成样本
    logger.info(f"Starting inference...")
    time_start = datetime.now()
    inputs_list = []
    truths_list = []
    preds_list = []

    latents_shape = (1, img_out_channels, img_shape[0], img_shape[1])  # (1, C_out, H, W)
    zero_input = torch.zeros(latents_shape, device=device)

    with torch.inference_mode():
        for i, (batch_y, batch_x) in enumerate(loader):
            batch_x = batch_x.to(device)  # img_lr: [1, C_in, H, W]
            batch_y = batch_y.to(device)  # img_clean: [1, C_out, H, W]

            pred = model(x=zero_input, img_lr=batch_x)  # [1, C_out, H, W]

            # 逆归一化
            input_phy = dataset.denormalize_input_tensor(batch_x)
            truth_phy = dataset.denormalize_output_tensor(batch_y)
            pred_phy = dataset.denormalize_output_tensor(pred)

            inputs_list.append(input_phy.cpu().numpy())
            truths_list.append(truth_phy.cpu().numpy())
            preds_list.append(pred_phy.cpu().numpy())

            if (i + 1) % 10 == 0:
                elapsed = (datetime.now() - time_start).total_seconds()
                time_start = datetime.now()
                remaining = (elapsed / 10) * (len(loader) - (i + 1))  # 预计剩余时间
                logger.info(f"进度 {i + 1}/{len(loader)}, {elapsed/10:.1f} s/个, 剩余: {remaining/3600:.2f} 小时")

    # 合并所有批次
    inputs_arr = np.concatenate(inputs_list, axis=0)  # [N, C_in, H, W]
    truths_arr = np.concatenate(truths_list, axis=0)  # [N, C_out, H, W]
    preds_arr = np.concatenate(preds_list, axis=0)  # [N, C_out, H, W]

    # 添加预测的集合 ensemble 维度: [N, 1, C_out, H, W]
    preds_arr = np.expand_dims(preds_arr, axis=1)

    logger.info(f"{inputs_arr.shape=}")
    logger.info(f"{truths_arr.shape=}")
    logger.info(f"{preds_arr.shape=}")

    # 5. 保存
    logger.info(f"Saving results to NetCDF...")

    # Coordinates
    selected_times = dataset.times[indices]
    image_shape = dataset.image_shape()
    lats = range(image_shape[0])
    lons = range(image_shape[1])

    # Input Group
    if False:
        ds_input = xr.Dataset(
            data_vars={
                v.name: (("time", "y", "x"), inputs_arr[:, i, :, :]) for i, v in enumerate(dataset.input_channels())
            },
            coords={"time": selected_times, "y": lats, "x": lons},
        )

    # Truth Group
    if False:
        ds_truth = xr.Dataset(
            data_vars={
                v.name: (("time", "y", "x"), truths_arr[:, i, :, :]) for i, v in enumerate(dataset.output_channels())
            },
            coords={"time": selected_times, "y": lats, "x": lons},
        )

    # Prediction Group
    ds_pred = xr.Dataset(
        data_vars={
            v.name: (("time", "ensemble", "y", "x"), preds_arr[:, :, i, :, :])
            for i, v in enumerate(dataset.output_channels())
        },
        coords={"time": selected_times, "ensemble": [0], "y": lats, "x": lons},
    )

    ckpt_name = Path(ckpt_path).stem
    dataset_type = cfg.dataset.type
    dataset_name = os.path.basename(os.path.dirname(cfg.dataset.input_path))  # 数据集名称 part1-US-Rocky
    output_base_path = cfg.generation.output_path

    # 解决输出目录无法区分不同输入/不同checkpoint的问题：
    # 1) input 文件名区分 test_data_in.nc vs test_data_in_ERA5.nc
    # 2) checkpoint 所在的训练 run 目录名区分不同数据集/不同训练配置得到的模型
    input_tag = Path(cfg.dataset.input_path).stem
    ckpt_run_tag = Path(ckpt_path).parent.name or "checkpoint"

    output_path = os.path.join(
        output_base_path,
        dataset_name,
        input_tag,
        ckpt_run_tag,
        f"generation-{ckpt_name}.nc",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 删除已存在的文件以避免组的追加问题
    if os.path.exists(output_path):
        os.remove(output_path)

    # 保存 groups
    # ds_input.to_netcdf(output_path, group="input", mode="w")
    # ds_truth.to_netcdf(output_path, group="truth", mode="a")
    # ds_pred.to_netcdf(output_path, group="prediction", mode="a")
    ds_pred.to_netcdf(output_path, mode="w")  # 只保存预测结果，输入和真实值可以通过原数据集获取

    logger.info(f"保存生成结果到 {output_path}")


if __name__ == "__main__":
    # 设置全局seed
    torch.manual_seed(42)
    np.random.seed(42)
    main()

"""
# 推理
# 指定 checkpoint_dir 为训练好的模型权重路径, 以 .pkl 结尾
CUDA_VISIBLE_DEVICES=0  python generate_regression.py --config-name=config_generate_hrrr_era5.yaml \
    generation.checkpoint_dir=""

"""
