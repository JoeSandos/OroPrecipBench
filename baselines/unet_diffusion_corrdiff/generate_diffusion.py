import os
import hydra
import torch
import pickle
import warnings
import numpy as np
import xarray as xr
import setproctitle
from pathlib import Path

from datetime import datetime
from functools import partial
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from physicsnemo import Module
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.corrdiff.utils import diffusion_step
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.models.diffusion.preconditioning import EDMPrecondSuperResolution
from physicsnemo.utils.diffusion import deterministic_sampler, stochastic_sampler

# 自定义模块
from dataset import DATASETS
from models import UNetBlock
from models import PhysicsnemoSongUNet, DhariwalUNet
from models import EDMPrecondSuperResolution

warnings.filterwarnings("ignore", category=UserWarning)


def instantiate_dataset(cfg_group, dataset_type):
    kwargs = OmegaConf.to_container(cfg_group, resolve=True)
    kwargs.pop("type", None)
    return dataset_type(**kwargs)


def generate_fn(model, sampler_fn, img_shape, img_out_channels, rank_batches, img_lr, dist):
    pred = diffusion_step(
        net=model,
        sampler_fn=sampler_fn,
        img_shape=img_shape,
        img_out_channels=img_out_channels,
        rank_batches=rank_batches,
        img_lr=img_lr,
        rank=dist.rank,
        device=dist.device,
    )

    # 使用 gather 到主进程
    if dist.world_size > 1:
        # 准备收集列表
        if dist.rank == 0:
            output_list = [
                torch.zeros_like(
                    pred,
                    dtype=pred.dtype,
                    device=pred.device,
                )
                for _ in range(dist.world_size)
            ]
        else:
            output_list = None

        # 收集所有进程的 pred
        torch.distributed.gather(pred, output_list, dst=0)

        if dist.rank == 0:
            # 合并所有进程的结果
            gatherd_preds = torch.cat(output_list, dim=0)
            return gatherd_preds
        else:
            return None
    else:
        return pred


@hydra.main(version_base=None, config_path="conf/diffusion", config_name="main")
def main(cfg: DictConfig):
    # ---------------- 配置验证区域 ----------
    dataset_cls = DATASETS[cfg.dataset.type]
    # ---------------- 配置验证区域 ----------

    # 验证 checkpoint 存在性
    ckpt_path = cfg.generation.checkpoint_dir
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[{datetime.now():%Y-%m-%d %H:%M:%S}][Error] Checkpoint not found at {ckpt_path}")

    # 验证 checkpoint 以 pkl 结尾
    if not ckpt_path.endswith(".pkl"):
        raise ValueError(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}][Error] Checkpoint file must have a .pkl extension, got {ckpt_path}"
        )

    # 1. 初始化分布式管理器
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # 初始化 logger
    os.makedirs(f"logs/{Path(__file__).stem}", exist_ok=True)  # 确保日志目录存在
    logger = PythonLogger("generate_diffusion")  # General python logger
    logger.file_logging(f"logs/{Path(__file__).stem}/{datetime.now().strftime('%y%m%d_%H%M%S')}.log")
    logger0 = RankZeroLoggingWrapper(logger, dist=dist)

    logger0.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    # 2. 划分批次
    num_ensembles = cfg.generation.num_ensembles  # 每个低分辨率样本生成的高分辨率样本数量
    seed_batch_size = cfg.generation.seed_batch_size  # 每个批次的种子数量
    seeds = list(np.arange(num_ensembles))
    num_batches = ((len(seeds) - 1) // (seed_batch_size * dist.world_size) + 1) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # 同步, 确保所有进程在继续之前都完成了批次划分
    if dist.world_size > 1:
        torch.distributed.barrier()

    # 3. 数据集
    logger0.info(f"初始化数据集...")
    dataset_cls = DATASETS[cfg.dataset.type]
    dataset = instantiate_dataset(cfg.dataset, dataset_cls)

    img_shape = dataset.image_shape()  # (x, y)
    img_in_channels = len(dataset.input_channels())
    img_out_channels = len(dataset.output_channels())

    # 4. 样本采样
    total_samples = len(dataset)
    max_samples = cfg.generation.get("max_samples", total_samples)

    if max_samples < total_samples:
        # 获取均匀采样的索引
        indices = np.linspace(0, total_samples - 1, max_samples, dtype=int)
        logger0.info(f"Subsampling {max_samples} / {total_samples}")
    else:
        indices = np.arange(total_samples)
        logger0.info(f"Using all {total_samples} samples.")

    # batch_size=1 每次只取1个样本进行生成
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=cfg.generation.num_workers)

    # 5. 模型, 加载 checkpoint
    logger0.info(f"Loading model...")
    logger0.info(f"Loading model from checkpoint={ckpt_path}")
    with open(ckpt_path, "rb") as f:
        model = pickle.load(f)["ema"]
    model.eval().to(device)
    logger0.info(f"Loaded model type: {type(model)}")

    if use_torch_compile := cfg.generation.get("use_torch_compile", False):
        logger0.info(f"Compiling model with torch.compile...")
        model = torch.compile(model)

    # 同步, 确保所有进程在继续之前都完成了 初始化数据集, 模型加载
    if dist.world_size > 1:
        torch.distributed.barrier()

    # 6. 采样器
    sigma_min, sigma_max = 0.002, 80  # EDM论文推荐值
    if cfg.generation.sampler_type == "deterministic":
        # 确定性采样
        sampler_fn = partial(
            deterministic_sampler,
            num_steps=9,
            solver="euler",  # [euler, heun]
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    elif cfg.generation.sampler_type == "stochastic":
        # 随机采样
        sampler_fn = partial(
            stochastic_sampler,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    else:
        raise ValueError(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}][Error] Unknown sampler_type: {cfg.generation.sampler_type}"
        )

    # 4. 推理, 生成样本
    time_start = datetime.now()
    inputs_list = []
    truths_list = []
    preds_list = []

    with torch.inference_mode():
        for i, (batch_y, batch_x) in enumerate(loader):
            batch_x = batch_x.to(device)  # [1, C_in, H, W]
            batch_y = batch_y.to(device)  # [1, C_out, H, W]

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

            if dist.rank != 0:
                continue  # 仅主进程处理结果

            # 逆归一化
            input_phy = dataset.denormalize_input_tensor(batch_x)  # [1, C_in, H, W]
            truth_phy = dataset.denormalize_output_tensor(batch_y)  # [1, C_out, H, W]
            pred_phy = dataset.denormalize_output_tensor(pred)  # [num_ensembles, C_out, H, W]

            # pred_phy 添加维度
            pred_phy = pred_phy.unsqueeze(0)  # [1, num_ensembles, C_out, H, W]

            inputs_list.append(input_phy.cpu().numpy())
            truths_list.append(truth_phy.cpu().numpy())
            preds_list.append(pred_phy.cpu().numpy())

            if (i + 1) % 10 == 0:
                elapsed = (datetime.now() - time_start).total_seconds()
                time_start = datetime.now()
                remaining = (elapsed / 10) * (len(loader) - (i + 1))  # 预计剩余时间
                logger0.info(f"进度 {i + 1}/{len(loader)}, {elapsed/10:.1f} s/个, 剩余: {remaining/3600:.2f} 小时")

    if dist.rank != 0:
        return  # 仅主进程继续

    # 合并所有批次
    inputs_arr = np.concatenate(inputs_list, axis=0)  # [N, C_in, H, W]
    truths_arr = np.concatenate(truths_list, axis=0)  # [N, C_out, H, W]
    preds_arr = np.concatenate(preds_list, axis=0)  # [N, num_ensembles, C_out, H, W]

    logger0.info(f"{inputs_arr.shape=}")
    logger0.info(f"{truths_arr.shape=}")
    logger0.info(f"{preds_arr.shape=}")

    # 5. 保存
    logger0.info(f"Saving results to NetCDF...")

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
        coords={"time": selected_times, "ensemble": range(num_ensembles), "y": lats, "x": lons},
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

    logger0.info(f"保存生成结果到 {output_path}")


if __name__ == "__main__":
    # 设置全局seed
    torch.manual_seed(42)
    np.random.seed(42)
    main()


"""
# 运行脚本示例：

## 单进程
python generate_diffusion.py --config-name=config_training_part1_US_Rocky.yaml

## 多进程
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_training_part1_US_Rocky.yaml



"""
