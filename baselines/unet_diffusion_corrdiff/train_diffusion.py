import os
import re
import json
import torch
import hydra
import swanlab
import warnings
import numpy as np
from setproctitle import setproctitle
from omegaconf import DictConfig, OmegaConf

# 自定义模块
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
from models import EDMPrecondSuperResolution
from training.loss import EDMLossSR
from training.augment import AugmentPipe
from dataset import DATASETS_import_path as DATASETS



# 忽略 PyTorch 1.12 打印的误报警告：“梯度步幅与桶视图步幅不匹配”。
# False warning printed by PyTorch 1.12.
warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")
warnings.filterwarnings("ignore", category=SyntaxWarning)


@hydra.main(version_base=None, config_path="conf/diffusion")
def main(opts: DictConfig):
    """
    降尺度扩散模型训练脚本。
    """
    # ---------------- 配置验证区域 ----------
    assert opts.batch >= opts.batch_gpu, "batch must be >= batch_gpu"
    assert opts.batch % opts.batch_gpu == 0, "batch must be divisible by batch_gpu"
    # ---------------- 配置验证区域 ----------

    # 初始化分布式训练
    torch.multiprocessing.set_start_method("spawn")  # spawn: 跨平台,独立的内存空间; fork: Linux,父进程和子进程共享内存
    dist.init()

    # Initialize config dict.
    # 初始化配置字典。
    c = dnnlib.EasyDict()

    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    # c.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", lr=opts.lr, betas=[0.9, 0.999], eps=1e-8)
    c.optimizer_kwargs = dnnlib.EasyDict(OmegaConf.to_container(opts.optim, resolve=True))
    c.generation_kwargs = dnnlib.EasyDict(OmegaConf.to_container(opts.generation, resolve=True))

    # 数据集配置
    c.dataset_kwargs = dnnlib.EasyDict(
        class_name=DATASETS[opts.dataset.type],
        input_path=opts.dataset.input_path,
        output_path=opts.dataset.output_path,
        invariant_path=opts.dataset.invariant_path,
        stats_path=opts.dataset.stats_path,
        input_variables=OmegaConf.to_container(opts.dataset.input_variables, resolve=True),
        output_variables=OmegaConf.to_container(opts.dataset.output_variables, resolve=True),
        invariant_variables=OmegaConf.to_container(opts.dataset.invariant_variables, resolve=True),
        input_log_variables=OmegaConf.to_container(opts.dataset.input_log_variables, resolve=True),
        output_log_variables=OmegaConf.to_container(opts.dataset.output_log_variables, resolve=True),
        max_samples=opts.dataset.get("max_samples", 1000000),
    )

    # 验证集配置
    if "validation" in opts:
        val_opts = OmegaConf.merge(opts.dataset, opts.get("validation", {}))
        c.validation_dataset_kwargs = dnnlib.EasyDict(
            class_name=DATASETS[val_opts.type],
            input_path=opts.dataset.input_path,
            output_path=opts.dataset.output_path,
            invariant_path=opts.dataset.invariant_path,
            stats_path=opts.dataset.stats_path,
            input_variables=OmegaConf.to_container(val_opts.input_variables, resolve=True),
            output_variables=OmegaConf.to_container(val_opts.output_variables, resolve=True),
            invariant_variables=OmegaConf.to_container(val_opts.invariant_variables, resolve=True),
            input_log_variables=OmegaConf.to_container(val_opts.input_log_variables, resolve=True),
            output_log_variables=OmegaConf.to_container(val_opts.output_log_variables, resolve=True),
            max_samples=opts.dataset.get("max_samples", 1000000),
        )
    else:
        c.validation_dataset_kwargs = None

    # Network architecture.
    # 网络架构。
    if opts.arch == "ddpmpp":
        c.network_kwargs.update(
            model_type="PhysicsnemoSongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    elif opts.arch == "ncsnpp":
        c.network_kwargs.update(
            model_type="PhysicsnemoSongUNet",
            embedding_type="fourier",
            encoder_type="residual",
            decoder_type="standard",
            channel_mult_noise=2,
            resample_filter=[1, 3, 3, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    elif opts.arch == "adm":
        c.network_kwargs.update(model_type="DhariwalUNet", model_channels=192, channel_mult=[1, 2, 3, 4])
    else:
        raise ValueError(f"--arch: unknown architecture '{opts.arch}'")

    # Preconditioning & loss function.
    # 预处理与损失函数。
    c.network_kwargs.class_name = "models.EDMPrecondSuperResolution"
    c.loss_kwargs.class_name = "training.loss.EDMLossSR"

    # Network options.
    # 网络选项。
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        # 数据增强
        c.augment_kwargs = dnnlib.EasyDict(class_name="training.augment.AugmentPipe", p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    # 设置 dropout 和 fp16 选项
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    # 训练选项。
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    # 随机种子。
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    # 迁移学习与恢复训练。
    if opts.transfer is not None:
        if opts.resume is not None:
            raise ValueError("--transfer and --resume cannot be specified at the same time")
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r"training-state-(\d+).pt", os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise ValueError("--resume must point to training-state-*.pt from a previous training run")
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f"network-snapshot-{match.group(1)}.pkl")
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    # 描述字符串。
    task_str = "DownScale"
    dtype_str = "fp16" if c.network_kwargs.use_fp16 else "fp32"
    dataset_name = os.path.basename(os.path.dirname(opts.dataset.input_path))  # 数据集名称 part1-US-Rocky
    desc = f"{opts.dataset.type:s}-{dataset_name:s}-{opts.arch:s}-EDMPrecondSR-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}"
    if opts.desc is not None:
        desc += f"-{opts.desc}"

    # Pick output directory.
    # 选择输出目录。
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f"{cur_run_id:05d}-{desc}")
        assert not os.path.exists(c.run_dir)

    # Print options.
    # 打印选项。
    dist.print0()
    dist.print0("Training options:")
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f"Output directory:        {c.run_dir}")
    dist.print0(f"Dataset type:            {opts.dataset.type}")
    dist.print0(f"Dataset name:            {dataset_name}")
    dist.print0(f"Network architecture:    {opts.arch}")
    dist.print0(f"Number of GPUs:          {dist.get_world_size()}")
    dist.print0(f"Batch size:              {c.batch_size}")
    dist.print0(f"Mixed-precision:         {c.network_kwargs.use_fp16}")
    dist.print0(f"Preconditioning & loss:  EDMPrecondSuperResolution & EDMLossSR")
    dist.print0()

    # 初始化 swanlab 记录器
    if dist.get_rank() == 0:
        swanlab.init(
            project="Benchmark-NIPS26",
            experiment_name=f"Diffusion-{dataset_name}",
            config=c,
            logdir="outputs/swanlog",
        )
        swanlab.config.architecture = opts.arch
        swanlab.config.precond = "EDMPrecondSuperResolution"
        swanlab.config.GPUs = dist.get_world_size()

    # Dry run?
    # 仅测试运行（不训练）？
    if opts.dry_run:
        dist.print0("Dry run; exiting.")
        return

    # Create output directory.
    # 创建输出目录。
    dist.print0("Creating output directory...")
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, "log.txt"), file_mode="a", should_flush=True)

    # Train. 开始训练。
    training_loop.training_loop(**c)


if __name__ == "__main__":
    # 设置全局seed
    torch.manual_seed(42)
    np.random.seed(42)
    main()
"""
# 正式训练脚本。
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part1_US_Rocky.yaml
"""
