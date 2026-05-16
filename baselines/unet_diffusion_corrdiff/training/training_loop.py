"""Main training loop."""

# 降尺度主训练循环。

import os
import time
import copy
import json
import pickle
import psutil
import torch
import swanlab
import numpy as np
from datetime import datetime

# 自定义模块
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.occupy_gpu_memory import occupy_available_gpu_memory
from training.validation import validate

# ----------------------------------------------------------------------------


def training_loop(
    run_dir=".",  # 输出目录。
    dataset_kwargs={},  # 训练集配置选项。
    validation_dataset_kwargs=None,  # 验证集配置选项，None 表示禁用。
    data_loader_kwargs={},  # DataLoader 配置选项。
    network_kwargs={},  # 模型与预处理配置。
    loss_kwargs={},  # 损失函数配置。
    optimizer_kwargs={},  # 优化器配置。
    generation_kwargs={},  # 生成配置选项。
    augment_kwargs=None,  # 数据增强管线配置，None 表示禁用。
    seed=0,  # 全局随机种子。
    batch_size=512,  # 单次训练迭代的总批大小。
    batch_gpu=None,  # 每块 GPU 的批大小上限，None 表示不限制。
    total_kimg=200000,  # 训练时长（以千张图像计）。
    ema_halflife_kimg=500,  # 模型权重 EMA 的半衰期（千张图像）。
    ema_rampup_ratio=0.05,  # EMA 预热系数，None 表示不预热。
    lr_rampup_kimg=10000,  # 学习率预热时长。
    loss_scaling=1,  # 损失缩放系数，用于降低 FP16 上下溢。
    kimg_per_tick=50,  # 进度打印间隔（千张图像）。
    snapshot_ticks=50,  # 网络快照保存频率，None 表示禁用。
    state_dump_ticks=500,  # 训练状态转储频率，None 表示禁用。
    resume_pkl=None,  # 从指定网络快照开始训练，None 表示随机初始化。
    resume_state_dump=None,  # 从指定训练状态开始，None 表示重置训练状态。
    resume_kimg=0,  # 从指定的训练进度开始。
    cudnn_benchmark=True,  # 是否启用 torch.backends.cudnn.benchmark。
    device=torch.device("cuda"),
    **kwargs,  # 其他参数（未使用）
):
    # Initialize.
    # 初始化。
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    # 选择每块 GPU 的批大小。
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    ########################
    #  1. 数据集
    ########################

    # 占用显存
    if OCCUPY_GPU_MEMORY := True:
        dummy_tensor = occupy_available_gpu_memory(reserve_gb=5)
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}][INFO] GPU memory reserved.")
    dist.print0("加载训练集...")
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs
        )
    )

    # 加载验证集
    if validation_dataset_kwargs is not None:
        dist.print0("加载验证集...")
        val_dataset_obj = dnnlib.util.construct_class_by_name(**validation_dataset_kwargs)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset_obj, batch_size=batch_gpu, **data_loader_kwargs)
    else:
        val_dataset_obj = None
        val_loader = None

    # 释放显存
    if OCCUPY_GPU_MEMORY:
        del dummy_tensor
        torch.cuda.empty_cache()
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}][INFO] GPU memory released.")

    # 打印 网络输入输出分辨率和通道数
    img_resolution = dataset_obj.image_shape()  # [H, W]
    img_in_channels = len(dataset_obj.input_channels())
    img_out_channels = len(dataset_obj.output_channels())
    dist.print0(f"img_resolution:          {img_resolution}")
    dist.print0(f"img_in_channels:         {img_in_channels}")
    dist.print0(f"img_out_channels:        {img_out_channels}")

    ####################################
    #  2. 模型 & 损失函数 & 优化器
    ####################################
    # Construct network.
    # 构建网络。
    dist.print0("Constructing network...")

    hr_mean_conditioning = loss_kwargs.get("hr_mean_conditioning", False)
    if hr_mean_conditioning:
        # 将 y_mean 与 y_lr 按通道连接 输入去噪网络
        img_in_channels = img_in_channels + img_out_channels

    interface_kwargs = dict(
        img_resolution=img_resolution,
        img_in_channels=img_in_channels,
        img_out_channels=img_out_channels,
    )
    # EDMPrecondSuperResolution: subclass of torch.nn.Module
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    # 设置优化器。
    dist.print0("Setting up optimizer...")
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)  # training.loss.(VP|VE|EDM)Loss
    # subclass of torch.optim.Optimizer
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    # training.augment.AugmentPipe
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    # 从先前的快照恢复训练。
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(src_module=data["ema"], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data["ema"], dst_module=ema, require_all=False)
        del data  # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"), weights_only=False)
        misc.copy_params_and_buffers(src_module=data["net"], dst_module=net, require_all=True)
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    ####################################
    #  3. 训练 & 验证
    ####################################
    # Train.
    # 开始训练。
    dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:

        # 记录每个 total_batch 每个rank的 loss
        loss_sum_local = torch.tensor(0.0, device=device)
        count_local = torch.tensor(0.0, device=device)
        ddp.train()

        # Accumulate gradients. 累积梯度。
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                # 获取数据, Dataset returns (hr, lr)
                images, lr_images = next(dataset_iterator)
                images = images.to(device)
                lr_images = lr_images.to(device)

                # Loss
                loss = loss_fn(net=ddp, img_clean=images, img_lr=lr_images, labels=None, augment_pipe=None)
                training_stats.report("Loss/loss", loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

                loss_sum_local += loss.detach().float().sum()
                count_local += loss.numel()  # 元素个数 [B, H, W] -> B*H*W

        # Update weights.
        # 更新权重。
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        # 更新 EMA: Exponential Moving Average（指数移动平均）
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # --- 这里：把本地 sum/count all_reduce 成全局平均 ---
        loss_sum = loss_sum_local
        cnt = torch.tensor(count_local, device=device, dtype=torch.float32)
        if dist.get_world_size() > 1:
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(cnt, op=torch.distributed.ReduceOp.SUM)

        if dist.get_rank() == 0:
            loss_mean_global = (loss_sum / cnt).item()
            swanlab.log({"train/loss": loss_mean_global}, step=int(cur_nimg))

        # Perform maintenance tasks once per tick.
        # 每个 tick 执行一次维护任务。
        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        # 打印状态行，并在 training_stats 中记录相同信息。
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

        # Check for abort.
        # 检查是否中止。
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # 验证 validation, 使用和保存网络快照相同的频率
        if (val_dataset_obj is not None) and (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            ema.eval()  # 评估模式
            val_loss_sum = torch.tensor(0.0, device=device)
            val_count = torch.tensor(0.0, device=device)
            with torch.no_grad():
                for images, lr_images in val_loader:
                    images = images.to(device)
                    lr_images = lr_images.to(device)

                    loss = loss_fn(net=ema, img_clean=images, img_lr=lr_images, labels=None, augment_pipe=None)

                    val_loss_sum += loss.sum()
                    val_count += loss.numel()

            if dist.get_world_size() > 1:
                torch.distributed.all_reduce(val_loss_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(val_count, op=torch.distributed.ReduceOp.SUM)

            if dist.get_rank() == 0:
                val_loss_mean = (val_loss_sum / val_count).item()
                swanlab.log({"val/loss": val_loss_mean}, step=int(cur_nimg))
                dist.print0(f"val/loss: {val_loss_mean:.4f}")

            """
            使用训练好的模型进行生成，并评估生成结果。
            """
            validate(
                model=ema,
                val_ds=val_dataset_obj,
                loss_kwargs=loss_kwargs,
                generation_kwargs=generation_kwargs,
                device=device,
                step=int(cur_nimg),
                dist=dist,
                num_workers=data_loader_kwargs.get("num_workers", 1),
            )

        # Save network snapshot.
        # 保存网络快照。
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f"network-snapshot-{cur_nimg//1000:06d}.pkl"), "wb") as f:
                    pickle.dump(data, f)
            del data  # conserve memory

        # Save full dump of the training state.
        # 保存完整的训练状态。
        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
            and dist.get_rank() == 0
        ):
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"training-state-{cur_nimg//1000:06d}.pt"),
            )

        # Update logs.
        # 更新日志。
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(
                json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + "\n"
            )
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        # 更新状态。
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    # 完成。
    dist.print0()
    dist.print0("Exiting...")


# ----------------------------------------------------------------------------
