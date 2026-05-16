import os
import time
import torch
import torch
import pynvml
import numpy as np


def get_gpu_memory_detailed():
    """
    只读取 CUDA_VISIBLE_DEVICES 指定的那一张卡的显存信息。
    如果环境变量没设，就默认读第 0 号卡。
    """
    pynvml.nvmlInit()

    # 1. 取出可见列表里的第一张卡
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    idx_in_nvml = int(visible[0])  # 在 NVML 里的物理编号

    handle = pynvml.nvmlDeviceGetHandleByIndex(idx_in_nvml)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    print(f"GPU {idx_in_nvml} (CUDA_VISIBLE_DEVICES={visible[0]})")
    print(f"  Total : {info.total/1024**3:.2f} GB")
    print(f"  Used  : {info.used/1024**3:.2f} GB")
    print(f"  Free  : {info.free/1024**3:.2f} GB")
    print(f"  Usage : {info.used/info.total*100:.1f}%")

    pynvml.nvmlShutdown()
    return info.free / 1024**3  # 返回空闲 GB


def occupy_gpu_memory(target_gb):
    """占用指定GB数的GPU显存"""
    bytes_per_gb = 1024**3
    target_bytes = int(target_gb * bytes_per_gb)

    # 计算需要的元素数量（float32每个元素4字节）
    num_elements = target_bytes // 4

    # 创建tensor
    dummy_tensor = torch.empty(num_elements, device="cuda", dtype=torch.float32)

    # 实际占用的显存
    actual_gb = dummy_tensor.element_size() * dummy_tensor.nelement() / (1024**3)
    print(f"目标占用: {target_gb} GB")
    print(f"实际占用: {actual_gb:.2f} GB")

    return dummy_tensor


def occupy_available_gpu_memory(reserve_gb=1):
    """占用除保留量外的所有可用显存"""
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return None

    # 获取当前GPU信息
    free_gb = get_gpu_memory_detailed()

    # 计算要占用的显存（保留指定GB）
    target_occupy_gb = max(0, free_gb - reserve_gb)

    if target_occupy_gb > 0:
        dummy_tensor = occupy_gpu_memory(target_occupy_gb)
        return dummy_tensor
    else:
        print("可用显存不足")
        return None


if __name__ == "__main__":
    # 占用几乎所有可用显存（保留1GB）
    dummy_tensor = occupy_available_gpu_memory(reserve_gb=1)
    print("GPU memory reserved.")

    time.sleep(1)  # 模拟一些计算过程

    del dummy_tensor
    torch.cuda.empty_cache()
