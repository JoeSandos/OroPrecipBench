import os
import json
import torch
import numpy as np
import xarray as xr
import pandas as pd

# from termcolor import colored
from typing import List, Tuple, Union, Literal
from datetime import datetime, timezone

# from physicsnemo.utils.diffusion import convert_datetime_to_cftime

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class ChannelMetadata:
    """Metadata describing a data channel."""

    name: str
    level: str = ""
    auxiliary: bool = False


class DownscalingDataset(torch.utils.data.Dataset, ABC):
    """Abstract class for dataset with downscaling paired data

    A DownscalingDataset has high-resolution output data (target) paired with
    low-resolution input data.
    """

    @abstractmethod
    def longitude(self) -> np.ndarray:
        """Get longitude values from the dataset."""
        pass

    @abstractmethod
    def latitude(self) -> np.ndarray:
        """Get latitude values from the dataset."""
        pass

    @abstractmethod
    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        pass

    @abstractmethod
    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""
        pass

    @abstractmethod
    def time(self) -> List:
        """Get time values from the dataset."""
        pass

    @abstractmethod
    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        pass

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return x

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return x

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x

    def info(self) -> dict:
        """Get information about the dataset."""
        return {}


class MyDataset(DownscalingDataset):
    """
    降尺度数据集

    输入:  LR 24×24
    输出:  HR 192×192
    不变量: 与输出同分辨率 HR 192×192

    参数:

        - input_path: 输入文件路径, 例如 path/to/dataset_root/part1-US-Rocky/train_data_in.nc
        - output_path: 输出文件路径, 例如 path/to/dataset_root/part1-US-Rocky/train_data_out.nc
        - invariant_path: 不变量文件路径, 例如 path/to/dataset_root/part1-US-Rocky/invariant.nc
        - stats_path: 统计文件路径, 例如 path/to/dataset_root/part1-US-Rocky/stats.json

        - input_variables: 输入变量列表, 默认 None 表示使用全部变量
            default_vars = ["tp", "t2m", "u10", "v10"]

        - output_variables: 输出变量列表, 默认 ["tp"]

        - invariant_variables: 不变量列表, 默认 ["elevation", "lsm"]

        - input_log_variables: 输入变量中需要对数变换的变量列表, 默认 ["tp"]
        - output_log_variables: 输出变量中需要对数变换的变量列表, 默认 ["tp"]
        - log_eps: 对数变换的平滑参数, 默认 1
        - max_samples: 最大样本数
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        invariant_path: str,
        stats_path: str,
        input_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = ["tp"],
        invariant_variables: Union[List[str], None] = ["elevation", "lsm"],
        input_log_variables: Union[List[str], None] = ["tp"],
        output_log_variables: Union[List[str], None] = ["tp"],
        log_eps: float = 1,
        max_samples: int = 1_000_000_000,
    ):
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}][INFO] MyDataset 初始化...")

        self.input_variables = input_variables
        self.output_variables = output_variables
        self.invariant_variables = invariant_variables or []
        self.input_log_variables = input_log_variables or []
        self.output_log_variables = output_log_variables or []
        self.log_eps = log_eps
        self.max_samples = max_samples

        # 文件路径
        self.file_input = input_path
        self.file_output = output_path
        self.file_invariant = invariant_path
        self.stats_path = stats_path

        # 验证文件
        assert os.path.exists(self.file_input), f"输入文件不存在: {self.file_input}"
        assert os.path.exists(self.file_output), f"输出文件不存在: {self.file_output}"
        assert os.path.exists(self.file_invariant), f"不变量文件不存在: {self.file_invariant}"
        assert os.path.exists(self.stats_path), f"统计文件不存在: {self.stats_path}"

        # 加载数据
        ds_input = xr.open_dataset(self.file_input)
        ds_output = xr.open_dataset(self.file_output)
        ds_inv = xr.open_dataset(self.file_invariant)

        # 如果变量列表为 None, 使用全部变量
        if self.input_variables is None:
            self.input_variables = list(ds_input.data_vars)
        if self.output_variables is None:
            self.output_variables = list(ds_output.data_vars)

        # 选择变量
        ds_input = ds_input[self.input_variables]
        ds_output = ds_output[self.output_variables]
        ds_inv = ds_inv[self.invariant_variables]

        # tp 索引
        if "tp" in self.input_variables:
            self.tp_idx = self.input_variables.index("tp")
        else:
            self.tp_idx = -1

        # log 变换通道索引
        self._input_log_idx = [i for i, v in enumerate(self.input_variables) if v in self.input_log_variables]
        self._output_log_idx = [i for i, v in enumerate(self.output_variables) if v in self.output_log_variables]

        # 时间坐标
        self.times = ds_input["time"].values

        # 转为 numpy
        self.invariants = ds_inv.to_array().transpose("variable", "latitude", "longitude").values
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}][INFO] invariant 加载完成")
        self.output = ds_output.to_array().transpose("time", "variable", "latitude", "longitude").values
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}][INFO] output 加载完成")
        self.input = ds_input.to_array().transpose("time", "variable", "latitude", "longitude").values
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}][INFO] input 加载完成")

        self.img_shape = self.output.shape[-2:]

        # 加载统计量
        self.input_mean, self.input_std, self.output_mean, self.output_std = self._load_stats()

        print(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}][INFO] MyDataset 初始化成功！"
            f" samples={len(self.times)}, img_shape={self.img_shape}"
        )

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        x_lr = self.input[idx]  # [Cin, 24, 24]
        y = self.output[idx]  # [Cout, 192, 192]

        # 上采样到高分辨率
        x_lr_up = self.upsample(x_lr)

        # tp 上采样后可能为负, 截断为 0
        if self.tp_idx >= 0:
            x_lr_up[self.tp_idx] = np.clip(x_lr_up[self.tp_idx], a_min=0.0, a_max=None)

        # 拼接不变量
        x = np.concatenate([x_lr_up, self.invariants], axis=0)

        # 归一化
        x = self.normalize_input(x)
        y = self.normalize_output(y)

        return (y, x)

    def _load_stats(self):
        with open(self.stats_path, "r") as f:
            stats = json.load(f)

        def _load(group, variables):
            mean, std = [], []
            for v in variables:
                if (v in self.input_log_variables and group == "input") or (
                    v in self.output_log_variables and group == "output"
                ):
                    mean.append(stats[group][v]["log_mean"])
                    std.append(stats[group][v]["log_std"])
                else:
                    mean.append(stats[group][v]["mean"])
                    std.append(stats[group][v]["std"])
            mean = np.array(mean)[:, None, None].astype(np.float32)
            std = np.array(std)[:, None, None].astype(np.float32)
            return mean, std

        in_mean, in_std = _load("input", self.input_variables)
        out_mean, out_std = _load("output", self.output_variables)
        inv_mean, inv_std = _load("invariant", self.invariant_variables)

        input_mean = np.concatenate([in_mean, inv_mean], axis=0)
        input_std = np.concatenate([in_std, inv_std], axis=0)

        return input_mean, input_std, out_mean, out_std

    # ============================================================
    # 归一化
    # ============================================================

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        x_proc = x.copy()
        for idx in self._input_log_idx:
            x_proc[idx] = self._log(x_proc[idx])
        x_proc = (x_proc - self.input_mean) / self.input_std
        return x_proc

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        x = x.copy().squeeze(0)
        x_proc = x * self.input_std + self.input_mean
        for idx in self._input_log_idx:
            x_proc[idx] = self._exp(x_proc[idx])
        x_proc = np.expand_dims(x_proc, axis=0)
        return x_proc

    def denormalize_input_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {x.shape}")
        input_mean = torch.tensor(self.input_mean, device=x.device, dtype=x.dtype)
        input_std = torch.tensor(self.input_std, device=x.device, dtype=x.dtype)
        x_proc = x * input_std + input_mean
        for idx in self._input_log_idx:
            x_proc[:, idx, :, :] = self._exp(x_proc[:, idx, :, :])
        return x_proc

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        x_proc = x.copy()
        for idx in self._output_log_idx:
            x_proc[idx] = self._log(x_proc[idx])
        x_proc = (x_proc - self.output_mean) / self.output_std
        return x_proc

    def normalize_output_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {x.shape}")
        for idx in self._output_log_idx:
            x[:, idx, :, :] = self._log(x[:, idx, :, :])
        x_proc = (x - self.output_mean) / self.output_std
        return x_proc

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        x = x.copy().squeeze(0)
        x_proc = x * self.output_std + self.output_mean
        for idx in self._output_log_idx:
            x_proc[idx] = self._exp(x_proc[idx])
        x_proc = np.expand_dims(x_proc, axis=0)
        return x_proc

    def denormalize_output_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {x.shape}")
        output_mean = torch.tensor(self.output_mean, device=x.device, dtype=x.dtype)
        output_std = torch.tensor(self.output_std, device=x.device, dtype=x.dtype)
        x_proc = x * output_std + output_mean
        for idx in self._output_log_idx:
            x_proc[:, idx, :, :] = self._exp(x_proc[:, idx, :, :])
        return x_proc

    # ============================================================
    # 对数变换
    # ============================================================

    def _log(self, x):
        if isinstance(x, torch.Tensor):
            return torch.log1p(x) if self.log_eps == 1 else torch.log(x + self.log_eps)
        elif isinstance(x, np.ndarray):
            return np.log1p(x) if self.log_eps == 1 else np.log(x + self.log_eps)
        else:
            raise TypeError(f"Expected ndarray or Tensor, got {type(x)}")

    def _exp(self, x):
        if isinstance(x, torch.Tensor):
            return torch.expm1(x) if self.log_eps == 1 else torch.exp(x) - self.log_eps
        elif isinstance(x, np.ndarray):
            return np.expm1(x) if self.log_eps == 1 else np.exp(x) - self.log_eps
        else:
            raise TypeError(f"Expected ndarray or Tensor, got {type(x)}")

    # ============================================================
    # 上采样 (24 → 192)
    # ============================================================

    def upsample(self, arr):
        """
        mode: nearest, linear, bilinear, bicubic, trilinear, area, nearest-exact
        """
        t = torch.tensor(arr[None, ...], dtype=torch.float32)  # [1, C, 24, 24]
        t_up = torch.nn.functional.interpolate(t, size=self.img_shape, mode="bilinear")
        return t_up[0].numpy()

    # ============================================================
    # 元数据 API
    # ============================================================

    def input_channels(self):
        inputs = [ChannelMetadata(name=v) for v in self.input_variables]
        invariants = [ChannelMetadata(name=v, auxiliary=True) for v in self.invariant_variables]
        return inputs + invariants

    def output_channels(self):
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def longitude(self):
        return np.full(self.img_shape, np.nan)

    def latitude(self):
        return np.full(self.img_shape, np.nan)

    def time(self):
        return self.times

    def image_shape(self):
        return self.img_shape
