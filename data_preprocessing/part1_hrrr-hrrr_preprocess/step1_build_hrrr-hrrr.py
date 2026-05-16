"""
构建 HRRR-HRRR 降尺度数据集

该脚本用于构建 HRRR 自降尺度 (self-downscaling) 数据集:
  - 低分辨率输入: HRRR 3km 粗化至 25km (shape: 24x24), 变量: tp, t2m, u10, v10
  - 高分辨率输出: HRRR 3km 原始分辨率 (shape: 192x192), 变量: tp
  - 静态地理变量: elevation (高程), lsm (海陆掩码), shape: 192x192

数据来源:
  - HRRR grib2:  {hrrr_root}/{var}/hrrr/{yyyymmdd}/*.grib2
  - DEM:         SRTM_30m_CONUS.tif
  - LSM:         hrrr 海陆掩码 grib2 文件
  - 区域网格:    {region}_lat_lon_192_192.nc (由 grid.ipynb 生成)

输出目录结构:
  dataset_root/{region}/
  ├── train_data_in.nc       训练集输入 (低分辨率, 24x24)
  ├── train_data_out.nc      训练集输出 (高分辨率, 192x192)
  ├── val_data_in.nc         验证集输入
  ├── val_data_out.nc        验证集输出
  ├── test_data_in.nc        测试集输入
  ├── test_data_out.nc       测试集输出
  ├── static_variables.nc    静态地理变量 (高分辨率, 192x192)
  └── stats.json             归一化统计量 (由 step2_cal_stats.py 生成)

使用方式:
  # 处理单个区域 (默认 part1-US-Rocky)
  python step1_build_hrrr-hrrr.py

  # 处理指定区域
  python step1_build_hrrr-hrrr.py --region part1-US-Cascades

  # 自定义参数
  python step1_build_hrrr-hrrr.py --region part1-US-Rocky --train-years 2015 2016 --months 1 2 3
"""

import os
import glob
import json
import itertools
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import traceback
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import warnings
import cfgrib

# 抑制 cfgrib 相关的警告信息 (索引文件不兼容、EOFError 等)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*index file.*")
warnings.filterwarnings("ignore", message=".*EOFError.*")

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """配置全局日志输出格式：时间戳 + 级别 + logger 名称 + 消息"""
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt="{asctime} [{levelname}]: {message}", style="{", datefmt="%Y-%m-%d %H:%M:%S")

    # 避免在交互环境/重复调用时重复添加 handler；但无论如何都强制统一 formatter
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root.addHandler(handler)
        return

    for h in root.handlers:
        h.setLevel(level)
        h.setFormatter(formatter)


def to_datetime64_index(arr) -> pd.DatetimeIndex:
    """将时间数组转换为 pandas DatetimeIndex, 用于时间对齐操作"""
    return pd.to_datetime(arr.values if hasattr(arr, "values") else arr)


class HRRRHRRRBuilder:
    """
    HRRR-HRRR 降尺度数据集构建器

    核心思路:
      1. 从区域网格文件加载目标区域的 192x192 经纬度网格
      2. 在 HRRR 全网格 (1059x1799) 中定位该区域的 y/x 范围
      3. 生成静态地理变量 (高程 + 海陆掩码)
      4. 逐月加载 HRRR 数据 → 裁剪到区域 → 分别生成高分辨率输出和低分辨率输入
      5. 合并月度分片为最终的训练/验证/测试集
    """

    def __init__(
        self,
        hrrr_root: str,
        dem_path: str,
        lsm_path: str,
        region_grid_dir: str,
        out_dir: str,
        region: str = "part1-US-Rocky",
        input_vars: List[str] = None,
        output_vars: List[str] = None,
        split_years: Dict[str, List[int]] = None,
        months: List[int] = None,
        coarsen_factor: int = 8,
        tmp_dir: Optional[str] = None,
        merge_flag: bool = True,
    ):
        """
        参数:
            hrrr_root:       HRRR 数据根目录, 下含 {var}/hrrr/{yyyymmdd}/*.grib2
            dem_path:        SRTM DEM GeoTIFF 文件路径
            lsm_path:        HRRR 海陆掩码 grib2 文件路径
            region_grid_dir: 区域网格文件目录, 包含 {region}_lat_lon_192_192.nc
            out_dir:         输出根目录, 会在其下创建 {region} 子目录
            region:          区域名称, 如 "part1-US-Rocky"
            input_vars:      低分辨率输入变量列表 (默认: tp, t2m, u10, v10)
            output_vars:     高分辨率输出变量列表 (默认: tp)
            split_years:     数据集划分, 字典格式 {"train": [2015,...], "val": [2023], "test": [2024]}
            months:          要处理的月份列表 (默认: 1-12)
            coarsen_factor:  粗化因子 (192/24=8, 即每 8x8 像素块取均值)
            tmp_dir:         临时分片文件目录 (默认: {out_dir}/tmp)
            merge_flag:      是否在处理完成后合并分片文件
        """
        self.hrrr_root = Path(hrrr_root)
        self.dem_path = Path(dem_path)
        self.lsm_path = Path(lsm_path)
        self.region_grid_dir = Path(region_grid_dir)
        self.out_dir = Path(out_dir)
        self.region = region

        # 低分辨率输入变量 (默认: 降水、温度、风速)
        self.input_vars = input_vars or ["tp", "t2m", "u10", "v10"]
        # 高分辨率输出变量 (默认: 仅降水)
        self.output_vars = output_vars or ["tp"]
        # 合并所有需要从 HRRR 加载的变量 (去重保序)
        self.all_hrrr_vars = list(dict.fromkeys(self.input_vars + self.output_vars))
        # 累积量变量: valid_time = time + step，需要特殊处理时间轴
        self.accum_vars = {"tp"}
        self.fxx = 1  # 预报步长 (小时)

        # 数据集时间划分
        self.split_years = split_years or {
            "test": [2024],  # 测试集: 2024
            "val": [2023],  # 验证集: 2023
            "train": list(range(2015, 2023)),  # 训练集: 2015-2022
        }
        self.months = months or list(range(1, 13))
        self.coarsen_factor = coarsen_factor  # 粗化因子: 192 → 24, factor=8
        self.merge_flag = merge_flag

        # 创建输出目录: dataset_root/{region}/
        self.region_dir = self.out_dir / self.region
        self.region_dir.mkdir(parents=True, exist_ok=True)
        # 创建临时分片目录: tmp/{region}/
        self.tmp_base = Path(tmp_dir) if tmp_dir else self.out_dir / "tmp"
        self.tmp_base.mkdir(parents=True, exist_ok=True)

        # 以下属性在 build_all() 过程中填充
        self.lat2d = None  # 目标区域的二维纬度数组 (192, 192)
        self.lon2d = None  # 目标区域的二维经度数组 (192, 192)
        self.region_range = None  # 区域在全网格中的 y/x 切片范围 (y0, y1, x0, x1)
        self.missing_log: list[str] = []  # 记录缺失数据的日志

    # ------------------------------------------------------------------ 网格加载
    def _load_target_grid(self):
        """
        从区域网格文件加载 192x192 的二维经纬度网格

        网格文件由 grid.ipynb 预先生成, 包含目标区域每个格点的精确经纬度。
        HRRR 使用 Lambert 投影, 经纬度是二维不规则网格, 不是等间距的。
        """
        grid_path = self.region_grid_dir / f"{self.region}_lat_lon_192_192.nc"
        if not grid_path.exists():
            raise FileNotFoundError(f"Region grid file not found: {grid_path}")

        ds = xr.open_dataset(grid_path)
        self.lat2d = ds["latitude"].values.astype(np.float32)
        self.lon2d = ds["longitude"].values.astype(np.float32)

        # 确保经度范围为 0°-360° (部分 HRRR 数据使用 -180~180)
        if self.lon2d.min() < 0:
            self.lon2d = self.lon2d + 360.0

        logger.info(
            f"[Grid] Loaded {grid_path.name}, "
            f"lat=[{float(self.lat2d.min()):.2f}, {float(self.lat2d.max()):.2f}], "
            f"lon=[{float(self.lon2d.min()):.2f}, {float(self.lon2d.max()):.2f}]"
        )

    def _find_region_range(self):
        """
        在 HRRR 全网格 (1059x1799) 中定位目标区域对应的 y/x 切片范围

        方法: 通过区域网格中心点, 在全网格中找最近邻格点,
              然后以此为圆心向外扩展 96 个格点, 得到 192x192 的切片范围。
        """
        # 使用第一个变量目录下的任意一个 grib2 文件来读取全网格坐标
        pattern = str(self.hrrr_root / self.all_hrrr_vars[0] / "hrrr" / "********" / "*.grib2")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError("Cannot find any HRRR grib2 file to extract grid.")

        ds = xr.open_dataset(files[0], engine="cfgrib")
        full_lat = ds["latitude"].values  # (1059, 1799) 二维纬度
        full_lon = ds["longitude"].values  # (1059, 1799) 二维经度
        ds.close()

        # 统一经度范围到 0-360
        if full_lon.min() < 0:
            full_lon = full_lon + 360.0

        # 取区域网格的中心点坐标
        center_lat = float(self.lat2d[self.lat2d.shape[0] // 2, self.lat2d.shape[1] // 2])
        center_lon = float(self.lon2d[self.lon2d.shape[0] // 2, self.lon2d.shape[1] // 2])

        # 在全网格中找到距离中心点最近的格点
        dist = (full_lat - center_lat) ** 2 + (full_lon - center_lon) ** 2
        yc, xc = np.unravel_index(dist.argmin(), dist.shape)

        # 以该点为中心, 向外扩展 96 个格点, 得到 192x192 的区域
        half = self.lat2d.shape[0] // 2  # 96
        y0, y1 = yc - half, yc + half
        x0, x1 = xc - half, xc + half

        # 边界检查
        if y0 < 0 or y1 > full_lat.shape[0] or x0 < 0 or x1 > full_lat.shape[1]:
            raise ValueError(f"Region slice ({y0}:{y1}, {x0}:{x1}) out of bounds " f"for full grid ({full_lat.shape})")

        self.region_range = (y0, y1, x0, x1)
        logger.info(f"[Grid] Region range: y={y0}:{y1}, x={x0}:{x1}")

    # -------------------------------------------------------------- 静态变量
    def compute_and_save_static(self):
        """
        生成 static_variables.nc, 包含高分辨率的静态地理变量:
          - elevation: 高程/海拔 (从 SRTM 30m DEM 插值到 HRRR 3km 网格)
          - lsm:       海陆掩码 0~1 (从 HRRR 原始 LSM 裁剪)

        该文件只需生成一次, 训练/验证/测试集共用。
        如果文件已存在则跳过。
        """
        out_path = self.region_dir / "static_variables.nc"
        if out_path.exists():
            logger.info(f"[Static] Already exists, skipping: {out_path}")
            return out_path

        logger.info("[Static] Computing static variables ...")
        lat2d = self.lat2d
        lon2d = self.lon2d
        ny, nx = lat2d.shape  # 192, 192

        variables = {}

        # --- 处理高程数据 (elevation) ---
        if "elevation" in (self.input_vars + self.output_vars + ["elevation"]):
            logger.info("[Static]   Interpolating DEM ...")
            # 使用 rioxarray 读取 SRTM 30m DEM GeoTIFF
            dem_da = rioxarray.open_rasterio(str(self.dem_path))
            dem_data = dem_da.values[0]  # (36000, 84000) 即美国本土 30m 分辨率
            dem_y = dem_da.y.values  # 纬度数组 (22°N ~ 52°N)
            dem_x = dem_da.x.values  # 经度数组 (-130°E ~ -60°E)

            # 处理 DEM 的 nodata 值 (SRTM 使用 -32768 表示海洋/无数据区域)
            nodata = dem_da.rio.nodata
            if nodata is not None:
                dem_data = np.where(dem_data == nodata, 0.0, dem_data)

            # 确保 y 轴为升序 (由南到北), RegularGridInterpolator 要求单调递增
            if dem_y[0] > dem_y[-1]:
                dem_data = dem_data[::-1]
                dem_y = dem_y[::-1]

            # 使用最近邻插值将 DEM 采样到 HRRR 网格点
            # DEM 30m 远比 HRRR 3km 精细 (~100倍), 最近邻已足够精确
            interp = RegularGridInterpolator(
                (dem_y, dem_x),
                dem_data,
                method="nearest",
                bounds_error=False,  # 区域网格可能部分超出 DEM 范围, 不报错
                fill_value=np.nan,  # 超出范围填充 NaN
            )

            # 将区域经度从 0-360 转换为 -180~180, 匹配 DEM 的坐标系
            lon_180 = np.where(lon2d > 180, lon2d - 360.0, lon2d)
            # 构造查询点: (lat, lon) 对
            points = np.stack([lat2d.ravel(), lon_180.ravel()], axis=-1)
            elevation = interp(points).reshape(ny, nx).astype(np.float32)
            variables["elevation"] = elevation

        # --- 处理海陆掩码 (lsm) ---
        if "lsm" in (self.input_vars + self.output_vars + ["lsm"]):
            logger.info("[Static]   Loading LSM ...")
            # 读取 HRRR 原始 LSM (1059x1799), 直接裁剪到区域范围
            ds_lsm = xr.open_dataset(str(self.lsm_path), engine="cfgrib")
            lsm_full = ds_lsm["lsm"]
            y0, y1, x0, x1 = self.region_range
            lsm = lsm_full.isel(y=slice(y0, y1), x=slice(x0, x1)).values.astype(np.float32)
            ds_lsm.close()
            variables["lsm"] = lsm

        # 构建 Dataset 并保存
        # 维度命名为 (latitude, longitude), 经纬度坐标为二维数组
        ds_static = xr.Dataset(
            {name: (["latitude", "longitude"], data) for name, data in variables.items()},
            coords={
                "latitude": (["latitude", "longitude"], lat2d),
                "longitude": (["latitude", "longitude"], lon2d),
            },
        )
        ds_static.to_netcdf(str(out_path))
        logger.info(f"[Static] Saved: {out_path}")
        return out_path

    # ----------------------------------------------------------- HRRR 数据加载
    def open_hrrr_month(self, year: int, month: int) -> Optional[xr.Dataset]:
        """
        读取指定年月的 HRRR 数据, 裁剪到目标区域, 返回合并后的 xarray.Dataset

        处理流程:
          1. 逐变量遍历, 每个变量按天加载 (避免一次性加载过多数据)
          2. 合并同一个月内所有天的数据
          3. 各变量合并为一个 Dataset
          4. 裁剪到目标区域的 192x192 范围

        参数:
            year:  年份, 如 2015
            month: 月份, 如 1

        返回:
            裁剪后的 xr.Dataset, 维度: (time, y, x), 或 None (如果无数据)
        """
        parts: dict[str, xr.DataArray] = {}

        for var in self.all_hrrr_vars:
            logger.info(f"Loading variable: {var} ...")
            month_dir = self.hrrr_root / var / "hrrr"
            if not month_dir.exists():
                logger.warning(f"HRRR directory not found: {month_dir}")
                self.missing_log.append(f"{year}-{month:02d} var={var}: directory not found")
                continue

            # 搜索该月所有天的文件夹, 如 data/hrrr/tp/hrrr/20150101, 20150102, ...
            daily_dirs = sorted(glob.glob(str(month_dir / f"{year:04d}{month:02d}??")))
            if not daily_dirs:
                msg = f"{year}-{month:02d} var={var}: no daily folders"
                logger.warning(f"{msg}")
                self.missing_log.append(msg)
                continue

            # 累积量变量需要多取上个月最后一天，使 valid_time 覆盖当月 00:00
            if var in self.accum_vars:
                prev_year, prev_month = (year, month - 1) if month > 1 else (year - 1, 12)
                prev_pattern = str(month_dir / f"{prev_year:04d}{prev_month:02d}??")
                prev_dirs = sorted(glob.glob(prev_pattern))
                if prev_dirs:
                    daily_dirs = [prev_dirs[-1]] + daily_dirs

            # 逐天加载并合并
            daily_datasets = []
            for daily_dir in daily_dirs:
                # 每天文件夹下的所有 grib2 文件 (通常每个文件对应一个预报时次)
                daily_files = sorted(glob.glob(str(Path(daily_dir) / "*.grib2")))
                if not daily_files:
                    continue
                try:
                    # 使用 xr.open_mfdataset 批量打开多个 grib2 文件
                    daily_ds = xr.open_mfdataset(
                        daily_files,
                        combine="nested",
                        concat_dim="time",
                        engine="cfgrib",
                        parallel=True,  # 并行读取加速
                    )
                    # 按有效时间 (valid_time) 排序
                    daily_ds = daily_ds.sortby("valid_time")
                    # 删除 time 维，只保留 valid_time 作为统一时间轴
                    if "time" in daily_ds.dims:
                        daily_ds = daily_ds.swap_dims({"time": "valid_time"})
                    if "time" in daily_ds.coords:
                        daily_ds = daily_ds.drop_vars("time")

                    if var not in daily_ds.data_vars:
                        logger.warning(f"Variable {var} not found in {daily_dir}")
                        continue
                    # 移除 step 和 surface 坐标, 避免后续 concat 时产生冲突
                    # (不同时次的 step 值不同, 会导致 "conflicting values" 错误)
                    for c in ["step", "surface", "heightAboveGround"]:
                        if c in daily_ds.coords:
                            daily_ds = daily_ds.drop_vars(c)
                    daily_datasets.append(daily_ds[[var]])
                except Exception as e:
                    logger.warning(f"Failed to load {daily_dir} var={var}: {e}")
                    self.missing_log.append(f"{daily_dir} var={var}: {e}")
                    continue

            if not daily_datasets:
                msg = f"{year}-{month:02d} var={var}: no valid data"
                logger.warning(f"{msg}")
                self.missing_log.append(msg)
                continue

            # 将该月所有天的数据沿 valid_time 维拼接
            ds_month = xr.concat(daily_datasets, dim="valid_time")
            parts[var] = ds_month[var]

        if not parts:
            logger.warning(f"No data at all for {year:04d}-{month:02d}")
            return None

        # 取所有变量 valid_time 的交集
        valid_times = None
        for var_name, da in parts.items():
            vt = set(da.valid_time.values)
            valid_times = vt if valid_times is None else valid_times & vt

        if not valid_times:
            logger.warning(f"No overlapping valid_time for {year:04d}-{month:02d}")
            return None

        common_times = sorted(valid_times)
        logger.info(f"Common valid_time count: {len(common_times)}, " f"range: {common_times[0]} ~ {common_times[-1]}")

        # 裁剪每个变量到交集时间范围，然后合并
        aligned_parts = {}
        for var_name, da in parts.items():
            aligned_parts[var_name] = da.sel(valid_time=common_times)

        merged = xr.Dataset(aligned_parts)

        # 清理无用坐标
        for coord_name in ["surface", "step"]:
            if coord_name in merged.coords:
                merged = merged.drop_vars(coord_name)

        # 统一时间坐标: valid_time → time
        merged = merged.rename({"valid_time": "time"})

        # 裁剪到目标区域: 从全网格 (1059x1799) 切片到 (192x192)
        y0, y1, x0, x1 = self.region_range
        merged = merged.isel(y=slice(y0, y1), x=slice(x0, x1))

        return merged

    # ------------------------------------------------------------ 粗化降分辨率
    def coarsen_to_lowres(self, ds: xr.Dataset) -> xr.Dataset:
        """
        将高分辨率数据 (192x192) 粗化为低分辨率数据 (24x24)

        粗化方法: 对每 8x8 像素块取均值 (coarsen factor = 8)
        同时对经纬度坐标也取均值, 保持二维网格形式。

        参数:
            ds: 高分辨率 xr.Dataset, 维度 (time, y, x), shape: (*, 192, 192)

        返回:
            低分辨率 xr.Dataset, 维度 (time, latitude, longitude), shape: (*, 24, 24)
        """
        cf = self.coarsen_factor  # 8

        # 对经纬度坐标进行粗化: 每 8x8 块取均值, 得到 24x24 的坐标网格
        lat_lowres = xr.DataArray(self.lat2d, dims=("y", "x")).coarsen(y=cf, x=cf).mean().values.astype(np.float32)
        lon_lowres = xr.DataArray(self.lon2d, dims=("y", "x")).coarsen(y=cf, x=cf).mean().values.astype(np.float32)

        # 对数据变量进行粗化: 每 8x8 块取均值
        ds_lowres = ds.coarsen(y=cf, x=cf).mean()

        # 将维度名从 (y, x) 重命名为 (latitude, longitude), 符合输出规范
        ds_lowres = ds_lowres.rename({"y": "latitude", "x": "longitude"})
        # 赋值粗化后的二维经纬度坐标
        ds_lowres = ds_lowres.assign_coords(
            latitude=(["latitude", "longitude"], lat_lowres),
            longitude=(["latitude", "longitude"], lon_lowres),
        )
        return ds_lowres

    # -------------------------------------------------------- 逐月处理
    def process_month(self, year: int, month: int, split: str):
        """
        处理单个月份的数据: 加载 → 分离高/低分辨率 → 保存为月度分片

        处理流程:
          1. 加载该月的 HRRR 数据 (已裁剪到 192x192 区域)
          2. 高分辨率输出: 选择 output_vars (tp), 保持 192x192
          3. 低分辨率输入: 选择 input_vars (tp, t2m, u10, v10), 粗化到 24x24
          4. 分别保存为临时分片文件

        如果分片文件已存在则跳过 (支持断点续处理)。

        参数:
            year:  年份
            month: 月份
            split: 数据集划分名称 ("train", "val", "test")
        """
        # 分片文件保存目录: tmp/{region}/{split}/
        shard_dir = self.tmp_base / self.region / split
        shard_dir.mkdir(parents=True, exist_ok=True)

        # 分片文件路径: data_in_{YYYY}_{MM}.nc, data_out_{YYYY}_{MM}.nc
        shard_in = shard_dir / f"data_in_{year:04d}_{month:02d}.nc"
        shard_out = shard_dir / f"data_out_{year:04d}_{month:02d}.nc"

        # 如果分片文件已存在, 跳过 (支持断点续处理)
        if shard_in.exists() and shard_out.exists():
            logger.info(f"[Skip] {year:04d}-{month:02d} {split}: shard files already exist")
            return

        # 加载该月的 HRRR 数据并裁剪到区域
        ds = self.open_hrrr_month(year, month)
        if ds is None:
            return

        # 检查实际可用的输入/输出变量 (某些变量可能部分缺失)
        available_in = [v for v in self.input_vars if v in ds.data_vars]
        available_out = [v for v in self.output_vars if v in ds.data_vars]

        # 输出变量 (tp) 缺失则跳过该月
        if not available_out:
            msg = f"{year}-{month:02d}: output vars {self.output_vars} not available, skipping"
            logger.warning(f"{msg}")
            self.missing_log.append(msg)
            return

        # ---- 构建高分辨率输出数据 (192x192) ----
        ds_out = ds[available_out].copy()
        # 移除 HRRR 原始的经纬度坐标, 后面替换为目标区域的经纬度
        if "latitude" in ds_out.coords:
            ds_out = ds_out.drop_vars(["latitude", "longitude"])
        # 将维度名从 (y, x) 重命名为 (latitude, longitude)
        ds_out = ds_out.rename({"y": "latitude", "x": "longitude"})
        # 赋值二维经纬度坐标 (0-360 经度范围)
        ds_out = ds_out.assign_coords(
            latitude=(["latitude", "longitude"], self.lat2d),
            longitude=(["latitude", "longitude"], self.lon2d),
        )
        # 确保降水非负 (HRRR 偶尔出现极小的负值)
        if "tp" in ds_out.data_vars:
            ds_out["tp"] = ds_out["tp"].clip(min=0.0)

        # ---- 构建低分辨率输入数据 (24x24) ----
        ds_in = self.coarsen_to_lowres(ds[available_in])
        if "tp" in ds_in.data_vars:
            ds_in["tp"] = ds_in["tp"].clip(min=0.0)

        # 保存分片文件
        ds_out.to_netcdf(str(shard_out))
        ds_in.to_netcdf(str(shard_in))
        logger.info(f"[OK] {year:04d}-{month:02d} {split}: {shard_in.name}, {shard_out.name}")

    # -------------------------------------------------------------- 合并分片
    def merge_splits(self):
        """
        将月度分片文件合并为最终的训练/验证/测试集文件

        对每个数据集划分 (train/val/test):
          1. 收集该划分下所有月度分片文件
          2. 沿 time 维度拼接并按时间排序
          3. 保存为 {split}_data_in.nc 和 {split}_data_out.nc
        """
        for split, years in self.split_years.items():
            if not years:
                continue

            # 最终输出文件路径
            out_in = self.region_dir / f"{split}_data_in.nc"
            out_out = self.region_dir / f"{split}_data_out.nc"

            # 如果最终文件已存在, 跳过
            if out_in.exists() and out_out.exists():
                logger.info(f"[Merge] Already exists, skipping: {split}")
                continue

            # 收集该划分下所有月度分片文件
            shard_dir = self.tmp_base / self.region / split
            in_shards = sorted(glob.glob(str(shard_dir / "data_in_*.nc")))
            out_shards = sorted(glob.glob(str(shard_dir / "data_out_*.nc")))

            if not in_shards or not out_shards:
                logger.warning(f"No shards found for {split}")
                continue

            logger.info(f"[Merge] Merging {len(in_shards)} shards for {split} ...")

            # 打开所有分片文件
            ds_in_list = [xr.open_dataset(f) for f in tqdm(in_shards, desc=f"Merge {split} input")]
            ds_out_list = [xr.open_dataset(f) for f in tqdm(out_shards, desc=f"Merge {split} output")]

            # 沿 time 维拼接并排序
            ds_in = xr.concat(ds_in_list, dim="time").sortby("time")
            ds_out = xr.concat(ds_out_list, dim="time").sortby("time")

            # 关闭分片文件, 释放资源
            for ds_item in ds_in_list + ds_out_list:
                ds_item.close()

            # 保存最终合并后的文件
            ds_in.to_netcdf(str(out_in))
            ds_out.to_netcdf(str(out_out))
            logger.info(f"[Merge] Saved: {out_in.name}, {out_out.name}")

    # ---------------------------------------------------------- 主入口
    def build_all(self):
        """
        主构建流程

        执行顺序:
          1. 加载目标区域的经纬度网格 (192x192)
          2. 在 HRRR 全网格中定位区域范围
          3. 生成静态地理变量 (elevation, lsm)
          4. 逐 split × 年 × 月 处理数据
          5. 合并月度分片为最终数据集
          6. 保存缺失数据日志
        """
        logger.info(f"{'=' * 60}")
        logger.info(f"Building HRRR-HRRR dataset for region: {self.region}")
        logger.info(f"{'=' * 60}")

        # 1-2. 加载网格 & 定位区域范围
        self._load_target_grid()
        self._find_region_range()

        # 3. 生成静态变量 (只需一次)
        self.compute_and_save_static()

        # 4. 逐 split × 年 × 月 处理
        for split, years in self.split_years.items():
            if not years:
                continue
            logger.info(f"--- Processing: {split} ({len(years)} years) ---")
            for y, m in tqdm(
                itertools.product(years, self.months),
                total=len(years) * len(self.months),
                desc=split,
            ):
                try:
                    self.process_month(int(y), int(m), split)
                except Exception as e:
                    traceback.print_exc()
                    msg = f"[ERROR] {y}-{m:02d}: {e}"
                    logger.error(msg)
                    self.missing_log.append(msg)

        # 5. 合并分片文件
        if self.merge_flag:
            self.merge_splits()

        # 6. 保存缺失数据日志
        if self.missing_log:
            log_path = self.region_dir / "missing_data.log"
            with open(log_path, "w") as f:
                f.write("\n".join(self.missing_log))
            logger.warning(f"{len(self.missing_log)} issues logged to {log_path}")


# ------------------------------------------------------------------- 命令行接口
def parse_args():
    """解析命令行参数, 所有参数都有合理的默认值"""
    p = argparse.ArgumentParser(description="Build HRRR-HRRR downscaling dataset")
    p.add_argument("--hrrr-root", default="/path/to/project_climate/Benchmark_process/data/hrrr")
    p.add_argument(
        "--dem-path", default="/path/to/project_climate/Benchmark_process/data/DEM/SRTM_30m_CONUS.tif"
    )
    p.add_argument(
        "--lsm-path",
        default="/path/to/project_climate/Benchmark_process/data/hrrr/lsm/subset_0aef124b__hrrr.t12z.wrfsfcf00.grib2",
    )
    p.add_argument(
        "--region-grid-dir", default="/path/to/project_climate/Benchmark_process/data/regions_lat_lon"
    )
    p.add_argument("--out-dir", default="/path/to/project_climate/Benchmark_process/hrrr-hrrr/dataset_root")
    p.add_argument("--region", default="part1-US-Rocky")
    p.add_argument("--input-vars", nargs="+", default=["tp", "t2m", "u10", "v10"])
    p.add_argument("--output-vars", nargs="+", default=["tp"])
    p.add_argument("--train-years", nargs="+", type=int, default=list(range(2015, 2023)))
    p.add_argument("--val-years", nargs="+", type=int, default=[2023])
    p.add_argument("--test-years", nargs="+", type=int, default=[2024])
    p.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    p.add_argument("--coarsen-factor", type=int, default=8)
    p.add_argument("--tmp-dir", default=None)
    p.add_argument("--no-merge", action="store_true")
    return p.parse_args()


def main():
    """命令行入口: 使用 argparse 参数构建数据集"""
    setup_logging(logging.INFO)
    args = parse_args()
    splits = {"test": args.test_years, "val": args.val_years, "train": args.train_years}
    builder = HRRRHRRRBuilder(
        hrrr_root=args.hrrr_root,
        dem_path=args.dem_path,
        lsm_path=args.lsm_path,
        region_grid_dir=args.region_grid_dir,
        out_dir=args.out_dir,
        region=args.region,
        input_vars=args.input_vars,
        output_vars=args.output_vars,
        split_years=splits,
        months=args.months,
        coarsen_factor=args.coarsen_factor,
        tmp_dir=args.tmp_dir,
        merge_flag=not args.no_merge,
    )
    builder.build_all()


def run():
    """
    批量处理入口: 遍历所有 5 个区域逐一构建数据集

    使用方式: 修改脚本底部为 run() 即可
      if __name__ == "__main__":
          run()
    """
    regions = [
        "part1-US-Rocky",
        "part1-US-Cascades",
        "part1-US-Nevada",
        "part1-US-Appalachian",
        "part1-US-Great_Plains",
    ]
    setup_logging(logging.INFO)
    for region in regions:
        builder = HRRRHRRRBuilder(
            hrrr_root="/path/to/project_climate/Benchmark_process/data/hrrr",
            dem_path="/path/to/project_climate/Benchmark_process/data/DEM/SRTM_30m_CONUS.tif",
            lsm_path="/path/to/project_climate/Benchmark_process/data/hrrr/lsm/subset_0aef124b__hrrr.t12z.wrfsfcf00.grib2",
            region_grid_dir="/path/to/project_climate/Benchmark_process/data/regions_lat_lon",
            out_dir="/path/to/project_climate/Benchmark_process/hrrr-hrrr/dataset_root",
            region=region,
        )
        builder.build_all()


if __name__ == "__main__":
    # main()
    run()
