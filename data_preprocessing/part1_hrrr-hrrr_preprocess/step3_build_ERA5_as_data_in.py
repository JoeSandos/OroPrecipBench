"""
构建 ERA5-HRRR 降尺度数据集的 ERA5 低分辨率输入数据

该脚本将 ERA5 全球再分析数据裁剪并插值到 HRRR 低分辨率网格 (24x24),
使其与 hrrr-hrrr 数据集的格式完全一致, 可直接替换低分辨率输入部分。

核心思路:
  1. 从 regions_lat_lon 加载 192x192 高分辨率网格
  2. 对高分辨率网格粗化 (8x8 均值) 得到 24x24 的低分辨率网格 (与 HRRR 低分辨率输入一致)
  3. 读取已有的 HRRR {split}_data_in.nc 获取目标 valid_time 序列
  4. 逐月读取 ERA5 → 插值到 24x24 网格 → 对齐时间 → 保存月度分片
  5. 合并月度分片为最终的 {split}_data_in_ERA5.nc

输出目录结构:
  dataset_root/{region}/
  ├── train_data_in_ERA5.nc
  ├── val_data_in_ERA5.nc
  └── test_data_in_ERA5.nc

使用方式:
  # 处理单个区域 (默认 part1-US-Rocky)
  python step3_build_ERA5_as_data_in.py

  # 处理指定区域
  python step3_build_ERA5_as_data_in.py --region part1-US-Cascades

  # 自定义参数
  python step3_build_ERA5_as_data_in.py --region part1-US-Rocky --train-years 2015 2016 --months 1 2
"""

import glob
import itertools
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


# ── ERA5 变量名映射 ────────────────────────────────────────────────────────
# ERA5 文件名中使用的变量名 → 数据集内部统一变量名
ERA5_VAR_MAP = {
    "tp": "tp",  # 总降水量 (m → mm, 需要乘以 1000)
    "t2m": "t2m",  # 2m 气温 (K, 不变)
    "u10": "u10",  # 10m 纬向风 (m/s, 不变)
    "v10": "v10",  # 10m 经向风 (m/s, 不变)
}

# 需要单位转换的变量: m → mm
ERA5_PRECIP_VARS = {"tp"}
ERA5_PRECIP_SCALE = 1000.0  # m → mm


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(
        fmt="{asctime} [{levelname}]: {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not root.handlers:
        h = logging.StreamHandler()
        h.setLevel(level)
        h.setFormatter(fmt)
        root.addHandler(h)
        return
    for h in root.handlers:
        h.setLevel(level)
        h.setFormatter(fmt)


class ERA5DataInBuilder:
    """
    ERA5 低分辨率输入数据构建器

    将 ERA5 全球数据处理为与 hrrr-hrrr 数据集低分辨率输入格式一致的 NetCDF 文件。
    """

    def __init__(
        self,
        era5_root: str,
        region_grid_dir: str,
        hrrr_dataset_dir: str,
        out_dir: str,
        region: str = "part1-US-Rocky",
        input_vars: List[str] = None,
        split_years: Dict[str, List[int]] = None,
        months: List[int] = None,
        coarsen_factor: int = 8,
        tmp_dir: Optional[str] = None,
        merge_flag: bool = True,
    ):
        """
        参数:
            era5_root:        ERA5 数据根目录
                              结构: {era5_root}/{var}/{yyyy}/{mm}/{var}_{yyyymmdd}_{hh}00.nc
            region_grid_dir:  高分辨率网格文件目录
                              包含 {region}_lat_lon_192_192.nc
            hrrr_dataset_dir: 已构建的 hrrr-hrrr 数据集根目录
                              用于读取 {split}_data_in.nc 获取目标 valid_time
            out_dir:          输出根目录 (与 hrrr-hrrr 数据集同目录即可)
            region:           区域名称, 如 "part1-US-Rocky"
            input_vars:       ERA5 输入变量列表 (默认: tp, t2m, u10, v10)
            split_years:      数据集划分
            months:           要处理的月份列表 (默认: 1-12)
            coarsen_factor:   粗化因子, 与 hrrr-hrrr 一致 (默认 8: 192→24)
            tmp_dir:          临时分片目录 (默认: {out_dir}/tmp_era5)
            merge_flag:       是否合并月度分片
        """
        self.era5_root = Path(era5_root)
        self.region_grid_dir = Path(region_grid_dir)
        self.hrrr_dataset_dir = Path(hrrr_dataset_dir)
        self.out_dir = Path(out_dir)
        self.region = region
        self.input_vars = input_vars or ["tp", "t2m", "u10", "v10"]
        self.split_years = split_years or {
            # "train": list(range(2015, 2023)),
            # "val": [2023],
            "test": [2024],
        }
        self.months = months or list(range(1, 13))
        self.coarsen_factor = coarsen_factor
        self.merge_flag = merge_flag

        # 输出目录
        self.region_dir = self.out_dir / self.region
        self.region_dir.mkdir(parents=True, exist_ok=True)

        # 临时分片目录
        self.tmp_base = Path(tmp_dir) if tmp_dir else self.out_dir / "tmp_era5"
        self.tmp_base.mkdir(parents=True, exist_ok=True)

        # 以下属性在 build_all() 中填充
        self.lat2d_hi = None  # 192x192 高分辨率纬度
        self.lon2d_hi = None  # 192x192 高分辨率经度 (0-360)
        self.lat2d_lo = None  # 24x24 低分辨率纬度 (粗化后)
        self.lon2d_lo = None  # 24x24 低分辨率经度 (0-360)

        # 各 split 的目标 valid_time (从已有 HRRR 数据集读取)
        self.split_target_times: Dict[str, pd.DatetimeIndex] = {}

        self.missing_log: List[str] = []

    # ── Step 1: 加载高分辨率网格 & 粗化得到低分辨率网格 ──────────────────
    def _load_and_coarsen_grid(self):
        """
        从区域网格文件加载 192x192 网格, 然后粗化得到 24x24 低分辨率网格。
        低分辨率网格与 hrrr-hrrr step1 中的 coarsen_to_lowres 逻辑完全一致。
        """
        grid_path = self.region_grid_dir / f"{self.region}_lat_lon_192_192.nc"
        if not grid_path.exists():
            raise FileNotFoundError(f"Region grid file not found: {grid_path}")

        ds = xr.open_dataset(grid_path)
        lat2d = ds["latitude"].values.astype(np.float32)
        lon2d = ds["longitude"].values.astype(np.float32)

        # 统一经度到 0-360
        if lon2d.min() < 0:
            lon2d = lon2d + 360.0

        self.lat2d_hi = lat2d
        self.lon2d_hi = lon2d

        cf = self.coarsen_factor  # 8
        # 粗化: 每 8x8 块取均值 → 24x24
        self.lat2d_lo = xr.DataArray(lat2d, dims=("y", "x")).coarsen(y=cf, x=cf).mean().values.astype(np.float32)
        self.lon2d_lo = xr.DataArray(lon2d, dims=("y", "x")).coarsen(y=cf, x=cf).mean().values.astype(np.float32)

        logger.info(
            f"[Grid] Loaded {grid_path.name}  hi={lat2d.shape}  lo={self.lat2d_lo.shape}\n"
            f"       lat_lo=[{self.lat2d_lo.min():.2f}, {self.lat2d_lo.max():.2f}]  "
            f"lon_lo=[{self.lon2d_lo.min():.2f}, {self.lon2d_lo.max():.2f}]"
        )

    # ── Step 2: 读取各 split 的目标 valid_time ───────────────────────────
    def _load_target_times(self):
        """
        从已有的 hrrr-hrrr {split}_data_in.nc 读取 valid_time (time) 序列,
        作为 ERA5 数据需要对齐的目标时间轴。
        """
        hrrr_region_dir = self.hrrr_dataset_dir / self.region

        for split in self.split_years:
            nc_path = hrrr_region_dir / f"{split}_data_in.nc"
            if not nc_path.exists():
                logger.warning(
                    f"[Time] HRRR reference file not found: {nc_path}, "
                    f"will use all available ERA5 times for {split}"
                )
                self.split_target_times[split] = None
                continue

            ds = xr.open_dataset(nc_path)
            times = pd.to_datetime(ds["time"].values)
            ds.close()
            self.split_target_times[split] = times
            logger.info(
                f"[Time] {split}: loaded {len(times)} target times from {nc_path.name}  "
                f"range={times[0]}~{times[-1]}"
            )

    # ── Step 3: 读取单个 ERA5 文件 ───────────────────────────────────────
    def _load_era5_file(self, var: str, year: int, month: int, day: int, hour: int) -> Optional[xr.Dataset]:
        """
        读取单个 ERA5 小时文件。
        路径格式: {era5_root}/{var}/{yyyy}/{mm}/{var}_{yyyymmdd}_{hh}00.nc
        """
        mm = f"{month:02d}"
        dd = f"{day:02d}"
        hh = f"{hour:02d}"
        fname = f"{var}_{year:04d}{mm}{dd}_{hh}00.nc"
        fpath = self.era5_root / var / f"{year:04d}" / mm / fname

        if not fpath.exists():
            return None

        try:
            ds = xr.open_dataset(fpath)
            return ds
        except Exception as e:
            logger.warning(f"[ERA5] Failed to open {fpath}: {e}")
            return None

    # ── Step 4: 插值单个变量到低分辨率网格 ──────────────────────────────
    def _interpolate_to_lowres(self, da: xr.DataArray, var: str) -> np.ndarray:
        """
        将 ERA5 的等经纬度 DataArray 插值到 24x24 的低分辨率网格。

        ERA5 是等经纬度网格 (lat 从大到小, lon 从 0 到 360),
        RegularGridInterpolator 要求经纬度单调递增, 需要翻转 lat 轴。

        参数:
            da:  ERA5 DataArray, 维度 (latitude, longitude)
            var: 变量名 (用于单位转换)

        返回:
            插值后的 24x24 numpy array (float32)
        """
        era5_lat = da.latitude.values  # 可能从大到小
        era5_lon = da.longitude.values  # 0-360
        data = da.values.astype(np.float64)

        # 单位转换: ERA5 降水单位 m → mm
        if var in ERA5_PRECIP_VARS:
            data = data * ERA5_PRECIP_SCALE
            data = np.clip(data, 0.0, None)  # 确保非负

        # 确保 lat 单调递增 (RegularGridInterpolator 要求)
        if era5_lat[0] > era5_lat[-1]:
            era5_lat = era5_lat[::-1]
            data = data[::-1, :]

        # 构建插值器
        interp = RegularGridInterpolator(
            (era5_lat, era5_lon),
            data,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # 目标点: 低分辨率网格的经纬度
        # ERA5 经度是 0-360, 低分辨率网格经度也统一为 0-360, 直接用
        target_lat = self.lat2d_lo.ravel()
        target_lon = self.lon2d_lo.ravel()
        points = np.stack([target_lat, target_lon], axis=-1)

        result = interp(points).reshape(self.lat2d_lo.shape).astype(np.float32)
        return result

    # ── Step 5: 读取并处理单个月份的 ERA5 数据 ───────────────────────────
    def open_era5_month(
        self,
        year: int,
        month: int,
        target_times: Optional[pd.DatetimeIndex],
    ) -> Optional[xr.Dataset]:
        """
        读取指定年月的 ERA5 数据, 插值到低分辨率网格, 对齐目标时间轴。

        处理流程:
          1. 逐变量、逐小时文件读取 ERA5
          2. 插值到 24x24 低分辨率网格
          3. 合并所有变量为一个 Dataset
          4. 将时间轴对齐到 target_times (HRRR valid_time 的交集)

        参数:
            year:         年份
            month:        月份
            target_times: 需要对齐的目标时间序列 (来自 HRRR data_in.nc)
                          None 表示保留所有时次

        返回:
            xr.Dataset (time, latitude, longitude) 或 None
        """
        import calendar

        n_days = calendar.monthrange(year, month)[1]

        # 需要读取的时间列表: 当月所有小时
        all_hours = [
            pd.Timestamp(year=year, month=month, day=d, hour=h) for d in range(1, n_days + 1) for h in range(24)
        ]

        # 如果有 target_times, 只读取其中属于本月的时次, 避免多余 IO
        if target_times is not None:
            month_mask = (target_times.year == year) & (target_times.month == month)
            needed_times = set(target_times[month_mask])
            if not needed_times:
                logger.info(f"[ERA5] {year}-{month:02d}: no target times in this month, skipping")
                return None
            all_hours = [t for t in all_hours if t in needed_times]

        if not all_hours:
            return None

        # 逐变量收集数据: {var: list of (timestamp, 24x24 array)}
        var_data: Dict[str, List] = {v: [] for v in self.input_vars}
        timestamps: List[pd.Timestamp] = []

        for ts in tqdm(all_hours, desc=f"  ERA5 {year}-{month:02d}", leave=False):
            hour_ok = True
            hour_arrays = {}

            for var in self.input_vars:
                ds = self._load_era5_file(var, ts.year, ts.month, ts.day, ts.hour)
                if ds is None:
                    msg = f"{year}-{month:02d}-{ts.day:02d}T{ts.hour:02d} var={var}: file not found"
                    logger.debug(f"[ERA5] {msg}")
                    self.missing_log.append(msg)
                    hour_ok = False
                    break

                # ERA5 文件只有一个时次, 取第一个 valid_time
                era5_var_name = var  # ERA5 文件内的变量名与外部一致
                if era5_var_name not in ds.data_vars:
                    # 有些文件内变量名可能不同, 尝试取第一个变量
                    era5_var_name = list(ds.data_vars)[0]

                da = ds[era5_var_name]
                # 如果有时间维度, squeeze 掉
                if "valid_time" in da.dims:
                    da = da.isel(valid_time=0)
                elif "time" in da.dims:
                    da = da.isel(time=0)

                # 确保有 latitude/longitude 维度
                if "latitude" not in da.dims or "longitude" not in da.dims:
                    logger.warning(f"[ERA5] {ts} {var}: unexpected dims {da.dims}, skipping")
                    hour_ok = False
                    break

                arr = self._interpolate_to_lowres(da, var)
                hour_arrays[var] = arr
                ds.close()

            if hour_ok:
                timestamps.append(ts)
                for var in self.input_vars:
                    var_data[var].append(hour_arrays[var])

        if not timestamps:
            logger.warning(f"[ERA5] {year}-{month:02d}: no valid hours loaded")
            return None

        logger.info(f"[ERA5] {year}-{month:02d}: loaded {len(timestamps)} hours")

        # 构建 Dataset
        time_arr = np.array([np.datetime64(t) for t in timestamps])
        ny, nx = self.lat2d_lo.shape  # 24, 24

        data_vars = {}
        for var in self.input_vars:
            arr_3d = np.stack(var_data[var], axis=0)  # (T, 24, 24)
            data_vars[var] = (["time", "latitude", "longitude"], arr_3d)

        ds_out = xr.Dataset(
            data_vars,
            coords={
                "time": ("time", time_arr),
                "latitude": (["latitude", "longitude"], self.lat2d_lo),
                "longitude": (["latitude", "longitude"], self.lon2d_lo),
            },
        )

        # 对齐到 target_times: 取交集并排序
        if target_times is not None:
            month_mask = (target_times.year == year) & (target_times.month == month)
            needed = target_times[month_mask]
            ds_times = pd.to_datetime(ds_out["time"].values)
            common = ds_times.intersection(needed)

            if len(common) == 0:
                logger.warning(f"[ERA5] {year}-{month:02d}: no common times after alignment")
                return None

            ds_out = ds_out.sel(time=common)
            if len(common) < len(needed):
                logger.warning(
                    f"[ERA5] {year}-{month:02d}: {len(needed) - len(common)} target times missing "
                    f"({len(common)}/{len(needed)} matched)"
                )

        ds_out = ds_out.sortby("time")
        # 确保降水非负
        if "tp" in ds_out.data_vars:
            ds_out["tp"] = ds_out["tp"].clip(min=0.0)

        return ds_out

    # ── Step 6: 处理单个月份 (保存分片) ─────────────────────────────────
    def process_month(self, year: int, month: int, split: str):
        """
        处理单个月份: 读取 ERA5 → 插值 → 对齐时间 → 保存分片文件

        如果分片文件已存在则跳过 (支持断点续处理)。
        """
        shard_dir = self.tmp_base / self.region / split
        shard_dir.mkdir(parents=True, exist_ok=True)

        shard_path = shard_dir / f"era5_data_in_{year:04d}_{month:02d}.nc"
        if shard_path.exists():
            logger.debug(f"[Skip] {shard_path.name} already exists")
            return

        target_times = self.split_target_times.get(split)

        ds = self.open_era5_month(year, month, target_times)
        if ds is None:
            msg = f"{year}-{month:02d} split={split}: no ERA5 data"
            logger.warning(f"[WARN] {msg}")
            self.missing_log.append(msg)
            return

        ds.to_netcdf(str(shard_path))
        logger.info(f"[OK] {split} {year:04d}-{month:02d}: saved {shard_path.name}  " f"shape={dict(ds.sizes)}")

    # ── Step 7: 合并月度分片 ─────────────────────────────────────────────
    def merge_splits(self):
        """
        将月度分片合并为 {split}_data_in_ERA5.nc
        """
        for split, years in self.split_years.items():
            if not years:
                continue

            out_path = self.region_dir / f"{split}_data_in_ERA5.nc"
            if out_path.exists():
                logger.info(f"[Merge] Already exists, skipping: {out_path.name}")
                continue

            shard_dir = self.tmp_base / self.region / split
            shard_files = sorted(glob.glob(str(shard_dir / "era5_data_in_*.nc")))

            if not shard_files:
                logger.warning(f"[Merge] No shards found for {split}")
                continue

            logger.info(f"[Merge] Merging {len(shard_files)} shards for {split} ...")

            ds_list = [xr.open_dataset(f) for f in tqdm(shard_files, desc=f"Merge {split}")]
            ds_merged = xr.concat(ds_list, dim="time").sortby("time")

            for ds_item in ds_list:
                ds_item.close()

            ds_merged.to_netcdf(str(out_path))
            logger.info(
                f"[Merge] Saved: {out_path}  "
                f"time={ds_merged.sizes['time']}  "
                f"shape=({ds_merged.sizes['latitude']}, {ds_merged.sizes['longitude']})"
            )

    # ── 主入口 ────────────────────────────────────────────────────────────
    def build_all(self):
        """
        主构建流程:
          1. 加载 192x192 高分辨率网格 & 粗化得到 24x24 低分辨率网格
          2. 读取各 split 的目标 valid_time (来自 HRRR data_in.nc)
          3. 逐 split × 年 × 月 处理 ERA5 数据
          4. 合并月度分片
          5. 保存缺失日志
        """
        logger.info("=" * 60)
        logger.info(f"Building ERA5 data_in for region: {self.region}")
        logger.info("=" * 60)

        # 1. 加载网格
        self._load_and_coarsen_grid()

        # 2. 读取目标时间
        self._load_target_times()

        # 3. 逐 split × 年 × 月 处理
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
                    import traceback

                    msg = f"[ERROR] {y}-{m:02d}: {e}\n{traceback.format_exc()}"
                    logger.error(msg)
                    self.missing_log.append(msg)

        # 4. 合并分片
        if self.merge_flag:
            self.merge_splits()

        # 5. 保存缺失日志
        if self.missing_log:
            log_path = self.region_dir / "missing_data_era5.log"
            with open(log_path, "w") as f:
                f.write("\n".join(self.missing_log))
            logger.warning(f"{len(self.missing_log)} issues logged to {log_path}")

        logger.info("Done.")


# ── 命令行接口 ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Build ERA5 low-res input data for hrrr-hrrr downscaling dataset")
    p.add_argument(
        "--era5-root",
        default="/path/to/project_climate/data/ERA5/ERA5_data_in_rootdata6/hourly/global",
        help="ERA5 数据根目录, 结构: {var}/{yyyy}/{mm}/{var}_{yyyymmdd}_{hh}00.nc",
    )
    p.add_argument(
        "--region-grid-dir",
        default="/path/to/project_climate/Benchmark/preprocess/regions_lat_lon",
        help="高分辨率区域网格文件目录",
    )
    p.add_argument(
        "--hrrr-dataset-dir",
        default="",
        help="已构建的 hrrr-hrrr 数据集根目录, 用于读取目标 valid_time",
    )
    p.add_argument(
        "--out-dir",
        default="/path/to/project_climate/Benchmark/preprocess/hrrr-hrrr/dataset_root",
        help="输出根目录 (ERA5 data_in 文件将保存到 {out_dir}/{region}/)",
    )
    p.add_argument("--region", default="part1-US-Rocky")
    p.add_argument("--input-vars", nargs="+", default=["tp", "t2m", "u10", "v10"])
    p.add_argument("--train-years", nargs="+", type=int, default=list(range(2015, 2023)))
    p.add_argument("--val-years", nargs="+", type=int, default=[2023])
    p.add_argument("--test-years", nargs="+", type=int, default=[2024])
    p.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    p.add_argument("--coarsen-factor", type=int, default=8)
    p.add_argument("--tmp-dir", default=None)
    p.add_argument("--no-merge", action="store_true")
    return p.parse_args()


def main():
    setup_logging(logging.INFO)
    args = parse_args()

    builder = ERA5DataInBuilder(
        era5_root=args.era5_root,
        region_grid_dir=args.region_grid_dir,
        hrrr_dataset_dir=args.hrrr_dataset_dir,
        out_dir=args.out_dir,
        region=args.region,
        input_vars=args.input_vars,
        split_years={
            # "train": args.train_years,
            # "val": args.val_years,
            "test": args.test_years,
        },
        months=args.months,
        coarsen_factor=args.coarsen_factor,
        tmp_dir=args.tmp_dir,
        merge_flag=not args.no_merge,
    )
    builder.build_all()


def run():
    """批量处理所有 5 个区域"""
    regions = [
        # "part1-US-Rocky",
        "part1-US-Cascades",
        "part1-US-Nevada",
        "part1-US-Appalachian",
        "part1-US-Great_Plains",
    ]
    setup_logging(logging.INFO)
    for region in regions:
        builder = ERA5DataInBuilder(
            era5_root="/path/to/project_climate/data/ERA5/ERA5_data_in_rootdata6/hourly/global",
            region_grid_dir="/path/to/project_climate/Benchmark/preprocess/regions_lat_lon",
            hrrr_dataset_dir="",
            out_dir="/path/to/project_climate/Benchmark/preprocess/hrrr-hrrr/dataset_root",
            region=region,
        )
        builder.build_all()


if __name__ == "__main__":
    # main()
    run()
