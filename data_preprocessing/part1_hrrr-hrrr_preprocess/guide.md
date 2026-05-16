<!-- 根据 @hrrr-hrrr/guide.md  实现 @hrrr-hrrr/step1_build_hrrr-hrrr.py 和  @hrrr-hrrr/step2_cal_stats.py   -->

# 构建 hrrr-hrrr 小时级 降尺度数据集

- 低分辨率输入数据: HRRR 3km 粗化至 25km, 变量: tp, t2m, u10, v10, shape: (24, 24)
- 高分辨率输出数据: HRRR 3km 原始分辨率, 变量: tp, shape: (192, 192)
- 高分辨率地形数据: HRRR 3km 原始分辨率的地形数据, 变量: elevation, lsm, shape: (192, 192)

## A. 数据说明

### hrrr 3km 
    ● 路径: /path/to/project_climate/Benchmark_process/data/hrrr/{var}/hrrr/{yyyymmdd}/{*.grib2}

    sizes: Frozen({'time': 24, 'y': 1059, 'x': 1799})

    Coordinates:
        * time        (time) datetime64[ns] 192B 2015-01-01 ... 2015-01-01T23:00:00
        step        timedelta64[ns] 8B 01:00:00
        surface     float64 8B 0.0
        latitude    (y, x) float64 15MB dask.array<chunksize=(1059, 1799), meta=np.ndarray>
        longitude   (y, x) float64 15MB dask.array<chunksize=(1059, 1799), meta=np.ndarray>
        valid_time  (time) datetime64[ns] 192B 2015-01-01T01:00:00 ... 2015-01-02

    Data variables:
        {var}       (time, y, x) float32 183MB dask.array<chunksize=(1, 1059, 1799), meta=np.ndarray>
    
    注意：{var} 可以是 tp, t2m, u10, v10 等变量
    注意：hrrr的不同变量数据存在不同程度的缺失, 因此 merge 不同 变量的时候需要取交集, 以保证时间维度一致.
    
#### hrrr 数据的时间维度说明：
+ 瞬时值 instant：风速、温度、...
    - 正常读取 【默认：filter_by_keys={'stepType': 'instant'}】
    - 预报时效：fxx=0
+ 累积量 accum：降水、辐射、...
    - 注意①：读取：filter_by_keys={'stepType': 'accum'}
    - 注意②：
        - fxx（step）：预报时效
        - time：起报时间
        - valid_time = time + step → 预报对应的实际时间 → ERA5用的时间


### 海陆掩码：hrrr 3km 原始分辨率的海陆掩码数据, 变量: lsm
    ● 路径: /path/to/project_climate/Benchmark_process/data/hrrr/lsm/subset_0aef124b__hrrr.t12z.wrfsfcf00.grib2

    sizes: Frozen({'y': 1059, 'x': 1799})

    Coordinates:
        time        datetime64[ns] 
        step        timedelta64[ns] 
        surface     float64 
        latitude    (y, x) float64 
        longitude   (y, x) float64 
        valid_time  datetime64[ns] 
    
    Data variables:
        lsm      (y, x) float32 


## SRTM_30m: 高精度的全球的地形数据.
    ● 路径: /path/to/project_climate/Benchmark_process/data/DEM/SRTM_30m_CONUS.tif
    ● 建议使用 rioxarray 处理

        ```python
        # 这是 rioxarray 读取方法
        dem_us_da = rioxarray.open_rasterio(US_DEM_PATH) # xarray.DataArray ({'band': 1, 'y': 36000, 'x': 84000})
        '''
        Coordinates:
        * band         (band) int64 8B 1
        * y            (y) float64 288kB 52.0 52.0 52.0 52.0 ... 22.0 22.0 22.0 22.0
        * x            (x) float64 672kB -130.0 -130.0 -130.0 ... -60.0 -60.0 -60.0
            spatial_ref  int64 8B 0
        '''
        
        # 这是 rasterio 读取方法
        with rasterio.open(US_DEM_PATH) as src:
            dem_us = src.read(1)
            transform_us = src.transform
        ```

## 高分辨率网格信息: 
    通过 `grid.ipynb` 生成的高分辨率网格信息文件，包含每个区域的经纬度坐标和网格尺寸等信息。
    regions_lat_lon/
    ├── part1-US-Appalachian_lat_lon_192_192.nc
    ├── part1-US-Cascades_lat_lon_192_192.nc
    ├── part1-US-Great_Plains_lat_lon_192_192.nc
    ├── part1-US-Nevada_lat_lon_192_192.nc
    └── part1-US-Rocky_lat_lon_192_192.nc


## B. 数据集规范

### 一、目录结构
```markdown
dataset_root/
├── part1-US-Rocky/
│   ├── train_data_in.nc        # 训练集输入（低分辨率）
│   ├── train_data_out.nc       # 训练集输出（高分辨率）
│   ├── val_data_in.nc          # 验证集输入（低分辨率）
│   ├── val_data_out.nc         # 验证集输出（高分辨率）
│   ├── test_data_in.nc         # 测试集输入（低分辨率）
│   ├── test_data_out.nc        # 测试集输出（高分辨率）
│   ├── static_variables.nc     # 静态地理变量（高分辨率）
│   └── stats.json              # 归一化统计量
├── part1-US-Cascades\
├── part1-US-Nevada\
├── part1-US-Appalachian\
└── part1-US-Great_Plains\
```

<font style="color:rgb(52, 73, 94);">每个区域（region）是一个独立子目录，包含上述 8 个文件。所有时间序列数据均使用 </font>**<font style="color:rgb(44, 62, 80);">NetCDF4 (.nc)</font>**<font style="color:rgb(52, 73, 94);"> 格式存储。</font>

### 二、时空分辨率规定
| **数据集** | **时间范围** | **时间分辨率** | **输入数据的shape**<br/>**（低分辨率）** | **输出/真值/DEM的shape**<br/>**（高分辨率）**<br/>**<font style="color:#DF2A3F;">所有保持一致</font>** |
| --- | --- | --- | --- | --- |
| part1-US-* | 训练期：2015-2022<br/>验证期：2023<br/>测试期：2024 | 小时 | 24*24 | 192*192 |

### 三、文件详细规格
#### 3.1 气象数据
+ 输入：`train_data_in.nc` / `val_data_in.nc`/ `test_data_in.nc`
+ 输出：`train_data_out.nc` / `val_data_out.nc`/ `test_data_out.nc`

**维度（Dimensions）：**`**（time, latitude, longitude）**`

**坐标（Coordinates）：**

+ time：datetime64[ns], 逐小时或逐日
+ latitude：（longitude，latitude）float32，
+ longitude：（longitude，latitude）float32，范围<font style="color:#DF2A3F;">0°-360°</font>。
+ PS：经纬度网格使用二维的形式

**数据变量（Data Variables）：**

+ 所有变量维度为： `**(time, latitude, longitude)**`
+ 输入数据，有**多个变量**
+ 输出数据，仅有**一个变量**`**tp**`
+ **<font style="color:#DF2A3F;">注意：降水单位统一使用 mm/h。</font>** 
+ hrrr的降水单位是 kg/m² 等价于 mm. 逐小时降水量即为 mm/h.
+ **统一变量名和单位（欢迎补充）**

| **变量名** | **全称** | **单位** |
| --- | --- | --- |
| `**tp**` | **总降水量** | **mm/h** |
| `**u10**` | **10m 纬向风** | **m/s** |
| `**v10**` | **10m 经向风** | **m/s** |
| `**sp**` | **地表气压** | **Pa** |
| `**t2m**` | **2m 气温** | **K** |
| …… | **** | **** |


#### 3.2 静态地理变量：`static_variables.nc`
**维度（Dimensions）：**`**（latitude, longitude）**`

**坐标（Coordinates）：**

+ latitude：（longitude，latitude）float32，
+ longitude：（longitude，latitude）float32，范围<font style="color:#DF2A3F;">0°-360°</font>。
+ PS：经纬度网格使用二维的形式

**数据变量（Data Variables）：**

+ 2个变量：
    - **elevation：**<font style="background-color:rgb(248, 248, 248);">高程/海拔</font>
    -  **lsm**：陆海掩膜（0~1）
+ 维度为： `**(latitude, longitude)**`


### 3.3  归一化统计量：`stats.json`
JSON 文件，存储**<font style="color:#DF2A3F;">训练集</font>**各变量的全局统计量，用于数据预处理（归一化/标准化）。

**结构**

```json
{
  "input": {
    "<var_name>": {
      "mean": <float>,
      "std": <float>,
      "min": <float>,
      "max": <float>
    },
    "tp": {
      "mean": <float>,
      "std": <float>,
      "min": <float>,
      "max": <float>,
      "log_mean": <float>,
      "log_std": <float>,
      "log_min": <float>,
      "log_max": <float>,
      "log_eps": <float>
    }
  },
  "output": {
    "tp": {
      "mean": <float>,
      "std": <float>,
      "min": <float>,
      "max": <float>,
      "log_mean": <float>,
      "log_std": <float>,
      "log_min": <float>,
      "log_max": <float>,
      "log_eps": <float>
    }
  },
  "invariant": {
    "<var_name>": {
      "mean": <float>,
      "std": <float>,
      "min": <float>,
      "max": <float>
    }
  }
}
```

**字段说明**

| 字段 | 说明 |
| --- | --- |
| `mean` | 该变量在训练集上的全局均值 |
| `std` | 该变量在训练集上的全局标准差 |
| `min` | 该变量在训练集上的全局最小值 |
| `max` | 该变量在训练集上的全局最大值 |
| `log_mean` | <font style="color:#DF2A3F;">log(1 + x) </font>变换后的均值（仅降水类变量） |
| `log_std` | <font style="color:#DF2A3F;">log(1 + x) </font>变换后的标准差（仅降水类变量） |
| `log_min` / `log_max` | log 变换后的极值（仅降水类变量） |
| `log_eps` | log 变换的偏移量，默认为 1.0（仅降水类变量） |


## C. 要求

### C.1 hrrr-hrrr/step1_build_hrrr-hrrr.py

1. 写一个 Python 类用于构建 HRRR-HRRR 的数据集。
    - 主函数一个 python 类
2. 优先生成 `static_variables.nc`, 包含高分辨率的地形数据（elevation）和海陆掩码（lsm）。 
    - 这部分数据只需生成一次，后续训练/验证/测试集共用。
    - 生成前判断 `static_variables.nc` 是否存在，若存在则跳过
3. 我建议一次只处理一个月份的数据，使用 glob 搜索。然后存储到硬盘上，最后再合并成一个文件。
    - 每次处理前判断一下文件是否存在，如果存在则跳过。
4. 需要能够指定：低分辨率输入数据、高分辨率输出数据、高分辨率地形数据 的变量 {var}
    - 默认：
        - 低分辨率输入数据：tp, t2m, u10, v10
        - 高分辨率输出数据：tp
        - 高分辨率地形数据：elevation, lsm
5. 需要能够指定：训练/验证/测试集的时间范围
    - 默认：
        - 训练集：2015-2022
        - 验证集：2023
        - 测试集：2024
6. 需要能够指定：处理哪个区域（region）
    - 默认：part1-US-Rocky
7. 需要自由指定输入输出路径
    - 默认输入路径：参考上面的数据说明部分
    - 默认输出路径：/path/to/project_climate/Benchmark_process/hrrr-hrrr/dataset_root
        - 需要在输出路径下自动创建区域子目录（例如 part1-US-Rocky）
8. 跳过不存在的数据文件，并记录日志
9. 需要在处理过程中显示进度（例如使用 tqdm）
10. 需要对原始分辨率的 HRRR 粗化至 25km. shape 从 (192, 192) 粗化至 (24, 24)
11. 所有数据的经纬度网格使用二维的形式，经度范围为 0°-360°。ps：HRRR 是 WRF 格式的网格
12. 可以参考 `era5-hrrr/build_era5_hrrr.py`: 构建 ERA5-HRRR 降尺度数据集
13. 考虑使用 dask 加速
14. 环境：conda activate climate. 已安装部分包, 可以安装更多的包.


### C.2 hrrr-hrrr/step2_cal_stats.py

1. 写一个 Python 脚本用于计算训练集的统计量，并保存为 `stats.json`。
2. 统计量包括：均值、标准差、最小值、最大值等基本统计量，以及针对降水类变量的 log(1 + x) 变换后的统计量。
3. 需要能够指定：输入数据路径、输出路径、变量列表
4. 需要能够指定：处理哪个区域（region）
5. 可以参考 `era5-hrrr/cal_stats.py`

## D. 可参考代码

### 1. 伪代码
```python
# 伪代码
class HRRREra5Builder:
    def __init__():
        pass

    def build_all(self):
        # 主入口
        
        # 1) 准备目标插值网格
        self._load_target_grid()
        
        # 2) 不变量
        self.compute_and_save_invariant()
        
        # 3) 逐 split 遍历 年-月
        for split, years in self.split_years.items():
            if len(years) == 0:
                continue
            print(f"处理: {split} 集")
            for y, m in tqdm(itertools.product(years, self.months), total=len(years) * len(self.months)):
                self.process_month(int(y), int(m))
        
        # 4) 合并 split
        if self.merge_flag:
            self.merge_splits()
```

### 2. 读取指定年月的 HRRR 数据, 返回合并后的 xarray.Dataset

``` python
    def open_hrrr_month_day_by_day(self, year: int, month: int) -> xr.Dataset:
        """
        读取指定年月的 HRRR 数据, 返回合并后的 xarray.Dataset
        """
        parts: dict[str, xr.DataArray] = {}
        for var in self.hrrr_vars:
            month_dir = self.hrrr_root / var / "hrrr"
            if not month_dir.exists():
                raise FileNotFoundError(f"HRRR variable directory not found: {month_dir}")

            # 搜索该月所有天的文件夹，例如 data/hrrr/tp/hrrr/yyyymmdd/*.grib2
            daily_dirs = sorted(glob.glob(str(month_dir / f"{year:04d}{month:02d}??")))
            if not daily_dirs:
                print(f"No daily HRRR folders for {year}-{month:02d} var={var}")
                continue

            # 读取每一天的数据
            daily_datasets = []
            for daily_dir in daily_dirs:
                # print(f"\t\t加载 HRRR 变量 {var} 日期 {os.path.basename(daily_dir)} ...")
                daily_files = sorted(glob.glob(str(Path(daily_dir) / "*.grib2")))
                if not daily_files:
                    print(f"No daily HRRR files for {daily_dir}")
                    continue

                daily_ds = xr.open_mfdataset(
                    daily_files,
                    combine="nested",
                    concat_dim="time",
                    engine="cfgrib",
                    parallel=True,
                )

                daily_ds = daily_ds.sortby("valid_time")
                if var not in daily_ds.data_vars:
                    raise ValueError(f"Variable {var} not found in HRRR files for {year}-{month:02d}")
                daily_datasets.append(daily_ds[[var]])

            # 合并每天的数据
            ds_month = xr.concat(daily_datasets, dim="time")
            parts[var] = ds_month[var]

        # 合并
        merged = xr.Dataset(parts)
        # 去除无用坐标
        if "surface" in merged.coords:
            merged = merged.drop_vars("surface")
        if "step" in merged.coords:
            merged = merged.drop_vars("step")

        # 统一时间坐标名称为 time
        merged = merged.swap_dims({"time": "valid_time"})
        merged = merged.drop_vars("time")
        merged = merged.rename({"valid_time": "time"})

        return merged
```

