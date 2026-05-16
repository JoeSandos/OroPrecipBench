import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

MAX_ENS = 4


def _infer_pred_da(arr: Any) -> xr.DataArray:
    """将 numpy/xarray 输入转换为具有标准预测维度的 xarray DataArray。

    预测维度的目标结构：(time, ensemble, y, x)
    - 如果是 4D：假定为 (time, ensemble, y, x)。
    - 如果是 3D：假定为 (time, y, x)，并添加大小为 1 的 ensemble 维度。
    """
    if isinstance(arr, xr.DataArray):
        da = arr
        # 将各种空间维度名称映射为标准名称 (y, x)
        rename_mapping = {
            "y_grid": "y",
            "south_north": "y",
            "x_grid": "x",
            "west_east": "x",
        }
        # 仅在目标维度名不存在时才进行重命名，避免冲突
        actual_rename = {k: v for k, v in rename_mapping.items() if k in da.dims and v not in da.dims}
        if actual_rename:
            da = da.rename(actual_rename)

        # 如果维度名称是以 "dim_" 开头的默认名称，则进行重命名
        if all(str(d).startswith("dim_") for d in da.dims):
            if da.ndim == 4:
                da = da.rename(
                    {
                        da.dims[0]: "time",
                        da.dims[1]: "ensemble",
                        da.dims[2]: "y",
                        da.dims[3]: "x",
                    }
                )
            elif da.ndim == 3:
                da = da.rename(
                    {
                        da.dims[0]: "time",
                        da.dims[1]: "y",
                        da.dims[2]: "x",
                    }
                )
        # 统一处理 3D 情况，增加 ensemble 维度
        if da.ndim == 3:
            da = da.expand_dims({"ensemble": [0]})

        # 尝试按照标准顺序排列维度
        target_dims = ("time", "ensemble", "y", "x")
        existing_dims = [d for d in target_dims if d in da.dims]
        if len(existing_dims) == da.ndim:
            da = da.transpose(*existing_dims)

        return da

    arr = np.asarray(arr)
    if arr.ndim == 4:
        da = xr.DataArray(arr, dims=("time", "ensemble", "y", "x"))
    elif arr.ndim == 3:
        da = xr.DataArray(arr, dims=("time", "y", "x"))
        da = da.expand_dims({"ensemble": [0]})
        da = da.transpose("time", "ensemble", "y", "x")
    else:
        raise ValueError(f"预测数组必须是 3D 或 4D，当前形状为 {arr.shape}")
    return da


def _infer_truth_da(arr: Any) -> xr.DataArray:
    """将 numpy/xarray 输入转换为具有标准真值维度的 xarray DataArray (time, y, x)。
    如果存在 ensemble 维度，则取第一个成员。
    """
    if isinstance(arr, xr.DataArray):
        da = arr
        # 将各种空间维度名称映射为标准名称 (y, x)
        rename_mapping = {
            "y_grid": "y",
            "south_north": "y",
            "x_grid": "x",
            "west_east": "x",
        }
        actual_rename = {k: v for k, v in rename_mapping.items() if k in da.dims and v not in da.dims}
        if actual_rename:
            da = da.rename(actual_rename)

        if all(str(d).startswith("dim_") for d in da.dims):
            if da.ndim == 4:
                da = da.rename(
                    {
                        da.dims[0]: "time",
                        da.dims[1]: "ensemble",
                        da.dims[2]: "y",
                        da.dims[3]: "x",
                    }
                )
            elif da.ndim == 3:
                da = da.rename(
                    {
                        da.dims[0]: "time",
                        da.dims[1]: "y",
                        da.dims[2]: "x",
                    }
                )
        # 真值不包含 ensemble 维度
        if "ensemble" in da.dims:
            da = da.isel(ensemble=0)

        # 尝试按照标准顺序排列维度 (time, y, x)
        target_dims = ("time", "y", "x")
        existing_dims = [d for d in target_dims if d in da.dims]
        if len(existing_dims) == da.ndim:
            da = da.transpose(*existing_dims)

        return da

    arr = np.asarray(arr)
    if arr.ndim == 4:
        da = xr.DataArray(arr, dims=("time", "ensemble", "y", "x"))
        da = da.isel(ensemble=0)
    elif arr.ndim == 3:
        da = xr.DataArray(arr, dims=("time", "y", "x"))
    else:
        raise ValueError(f"真值数组必须是 3D 或 4D，当前形状为 {arr.shape}")
    return da


def _load_nc(entry: Dict[str, Any]) -> xr.DataArray:
    """从 NetCDF 加载指定变量。支持 group 路径。"""
    path = Path(entry["path"])
    group = entry.get("group", "")
    var = entry.get("var")
    ds = (
        xr.open_dataset(path, group=group, chunks="auto", mask_and_scale=False)
        if group
        else xr.open_dataset(path, chunks="auto", mask_and_scale=False)
    )
    if var is None:
        # 如果未指定变量，则尝试获取第一个数据变量
        if not ds.data_vars:
            raise ValueError(f"在 {path} 中未找到数据变量")
        var = next(iter(ds.data_vars))
    if var not in ds:
        raise ValueError(f"在 {path} (group={group}) 中未找到变量 {var}")
    return ds[var]


def _load_npz(entry: Dict[str, Any]) -> Any:
    """从 NPZ 文件中加载变量。"""
    path = Path(entry["path"])
    var = entry.get("var")
    with np.load(path, allow_pickle=True) as data:
        if var is None:
            if not getattr(data, "files", []):
                raise ValueError(f"在 {path} 中未找到变量")
            var = data.files[0]
        if var not in data:
            raise ValueError(f"在 {path} 中未找到变量 {var}")
        return data[var]


def limit_ensemble_size(da: xr.DataArray, max_ens: Optional[int] = MAX_ENS) -> xr.DataArray:
    """限制预测 DataArray 的集合数量，仅保留前 max_ens 个成员。

    如果 max_ens 为 None，则不限制。
    """
    if max_ens is None:
        return da
    if "ensemble" not in da.dims:
        return da
    ens_size = int(da.sizes["ensemble"])
    if ens_size <= max_ens:
        return da
    print(f"正在将集合大小从 {ens_size} 限制为 {max_ens}")
    return da.isel(ensemble=slice(0, max_ens))


def _load_entry(entry: Dict[str, Any]) -> xr.DataArray:
    """根据文件后缀分发加载任务。"""
    suffix = Path(entry["path"]).suffix.lower()
    if suffix == ".nc":
        return _load_nc(entry)
    if suffix == ".npz":
        return xr.DataArray(_load_npz(entry))
    raise ValueError(f"不支持的文件类型: {entry['path']}")


def _load_entry_with_keys(entry: Dict[str, Any], *, var_key: str = "var", group_key: str = "group") -> xr.DataArray:
    """加载配置项，允许使用替代的变量或组键名（常用于真值对应条目中的参考预测）。"""
    spec = {
        "path": entry["path"],
        "var": entry.get(var_key, entry.get("var")),
        "group": entry.get(group_key, entry.get("group")),
    }
    return _load_entry(spec)


def _standardize_shape(
    preds: List[Dict[str, Any]], truth_entry: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """将所有预测值和真值裁剪为相同的最小空间分辨率（高x宽）。"""
    truth_shape_2d = truth_entry["truth"].shape[-2:]

    shape_list: List[tuple[int, int]] = [truth_shape_2d]
    shape_list.extend([p["pred"].shape[-2:] for p in preds])
    if truth_entry.get("ref_pred") is not None:
        shape_list.append(truth_entry["ref_pred"].shape[-2:])

    shape_arr = np.asarray(shape_list, dtype=int)
    small_h = int(shape_arr[:, 0].min())
    small_w = int(shape_arr[:, 1].min())

    def _crop(da: xr.DataArray, shape: tuple[int, int]) -> xr.DataArray:
        new_h, new_w = shape
        if len(da.shape) == 4:
            _, _, height, width = da.shape
        elif len(da.shape) == 3:
            _, height, width = da.shape
        else:
            raise ValueError(f"数组必须是 3D 或 4D，当前形状为 {da.shape}")
        if height == new_h and width == new_w:
            return da

        # 居中裁剪
        start_h = (height - new_h) // 2
        start_w = (width - new_w) // 2
        return da.isel(
            {
                da.dims[-2]: slice(start_h, start_h + new_h),
                da.dims[-1]: slice(start_w, start_w + new_w),
            }
        )

    truth_entry["truth"] = _crop(truth_entry["truth"], (small_h, small_w))
    if truth_entry.get("ref_pred") is not None:
        truth_entry["ref_pred"] = _crop(truth_entry["ref_pred"], (small_h, small_w))
    for p in preds:
        p["pred"] = _crop(p["pred"], (small_h, small_w))

    print(f"已将所有数组标准化为形状: {small_h} x {small_w}")
    return preds, truth_entry


def load_config_predictions_truth(
    config_path: Path,
    *,
    max_ens: Optional[int] = MAX_ENS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """从 JSON 配置文件加载预测值和真值。

    JSON 格式示例：
    {
      "entries": [
        {"label": "truth", "path": "...", "group": "output", "var": "tp"},
        {"label": "model_a", "path": "...", "group": "prediction", "var": "tp"},
        {"label": "model_b", "path": "...", "var": "pred"}
      ]
    }
    - 对于 .nc 文件：可选择指定 group，必须指定 var。
    - 对于 .npz 文件：若不是第一个键名，必须指定 var。
    - 真值条目的标签（label）必须为 "truth"。

    返回：(pred_list, truth_entry)
    pred_list: [{"label": str, "pred": xr.DataArray}]
    truth_entry: {"label": "truth", "truth": xr.DataArray, "ref_pred": Optional[xr.DataArray]}
    """
    cfg = json.loads(Path(config_path).read_text())
    entries = cfg.get("entries", [])
    if not entries:
        raise ValueError("配置文件 JSON 必须包含 'entries' 字段")

    truth_entry: Optional[Dict[str, Any]] = None
    preds: List[Dict[str, Any]] = []

    for e in entries:
        if e["path"] is None:
            continue
        if "path" not in e or "label" not in e:
            raise ValueError("每个条目必须包含 'path' 和 'label' 字段")
        label = str(e["label"])
        arr = _load_entry(e)
        if label == "truth":
            truth_entry = {
                "label": label,
                "path": Path(e["path"]),
                "truth": limit_ensemble_size(_infer_truth_da(arr), max_ens=max_ens),
            }
            # 特殊情况：truth 条目可能包含对应的参考预测字段 (pred_var/pred_group)
            if "pred_var" in e or "pred_group" in e:
                ref_arr = _load_entry_with_keys(e, var_key="pred_var", group_key="pred_group")
                truth_entry["ref_pred"] = limit_ensemble_size(_infer_pred_da(ref_arr), max_ens=max_ens)
        else:
            preds.append(
                {
                    "label": label,
                    "path": Path(e["path"]),
                    "pred": limit_ensemble_size(_infer_pred_da(arr), max_ens=max_ens),
                }
            )

    if truth_entry is None:
        raise ValueError("配置中必须包含一个标签为 'truth' 的条目")
    # 调整并统一所有数据块的空间大小
    preds, truth_entry = _standardize_shape(preds, truth_entry)

    return preds, truth_entry
