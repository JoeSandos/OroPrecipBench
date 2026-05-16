#!/usr/bin/env python3
import argparse
import os
import pwd
import subprocess
from typing import List, Tuple
from typing import Optional, Dict, Any


def get_current_user() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except Exception:
        return os.getenv("USER", "")


def run_ps(user: str) -> List[str]:
    # Use ps to list processes for the user with desired columns; remove headers for easier parsing
    cmd = [
        "ps",
        "-u",
        user,
        "--sort=-%mem",
        "-o",
        "pid=",
        "-o",
        "%mem=",
        "-o",
        "rss=",
        "-o",
        "vsz=",
        "-o",
        "cmd=",
    ]
    out = subprocess.check_output(cmd, text=True, errors="replace")
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines


def parse_ps_line(line: str) -> Tuple[int, float, int, int, str]:
    # pid, %mem, rss, vsz, cmd (rss/vsz are in KiB per ps man)
    parts = line.split(None, 4)
    if len(parts) < 5:
        # If command column is empty, fill with empty string
        parts = (parts + [""])[:5]
    pid = int(parts[0])
    pmem = float(parts[1])
    rss_kb = int(parts[2])
    vsz_kb = int(parts[3])
    cmd = parts[4]
    return pid, pmem, rss_kb, vsz_kb, cmd


def read_smaps_rollup(pid: int) -> Tuple[int, int]:
    """从 /proc/<pid>/smaps_rollup 读取 Rss/Pss (单位 KiB)。失败时返回 (0, 0)。"""
    try:
        with open(f"/proc/{pid}/smaps_rollup", "r") as f:
            smap = f.read()
        rss = int([L for L in smap.splitlines() if L.startswith("Rss")][0].split()[1])
        pss = int([L for L in smap.splitlines() if L.startswith("Pss")][0].split()[1])
        return rss, pss
    except Exception:
        return 0, 0


def kb_to_human_mbgb(kb: int) -> str:
    """自动将 KiB 转换为 KB/MB/GB 字符串。"""
    if kb < 1024:
        return f"{kb} KB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.1f} MB"
    gb = mb / 1024.0
    return f"{gb:.2f} GB"


def format_row(
    pid: int,
    pmem: float,
    rss_kb: int,
    vsz_kb: int,
    cmd: str,
    pss_kb: int = 0,
    shared_ratio: float | None = None,
    width_cmd: int = 60,
) -> str:
    rss_h = kb_to_human_mbgb(rss_kb)
    pss_h = kb_to_human_mbgb(pss_kb) if pss_kb > 0 else "--"
    shr = f"{shared_ratio*100:5.1f}%" if (shared_ratio is not None) else "   --"
    vsz_h = kb_to_human_mbgb(vsz_kb)
    cmd_show = (cmd[: width_cmd - 3] + "...") if len(cmd) > width_cmd else cmd
    return f"{pid:>7}  {pmem:6.2f}%  {rss_h:>10}  {pss_h:>10}  {shr:>8}  {vsz_h:>10}  {cmd_show}"


def format_group_row(
    cmd: str,
    count: int,
    pmem_sum: float,
    rss_sum_kb: int,
    pss_sum_kb: int,
    shared_ratio: float | None,
    width_cmd: int = 80,
) -> str:
    cmd_show = (cmd[: width_cmd - 3] + "...") if len(cmd) > width_cmd else cmd
    rss_h = kb_to_human_mbgb(rss_sum_kb)
    pss_h = kb_to_human_mbgb(pss_sum_kb) if pss_sum_kb > 0 else "--"
    shr = f"{shared_ratio*100:5.1f}%" if (shared_ratio is not None) else "   --"
    return f"{cmd_show:<{width_cmd}}  {count:>6}  {pmem_sum:6.2f}%  {rss_h:>10}  {pss_h:>10}  {shr:>8}"


def get_grouped_mem_usage(
    top: int = 20,
    user: Optional[str] = None,
) -> Dict[str, Any]:
    """
    返回按命令(`cmd`)合并后的内存使用情况与汇总，供其他脚本调用。

    参数:
    - top: 选取前 N 个进程（按 `%MEM` 降序）；若 <=0 则使用全部进程。
    - user: 指定用户名；为空则自动获取当前用户。

    返回字典包含:
    - groups: List[Dict]，每项字段如下:
        - cmd: 进程命令字符串
        - count: 分组内进程数
        - pmem: 分组内 `%MEM` 合计（浮点，单位为百分比值相加）
        - rss_kb: 分组常驻内存 RSS 合计（KiB）
        - pss_kb: 分组比例集内存 PSS 合计（KiB）
        - sm_rss_kb: 分组 smaps_rollup 中 Rss 合计（KiB）
        - rss_h: 格式化后的 RSS 字符串
        - pss_h: 格式化后的 PSS 字符串
        - sm_rss_h: 格式化后的 smaps_rollup Rss 字符串
        - shared_ratio: 共享比例估计（0~1；None 表示不可计算），按 1 - PSS/SMAPS_RSS 估计
    - totals: Dict，总计（针对选取的 top 行）：
        - pmem: `%MEM` 合计（浮点）
        - rss_kb: RSS 合计（KiB）
        - vsz_kb: VSZ 合计（KiB）
        - pss_kb: PSS 合计（KiB）
        - rss_h: 格式化后的 RSS 字符串
        - vsz_h: 格式化后的 VSZ 字符串
        - pss_h: 格式化后的 PSS 字符串

    说明:
    - `%MEM` 由 `ps` 提供并已按降序排序。
    - RSS/VSZ 单位为 KiB（来自 `ps`），PSS/SMAPS_RSS 来自 `/proc/<pid>/smaps_rollup`，单位为 KiB。
    - `shared_ratio` 为粗略估计值，仅在 SMAPS_RSS 与 PSS 均可用时计算。
    """

    # 解析用户
    u = user or get_current_user()
    if not u:
        raise RuntimeError("无法确定用户，需显式传入 user 或设置环境变量 USER")

    # 收集进程行
    lines = run_ps(u)
    rows: List[Tuple[int, float, int, int, str]] = []
    for line in lines:
        try:
            pid, pmem, rss_kb, vsz_kb, cmd = parse_ps_line(line)
            rows.append((pid, pmem, rss_kb, vsz_kb, cmd))
        except Exception:
            continue

    # 选取 top 行（ps 已按 %MEM 降序）
    top_rows = rows[:top] if (isinstance(top, int) and top > 0) else rows

    # 读取 PSS 与共享比例（仅对选取行）
    annotated: List[Tuple[int, float, int, int, str, int, Optional[float], int]] = []
    total_pss_kb = 0
    for r in top_rows:
        pid, pmem, rss_kb, vsz_kb, cmd = r
        sm_rss_kb, pss_kb = read_smaps_rollup(pid)
        shared_ratio: Optional[float] = None
        if sm_rss_kb > 0 and pss_kb > 0:
            try:
                shared_ratio = 1.0 - (pss_kb / sm_rss_kb)
            except Exception:
                shared_ratio = None
        annotated.append((pid, pmem, rss_kb, vsz_kb, cmd, pss_kb, shared_ratio, sm_rss_kb))
        if pss_kb > 0:
            total_pss_kb += pss_kb

    # 汇总 totals（针对选取行）
    totals = {
        "pmem": sum(r[1] for r in annotated),
        "rss_kb": sum(r[2] for r in annotated),
        "vsz_kb": sum(r[3] for r in annotated),
        "pss_kb": total_pss_kb,
    }
    totals["rss_h"] = kb_to_human_mbgb(totals["rss_kb"])
    totals["vsz_h"] = kb_to_human_mbgb(totals["vsz_kb"])
    totals["pss_h"] = kb_to_human_mbgb(totals["pss_kb"])

    # 分组（按 CMD 合并，仅对选取行）
    groups_map: Dict[str, Dict[str, Any]] = {}
    for r in annotated:
        pid, pmem, rss_kb, vsz_kb, cmd, pss_kb, shared_ratio, sm_rss_kb = r
        g = groups_map.setdefault(
            cmd,
            {
                "cmd": cmd,
                "count": 0,
                "pmem": 0.0,
                "rss_kb": 0,
                "pss_kb": 0,
                "sm_rss_kb": 0,
            },
        )
        g["count"] += 1
        g["pmem"] += pmem
        g["rss_kb"] += rss_kb
        g["pss_kb"] += pss_kb
        g["sm_rss_kb"] += sm_rss_kb

    # 计算每组共享比例和格式化内存字符串
    for g in groups_map.values():
        sm = g.get("sm_rss_kb", 0)
        pss = g.get("pss_kb", 0)
        if sm > 0 and pss > 0:
            try:
                g["shared_ratio"] = 1.0 - (pss / sm)
            except Exception:
                g["shared_ratio"] = None
        else:
            g["shared_ratio"] = None
        g["rss_h"] = kb_to_human_mbgb(g["rss_kb"])
        g["pss_h"] = kb_to_human_mbgb(g["pss_kb"])
        g["sm_rss_h"] = kb_to_human_mbgb(g["sm_rss_kb"])

    # 按 pmem 合计降序返回列表
    groups_sorted = sorted(groups_map.values(), key=lambda d: d["pmem"], reverse=True)

    return {
        "groups": groups_sorted,
        "totals": totals,
    }


def main():
    parser = argparse.ArgumentParser(description="显示当前用户各进程的内存占用（百分比与实际容量 MB/GB）")
    parser.add_argument("--user", default=None, help="指定用户（默认当前用户）")
    parser.add_argument("--top", type=int, default=20, help="显示前 N 个进程，按内存百分比降序")
    parser.add_argument("--cmd-width", type=int, default=80, help="命令列最大显示宽度")
    args = parser.parse_args()

    user = args.user or get_current_user()
    if not user:
        raise SystemExit("无法确定用户，请使用 --user 指定用户")

    lines = run_ps(user)
    rows = []
    for line in lines:
        try:
            pid, pmem, rss_kb, vsz_kb, cmd = parse_ps_line(line)
            rows.append((pid, pmem, rss_kb, vsz_kb, cmd))
        except Exception:
            # Skip malformed lines
            continue

    # Already sorted by ps; optionally slice top N
    top_rows = rows[: args.top] if args.top and args.top > 0 else rows

    # 读取 PSS 与共享比例（仅对显示的行）
    annotated = []
    total_pss_kb = 0
    for r in top_rows:
        pid, pmem, rss_kb, vsz_kb, cmd = r
        sm_rss_kb, pss_kb = read_smaps_rollup(pid)
        shared_ratio = None
        if sm_rss_kb > 0 and pss_kb > 0:
            try:
                shared_ratio = 1.0 - (pss_kb / sm_rss_kb)
            except Exception:
                shared_ratio = None
        annotated.append((pid, pmem, rss_kb, vsz_kb, cmd, pss_kb, shared_ratio, sm_rss_kb))
        if pss_kb > 0:
            total_pss_kb += pss_kb

    # Totals (for displayed rows)
    total_pmem = sum(r[1] for r in annotated)
    total_rss_kb = sum(r[2] for r in annotated)
    total_vsz_kb = sum(r[3] for r in annotated)

    header = f"{'PID':>7}  {'%MEM':>6}  {'RSS':>10}  {'PSS':>10}  {'共享比例':>8}  {'VSZ':>10}  CMD"
    print(header)
    print("-" * len(header))
    for r in annotated:
        print(format_row(r[0], r[1], r[2], r[3], r[4], pss_kb=r[5], shared_ratio=r[6], width_cmd=args.cmd_width))

    # 分组（按 CMD 合并，仅对显示行）
    groups = {}
    for r in annotated:
        pid, pmem, rss_kb, vsz_kb, cmd, pss_kb, shared_ratio, sm_rss_kb = r
        g = groups.setdefault(cmd, {"count": 0, "pmem": 0.0, "rss": 0, "pss": 0, "sm_rss": 0})
        g["count"] += 1
        g["pmem"] += pmem
        g["rss"] += rss_kb
        g["pss"] += pss_kb
        g["sm_rss"] += sm_rss_kb

    # 根据合计 %MEM 降序打印分组
    sorted_groups = sorted(groups.items(), key=lambda kv: kv[1]["pmem"], reverse=True)

    if sorted_groups:
        group_header = (
            f"{'CMD(合并)':<{args.cmd_width}}  {'进程数':>6}  {'%MEM':>6}  {'RSS':>10}  {'PSS':>10}  {'共享比例':>8}"
        )
        print("\n" + group_header)
        print("-" * len(group_header))
        for cmd, g in sorted_groups:
            shr = None
            if g["sm_rss"] > 0 and g["pss"] > 0:
                try:
                    shr = 1.0 - (g["pss"] / g["sm_rss"])
                except Exception:
                    shr = None
            print(
                format_group_row(
                    cmd,
                    g["count"],
                    g["pmem"],
                    g["rss"],
                    g["pss"],
                    shr,
                    width_cmd=args.cmd_width,
                )
            )

    print("\n汇总(显示行):")
    print(f"  总内存百分比: {total_pmem:.2f}%")
    print(f"  总常驻内存(RSS): {kb_to_human_mbgb(total_rss_kb)}")
    print(f"  总虚拟内存(VSZ): {kb_to_human_mbgb(total_vsz_kb)}")
    print(f"  总比例集内存(PSS): {kb_to_human_mbgb(total_pss_kb)}")


if __name__ == "__main__":
    main()

"""
python3 mem_usage.py --top 10

"""
