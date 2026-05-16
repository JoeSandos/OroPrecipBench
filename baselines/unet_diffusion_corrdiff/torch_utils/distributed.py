# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import torch
from . import training_stats

# ----------------------------------------------------------------------------


class _DistributedModule:
    def init(self):
        """Initialize distributed training environment."""
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"

        backend = "gloo" if os.name == "nt" else "nccl"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

        sync_device = torch.device("cuda") if self.world_size > 1 else None
        training_stats.init_multiprocessing(rank=self.rank, sync_device=sync_device)

    # ----------------------------------------------------------------------------

    def get_rank(self):
        """Get current process rank."""
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # ----------------------------------------------------------------------------

    def get_world_size(self):
        """Get total number of processes."""
        return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # ----------------------------------------------------------------------------

    def should_stop(self):
        """Check if training should stop."""
        return False

    # ----------------------------------------------------------------------------

    def update_progress(self, cur, total):
        """Update training progress."""
        _ = cur, total

    # ----------------------------------------------------------------------------

    def print0(self, *args, **kwargs):
        """Print only from rank 0."""
        if self.rank == 0:
            print(*args, **kwargs)

    # ----------------------------------------------------------------------------
    # 新增属性访问方式
    @property
    def rank(self):
        """Current process rank (property)."""
        return self.get_rank()

    @property
    def world_size(self):
        """Total number of processes (property)."""
        return self.get_world_size()

    @property
    def local_rank(self):
        """Local rank of current process (property)."""
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def device(self):
        """Current device for the process (property).

        Returns:
            torch.device: CUDA device if available, otherwise CPU device.
        """
        # return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")


# ----------------------------------------------------------------------------
# 将模块替换为实例，使属性和方法都能直接访问
import sys

sys.modules[__name__] = _DistributedModule()
