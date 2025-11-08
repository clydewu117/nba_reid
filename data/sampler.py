#!/usr/bin/env python
"""
RandomIdentitySampler for Basketball ReID (single-GPU / non-DDP version)
Compatible with DataLoader(batch_sampler=...)
"""

import copy
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample P identities, and for each identity, randomly sample K instances.
    Each iteration yields a batch of indices with size P*K.

    Args:
        data_source (list): dataset, each element must have key 'pid'
        batch_size (int): total batch size (P*K)
        num_instances (int): number of instances per identity (K)
        seed (int): random seed
    """
    def __init__(self, data_source, batch_size, num_instances, seed=42):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.seed = seed

        # 构建 pid → [样本索引] 映射
        self.index_dic = defaultdict(list)
        for index, item in enumerate(self.data_source):
            pid = item["pid"]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        random.seed(self.seed)
        np.random.seed(self.seed)

        print(f"[RandomIdentitySampler] Initialized:")
        print(f"  - Total identities: {len(self.pids)}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Instances per ID: {self.num_instances}")
        print(f"  - PIDs per batch: {self.num_pids_per_batch}")

    def __iter__(self):
        """Yield a list[int] for each batch (required for batch_sampler)."""
        batch_idxs_dict = defaultdict(list)

        # Step 1: 按 identity 打包为若干组（每组 num_instances）
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                # 样本不足时进行有放回采样
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            else:
                random.shuffle(idxs)

            # 按 num_instances 切分
            groups = [idxs[i:i + self.num_instances] for i in range(0, len(idxs), self.num_instances)]
            batch_idxs_dict[pid].extend(groups)

        # Step 2: 随机选 P 个 identity 组成一个 batch
        avai_pids = copy.deepcopy(self.pids)
        final_batches = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            batch = []

            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                batch.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

            final_batches.append(batch)

        return iter(final_batches)

    def __len__(self):
        """
        返回实际 batch 数量（与 __iter__ 逻辑一致）
        确保 len(train_loader) 与 实际循环次数匹配
        """
        total_groups = 0
        for pid in self.pids:
            num_samples = len(self.index_dic[pid])
            num_groups = int(np.ceil(num_samples / self.num_instances))
            total_groups += num_groups

        # 每个 batch 由 num_pids_per_batch 个身份组成
        num_batches = int(np.floor(total_groups / self.num_pids_per_batch))
        return max(1, num_batches)


# ---------------------------------------------------------------------------
# 简化版 Sampler (非 batch_sampler 模式)
# ---------------------------------------------------------------------------
class RandomIdentitySampler_AlignedReID(Sampler):
    """Flat version (used when sampler=, not batch_sampler=)."""
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            pid = item["pid"]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = np.random.permutation(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = len(t) < self.num_instances
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
