#!/usr/bin/env python3
"""
双线程流水线实时性能基准测试脚本

两个独立线程各自按目标频率运行：
  - 补全线程 (10 Hz): 预处理 + LidarCompletionNet 推理 + 后处理
  - 采样线程 (50 Hz): HeightMapSampler 推理

数据依赖通过互斥锁 + 退避策略解耦：
  - 补全线程每次产出新点云后写入共享缓冲区（加锁）
  - 采样线程 trylock 尝试读取最新点云；若锁被占用则退避，
    直接使用上一次缓存的点云继续推理，保证不阻塞

用法:
    python scripts/benchmark_realtime.py [--duration 30] [--warmup 3]
"""

import os
import sys
import argparse
import logging
import threading
import numpy as np
from time import time, sleep

import torch
import torch.nn as nn

import MinkowskiEngine as ME

# ---------------------------------------------------------------------------
# 导入项目已有模块
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "output", "checkpoint", "lidar_completion_x64_v2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "output", "checkpoint", "heightmap_sampler_v0"))

from lidar_sparse_completion_x64 import (
    LidarCompletionNet,
    ConstructTerrainDataset,
    InfSampler,
    CollationAndTransformation,
    voxelization,
    devoxelization,
    compute_feats,
    points_transform_and_normclip,
    StridedSamplingTransform,
)

from lidar_heightmap_sampling_x64 import (
    HeightMapSampler,
    generate_grid_xy,
    sample_random_center,
    sample_heightmap_from_points,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

COMPLETION_RESOLUTION = 64
PURNING_THRESHOLD = 0.35
ENC_CHANNELS = [16, 32, 64, 128, 256, 512]
DEC_CHANNELS = [16, 32, 64, 128, 256, 512]

PHYS_SCALE = 3.2
GRID_RES_PHYS = 0.1
GRID_SIZE_PHYS = [1.6, 1.0]
K_NEIGHBORS = 8
HIDDEN_DIM = 512
NUM_BINS = 32

parser = argparse.ArgumentParser(description="双线程流水线基准测试")
parser.add_argument("--duration", type=float, default=60.0,
                    help="测试持续时间 (秒)")
parser.add_argument("--warmup", type=float, default=5.0,
                    help="预热时间 (秒)")
parser.add_argument("--completion_hz", type=float, default=10.0,
                    help="补全线程目标频率")
parser.add_argument("--sampler_hz", type=float, default=50.0,
                    help="采样线程目标频率")
parser.add_argument("--completion_checkpoint", type=str,
                    default="./output/checkpoint/lidar_completion_x64_v2/export/model.pth")
parser.add_argument("--sampler_checkpoint", type=str,
                    default="./output/checkpoint/heightmap_sampler_v0/sampler_2000.pth")
parser.add_argument("--batch_size", type=int, default=1)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def make_benchmark_dataloader(batch_size=1):
    class _Cfg:
        resolution = COMPLETION_RESOLUTION
        cache_use = False

    dataset = ConstructTerrainDataset(
        phase="train",
        config=_Cfg(),
        augment_data=True,
        transforms=[
            lambda x: StridedSamplingTransform(x, stride_list=[1]*11 + [2]*7 + [3]*4 + [4]*2 + [5]*1),
        ],
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=CollationAndTransformation(),
        pin_memory=True,
        drop_last=False,
        sampler=InfSampler(dataset, shuffle=True),
    )
    return loader


# ---------------------------------------------------------------------------
# 共享数据缓冲区
# ---------------------------------------------------------------------------
class SharedPointCloud:
    """补全线程写入、采样线程读取的共享点云缓冲区。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.coords_float = None   # (N, 3) x64 坐标
        self.frame_id = -1         # 单调递增的帧编号
        self.ready = False         # 至少有一帧数据

    def update(self, coords_float, frame_id):
        """补全线程调用：写入新点云。"""
        with self.lock:
            self.coords_float = coords_float
            self.frame_id = frame_id
            self.ready = True

    def try_read(self):
        """采样线程调用：非阻塞读取。
        返回 (coords_float, frame_id, is_fresh)
          - is_fresh=True  表示成功拿到锁，读到（可能更新的）数据
          - is_fresh=False 表示锁被占用，调用者应退避使用上次缓存
        """
        acquired = self.lock.acquire(blocking=False)
        if acquired:
            try:
                return self.coords_float, self.frame_id, True
            finally:
                self.lock.release()
        else:
            return None, -1, False


# ---------------------------------------------------------------------------
# 补全线程
# ---------------------------------------------------------------------------
def completion_thread_fn(
    completion_net, device, shared_buf, config,
    data_iter, stop_event, warmup_end_time, stats
):
    """补全线程：目标 completion_hz，循环 预处理→推理→后处理→写入共享缓冲。"""
    target_period = 1.0 / config.completion_hz
    res = COMPLETION_RESOLUTION

    prev_frame_data = [None] * config.batch_size
    frame_id = 0
    data_dict = None
    frame_idx = 0
    num_frames = 0

    next_tick = time()  # 绝对时间戳定频基准

    while not stop_event.is_set():
        t_loop_start = time()

        # ---- 加载数据帧 ----
        if data_dict is None or frame_idx >= num_frames:
            data_dict = next(data_iter)
            num_frames = data_dict['num_frames']
            frame_idx = 0
            prev_frame_data = [None] * config.batch_size

        t = frame_idx
        frame_idx += 1

        # ---- 预处理 ----
        t_pre_start = time()

        t_p_points_list_np = data_dict['partial'][t]
        t_c_points_list_np = data_dict['complete'][t]
        t_pos_data_list_np = data_dict['pos_data'][t]
        t_quat_data_list_np = data_dict['quat_data'][t]
        _t_pos_data_list_np = data_dict['pos_data'][(t - 1) if t != 0 else 0]
        _t_quat_data_list_np = data_dict['quat_data'][(t - 1) if t != 0 else 0]

        t_p_points_list = [torch.from_numpy(p).float().to(device) for p in t_p_points_list_np]
        t_c_points_list = [torch.from_numpy(p).float().to(device) for p in t_c_points_list_np]
        t_pos_data_list = [torch.from_numpy(p).float().to(device) for p in t_pos_data_list_np]
        t_quat_data_list = [torch.from_numpy(p).float().to(device) for p in t_quat_data_list_np]
        _t_pos_data_list = [torch.from_numpy(p).float().to(device) for p in _t_pos_data_list_np]
        _t_quat_data_list = [torch.from_numpy(p).float().to(device) for p in _t_quat_data_list_np]

        _, t_p_coords_float_list, t_p_coords_voxel_list, _ = voxelization(t_p_points_list, res)
        _, _, t_c_coords_voxel_list, _ = voxelization(t_c_points_list, res)

        curr_feats_list = compute_feats(
            coords_float_list=t_p_coords_float_list,
            coords_voxel_list=t_p_coords_voxel_list,
            time_encoding=0.0, prob=1.0,
        )

        all_input_coords = list(t_p_coords_voxel_list)
        all_input_feats = list(curr_feats_list)

        if t > 0 and any(d is not None for d in prev_frame_data):
            prev_points_list = [d[0] for d in prev_frame_data if d is not None]
            prev_probs_list = [d[1] for d in prev_frame_data if d is not None]
            transformed_points_list, transformed_probs_list = points_transform_and_normclip(
                points_prev_list=prev_points_list,
                pos_curr=torch.stack(t_pos_data_list),
                quat_curr=torch.stack(t_quat_data_list),
                pos_prev=torch.stack(_t_pos_data_list),
                quat_prev=torch.stack(_t_quat_data_list),
                feats_prev_list=prev_probs_list,
                scale=3.2, bound=0.5,
            )
            _, hist_float_list, hist_voxel_list, temp_feats_list = voxelization(
                transformed_points_list, res, transformed_probs_list
            )
            hist_feats_list = compute_feats(
                coords_float_list=hist_float_list,
                coords_voxel_list=hist_voxel_list,
                time_encoding=1.0, prob=temp_feats_list,
            )
            for b in range(config.batch_size):
                all_input_coords[b] = torch.cat([all_input_coords[b], hist_voxel_list[b]], dim=0)
                all_input_feats[b] = torch.cat([all_input_feats[b], hist_feats_list[b]], dim=0)

        batched_input_coords, batched_input_feats = ME.utils.sparse_collate(
            all_input_coords, all_input_feats, device=device
        )
        batched_gt_coords = ME.utils.batched_coordinates(t_c_coords_voxel_list).to(device)
        sin_fused = ME.SparseTensor(
            features=batched_input_feats, coordinates=batched_input_coords, device=device
        )
        cm = sin_fused.coordinate_manager
        in_target_key, _ = cm.insert_and_map(coordinates=batched_gt_coords, string_id="target")

        input_points = batched_input_feats.shape[0]
        torch.cuda.synchronize()
        t_pre_end = time()

        # ---- 补全推理 ----
        t_comp_start = time()
        with torch.no_grad():
            out_cls, out_targets, sout, occ_probs = completion_net(sin_fused, in_target_key)
        torch.cuda.synchronize()
        t_comp_end = time()

        # ---- 后处理 ----
        t_post_start = time()
        sout_coords_list, _ = sout.decomposed_coordinates_and_features
        sout_points_norm_list, sout_coords_floats_list = devoxelization(sout_coords_list, res=res)

        num_points_list = [pts.shape[0] for pts in sout_points_norm_list]
        occ_probs_list = torch.split(occ_probs.detach(), num_points_list)
        prev_frame_data = [
            (coords, probs.unsqueeze(1))
            for coords, probs in zip(sout_points_norm_list, occ_probs_list)
        ]
        output_points = sum(num_points_list)

        # 取 batch=0 的点云写入共享缓冲
        shared_buf.update(sout_coords_floats_list[0].clone(), frame_id)

        torch.cuda.synchronize()
        t_post_end = time()

        # ---- 记录统计 ----
        is_recording = time() >= warmup_end_time
        if is_recording:
            stats["preprocess"].append((t_pre_end - t_pre_start) * 1000)
            stats["completion"].append((t_comp_end - t_comp_start) * 1000)
            stats["postprocess"].append((t_post_end - t_post_start) * 1000)
            stats["input_points"].append(input_points)
            stats["output_points"].append(output_points)
            stats["comp_total"].append((time() - t_loop_start) * 1000)

        frame_id += 1

        # ---- 严格定频（绝对时间戳补偿） ----
        next_tick += target_period
        now = time()
        if next_tick > now:
            sleep(next_tick - now)
        else:
            # 超时：对齐到下一个周期，避免累积误差
            missed = int((now - next_tick) / target_period) + 1
            next_tick += missed * target_period
            stats["overrun_count"] = stats.get("overrun_count", 0) + 1


# ---------------------------------------------------------------------------
# 采样线程
# ---------------------------------------------------------------------------
def sampler_thread_fn(
    sampler_net, device, shared_buf, config,
    stop_event, warmup_end_time, stats
):
    """采样线程：目标 sampler_hz，trylock 读共享缓冲；失败则退避用上次缓存。"""
    target_period = 1.0 / config.sampler_hz

    grid_res = GRID_RES_PHYS / PHYS_SCALE * 64.0
    grid_size = [GRID_SIZE_PHYS[0] / PHYS_SCALE * 64.0,
                 GRID_SIZE_PHYS[1] / PHYS_SCALE * 64.0]

    cached_points = None
    cached_frame_id = -1

    # 等待第一帧补全数据就绪
    while not stop_event.is_set():
        if shared_buf.ready:
            break
        sleep(0.001)

    next_tick = time()  # 绝对时间戳定频基准

    while not stop_event.is_set():
        t_loop_start = time()

        # ---- trylock 读取最新点云 ----
        new_points, new_fid, is_fresh = shared_buf.try_read()
        if is_fresh and new_points is not None:
            cached_points = new_points
            cached_frame_id = new_fid
            fallback = False
        else:
            fallback = True  # 退避：使用上次缓存

        if cached_points is None or cached_points.shape[0] < 3:
            next_tick += target_period
            now = time()
            if next_tick > now:
                sleep(next_tick - now)
            else:
                next_tick = now
            continue

        # ---- 高度图推理 ----
        t_hm_start = time()
        query_center = sample_random_center(grid_size, margin=3.2, device=device, coord_max=64.0)
        yaw = torch.rand(1, device=device).item() * 2.0 * np.pi

        with torch.no_grad():
            pred_h, grid_xy, nx, ny, *_ = sampler_net(
                cached_points, query_center, yaw=yaw
            )
        torch.cuda.synchronize()
        t_hm_end = time()

        # ---- 记录统计 ----
        is_recording = time() >= warmup_end_time
        if is_recording:
            stats["heightmap"].append((t_hm_end - t_hm_start) * 1000)
            stats["fallback_count"] += (1 if fallback else 0)
            stats["sample_count"] += 1
            stats["source_frame_ids"].append(cached_frame_id)

        # ---- 严格定频（绝对时间戳补偿） ----
        next_tick += target_period
        now = time()
        if next_tick > now:
            sleep(next_tick - now)
        else:
            missed = int((now - next_tick) / target_period) + 1
            next_tick += missed * target_period
            stats["overrun_count"] = stats.get("overrun_count", 0) + 1


# ---------------------------------------------------------------------------
# 统计输出
# ---------------------------------------------------------------------------
def print_stats(name, values_ms):
    arr = np.array(values_ms)
    if len(arr) == 0:
        print(f"\n  [{name}]  无数据")
        return
    print(f"\n  [{name}]")
    print(f"    样本数:   {len(arr)}")
    print(f"    均值:     {arr.mean():.2f} ms")
    print(f"    中位数:   {np.median(arr):.2f} ms")
    print(f"    最小值:   {arr.min():.2f} ms")
    print(f"    最大值:   {arr.max():.2f} ms")
    print(f"    P95:      {np.percentile(arr, 95):.2f} ms")
    print(f"    P99:      {np.percentile(arr, 99):.2f} ms")
    print(f"    标准差:   {arr.std():.2f} ms")
    fps = 1000.0 / arr.mean() if arr.mean() > 0 else float('inf')
    print(f"    平均频率: {fps:.2f} FPS")


def print_point_stats(name, values):
    arr = np.array(values)
    if len(arr) == 0:
        return
    print(f"\n  [{name}]")
    print(f"    均值: {arr.mean():.0f}")
    print(f"    最小: {arr.min():.0f}")
    print(f"    最大: {arr.max():.0f}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main():
    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"设备: {device}")
    logging.info(f"补全目标频率: {config.completion_hz} Hz, 采样目标频率: {config.sampler_hz} Hz")
    logging.info(f"测试时长: {config.duration}s, 预热: {config.warmup}s")

    # ---- 1. 加载补全网络 ----
    logging.info("加载补全网络...")
    completion_net = LidarCompletionNet(
        resolution=COMPLETION_RESOLUTION,
        activeF=PURNING_THRESHOLD,
        encoder_channels=ENC_CHANNELS,
        decoder_channels=DEC_CHANNELS,
    ).to(device)
    ckpt = torch.load(config.completion_checkpoint, map_location=device)
    completion_net.load_state_dict(ckpt["state_dict"])
    completion_net.eval()
    for p in completion_net.parameters():
        p.requires_grad = False
    comp_params = sum(p.numel() for p in completion_net.parameters())
    logging.info(f"补全网络参数量: {comp_params:,} ({comp_params / 1e6:.2f}M)")

    # ---- 2. 加载高度图采样网络 ----
    logging.info("加载高度图采样网络...")
    grid_res = GRID_RES_PHYS / PHYS_SCALE * 64.0
    grid_size = (GRID_SIZE_PHYS[0] / PHYS_SCALE * 64.0,
                 GRID_SIZE_PHYS[1] / PHYS_SCALE * 64.0)
    sampler_net = HeightMapSampler(
        k=K_NEIGHBORS, hidden_dim=HIDDEN_DIM,
        grid_res=grid_res, grid_size=grid_size, num_bins=NUM_BINS,
    ).to(device)
    state = torch.load(config.sampler_checkpoint, map_location=device)
    sampler_net.load_state_dict(state["sampler_state_dict"])
    sampler_net.eval()
    for p in sampler_net.parameters():
        p.requires_grad = False
    sampler_net.jit_compile()
    logging.info("HeightMapSampler JIT compiled.")
    sampler_params = sum(p.numel() for p in sampler_net.parameters())
    logging.info(f"高度图采样网络参数量: {sampler_params:,} ({sampler_params / 1e6:.3f}M)")

    # ---- 3. 数据加载 ----
    logging.info("加载数据集...")
    dataloader = make_benchmark_dataloader(batch_size=config.batch_size)
    data_iter = iter(dataloader)

    # ---- 4. 共享缓冲区 & 统计容器 ----
    shared_buf = SharedPointCloud()

    comp_stats = {
        "preprocess": [], "completion": [], "postprocess": [],
        "input_points": [], "output_points": [], "comp_total": [],
    }
    samp_stats = {
        "heightmap": [], "fallback_count": 0, "sample_count": 0,
        "source_frame_ids": [],
    }

    stop_event = threading.Event()
    warmup_end_time = time() + config.warmup
    test_end_time = warmup_end_time + config.duration

    # ---- 5. 启动双线程 ----
    logging.info("启动双线程流水线...")
    print("=" * 70)

    t_comp = threading.Thread(
        target=completion_thread_fn,
        args=(completion_net, device, shared_buf, config,
              data_iter, stop_event, warmup_end_time, comp_stats),
        daemon=True,
    )
    t_samp = threading.Thread(
        target=sampler_thread_fn,
        args=(sampler_net, device, shared_buf, config,
              stop_event, warmup_end_time, samp_stats),
        daemon=True,
    )

    t_comp.start()
    t_samp.start()

    # ---- 主线程等待测试结束 ----
    try:
        while time() < test_end_time:
            remaining = test_end_time - time()
            elapsed_test = config.duration - remaining
            # 每 2 秒打印进度
            comp_frames = len(comp_stats["completion"])
            samp_frames = samp_stats["sample_count"]
            fallbacks = samp_stats["fallback_count"]
            print(f"\r  [{elapsed_test:.0f}/{config.duration:.0f}s] "
                  f"补全: {comp_frames} 帧 | "
                  f"采样: {samp_frames} 次 (退避: {fallbacks}) | "
                  f"共享帧ID: {shared_buf.frame_id}",
                  end="", flush=True)
            sleep(1.0)
    except KeyboardInterrupt:
        pass

    stop_event.set()
    t_comp.join(timeout=5)
    t_samp.join(timeout=5)
    print()

    # ---- 6. 统计输出 ----
    print("\n" + "=" * 70)
    print("  双线程流水线基准测试结果")
    print("=" * 70)

    print(f"\n  测试环境:")
    print(f"    设备:             {torch.cuda.get_device_name(0)}")
    print(f"    PyTorch:          {torch.__version__}")
    print(f"    CUDA:             {torch.version.cuda}")
    print(f"    补全目标频率:     {config.completion_hz} Hz")
    print(f"    采样目标频率:     {config.sampler_hz} Hz")
    print(f"    测试时长:         {config.duration}s (预热 {config.warmup}s)")

    print(f"\n  吞吐量:")
    print(f"    补全帧数:         {len(comp_stats['completion'])}")
    print(f"    采样次数:         {samp_stats['sample_count']}")
    if config.duration > 0:
        print(f"    补全实际频率:     {len(comp_stats['completion']) / config.duration:.2f} Hz")
        print(f"    采样实际频率:     {samp_stats['sample_count'] / config.duration:.2f} Hz")

    fallback_rate = (samp_stats["fallback_count"] / samp_stats["sample_count"] * 100
                     if samp_stats["sample_count"] > 0 else 0)
    print(f"\n  退避与超时统计:")
    print(f"    采样退避次数:     {samp_stats['fallback_count']}")
    print(f"    采样退避率:       {fallback_rate:.1f}%")
    comp_overruns = comp_stats.get("overrun_count", 0)
    samp_overruns = samp_stats.get("overrun_count", 0)
    print(f"    补全超时次数:     {comp_overruns} (目标周期 {1000/config.completion_hz:.0f}ms)")
    print(f"    采样超时次数:     {samp_overruns} (目标周期 {1000/config.sampler_hz:.0f}ms)")
    if samp_stats["source_frame_ids"]:
        fids = np.array(samp_stats["source_frame_ids"])
        unique_fids = len(np.unique(fids))
        print(f"    采样引用帧数:     {unique_fids} (共 {len(fids)} 次采样)")

    print("\n" + "-" * 70)
    print("  补全线程耗时统计")
    print("-" * 70)
    print_stats("数据预处理 (体素化 + 历史帧融合)", comp_stats["preprocess"])
    print_stats("补全网络推理 (LidarCompletionNet)", comp_stats["completion"])
    print_stats("补全后处理 (反体素化 + 缓存更新)", comp_stats["postprocess"])
    print_stats("补全单次总耗时 (预处理+推理+后处理)", comp_stats["comp_total"])

    print("\n" + "-" * 70)
    print("  采样线程耗时统计")
    print("-" * 70)
    print_stats("高度图网络推理 (HeightMapSampler)", samp_stats["heightmap"])

    print("\n" + "-" * 70)
    print("  点云规模统计")
    print("-" * 70)
    print_point_stats("输入点数 (体素化后)", comp_stats["input_points"])
    print_point_stats("补全输出点数", comp_stats["output_points"])

    # GPU 显存
    print("\n" + "-" * 70)
    print("  GPU 显存使用")
    print("-" * 70)
    print(f"    当前分配:  {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"    峰值分配:  {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    print(f"    当前缓存:  {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    print(f"    峰值缓存:  {torch.cuda.max_memory_reserved() / 1024**2:.1f} MB")

    print("\n" + "=" * 70)
    print("  测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
