#!/usr/bin/env python3
"""
实时性能基准测试脚本

测试预训练的 LidarCompletionNet 和 HeightMapSampler 两个网络在 Jetson 上的
实时推理性能。从数据集加载样本，连续运行多帧推理，统计：
  - 每帧总耗时（数据预处理 + 推理 + 后处理）
  - 纯推理（forward）耗时
  - 运行频率 (FPS)
  - 统计信息（均值、中位数、最小、最大、P95、P99）

用法:
    python scripts/benchmark_realtime.py [--num_samples 100] [--warmup 10]
"""

import os
import sys
import glob
import argparse
import logging
import numpy as np
from time import time

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

# HeightMapSampler 默认参数（与训练配置一致）
PHYS_SCALE = 3.2
GRID_RES_PHYS = 0.1
GRID_SIZE_PHYS = [1.6, 1.0]
K_NEIGHBORS = 8
HIDDEN_DIM = 512
NUM_BINS = 32

parser = argparse.ArgumentParser(description="Realtime Benchmark for Completion + HeightMap networks")
parser.add_argument("--num_samples", type=int, default=500,
                    help="总测试帧数（跨样本连续帧）")
parser.add_argument("--warmup", type=int, default=20,
                    help="预热帧数（不计入统计）")
parser.add_argument("--completion_checkpoint", type=str,
                    default="./output/checkpoint/lidar_completion_x64_v2/export/model.pth",
                    help="补全网络权重路径")
parser.add_argument("--sampler_checkpoint", type=str,
                    default="./output/checkpoint/heightmap_sampler_v0/sampler_2000.pth",
                    help="高度图采样网络权重路径")
parser.add_argument("--batch_size", type=int, default=1,
                    help="推理批大小（实时场景固定为 1）")
parser.add_argument("--heightmap_samples", type=int, default=5,
                    help="每帧高度图采样次数")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def make_benchmark_dataloader(batch_size=1):
    """创建用于基准测试的数据加载器，不做数据增强。"""
    class _Cfg:
        resolution = COMPLETION_RESOLUTION
        cache_use = False

    dataset = ConstructTerrainDataset(
        phase="train",
        config=_Cfg(),
        augment_data=True,
        transforms=[
            lambda x: StridedSamplingTransform(x, stride_list=[1] * 11 + [2] * 7 + [3] * 4 + [4] * 2 + [5] * 1),
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
# 单帧推理流水线
# ---------------------------------------------------------------------------
class InferencePipeline:
    """封装单帧推理流水线，模拟实时运行。"""

    def __init__(self, completion_net, sampler_net, device, config):
        self.completion_net = completion_net
        self.sampler_net = sampler_net
        self.device = device
        self.config = config
        self.resolution = COMPLETION_RESOLUTION

        # 历史帧缓存
        self.prev_frame_data = [None] * config.batch_size

        # 高度图网格参数 (x64 坐标)
        self.grid_res = GRID_RES_PHYS / PHYS_SCALE * 64.0
        self.grid_size = [GRID_SIZE_PHYS[0] / PHYS_SCALE * 64.0,
                          GRID_SIZE_PHYS[1] / PHYS_SCALE * 64.0]

    def reset(self):
        """重置历史帧缓存（新样本时调用）。"""
        self.prev_frame_data = [None] * self.config.batch_size

    def run_frame(self, data_dict, frame_idx):
        """
        执行单帧完整推理流水线，返回各阶段耗时。

        返回 dict:
            preprocess_ms:   数据预处理耗时 (ms)
            completion_ms:   补全网络推理耗时 (ms)
            heightmap_ms:    高度图网络推理耗时 (ms)
            postprocess_ms:  后处理耗时 (ms)
            total_ms:        总耗时 (ms)
            input_points:    输入点数
            output_points:   补全输出点数
        """
        device = self.device
        res = self.resolution

        torch.cuda.synchronize()
        t_total_start = time()

        # ================================================================
        # 1. 数据预处理
        # ================================================================
        t_pre_start = time()

        t_p_points_list_np = data_dict['partial'][frame_idx]
        t_c_points_list_np = data_dict['complete'][frame_idx]
        t_pos_data_list_np = data_dict['pos_data'][frame_idx]
        t_quat_data_list_np = data_dict['quat_data'][frame_idx]
        _t_pos_data_list_np = data_dict['pos_data'][(frame_idx - 1) if frame_idx != 0 else 0]
        _t_quat_data_list_np = data_dict['quat_data'][(frame_idx - 1) if frame_idx != 0 else 0]

        t_p_points_list = [torch.from_numpy(p).float().to(device) for p in t_p_points_list_np]
        t_c_points_list = [torch.from_numpy(p).float().to(device) for p in t_c_points_list_np]
        t_pos_data_list = [torch.from_numpy(p).float().to(device) for p in t_pos_data_list_np]
        t_quat_data_list = [torch.from_numpy(p).float().to(device) for p in t_quat_data_list_np]
        _t_pos_data_list = [torch.from_numpy(p).float().to(device) for p in _t_pos_data_list_np]
        _t_quat_data_list = [torch.from_numpy(p).float().to(device) for p in _t_quat_data_list_np]

        # 体素化
        _, t_p_coords_float_list, t_p_coords_voxel_list, _ = voxelization(
            t_p_points_list, res
        )
        _, _, t_c_coords_voxel_list, _ = voxelization(
            t_c_points_list, res
        )

        curr_feats_list = compute_feats(
            coords_float_list=t_p_coords_float_list,
            coords_voxel_list=t_p_coords_voxel_list,
            time_encoding=0.0,
            prob=1.0,
        )

        all_input_coords = list(t_p_coords_voxel_list)
        all_input_feats = list(curr_feats_list)

        # 历史帧融合
        if frame_idx > 0 and any(d is not None for d in self.prev_frame_data):
            prev_points_list = [data[0] for data in self.prev_frame_data if data is not None]
            prev_probs_list = [data[1] for data in self.prev_frame_data if data is not None]

            transformed_points_list, transformed_probs_list = points_transform_and_normclip(
                points_prev_list=prev_points_list,
                pos_curr=torch.stack(t_pos_data_list),
                quat_curr=torch.stack(t_quat_data_list),
                pos_prev=torch.stack(_t_pos_data_list),
                quat_prev=torch.stack(_t_quat_data_list),
                feats_prev_list=prev_probs_list,
                scale=3.2,
                bound=0.5,
            )

            _, hist_float_list, hist_voxel_list, temp_feats_list = voxelization(
                transformed_points_list, res, transformed_probs_list
            )
            hist_feats_list = compute_feats(
                coords_float_list=hist_float_list,
                coords_voxel_list=hist_voxel_list,
                time_encoding=1.0,
                prob=temp_feats_list,
            )

            for b in range(self.config.batch_size):
                all_input_coords[b] = torch.cat([all_input_coords[b], hist_voxel_list[b]], dim=0)
                all_input_feats[b] = torch.cat([all_input_feats[b], hist_feats_list[b]], dim=0)

        # 构建 SparseTensor
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

        # ================================================================
        # 2. 补全网络推理
        # ================================================================
        torch.cuda.synchronize()
        t_comp_start = time()

        with torch.no_grad():
            out_cls, out_targets, sout, occ_probs = self.completion_net(sin_fused, in_target_key)

        torch.cuda.synchronize()
        t_comp_end = time()

        # ================================================================
        # 3. 补全后处理 + 高度图推理
        # ================================================================
        torch.cuda.synchronize()
        t_post_start = time()

        sout_coords_list, _ = sout.decomposed_coordinates_and_features
        sout_points_norm_list, sout_coords_floats_list = devoxelization(
            sout_coords_list, res=res
        )

        # 更新历史帧缓存
        num_points_list = [pts.shape[0] for pts in sout_points_norm_list]
        occ_probs_list = torch.split(occ_probs.detach(), num_points_list)
        self.prev_frame_data = [
            (coords, probs.unsqueeze(1))
            for coords, probs in zip(sout_points_norm_list, occ_probs_list)
        ]

        output_points = sum(num_points_list)

        torch.cuda.synchronize()
        t_post_end = time()

        # ================================================================
        # 4. 高度图网络推理
        # ================================================================
        torch.cuda.synchronize()
        t_hm_start = time()

        for b in range(self.config.batch_size):
            rec_points = sout_coords_floats_list[b]  # (N, 3) x64 坐标

            if rec_points.shape[0] < 3:
                continue

            for _ in range(self.config.heightmap_samples):
                query_center = sample_random_center(
                    self.grid_size, margin=3.2, device=device, coord_max=64.0
                )
                yaw = torch.rand(1, device=device).item() * 2.0 * np.pi

                with torch.no_grad():
                    pred_h, grid_xy, nx, ny, *_ = self.sampler_net(
                        rec_points, query_center, yaw=yaw
                    )

        torch.cuda.synchronize()
        t_hm_end = time()

        # ================================================================
        # 汇总
        # ================================================================
        torch.cuda.synchronize()
        t_total_end = time()

        return {
            "preprocess_ms":  (t_pre_end - t_pre_start) * 1000,
            "completion_ms":  (t_comp_end - t_comp_start) * 1000,
            "postprocess_ms": (t_post_end - t_post_start) * 1000,
            "heightmap_ms":   (t_hm_end - t_hm_start) * 1000,
            "total_ms":       (t_total_end - t_total_start) * 1000,
            "input_points":   input_points,
            "output_points":  output_points,
        }


# ---------------------------------------------------------------------------
# 统计输出
# ---------------------------------------------------------------------------
def print_stats(name, values_ms):
    """打印一组耗时数据的统计信息。"""
    arr = np.array(values_ms)
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
    """打印点数统计。"""
    arr = np.array(values)
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
    logging.info(f"测试帧数: {config.num_samples}, 预热帧数: {config.warmup}")

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
        k=K_NEIGHBORS,
        hidden_dim=HIDDEN_DIM,
        grid_res=grid_res,
        grid_size=grid_size,
        num_bins=NUM_BINS,
    ).to(device)

    state = torch.load(config.sampler_checkpoint, map_location=device)
    sampler_net.load_state_dict(state["sampler_state_dict"])
    sampler_net.eval()
    for p in sampler_net.parameters():
        p.requires_grad = False

    sampler_params = sum(p.numel() for p in sampler_net.parameters())
    logging.info(f"高度图采样网络参数量: {sampler_params:,} ({sampler_params / 1e6:.3f}M)")

    # ---- 3. 数据加载 ----
    logging.info("加载数据集...")
    dataloader = make_benchmark_dataloader(batch_size=config.batch_size)
    data_iter = iter(dataloader)

    # ---- 4. 构建推理流水线 ----
    pipeline = InferencePipeline(completion_net, sampler_net, device, config)

    # ---- 5. 运行基准测试 ----
    all_results = []
    total_frames = config.warmup + config.num_samples
    frame_count = 0
    sample_count = 0

    logging.info(f"开始基准测试 ({config.warmup} 预热 + {config.num_samples} 测试)...")
    print("=" * 70)

    while frame_count < total_frames:
        # 加载新样本
        data_dict = next(data_iter)
        num_frames = data_dict['num_frames']
        pipeline.reset()
        sample_count += 1

        for t in range(num_frames):
            if frame_count >= total_frames:
                break

            is_warmup = frame_count < config.warmup
            phase = "warmup" if is_warmup else "test"

            result = pipeline.run_frame(data_dict, t)

            if not is_warmup:
                all_results.append(result)

            # 实时打印进度
            if frame_count % 10 == 0 or frame_count == total_frames - 1:
                print(f"  [{phase}] frame {frame_count + 1}/{total_frames} | "
                      f"total: {result['total_ms']:.1f}ms | "
                      f"completion: {result['completion_ms']:.1f}ms | "
                      f"heightmap: {result['heightmap_ms']:.1f}ms | "
                      f"in_pts: {result['input_points']} | "
                      f"out_pts: {result['output_points']}")

            frame_count += 1

    # ---- 6. 统计输出 ----
    print("\n" + "=" * 70)
    print("  实时性能基准测试结果")
    print("=" * 70)

    print(f"\n  测试环境:")
    print(f"    设备:           {torch.cuda.get_device_name(0)}")
    print(f"    PyTorch:        {torch.__version__}")
    print(f"    CUDA:           {torch.version.cuda}")
    print(f"    批大小:         {config.batch_size}")
    print(f"    高度图采样次数: {config.heightmap_samples}/帧")
    print(f"    总测试帧数:     {len(all_results)}")
    print(f"    总样本数:       {sample_count}")

    preprocess_times = [r["preprocess_ms"] for r in all_results]
    completion_times = [r["completion_ms"] for r in all_results]
    postprocess_times = [r["postprocess_ms"] for r in all_results]
    heightmap_times = [r["heightmap_ms"] for r in all_results]
    total_times = [r["total_ms"] for r in all_results]
    input_points = [r["input_points"] for r in all_results]
    output_points = [r["output_points"] for r in all_results]

    print("\n" + "-" * 70)
    print("  各阶段耗时统计")
    print("-" * 70)

    print_stats("数据预处理 (体素化 + 历史帧融合)", preprocess_times)
    print_stats("补全网络推理 (LidarCompletionNet)", completion_times)
    print_stats("补全后处理 (反体素化 + 缓存更新)", postprocess_times)
    print_stats("高度图网络推理 (HeightMapSampler)", heightmap_times)
    print_stats("单帧总耗时 (端到端)", total_times)

    print("\n" + "-" * 70)
    print("  点云规模统计")
    print("-" * 70)
    print_point_stats("输入点数 (体素化后)", input_points)
    print_point_stats("补全输出点数", output_points)

    # 综合频率
    total_arr = np.array(total_times)
    comp_arr = np.array(completion_times)
    hm_arr = np.array(heightmap_times)

    print("\n" + "-" * 70)
    print("  综合频率")
    print("-" * 70)
    print(f"    端到端 (含预处理):       {1000.0 / total_arr.mean():.2f} FPS (mean {total_arr.mean():.2f} ms)")
    print(f"    纯推理 (补全+高度图):    {1000.0 / (comp_arr.mean() + hm_arr.mean()):.2f} FPS (mean {comp_arr.mean() + hm_arr.mean():.2f} ms)")
    print(f"    补全网络单独:            {1000.0 / comp_arr.mean():.2f} FPS (mean {comp_arr.mean():.2f} ms)")
    print(f"    高度图网络单独:          {1000.0 / hm_arr.mean():.2f} FPS (mean {hm_arr.mean():.2f} ms)")

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
