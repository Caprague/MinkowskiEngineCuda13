# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
"""
基于预训练稀疏补全网络的轻量化地形高度图重采样网络训练脚本。

功能:
1. 加载预训练的 LidarCompletionNet，生成恢复后的稠密点云；
2. 在恢复点云后接入轻量的 HeightMapSampler 网络；
3. 以真值稠密点云为参考，通过 grid_pattern 俯近邻采样获取真值高度图；
4. 在一帧点云中模拟多次不同水平位置的采样，训练高度图预测网络。

输入:
- 补全网络恢复出的点云 (稀疏/稠密)；
- 机器人水平位置 (相对于恢复点云中心的偏移)。

输出:
- 固定分辨率网格上的高度值 (Z)，构成矩形点阵。
"""

"""
   设计概要

   1. 网络架构: HeightMapSampler

   轻量 PointNet 风格的 k-NN 查询网络，参数量约 ~0.03M：

   - Point Encoder: 3 → hidden_dim → hidden_dim 逐点编码恢复出的点云
   - Local MLP: 对每个网格查询点，找 XY 平面 k-NN 邻居，拼接 rel_xyz(3) + feat(hidden_dim) → MaxPool 聚合
   - Query MLP: local_feat + query_xy(2) + robot_rel(2) → hidden_dim → hidden_dim/2 → 1 输出预测高度

   2. 网格采样 (generate_grid_xy + sample_heightmap_from_points)

   - 物理默认: 分辨率 0.1m，尺寸 [1.6m, 1.0m]
   - 通过 --phys_scale=3.2 自动换算到归一化坐标（与 lidar_sparse_completion_x64.py 的 scale=3.2 一致）
   - 真值采样: 对每个网格中心，在 XY 平面找最近邻点，取该点 Z 作为高度真值，支持 max_nn_dist 阈值过滤

   3. 训练流程

   - 加载预训练 LidarCompletionNet（默认 冻结）
   - 每帧点云通过补全网络得到恢复点云
   - 每帧模拟多次采样: 在机器人位置 [0.5, 0.5]（归一化）附近加入随机 jitter（--position_jitter）
   - 输入网络: 恢复点云 + query_center + robot_rel
   - Loss: 仅对有效网格计算 L1 Loss

   4. 配置与运行
     1 # 训练
     2 python examples/lidar_heightmap_sampling.py \
     3     --completion_checkpoint ./output/checkpoint/lidar_completion_x64_v2/model_30000.pth \
     4     --completion_resolution 64 \
     5     --batch_size 4 \
     6     --samples_per_frame 4 \
     7     --grid_res_phys 0.1 \
     8     --grid_size_phys 1.6 1.0
     9
    10 # 可视化评估
    11 python examples/lidar_heightmap_sampling.py \
    12     --completion_checkpoint ./output/checkpoint/lidar_completion_x64_v2/model_30000.pth \
    13     --eval
   5. 关键可调参数
   ┌─────────────────────┬─────────┬─────────────────────────┐
   │ 参数                 │ 默认值   │ 说明                    │
   ├─────────────────────┼─────────┼─────────────────────────┤
   │ --grid_res_phys     │ 0.1     │ 网格物理分辨率 (m)        │
   │ --grid_size_phys    │ 1.6 1.0 │ 网格物理尺寸 (m)          │
   │ --k_neighbors       │ 16      │ 每个查询点的 k-NN 数      │
   │ --hidden_dim        │ 64      │ 网络隐藏维度              │
   │ --samples_per_frame │ 4       │ 每帧模拟采样次数          │
   │ --position_jitter   │ 0.15    │ 机器人位置扰动 (归一化)    │
   │ --freeze_completion │ True    │ 是否冻结补全网络          │
   └─────────────────────┴─────────┴─────────────────────────┘
   如需调整网络结构（例如加入 2D CNN 分支、改用 attention 聚合、或增加 valid mask 预测头），可以直接修改 HeightMapSampler 类。

"""


import os
import sys
import glob
import shutil
import argparse
import logging
import numpy as np
from time import time, strftime, localtime

# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import MinkowskiEngine as ME

assert (
    int(o3d.__version__.split(".")[1]) >= 8
), f"Requires open3d version >= 0.8, the current version is {o3d.__version__}"

# ------------------------------------------------------------------------------
# 导入已有工具与网络
# ------------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from lidar_sparse_completion_x64 import (
    LidarCompletionNet,
    ConstructTerrainDataset,
    InfSampler,
    CollationAndTransformation,
    voxelization,
    devoxelization,
    compute_feats,
    StridedSamplingTransform,
    VoxelFilterTransform,
)


###############################################################################
# Global configs
###############################################################################

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[ch],
)

parser = argparse.ArgumentParser()
# -- 补全网络配置 (用于加载预训练权重) --
parser.add_argument("--completion_resolution", type=int, default=64)
parser.add_argument("--completion_checkpoint", type=str, required=True,
                    help="预训练 LidarCompletionNet 的 checkpoint 路径")
parser.add_argument("--completion_activeF",    type=float, default=0.35)
parser.add_argument("--freeze_completion",     type=bool,  default=True,
                    help="是否冻结补全网络参数")

# -- 高度图网络配置 --
parser.add_argument("--grid_res_phys",         type=float, default=0.1,
                    help="网格物理分辨率 (米)")
parser.add_argument("--grid_size_phys",        type=float, nargs=2, default=[1.6, 1.0],
                    help="网格物理尺寸 [长, 宽] (米)")
parser.add_argument("--phys_scale",            type=float, default=3.2,
                    help="归一化坐标与物理尺度的换算系数: 1.0 (norm) = phys_scale (m)")
parser.add_argument("--k_neighbors",           type=int,   default=16,
                    help="每个查询点最近邻数量")
parser.add_argument("--hidden_dim",            type=int,   default=64,
                    help="轻量网络隐藏层维度")
parser.add_argument("--samples_per_frame",     type=int,   default=4,
                    help="每帧点云模拟的采样次数")
parser.add_argument("--position_jitter",       type=float, default=0.15,
                    help="机器人水平位置随机扰动范围 (归一化坐标)")
parser.add_argument("--max_nn_dist",           type=float, default=0.08,
                    help="真值近邻采样最大有效距离 (归一化坐标)")

# -- 训练配置 --
parser.add_argument("--max_iter",              type=int,   default=30001)
parser.add_argument("--stat_freq_iter",        type=int,   default=50)
parser.add_argument("--save_freq_iter",        type=int,   default=5000)
parser.add_argument("--batch_size",            type=int,   default=4)
parser.add_argument("--lr",                    type=float, default=1e-3)
parser.add_argument("--weight_decay",          type=float, default=1e-4)
parser.add_argument("--max_norm",              type=float, default=1.0)
parser.add_argument("--num_workers",           type=int,   default=4)
parser.add_argument("--log_dir",               type=str,   default="./output/logs_heightmap")
parser.add_argument("--save_dir",              type=str,   default="./output/checkpoint")
parser.add_argument("--model_name",            type=str,   default="heightmap_sampler_v1")
parser.add_argument("--resume",                action="store_true")
parser.add_argument("--eval",                  action="store_true")
parser.add_argument("--max_visualization",     type=int,   default=20)

# -- 补全网络通道配置 (需与训练时的 checkpoint 匹配) --
ENC_CHANNELS = [16, 32, 64, 128, 256, 512]
DEC_CHANNELS = [16, 32, 64, 128, 256, 512]


###############################################################################
# End of global configs
###############################################################################


###############################################################################
# Grid sampling utilities
###############################################################################

def generate_grid_xy(center_xy, grid_size, grid_res, device):
    """
    生成以 center_xy 为中心的二维矩形网格坐标。

    Args:
        center_xy: (2,) 网格中心在归一化坐标系中的位置
        grid_size: [size_x, size_y] 网格总尺寸 (归一化)
        grid_res:  网格分辨率 (归一化)
        device:    torch device

    Returns:
        grid_xy: (M, 2) 所有网格单元中心坐标
        nx:      x 方向网格数
        ny:      y 方向网格数
    """
    size_x, size_y = grid_size
    nx = max(1, int(round(size_x / grid_res)))
    ny = max(1, int(round(size_y / grid_res)))

    # 单元中心坐标，均匀分布，恰好覆盖 size_x / size_y
    x = torch.arange(
        -size_x / 2.0 + grid_res / 2.0,
        size_x / 2.0,
        grid_res,
        device=device,
        dtype=torch.float32,
    )
    y = torch.arange(
        -size_y / 2.0 + grid_res / 2.0,
        size_y / 2.0,
        grid_res,
        device=device,
        dtype=torch.float32,
    )

    # 若因浮点误差导致长度不符，强制截断/补齐到 nx/ny
    if len(x) != nx:
        x = torch.linspace(-size_x / 2.0 + grid_res / 2.0,
                           size_x / 2.0 - grid_res / 2.0, nx, device=device)
    if len(y) != ny:
        y = torch.linspace(-size_y / 2.0 + grid_res / 2.0,
                           size_y / 2.0 - grid_res / 2.0, ny, device=device)

    xx, yy = torch.meshgrid(x, y, indexing="ij")
    grid_xy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (nx*ny, 2)
    grid_xy = grid_xy + center_xy.unsqueeze(0)
    return grid_xy, nx, ny


def sample_heightmap_from_points(points, grid_xy, max_dist=None):
    """
    从稠密点云中通过俯视图 (XY 平面) 最近邻采样获取高度图真值。

    Args:
        points:   (N, 3) 点云，归一化坐标
        grid_xy:  (M, 2) 查询网格 XY 坐标
        max_dist: 可选的最大有效距离阈值 (归一化坐标)

    Returns:
        heights:     (M,) 各网格位置的高度值 (Z)
        valid_mask:  (M,) bool，表示该网格是否有有效近邻
    """
    if points.shape[0] == 0:
        M = grid_xy.shape[0]
        return (
            torch.zeros(M, device=grid_xy.device, dtype=torch.float32),
            torch.zeros(M, device=grid_xy.device, dtype=torch.bool),
        )

    # XY 平面欧氏距离 (M, N)
    dists = torch.cdist(grid_xy, points[:, :2])
    min_dists, nearest_idx = torch.min(dists, dim=1)  # (M,)
    heights = points[nearest_idx, 2]  # (M,)

    if max_dist is not None:
        valid_mask = min_dists <= max_dist
    else:
        valid_mask = torch.ones_like(heights, dtype=torch.bool)

    return heights, valid_mask


###############################################################################
# End of grid sampling utilities
###############################################################################


###############################################################################
# Network class
###############################################################################

class HeightMapSampler(nn.Module):
    """
    轻量化高度图重采样网络。

    对输入的恢复点云，以机器人水平位置为条件，
    通过局部 k-NN 聚合预测固定矩形网格上的高度值。
    """

    def __init__(
        self,
        k: int = 16,
        hidden_dim: int = 64,
        grid_res: float = 0.03125,
        grid_size: tuple = (0.5, 0.3125),
    ):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.grid_res = grid_res
        self.grid_size = grid_size

        # 逐点编码器: 3 -> hidden_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 局部邻居处理器: [rel_xyz(3), feat(hidden_dim)] -> hidden_dim
        self.local_mlp = nn.Sequential(
            nn.Linear(3 + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 查询解码器: [pooled_local(hidden_dim), query_xy(2), robot_rel(2)] -> 1 (height)
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2 + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def generate_grid(self, center_xy, device):
        """封装 generate_grid_xy，使用内部存储的 grid_size / grid_res。"""
        return generate_grid_xy(center_xy, self.grid_size, self.grid_res, device)

    def forward(self, points, center_xy, robot_rel_xy):
        """
        Args:
            points:        (N, 3) 恢复后的点云 (归一化坐标)
            center_xy:     (2,)   采样网格中心 (通常是机器人水平位置)
            robot_rel_xy:  (2,)   机器人位置相对于点云中心的偏移

        Returns:
            pred_heights:  (M,)   预测高度值
            grid_xy:       (M, 2) 网格 XY 坐标
            nx:            x 方向网格数
            ny:            y 方向网格数
        """
        # 1. 生成查询网格
        grid_xy, nx, ny = self.generate_grid(center_xy, points.device)
        M = grid_xy.shape[0]

        # 2. 编码输入点云
        p_feats = self.point_encoder(points)  # (N, hidden_dim)

        # 3. XY 平面 k-NN 搜索
        k_actual = min(self.k, points.shape[0])
        if points.shape[0] <= k_actual:
            # 点过少时全部复制
            nn_idx = torch.arange(points.shape[0], device=points.device).unsqueeze(0).expand(M, -1)
            k_actual = points.shape[0]
        else:
            dists = torch.cdist(grid_xy, points[:, :2])  # (M, N)
            _, nn_idx = torch.topk(dists, k_actual, largest=False, dim=1)  # (M, k_actual)

        # 4. 聚合局部邻居特征
        neighbor_pts = points[nn_idx]       # (M, k_actual, 3)
        neighbor_feats = p_feats[nn_idx]    # (M, k_actual, hidden_dim)

        # 查询点扩展为 (M, k_actual, 3) 用于计算相对位置
        queries_xyz = torch.cat([
            grid_xy,
            torch.zeros(M, 1, device=points.device, dtype=torch.float32)
        ], dim=1)
        rel_pos = neighbor_pts - queries_xyz.unsqueeze(1)  # (M, k_actual, 3)

        local_input = torch.cat([rel_pos, neighbor_feats], dim=-1)  # (M, k_actual, 3+hidden)
        local_output = self.local_mlp(local_input)                   # (M, k_actual, hidden)
        local_feat = local_output.max(dim=1)[0]                      # (M, hidden)

        # 5. 全局查询解码
        robot_exp = robot_rel_xy.unsqueeze(0).expand(M, -1)          # (M, 2)
        query_input = torch.cat([local_feat, grid_xy, robot_exp], dim=-1)  # (M, hidden+4)
        pred_heights = self.query_mlp(query_input).squeeze(-1)       # (M,)

        return pred_heights, grid_xy, nx, ny


###############################################################################
# End of network class
###############################################################################


###############################################################################
# Visualizer
###############################################################################

class HeightmapVisualizer:
    """高度图结果可视化工具 (基于 Open3D)。"""

    def __init__(self, completion_net, sampler_net, dataloader, device, config):
        self.completion_net = completion_net
        self.sampler_net = sampler_net
        self.dataloader = dataloader
        self.device = device
        self.config = config
        self.train_iter = iter(dataloader)

        self.completion_net.eval()
        self.sampler_net.eval()

        self.max_visualization = config.max_visualization
        self.curr_sample_idx = 0
        self.curr_frame_idx = 0
        self.num_frames = 0
        self.cnt_frames = 0
        self.data_dict = {}
        self.is_first_frame = True
        self.is_running = True
        self.last_next_time = time()

        # 固定归一化坐标系下的机器人位置 (robot frame 中心)
        self.robot_norm_pos = torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)

        print("\n高度图可视化布局:")
        print(" 红: 输入 (残缺) | 绿: 补全恢复")
        print(" 黄: 真值高度图  | 青: 预测高度图")

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="HeightMap Viewer", width=1400, height=900, left=0, top=0)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.2, 0.2, 0.2])
        opt.point_size = 4.0

        self.in_pcd = o3d.geometry.PointCloud()
        self.rec_pcd = o3d.geometry.PointCloud()
        self.gt_hm_pcd = o3d.geometry.PointCloud()
        self.pred_hm_pcd = o3d.geometry.PointCloud()

        for geo in (self.in_pcd, self.rec_pcd, self.gt_hm_pcd, self.pred_hm_pcd):
            self.vis.add_geometry(geo)

        self.vis.register_key_callback(ord("N"), self.render_sample)
        self.vis.register_key_callback(ord("P"), self.prev_sample)
        self.vis.register_key_callback(ord("R"), self.reset_visualization)
        self.vis.register_key_callback(ord("Q"), self.close)

    def visualize(self):
        print("\n[N] 下一帧 | [P] 上一帧 | [R] 重置 | [Q] 退出\n")
        self.render_sample(None)
        while self.is_running:
            self.vis.poll_events()
            self.vis.update_renderer()
        self.vis.destroy_window()

    def close(self, vis):
        self.is_running = False

    def prev_sample(self, vis):
        if abs(time() - self.last_next_time) < 0.02:
            return
        self.curr_frame_idx -= 1
        if self.curr_frame_idx < 0:
            self.curr_frame_idx = 0
            print("已经是第一帧了")
            return
        if self.data_dict is not None:
            print(f"可视化样本 {self.curr_sample_idx + 1}/{self.max_visualization} 第 {self.curr_frame_idx + 1}/{self.num_frames} 帧")
            self.render_frame(self.data_dict, self.curr_frame_idx)
        self.last_next_time = time()

    def reset_visualization(self, vis):
        if abs(time() - self.last_next_time) < 0.02:
            return
        self.curr_frame_idx = 0
        if self.data_dict is not None:
            print(f"可视化样本 {self.curr_sample_idx + 1}/{self.max_visualization} 第 {self.curr_frame_idx + 1}/{self.num_frames} 帧")
            self.render_frame(self.data_dict, self.curr_frame_idx)
        self.vis.reset_view_point(True)
        self.last_next_time = time()

    def render_sample(self, vis):
        if abs(time() - self.last_next_time) < 0.02:
            return

        if self.is_first_frame:
            self.data_dict = next(self.train_iter)
            assert len(self.data_dict["partial"][0]) == 1, "Error: batch_size should be 1 for visualization"
            self.num_frames = self.data_dict["num_frames"]
            self.curr_frame_idx = 0
            self.is_first_frame = False

        print(f"可视化样本 {self.curr_sample_idx + 1}/{self.max_visualization} 第 {self.curr_frame_idx + 1}/{self.num_frames} 帧")
        self.render_frame(self.data_dict, self.curr_frame_idx)

        if self.cnt_frames == 0:
            self.vis.reset_view_point(True)

        self.curr_frame_idx += 1
        self.cnt_frames += 1

        if self.curr_frame_idx >= self.num_frames:
            self.curr_sample_idx += 1
            self.is_first_frame = True

        if self.curr_sample_idx >= self.max_visualization:
            self.is_running = False

        self.last_next_time = time()

    def render_frame(self, data_dict, frame_idx):
        # ---- 数据准备: 当前帧 partial + complete ----
        t_p_points_list_np = data_dict["partial"][frame_idx]
        t_c_points_list_np = data_dict["complete"][frame_idx]

        t_p_points_list = [torch.from_numpy(p).float().to(self.device) for p in t_p_points_list_np]
        t_c_points_list = [torch.from_numpy(p).float().to(self.device) for p in t_c_points_list_np]

        # 体素化
        _, t_p_coords_float_list, t_p_coords_voxel_list, _ = voxelization(
            t_p_points_list, self.config.completion_resolution
        )
        _, _, t_c_coords_voxel_list, _ = voxelization(
            t_c_points_list, self.config.completion_resolution
        )

        curr_feats_list = compute_feats(
            coords_float_list=t_p_coords_float_list,
            coords_voxel_list=t_p_coords_voxel_list,
            time_encoding=0.0,
            prob=1.0,
        )

        batched_input_coords, batched_input_feats = ME.utils.sparse_collate(
            t_p_coords_voxel_list, curr_feats_list, device=self.device
        )
        batched_gt_coords = ME.utils.batched_coordinates(t_c_coords_voxel_list).to(self.device)

        sin_fused = ME.SparseTensor(
            features=batched_input_feats, coordinates=batched_input_coords, device=self.device
        )
        cm = sin_fused.coordinate_manager
        in_target_key, _ = cm.insert_and_map(coordinates=batched_gt_coords, string_id="target")

        # ---- 补全网络前向 ----
        with torch.no_grad():
            _, _, sout, occ_probs = self.completion_net(sin_fused, in_target_key)

        sout_coords_list, _ = sout.decomposed_coordinates_and_features
        sout_points_norm_list, _ = devoxelization(
            sout_coords_list, res=self.config.completion_resolution
        )

        # ---- 生成高度图并可视化 (仅展示第一项) ----
        rec_points = sout_points_norm_list[0]          # (N, 3)
        gt_points = t_c_points_list[0]                 # (M, 3)
        cloud_center = rec_points[:, :2].mean(dim=0)
        robot_rel = self.robot_norm_pos - cloud_center

        with torch.no_grad():
            pred_h, grid_xy, nx, ny = self.sampler_net(rec_points, self.robot_norm_pos, robot_rel)

        gt_h, gt_valid = sample_heightmap_from_points(
            gt_points, grid_xy, max_dist=self.config.max_nn_dist
        )

        # 构建可视化点云 (x, y, z)
        pred_pcd_pts = torch.cat([grid_xy, pred_h.unsqueeze(-1)], dim=1).cpu().numpy()
        gt_pcd_pts = torch.cat([grid_xy, gt_h.unsqueeze(-1)], dim=1).cpu().numpy()

        # 颜色: 根据高度着色
        def color_by_height(pts, z_min=None, z_max=None):
            z = pts[:, 2]
            if z_min is None:
                z_min, z_max = z.min(), z.max()
            if z_max <= z_min:
                z_max = z_min + 1e-6
            norm = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
            colors = np.zeros((len(pts), 3))
            colors[:, 0] = norm
            colors[:, 2] = 1.0 - norm
            return colors

        z_min = min(pred_pcd_pts[:, 2].min(), gt_pcd_pts[:, 2].min())
        z_max = max(pred_pcd_pts[:, 2].max(), gt_pcd_pts[:, 2].max())

        # 输入点云 (红色)
        in_pc = t_p_points_list[0].cpu().numpy()
        self.in_pcd.points = o3d.utility.Vector3dVector(in_pc)
        self.in_pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(in_pc), 1)))

        # 恢复点云 (绿色)
        self.rec_pcd.points = o3d.utility.Vector3dVector(rec_points.cpu().numpy())
        self.rec_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (len(rec_points), 1)))

        # 真值高度图 (左上，高度着色)
        self.gt_hm_pcd.points = o3d.utility.Vector3dVector(gt_pcd_pts)
        self.gt_hm_pcd.colors = o3d.utility.Vector3dVector(color_by_height(gt_pcd_pts, z_min, z_max))

        # 预测高度图 (右上，高度着色)
        self.pred_hm_pcd.points = o3d.utility.Vector3dVector(pred_pcd_pts)
        self.pred_hm_pcd.colors = o3d.utility.Vector3dVector(color_by_height(pred_pcd_pts, z_min, z_max))

        # 平移布局 (使用 relative=False 避免累积漂移)
        self.in_pcd.translate([-0.6, -0.6, 0.0], relative=False)
        self.rec_pcd.translate([0.6, -0.6, 0.0], relative=False)
        self.gt_hm_pcd.translate([-0.6, 0.6, 0.0], relative=False)
        self.pred_hm_pcd.translate([0.6, 0.6, 0.0], relative=False)

        for geo in (self.in_pcd, self.rec_pcd, self.gt_hm_pcd, self.pred_hm_pcd):
            self.vis.update_geometry(geo)

        # 打印误差
        if gt_valid.any():
            err = (pred_h[gt_valid] - gt_h[gt_valid]).abs()
            print(f"  有效格点数: {gt_valid.sum().item()} / {len(gt_valid)}")
            print(f"  高度 L1 误差: mean={err.mean().item():.4f}, max={err.max().item():.4f}")


###############################################################################
# End of visualizer
###############################################################################


###############################################################################
# Train function
###############################################################################

def make_data_loader(phase, batch_size, shuffle, num_workers, repeat, augment_data, transforms, completion_resolution=64, cache_use=False):
    """封装数据加载器创建 (复用已有 dataset)。"""
    class _TempConfig:
        pass
    _TempConfig.resolution = completion_resolution
    _TempConfig.cache_use = cache_use
    data_set = ConstructTerrainDataset(
        phase=phase, config=_TempConfig(), augment_data=augment_data, transforms=transforms
    )
    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": CollationAndTransformation(),
        "pin_memory": True,
        "drop_last": False,
    }
    if repeat:
        args["sampler"] = InfSampler(data_set, shuffle)
    else:
        args["shuffle"] = shuffle
    return torch.utils.data.DataLoader(data_set, **args)


def train(completion_net, sampler_net, dataloader, optimizer, scheduler, start_iter, start_step, device, config):
    """高度图采样网络训练主循环。"""
    timestamp = strftime("%Y%m%d_%H%M%S", localtime())
    run_log_dir = os.path.join(config.log_dir, timestamp)
    os.makedirs(run_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_log_dir)
    print(f"TensorBoard logs: {run_log_dir}")

    model_save_path = os.path.join(config.save_dir, config.model_name)
    os.makedirs(model_save_path, exist_ok=True)
    try:
        shutil.copy2(__file__, model_save_path)
    except Exception as e:
        logging.warning(f"Failed to save script: {e}")

    crit_l1 = nn.L1Loss(reduction="mean")

    completion_net.eval() if config.freeze_completion else completion_net.train()
    sampler_net.train()
    train_iter = iter(dataloader)

    train_steps = start_step
    data_time = 0

    # 归一化网格参数
    grid_res = config.grid_res_phys / config.phys_scale
    grid_size = [config.grid_size_phys[0] / config.phys_scale,
                 config.grid_size_phys[1] / config.phys_scale]

    # 固定机器人归一化位置 (robot frame 中心为 0.5, 0.5)
    robot_norm_pos = torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)

    beg_iter = start_iter
    end_iter = start_iter + config.max_iter

    for iter_idx in range(beg_iter, end_iter):
        start_time = time()
        data_dict = next(train_iter)
        data_time = time() - start_time
        writer.add_scalar("Time/DataLoading", data_time, iter_idx)

        step_losses = []
        step_valid_ratios = []

        optimizer.zero_grad()

        num_frames = data_dict["num_frames"]
        for t in range(num_frames):
            # ---- 数据加载: 当前帧 partial + complete ----
            t_p_points_list_np = data_dict["partial"][t]
            t_c_points_list_np = data_dict["complete"][t]

            t_p_points_list = [torch.from_numpy(p).float().to(device) for p in t_p_points_list_np]
            t_c_points_list = [torch.from_numpy(p).float().to(device) for p in t_c_points_list_np]

            # 体素化当前帧
            _, t_p_coords_float_list, t_p_coords_voxel_list, _ = voxelization(
                t_p_points_list, config.completion_resolution
            )
            _, _, t_c_coords_voxel_list, _ = voxelization(
                t_c_points_list, config.completion_resolution
            )

            curr_feats_list = compute_feats(
                coords_float_list=t_p_coords_float_list,
                coords_voxel_list=t_p_coords_voxel_list,
                time_encoding=0.0,
                prob=1.0,
            )

            batched_input_coords, batched_input_feats = ME.utils.sparse_collate(
                t_p_coords_voxel_list, curr_feats_list, device=device
            )
            batched_gt_coords = ME.utils.batched_coordinates(t_c_coords_voxel_list).to(device)

            sin_fused = ME.SparseTensor(
                features=batched_input_feats, coordinates=batched_input_coords, device=device
            )
            cm = sin_fused.coordinate_manager
            in_target_key, _ = cm.insert_and_map(coordinates=batched_gt_coords, string_id="target")

            # ---- 补全网络前向 (冻结，不计算梯度) ----
            with torch.set_grad_enabled(not config.freeze_completion):
                out_cls, out_targets, sout, occ_probs = completion_net(sin_fused, in_target_key)

            sout_coords_list, _ = sout.decomposed_coordinates_and_features
            sout_points_norm_list, _ = devoxelization(
                sout_coords_list, res=config.completion_resolution
            )

            # ---- 对每个 batch 项进行多次采样训练 ----
            batch_sampler_loss = 0.0
            batch_valid_count = 0
            total_samples = 0
            valid_samples = 0

            for b in range(config.batch_size):
                rec_points = sout_points_norm_list[b]      # 恢复点云 (N, 3)
                gt_points = t_c_points_list[b]             # 真值稠密点云 (M, 3)

                if rec_points.shape[0] < 3 or gt_points.shape[0] < 3:
                    continue

                cloud_center = rec_points[:, :2].mean(dim=0)

                for s in range(config.samples_per_frame):
                    # 模拟机器人水平位置扰动 (数据增强)
                    jitter = (torch.rand(2, device=device) - 0.5) * config.position_jitter
                    query_center = robot_norm_pos + jitter
                    robot_rel = query_center - cloud_center

                    # 网络预测
                    pred_h, grid_xy, nx, ny = sampler_net(rec_points, query_center, robot_rel)

                    # 真值采样
                    gt_h, gt_valid = sample_heightmap_from_points(
                        gt_points, grid_xy, max_dist=config.max_nn_dist
                    )

                    total_samples += len(gt_valid)
                    if gt_valid.any():
                        loss = crit_l1(pred_h[gt_valid], gt_h[gt_valid])
                        batch_sampler_loss += loss
                        batch_valid_count += gt_valid.sum().item()
                        valid_samples += 1

            if valid_samples > 0:
                batch_sampler_loss = batch_sampler_loss / valid_samples
                batch_sampler_loss.backward()

                step_losses.append(batch_sampler_loss.item())
                step_valid_ratios.append(batch_valid_count / total_samples if total_samples > 0 else 0.0)

                train_steps += 1
                writer.add_scalar("Loss/HeightmapL1", batch_sampler_loss.item(), train_steps)

        # ---- 梯度裁剪与参数更新 ----
        torch.nn.utils.clip_grad_norm_(sampler_net.parameters(), max_norm=config.max_norm)
        optimizer.step()
        scheduler.step()

        total_time = time() - start_time
        avg_loss = sum(step_losses) / len(step_losses) if step_losses else 0.0
        avg_valid = sum(step_valid_ratios) / len(step_valid_ratios) if step_valid_ratios else 0.0

        writer.add_scalar("Time/IterTotal", total_time, iter_idx)
        writer.add_scalar("Loss/IterHeightmapL1", avg_loss, iter_idx)
        writer.add_scalar("Metrics/ValidRatio", avg_valid, iter_idx)
        writer.add_scalar("Params/LR", scheduler.get_last_lr()[0], iter_idx)

        if iter_idx % config.stat_freq_iter == 0:
            logging.info(
                f"Iter: {iter_idx}, Avg Loss: {avg_loss:.3e}, "
                f"ValidRatio: {avg_valid:.2%}, DataTime: {data_time:.3e}, TotalTime: {total_time:.3e}"
            )

        if iter_idx % config.save_freq_iter == 0:
            save_file = os.path.join(model_save_path, f"sampler_{iter_idx}.pth")
            torch.save({
                "sampler_state_dict": sampler_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "curr_iter": iter_idx,
                "curr_step": train_steps,
                "config": vars(config),
            }, save_file)
            logging.info(f"Saved sampler checkpoint: {save_file}")
            sampler_net.train()

    writer.close()


###############################################################################
# End of train function
###############################################################################


###############################################################################
# Main Thread
###############################################################################

if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # 1. 初始化补全网络并加载预训练权重
    # -------------------------------------------------------------------------
    completion_net = LidarCompletionNet(
        resolution=config.completion_resolution,
        activeF=config.completion_activeF,
        encoder_channels=ENC_CHANNELS,
        decoder_channels=DEC_CHANNELS,
    ).to(device)

    if not os.path.exists(config.completion_checkpoint):
        raise FileNotFoundError(f"Completion checkpoint not found: {config.completion_checkpoint}")

    ckpt = torch.load(config.completion_checkpoint, map_location=device)
    completion_net.load_state_dict(ckpt["state_dict"])
    logging.info(f"Loaded completion network from: {config.completion_checkpoint}")

    if config.freeze_completion:
        for p in completion_net.parameters():
            p.requires_grad = False
        completion_net.eval()
        logging.info("Completion network is FROZEN.")
    else:
        logging.info("Completion network is UNFROZEN (joint training).")

    comp_params = sum(p.numel() for p in completion_net.parameters())
    print(f"补全网络参数量: {comp_params:,} ({comp_params / 1e6:.2f}M)")

    # -------------------------------------------------------------------------
    # 2. 初始化高度图采样网络
    # -------------------------------------------------------------------------
    grid_res = config.grid_res_phys / config.phys_scale
    grid_size = (config.grid_size_phys[0] / config.phys_scale,
                 config.grid_size_phys[1] / config.phys_scale)

    sampler_net = HeightMapSampler(
        k=config.k_neighbors,
        hidden_dim=config.hidden_dim,
        grid_res=grid_res,
        grid_size=grid_size,
    ).to(device)

    sampler_params = sum(p.numel() for p in sampler_net.parameters())
    trainable_sampler_params = sum(p.numel() for p in sampler_net.parameters() if p.requires_grad)
    print(f"高度图采样网络参数量: {sampler_params:,} ({sampler_params / 1e6:.3f}M)")
    print(f"可训练参数量: {trainable_sampler_params:,}")

    # -------------------------------------------------------------------------
    # 3. 数据加载器
    # -------------------------------------------------------------------------
    dataloader = make_data_loader(
        phase="train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        augment_data=True,
        transforms=[
            lambda x: StridedSamplingTransform(x, stride_list=[1] * 11 + [2] * 7 + [3] * 4 + [4] * 2 + [5] * 1),
            # lambda x: VoxelFilterTransform(x, voxel_size=0.015),
        ],
        completion_resolution=config.completion_resolution,
    )

    # -------------------------------------------------------------------------
    # 4. 训练或可视化模式
    # -------------------------------------------------------------------------
    if not config.eval:
        optimizer = optim.AdamW(
            sampler_net.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_iter, eta_min=1e-5
        )

        start_iter = 0
        start_step = 0
        if config.resume:
            checkpoint_dir = os.path.join(config.save_dir, config.model_name)
            ckpt_files = glob.glob(os.path.join(checkpoint_dir, "sampler_*.pth"))
            if ckpt_files:
                ckpt_files.sort(key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0]))
                latest = ckpt_files[-1]
                logging.info(f"Resuming sampler from: {latest}")
                state = torch.load(latest, map_location=device)
                sampler_net.load_state_dict(state["sampler_state_dict"])
                optimizer.load_state_dict(state["optimizer"])
                if "scheduler" in state:
                    scheduler.load_state_dict(state["scheduler"])
                start_iter = state["curr_iter"] + 1
                start_step = state["curr_step"] + 1
            else:
                logging.warning("No sampler checkpoint found. Starting from scratch.")

        train(completion_net, sampler_net, dataloader, optimizer, scheduler,
              start_iter, start_step, device, config)

    else:
        # 可视化模式: 加载采样器最新权重
        checkpoint_dir = os.path.join(config.save_dir, config.model_name)
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "sampler_*.pth"))
        if not ckpt_files:
            raise FileNotFoundError(f"No sampler checkpoints found in {checkpoint_dir}")
        ckpt_files.sort(key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0]))
        latest = ckpt_files[-1]
        logging.info(f"Evaluating sampler from: {latest}")
        state = torch.load(latest, map_location=device)
        sampler_net.load_state_dict(state["sampler_state_dict"])

        vis_dataloader = make_data_loader(
            phase="train",
            batch_size=1,
            shuffle=True,
            num_workers=0,
            repeat=True,
            augment_data=False,
            transforms=[],
            completion_resolution=config.completion_resolution,
        )
        vis = HeightmapVisualizer(completion_net, sampler_net, vis_dataloader, device, config)
        vis.visualize()

###############################################################################
# End of Main Thread
###############################################################################
