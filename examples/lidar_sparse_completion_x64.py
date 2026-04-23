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
import os
import re
import sys
import glob
import subprocess
import argparse
import logging
import numpy as np
from time import time, strftime, localtime
import urllib

# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import Sampler
from torch.utils.tensorboard import SummaryWriter

import MinkowskiEngine as ME

assert (
    int(o3d.__version__.split(".")[1]) >= 8
), f"Requires open3d version >= 0.8, the current version is {o3d.__version__}"


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
parser.add_argument("--resolution",         type=int,                   default=64)
parser.add_argument("--max_iter",           type=int,                   default=30001)
parser.add_argument("--stat_freq_iter",     type=int,                   default=50)
parser.add_argument("--save_freq_iter",     type=int,                   default=5000)
parser.add_argument("--batch_size",         type=int,                   default=4)
parser.add_argument("--lr",                 type=float,                 default=1e-3)
parser.add_argument("--weight_decay",       type=float,                 default=1e-4)
parser.add_argument("--voxel_coef",         type=float,                 default=1.00)
parser.add_argument("--chamfer_coef",       type=float,                 default=0.25)
parser.add_argument("--chamfer_p_coef",     type=float,                 default=0.5,    help="chamfer dist precision")
parser.add_argument("--chamfer_r_coef",     type=float,                 default=2.0,    help="chamfer dist recall")
parser.add_argument("--tv_coef",            type=float,                 default=0.01,   help="total variation regularization coef")
parser.add_argument("--max_norm",           type=float,                 default=1.0)
parser.add_argument("--num_workers",        type=int,                   default=4)
parser.add_argument("--log_dir",            type=str,                   default="./output/logs")
parser.add_argument("--save_dir",           type=str,                   default="./output/checkpoint")
parser.add_argument("--model_name",         type=str,                   default="lidar_completion_x64_v1")
parser.add_argument("--load_optimizer",     type=str,                   default=True)
parser.add_argument("--cache_use",          type=bool,                  default=False)
parser.add_argument("--max_visualization",  type=int,                   default=10)
parser.add_argument("--resume",             action="store_true")
parser.add_argument("--eval",               action="store_true")

PURNING_THRESHOLD = 0.35
# ENC_CHANNELS = [16, 32, 64, 128, 512, 1024]
# DEC_CHANNELS = [16, 32, 64, 128, 512, 1024]
ENC_CHANNELS = [16, 32, 64, 128, 256, 512]
DEC_CHANNELS = [16, 32, 64, 128, 256, 512]


###############################################################################
# End of global configs
###############################################################################


###############################################################################
# Utility functions
###############################################################################

def voxelization(points_list, res: int=1, feats_list=None):
    """
    辅助函数: 进行体素尺寸放大，并使用ME量化工具进行去重
    输入:
        points_list: List[torch.Tensor], 包含 Batch 中每个样本的点云坐标 (N_i, 3)。
                     数据类型应为 torch.float32。
        res: float, 体素分辨率
    返回:
        unique_points_list: List[torch.Tensor], 去重后的原始归一化坐标
        coords_float_list: List[torch.Tensor], 去重后的浮点放大坐标 (用于计算 Offset 特征)
        coords_int_list: List[torch.Tensor], 去重后的整型体素坐标 (用于构建 SparseTensor)
    """
    unique_points_list = []
    coords_float_list = []
    coords_int_list = []
    new_feats_list = []
    
    for i, points in enumerate(points_list):
        coords_float = points * res
        discrete_coords, indices = ME.utils.sparse_quantize(
            coordinates=coords_float, 
            return_index=True, 
            quantization_size=1,
            device=points.device # Use the device of input points
        )
        
        unique_points_list.append(points[indices])
        coords_float_list.append(coords_float[indices])
        coords_int_list.append(discrete_coords.int())
        if feats_list is not None:
            new_feats_list.append(feats_list[i][indices])
        
    return unique_points_list, coords_float_list, coords_int_list, new_feats_list


def devoxelization(coords_list, res: int=1):
    points_list = []
    points_norm_list = []
    
    for coords in coords_list:
        points = coords.float()
        points_norm = points / res
        
        points_list.append(points)
        points_norm_list.append(points_norm)
        
    return points_norm_list, points_list


def compute_feats(coords_float_list, coords_voxel_list, time_encoding, prob: float | list[torch.Tensor]):
    """
    辅助函数: 计算稀疏张量的特征 (适配 PyTorch Tensor)
    输入:
        coords_float_list: List[torch.Tensor], 浮点放大坐标列表 [(N1, 3), (N2, 3), ...]
        coords_voxel_list: List[torch.Tensor], 整型体素坐标列表 [(N1, 3), (N2, 3), ...]
        time_encoding: float, 时间编码 (curr: 0.0, hist: 1.0)
        prob_values: Optional[List[torch.Tensor]], 每个点的占据概率列表 [(N1,), (N2,), ...]
    返回:
        feats_list: List[torch.Tensor], 特征矩阵列表 [(N1, 5), (N2, 5), ...] (Offset 3 + Time 1 + Prob 1)
    """
    feats_list = []
    
    for i, (coords_float, coords_voxel) in enumerate(zip(coords_float_list, coords_voxel_list)):
        feats_offset = coords_float - coords_voxel.float()
        feats_temporal = torch.full(
            (coords_float.shape[0], 1), 
            time_encoding, 
            dtype=coords_float.dtype, 
            device=coords_float.device
        )
        if isinstance(prob, float):
            feats_prob = torch.full(
                (coords_float.shape[0], 1), 
                prob, 
                dtype=coords_float.dtype, 
                device=coords_float.device
            )
        else:
            feats_prob = prob[i]
        feats = torch.cat([feats_offset, feats_temporal, feats_prob], dim=1) # Concatenate: 3 + 1 + 1 = 5
        feats_list.append(feats)
        
    return feats_list


def PointCloud(points, color=None, translate_offset=None, rotate_matrix=None):
    """
    辅助函数，将输入的 Tensor 点云转化为 Open3D 中的 pcd 点云格式，方便可视化/保存等
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
    if translate_offset is not None:
        pcd.translate(translate_offset)
    if rotate_matrix is not None:
        pcd.rotate(rotate_matrix, center=(0,0,0))
    return pcd


def make_data_loader(phase, batch_size, shuffle, num_workers, repeat, config, augment_data, transforms):
    """
    辅助函数，用于获取 data loader
    """
    data_set = ConstructTerrainDataset(phase=phase, config=config, augment_data=augment_data, transforms=transforms)

    args = {
        "batch_size": batch_size,                       # 小批量样本尺寸
        "num_workers": num_workers,                     # 样本并行加载线程
        "collate_fn": CollationAndTransformation(),     # 小批量样本聚合函数
        "pin_memory": True,                             # 内存连续性对齐
        "drop_last": False,                             # 是否丢弃不完整的最后一个小批量样本
    }

    if repeat:
        args["sampler"] = InfSampler(data_set, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(data_set, **args)
    
    return loader


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def transform_points(
    points: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Transform input points in a given frame to a target frame.

    This function transform points from a source frame to a target frame. The transformation is defined by the
    position :math:`t` and orientation :math:`R` of the target frame in the source frame.

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    If the input `points` is a batch of points, the inputs `pos` and `quat` must be either a batch of
    positions and quaternions or a single position and quaternion. If the inputs `pos` and `quat` are
    a single position and quaternion, the same transformation is applied to all points in the batch.

    If either the inputs :attr:`pos` and :attr:`quat` are None, the corresponding transformation is not applied.

    Args:
        points: Points to transform. Shape is (N, P, 3) or (P, 3).
        pos: Position of the target frame. Shape is (N, 3) or (3,).
            Defaults to None, in which case the position is assumed to be zero.
        quat: Quaternion orientation of the target frame in (w, x, y, z). Shape is (N, 4) or (4,).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        Transformed points in the target frame. Shape is (N, P, 3) or (P, 3).

    Raises:
        ValueError: If the inputs `points` is not of shape (N, P, 3) or (P, 3).
        ValueError: If the inputs `pos` is not of shape (N, 3) or (3,).
        ValueError: If the inputs `quat` is not of shape (N, 4) or (4,).
    """
    points_batch = points.clone()
    # check if inputs are batched
    is_batched = points_batch.dim() == 3
    # -- check inputs
    if points_batch.dim() == 2:
        points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
    if points_batch.dim() != 3:
        raise ValueError(f"Expected points to have dim = 2 or dim = 3: got shape {points.shape}")
    if not (pos is None or pos.dim() == 1 or pos.dim() == 2):
        raise ValueError(f"Expected pos to have dim = 1 or dim = 2: got shape {pos.shape}")
    if not (quat is None or quat.dim() == 1 or quat.dim() == 2):
        raise ValueError(f"Expected quat to have dim = 1 or dim = 2: got shape {quat.shape}")
    # -- rotation
    if quat is not None:
        # convert to batched rotation matrix
        rot_mat = matrix_from_quat(quat)
        if rot_mat.dim() == 2:
            rot_mat = rot_mat[None]  # (3, 3) -> (1, 3, 3)
        # convert points to matching batch size (N, P, 3) -> (N, 3, P)
        # and apply rotation
        points_batch = torch.matmul(rot_mat, points_batch.transpose_(1, 2))
        # (N, 3, P) -> (N, P, 3)
        points_batch = points_batch.transpose_(1, 2)
    # -- translation
    if pos is not None:
        # convert to batched translation vector
        if pos.dim() == 1:
            pos = pos[None, None, :]  # (3,) -> (1, 1, 3)
        else:
            pos = pos[:, None, :]  # (N, 3) -> (N, 1, 3)
        # apply translation
        points_batch += pos
    # -- return points in same shape as input
    if not is_batched:
        points_batch = points_batch.squeeze(0)  # (1, P, 3) -> (P, 3)

    return points_batch


def points_transform_and_normclip(points_prev_list: list[torch.Tensor], 
                                  pos_curr: torch.Tensor, quat_curr: torch.Tensor, 
                                  pos_prev: torch.Tensor, quat_prev: torch.Tensor,
                                  feats_prev_list: list[torch.Tensor] = None,
                                  scale: float=3.2, bound: float=0.5):
    """
        N = num_envs
            time K_t:
                pos_curr shape (N, 3)
                quat_curr shape (N, 4)
            time K_t-1:
                points_prev list [(P1, 3), (P2, 3), ... , (Pn, 3)] 
                pos_prev shape (N, 3)
                quat_prev shape (N, 4)
    """
    lengths = [p.shape[0] for p in points_prev_list]
    points_prev = pad_sequence(points_prev_list, batch_first=True, padding_value=float('inf'))  # (N, P, 3)
    points_prev -= 0.5                                                                          # (0, 1) -> (-0.5, +0.5)
    
    rot_matrix_curr = matrix_from_quat(quat_curr)                                               # (N, 4) -> (N, 3, 3)

    points_world_prev = transform_points(points_prev * scale, pos_prev, quat_prev)              # (N, P, 3)
    points_world_prev_centered = points_world_prev - pos_curr.unsqueeze(1)                      # (N, P, 3)
    
    points_prev_view = torch.matmul(points_world_prev_centered, rot_matrix_curr)                # (N, P, 3)
    points_prev_view_norm = points_prev_view / scale                                            # (N, P, 3)

    mask_inside = (points_prev_view_norm.abs() < bound).all(dim=2)
    
    has_points = mask_inside.any(dim=1)
    if not has_points.all():
        bad_indices = torch.where(~has_points)[0]
        raise RuntimeError(f"数据异常：以下样本的点云在归一化裁剪后全部丢失 (Mask全为False): {bad_indices.tolist()}。请检查输入点云范围或增大 bound 值。")
    
    inside_lengths = mask_inside.sum(dim=1)
    all_inside_points = points_prev_view_norm[mask_inside]
    points_prev_inside_list = torch.split(all_inside_points + 0.5, inside_lengths.tolist())     # (-0.5, +0.5) -> (0, 1)

    if feats_prev_list is not None:
        feats_prev = pad_sequence(feats_prev_list, batch_first=True, padding_value=float('inf'))
        all_inside_feats = feats_prev[mask_inside]
        feats_prev_inside_list = torch.split(all_inside_feats, inside_lengths.tolist())
        return points_prev_inside_list, feats_prev_inside_list

    return points_prev_inside_list


###############################################################################
# End of utility functions
###############################################################################


###############################################################################
# Preprocess functions
###############################################################################

def VoxelFilterTransform(data_dict, voxel_size=0.05):
    """
    Transform 函数：对 data_dict['complete'] 中的点云序列进行体素下采样 (Voxel Grid Filter)
    
    Args:
        data_dict (dict): 包含 'partial', 'complete' 等键的字典。
        voxel_size (float): 体素的边长。值越大，点云越稀疏。
        
    Returns:
        dict: 处理后的 data_dict，其中 'complete' 已被下采样。
    """
    if 'complete' not in data_dict:
        return data_dict

    complete_sequence = data_dict['complete'] # List[N_frames] of np.ndarray (P_i, 3)
    filtered_complete_sequence = []

    for points in complete_sequence:
        if len(points) == 0:
            filtered_complete_sequence.append(points)
            continue
            
        # Open3D 体素滤波
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        points_down = np.asarray(pcd_down.points)
        filtered_complete_sequence.append(points_down)

    data_dict['complete'] = filtered_complete_sequence
    
    return data_dict


def RandomRotationTransform(data_dict, max_angle_deg=180.0):
    """
    Transform 函数：对 data_dict['partial'] 中的每帧点云，绕中心 (0.5,0.5,0.5) 进行随机旋转

    Args:
        data_dict (dict): 包含 'partial' 等键的字典。
        max_angle_deg (float): 最大旋转角度（度），实际旋转角度在 [-max_angle_deg, +max_angle_deg] 内均匀采样。
    """
    if 'partial' not in data_dict:
        return data_dict

    max_angle_rad = np.radians(max_angle_deg)
    # 随机旋转轴（单位向量）和旋转角度
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = np.random.uniform(-max_angle_rad, max_angle_rad)

    # Rodrigues 旋转公式: R = I*cos(θ) + (1-cos(θ))*n*n^T + sin(θ)*[n]_x
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * K

    center = np.array([0.5, 0.5, 0.5])
    rotated_sequence = []
    for points in data_dict['partial']:
        if len(points) == 0:
            rotated_sequence.append(points)
            continue
        # 平移到原点 → 旋转 → 平移回来
        rotated = (points - center) @ R.T + center
        # 裁剪到 [0, 1] 边界
        rotated = np.clip(rotated, 0.0, 1.0)
        rotated_sequence.append(rotated)

    data_dict['partial'] = rotated_sequence
    return data_dict


def RandomCylinderCutoutTransform(data_dict, max_cylinders=3, max_radius=0.15):
    """
    Transform 函数：对 data_dict['partial'] 中的每帧点云，在随机平面位置处，
    以随机半径的圆柱（Z轴方向无限延伸）挖去该区域内的点。
    可同时生成多个圆柱区域，数量在 [1, max_cylinders] 内随机。

    Args:
        data_dict (dict): 包含 'partial' 等键的字典。
        max_cylinders (int): 最大圆柱数量，实际数量在 [1, max_cylinders] 内随机。
        max_radius (float): 圆柱最大半径（在归一化坐标 0~1 空间下）。
    """
    if 'partial' not in data_dict:
        return data_dict

    # 随机生成 1~max_cylinders 个圆柱
    n_cylinders = np.random.randint(1, max_cylinders + 1)
    cylinders = []
    for _ in range(n_cylinders):
        cx = np.random.uniform(0.0, 1.0)
        cy = np.random.uniform(0.0, 1.0)
        radius = np.random.uniform(0.001, max_radius)
        cylinders.append((cx, cy, radius))

    cutout_sequence = []
    for points in data_dict['partial']:
        if len(points) == 0:
            cutout_sequence.append(points)
            continue
        # 逐个圆柱剔除，保留所有圆柱外部的点
        mask = np.ones(len(points), dtype=bool)
        for cx, cy, radius in cylinders:
            dist = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
            mask &= dist > radius
        cutout_sequence.append(points[mask])

    data_dict['partial'] = cutout_sequence
    return data_dict


def RandomNoiseTransform(data_dict, noise_std=0.005):
    """
    Transform 函数：对 data_dict['partial'] 中的每帧点云，添加高斯噪声，并裁剪到 [0, 1] 范围内。

    Args:
        data_dict (dict): 包含 'partial' 等键的字典。
        noise_std (float): 高斯噪声的标准差。
    """
    if 'partial' not in data_dict:
        return data_dict

    noisy_sequence = []
    for points in data_dict['partial']:
        if len(points) == 0:
            noisy_sequence.append(points)
            continue
        noise = np.random.normal(0, noise_std, size=points.shape)
        noisy = points + noise
        # 裁剪到 [0, 1] 边界
        noisy = np.clip(noisy, 0.0, 1.0)
        noisy_sequence.append(noisy)

    data_dict['partial'] = noisy_sequence
    return data_dict


def StridedSamplingTransform(data_dict, stride_list=[3, 8]):
    """
    Transform 函数：对 data_dict 中的时序数据进行间隔采样（Strided Sampling）。
    从 stride_list 中随机选择一个间隔，对 partial、complete、pos_data、quat_data
    按该间隔从首帧采到末帧。

    Args:
        data_dict (dict): 包含 'partial', 'complete', 'pos_data', 'quat_data' 等键的字典。
        stride_list (list[int]): 采样间隔候选列表，从中随机选择一个间隔。
    """
    if 'partial' not in data_dict or len(data_dict['partial']) == 0:
        return data_dict

    num_frames = len(data_dict['partial'])
    if num_frames < 2:
        return data_dict

    # 从 stride_list 中随机选择一个间隔
    stride = int(np.random.choice(stride_list))

    # 生成采样索引：从首帧开始，按间隔采到末帧
    indices = list(range(0, num_frames, stride))

    if len(indices) < 2:
        # 如果采样后帧数过少，则跳过采样，保留原数据
        return data_dict

    # 对各时序数据按索引采样
    data_dict['partial'] = [data_dict['partial'][i] for i in indices]
    data_dict['complete'] = [data_dict['complete'][i] for i in indices]
    data_dict['pos_data'] = data_dict['pos_data'][indices]
    data_dict['quat_data'] = data_dict['quat_data'][indices]
    data_dict['num_frames'] = len(indices)

    return data_dict


###############################################################################
# Preprocess functions
###############################################################################


###############################################################################
# Utility classes
###############################################################################

# 自定义数据集类
class ConstructTerrainDataset(torch.utils.data.Dataset):
    def __init__(self, phase="train", type=None, config=None, augment_data=False, transforms=None):
        self.type = type                        # "walk", ...
        self.phase = phase                      # "train", "test"
        self.augment_data = augment_data        # 是否启用数据增强
        self.transforms = transforms            # 预处理函数
        self.resolution = config.resolution     # 体素分辨率
        self.samples = []                       # 存储配对好的样本信息
        self.cache = {}                         # 样本加载缓存
        self.last_cache_percent = 0             # 样本缓存比例
        self.cache_use = config.cache_use
        
        # 修改根目录路径以适应新数据集结构
        self.root = "/workspace/Share/DataCollection/Processed"  # 根据实际数据集路径调整
        
        # 如果指定了 type (如 "walk")，则列表只包含该目录
        if self.type is not None:
            type_dirs = [self.type]
        # 不指定则获取全部目录
        else:
            all_types = glob.glob(os.path.join(self.root, "*/"))
            type_dirs = sorted([os.path.basename(os.path.normpath(d)) for d in all_types if os.path.isdir(d)])
        logging.info(f"Scanning dataset types: {type_dirs}")

        # 遍历每一个类型目录进行样本配对
        for current_type in type_dirs:
            partial_root = os.path.join(self.root, current_type, self.phase, "partial")
            complete_root = os.path.join(self.root, current_type, self.phase, "complete")
            transform_root = os.path.join(self.root, current_type, self.phase, "transform")
        
            # 检查目录是否存在，确保配置的运动类别有效
            if not os.path.exists(partial_root) or not os.path.exists(complete_root) or not os.path.exists(transform_root):
                error_msg = f"Directory not found for type '{current_type}'. Checked paths:\n  Partial: {partial_root}\n  Complete: {complete_root}\n  Transform: {transform_root}"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # 先获取 Partial 下全部序列样本的文件夹列表，并排序 (例如: ['0001', '0002'])
            partial_seq_dirs = sorted(glob.glob(os.path.join(partial_root, "*/")))
            # 遍历 Partial 序列样本文件夹列表
            for partial_seq_path in partial_seq_dirs:
                # 对应构建 Comple 序列样本文件夹路径
                seq_idx_s = os.path.basename(os.path.normpath(partial_seq_path))    # 获取 "0001"
                complete_seq_path = os.path.join(complete_root, seq_idx_s)          # 组合到 complete_root
                transform_path = os.path.join(transform_root, f"{seq_idx_s}.npz")   # 对应的变换文件路径
                
                # 检查 complete 目录是否存在，确保数据配对完整性
                if not os.path.exists(complete_seq_path):
                    error_msg = f"Missing complete data for {seq_idx_s} at path: {complete_seq_path}. Data integrity check failed."
                    logging.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                # 检查 transform 文件是否存在
                if not os.path.exists(transform_path):
                    error_msg = f"Missing transform data for {seq_idx_s} at path: {transform_path}. Data integrity check failed."
                    logging.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                # 获取该序列下所有的帧文件 (001.pcd ~ xxx.pcd)
                p_fnames = sorted(glob.glob(os.path.join(partial_seq_path, "*.pcd")))
                c_fnames = sorted(glob.glob(os.path.join(complete_seq_path, "*.pcd")))
            
                # 检查确保帧数一致
                if len(p_fnames) != len(c_fnames):
                    error_msg = f"Frame count mismatch in {seq_idx_s}: found {len(p_fnames)} partial frames and {len(c_fnames)} complete frames."
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                
                # 加载变换数据
                transform_data = np.load(transform_path)
                pos_data = transform_data['pos']  # (N, 3) 位置数据
                quat_data = transform_data['quat']  # (N, 4) 姿态数据
                
                # 检查变换数据的帧数是否匹配
                if len(pos_data) != len(p_fnames) or len(quat_data) != len(p_fnames):
                    error_msg = f"Transform data frame count mismatch in {seq_idx_s}: found {len(p_fnames)} pcd frames but {len(pos_data)} pos frames and {len(quat_data)} quat frames."
                    logging.error(error_msg)
                    raise ValueError(error_msg)
            
                # 将这一组时序样本加入列表
                self.samples.append(
                    {
                        'type': current_type,
                        'seq_idx': seq_idx_s,
                        'partial_paths': p_fnames,          
                        'complete_paths': c_fnames,         
                        'transform_path': transform_path,   
                        'pos_data': pos_data,               
                        'quat_data': quat_data,             
                    }
                )
        
        assert len(self.samples) > 0, "No paired samples loaded!"
        logging.info(f"Loaded {len(self.samples)} sequences for phase: {phase}")
        
        # 忽略 Open3D 警告
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
    def __len__(self):
        return len(self.samples)
    
    def _load_pcd_sequence(self, pcd_paths):
        """
        辅助函数：加载一组 PCD 文件路径，处理并合并为一个样本列表
        """
        data_list = []
        
        for pcd_path in pcd_paths:
            if self.cache_use:
                # 缓存检查
                if pcd_path in self.cache:
                    points = self.cache[pcd_path]
                    data_list.append(points)
                    continue
            
            # 读取 PCD 文件
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            # 检查点云是否为空，确保数据有效性
            if len(points) == 0:
                error_msg = f"Empty point cloud detected in file: {pcd_path}. Data integrity check failed."
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            # 检查是否在 (-0.5, 0.5) 范围内
            is_norm_rule1 = (points.min() > -0.50) and (points.max() < 0.50)
            is_norm_rule2 = (points.min() > -0.55) and (points.max() < 0.55)
            if is_norm_rule1:
                # (-0.5, 0.5) -> (0, 1)
                points += 0.5
            elif is_norm_rule2:
                mask = (points > -0.5) & (points < 0.5)
                valid_points_mask = np.all(mask, axis=1)
                points = points[valid_points_mask]

                if len(points) == 0:
                    error_msg = f"After filtering, no points remain in valid range (-0.5, 0.5) for file: {pcd_path}."
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                
                points += 0.5
            else:
                error_msg = (
                    f"\n--- 数据归一化检查失败 (Data Normalization Check Failed) ---\n"
                    f"File: {pcd_path}\n"
                    f"Coordinate Range: [{points.min():.3f}, {points.max():.3f}]\n"
                    f"Requirement: Must be in the range [-0.5, +0.5] \n"
                    f"Action: Please normalize your mesh/point cloud data before training.\n"
                    f"--- 建议：请检查预处理脚本是否对 {pcd_path} 执行了归一化 ---"
                )
                logging.error(error_msg)
                assert False, error_msg
            
            if self.cache_use:
                # 存入缓存
                self.cache[pcd_path] = points

            # 时序性存入列表
            data_list.append(points)
        
        return data_list

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        data_dict = {
            'type': sample_info['type'],
            'seq_idx': sample_info['seq_idx'],
            'num_frames': len(sample_info['partial_paths']),
            'partial': self._load_pcd_sequence(sample_info['partial_paths']),
            'complete': self._load_pcd_sequence(sample_info['complete_paths']),
            'pos_data': sample_info['pos_data'],
            'quat_data': sample_info['quat_data'],
        }

        if self.augment_data:
            for t in self.transforms:
                data_dict = t(data_dict)
        
        return data_dict


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        """
        构造函数，保存采样数据源、混洗标志位，生成样本采样序列
        """
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        """
        生成样本采样序列，后续会然此序列，逐个弹出样本
        """
        perm = len(self.data_source)    # 样本总数
        # 若开启数据混洗，则随机生成样本序列号列表
        if self.shuffle:
            self._perm = torch.randperm(perm).tolist()
        # 反之，生成顺序样本序列号列表
        else:
            self._perm = list(range(perm))

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.data_source)
    
    def __next__(self):
        # 若样本生成序列已空，则重新生成
        if len(self._perm) == 0:
            self.reset_permutation()
        # 弹出一个样本
        return self._perm.pop()


class CollationAndTransformation:
    """
    样本聚合类，用于聚合多 Worker 读取的单样本数据，生成小批量样本数据
    """
    def __init__(self):
        None

    # 回调处理函数，对样本离散坐标进行裁剪，并返回小批量聚合样本
    def __call__(self, list_data):            
        type_list = [data['type'] for data in list_data]
        seq_idx_list = [data['seq_idx'] for data in list_data]
        num_frames_list = [data['num_frames'] for data in list_data]
        partial_list = [data['partial'] for data in list_data]
        complete_list = [data['complete'] for data in list_data]
        pos_data_list = [data['pos_data'] for data in list_data]
        quat_data_list = [data['quat_data'] for data in list_data]
        
        num_frames_tensor = torch.tensor(num_frames_list)
        if torch.all(num_frames_tensor == num_frames_tensor[0]):
            final_num_frames = int(num_frames_tensor[0].item())
            turncated_p_list = partial_list
            turncated_c_list = complete_list
            turncated_pos_list = pos_data_list
            turncated_quat_list = quat_data_list
        else:
            min_frames = int(num_frames_tensor.min().item())
            final_num_frames = min_frames
            turncated_p_list = [seq[:min_frames] for seq in partial_list]
            turncated_c_list = [seq[:min_frames] for seq in complete_list]
            turncated_pos_list = [seq[:min_frames] for seq in pos_data_list]
            turncated_quat_list = [seq[:min_frames] for seq in quat_data_list]
            
        time_p_list = []
        time_c_list = []
        time_pos_list = []
        time_quat_list = []
        for t in range(final_num_frames):
            t_p_coords = [sample_seq[t] for sample_seq in turncated_p_list]
            t_c_coords = [sample_seq[t] for sample_seq in turncated_c_list]
            t_pos_data = [sample_seq[t] for sample_seq in turncated_pos_list]
            t_quat_data = [sample_seq[t] for sample_seq in turncated_quat_list]
            time_p_list.append(t_p_coords)
            time_c_list.append(t_c_coords)
            time_pos_list.append(t_pos_data)
            time_quat_list.append(t_quat_data)

        return {
            'type': type_list,
            'seq_idx': seq_idx_list,
            'num_frames': final_num_frames,
            'partial': time_p_list,
            'complete': time_c_list,
            'pos_data': time_pos_list,
            'quat_data': time_quat_list,
        }


class ChamferDistanceLoss(nn.Module):
    """
    计算两个点云集合之间的 Chamfer Distance。
    无需两个点云的点数相同，也无需点的顺序对应。
    """
    def __init__(self, precision, recall):
        super(ChamferDistanceLoss, self).__init__()
        self.precision_coef = precision
        self.recall_coef = recall

    def forward(self, pred, gt):
        """
        Args:
            pred: 预测点云，形状为 (B, N, 3) 或 (N, 3)
            gt: 真值点云，形状为 (B, M, 3) 或 (M, 3)
                N 和 M 可以不同 (点数不同)。
        Returns:
            loss: 标量 (Scalar) 损失值
        """
        # 处理单个样本的情况 (维度扩展)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0) # (N, 3) -> (1, N, 3)
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)     # (M, 3) -> (1, M, 3)

        # 计算 Batch 间的距离矩阵
        # (B, N, 3) -> (B, N, 1, 3)
        # (B, M, 3) -> (B, 1, M, 3)
        # 广播机制自动计算欧氏距离
        dist_matrix = torch.cdist(pred, gt, p=2) # (B, N, M)
        
        # 方向 1: 每个预测点到真值点的最小距离
        min_dist_pred_to_gt = torch.min(dist_matrix, dim=2)[0] # (B, N)
        # 方向 2: 每个真值点到预测点的最小距离
        min_dist_gt_to_pred = torch.min(dist_matrix, dim=1)[0] # (B, M)
        
        # 计算平均损失 (Chamfer Distance)
        # 系数调节精度与回召率
        loss = torch.mean(min_dist_pred_to_gt) * self.precision_coef + torch.mean(min_dist_gt_to_pred) * self.recall_coef
        
        return loss


class MinkowskiTVLoss(nn.Module):
    """
    Total Variation Loss for sparse voxel occupancy field.
    Uses MinkowskiConvolution (3x3x3 uniform averaging) to compute neighbor average,
    then penalizes the difference between each voxel's occupancy probability
    and its neighbors' average probability.

    L_tv = λ * Σ |σ(logit_v) - avg(σ(logits of neighbors))|
    """
    def __init__(self):
        super().__init__()
        self.tv_conv = ME.MinkowskiConvolution(
            1, 1, kernel_size=3, bias=False, dimension=3
        )
        # Freeze weights (TV conv is fixed, not learned)
        for param in self.tv_conv.parameters():
            param.requires_grad = False
        # Initialize with uniform averaging: all 27 positions (3x3x3) get weight 1/27
        with torch.no_grad():
            nn.init.constant_(self.tv_conv.kernel, 1.0 / 27)

    def forward(self, cls_sparse_tensor):
        """
        Args:
            cls_sparse_tensor: ME.SparseTensor with occupancy logits in .F, shape (N, 1)
        Returns:
            tv_loss: scalar TV loss value
        """
        prob = torch.sigmoid(cls_sparse_tensor.F)

        # Create new SparseTensor with probability features (reuse coordinate mapping)
        prob_st = ME.SparseTensor(
            features=prob,
            coordinate_map_key=cls_sparse_tensor.coordinate_map_key,
            coordinate_manager=cls_sparse_tensor.coordinate_manager,
        )

        # Compute 3x3x3 neighborhood average (including self)
        # In sparse conv: neighbor_avg = (1/27)*self_prob + (1/27)*sum_neighbor_probs
        neighbor_avg = self.tv_conv(prob_st)

        # Laplacian: avg_26_neighbors - self_prob
        # When all 26 neighbors exist:
        #   avg_26 = (27*avg_27 - self_prob) / 26
        #   laplacian = avg_26 - self_prob = (avg_27 - self_prob) * 27 / 26
        # When some neighbors are missing (sparse boundary):
        #   laplacian is slightly conservative (under-weights missing neighbors)
        #   This is desired behavior — less regularization at sparse boundaries
        laplacian = (neighbor_avg.F - prob) * (27.0 / 26.0)

        tv_loss = laplacian.abs().mean()

        return tv_loss


class Visualizer:
    def __init__(self, net, dataloader, device, config):
        self.train_iter = iter(dataloader)
        self.net = net
        self.net.eval()
        
        self.crit1 = ChamferDistanceLoss(config.chamfer_p_coef, config.chamfer_r_coef).to(device)
        self.crit2 = nn.BCEWithLogitsLoss().to(device)
        self.crit3 = MinkowskiTVLoss().to(device)

        self.voxel_coef = config.voxel_coef
        self.chamfer_coef = config.chamfer_coef
        self.tv_coef = config.tv_coef
        
        self.M = np.eye(3)
        
        self.max_visualization = config.max_visualization
        self.prev_frame_data = [None] * 1 
        self.curr_sample_idx = 0
        self.curr_frame_idx = 0
        self.num_frames = 0
        self.cnt_frames = 0
        self.data_dict = {}
        self.is_first_frame = True
        self.is_running = True # 控制主循环的标志
        self.last_next_time = time()
        
        print("🔍 可视化布局说明:")
        print(" 红: 当前 | 蓝: 历史")
        print(" 黄: 真值 | 绿: 输出")
        
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Main Viewer", width=1200, height=800, left=0, top=0)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.3, 0.3, 0.3])
        opt.point_size = 5.0
        
        self.sin_curr_pcd = o3d.geometry.PointCloud()
        self.sin_hist_pcd = o3d.geometry.PointCloud()
        self.gt_pcd = o3d.geometry.PointCloud()
        self.sout_pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.sin_curr_pcd)
        self.vis.add_geometry(self.sin_hist_pcd)
        self.vis.add_geometry(self.gt_pcd)
        self.vis.add_geometry(self.sout_pcd)

        # 初始化包围盒线框 (8顶点, 12边)
        def _init_bbox_lineset(color):
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.zeros((8, 3)))
            ls.lines = o3d.utility.Vector2iVector([
                [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                [0, 4], [1, 5], [2, 6], [3, 7],  # 立柱
            ])
            ls.paint_uniform_color(color)
            return ls

        # 静态固定 1x1x1 包围盒 (浅灰色)
        self.sin_curr_bbox = _init_bbox_lineset([0.7, 0.7, 0.7])
        self.sin_hist_bbox = _init_bbox_lineset([0.7, 0.7, 0.7])
        self.gt_bbox = _init_bbox_lineset([0.7, 0.7, 0.7])
        self.sout_bbox = _init_bbox_lineset([0.7, 0.7, 0.7])
        # 动态点云范围包围盒 (青色)
        self.sin_curr_bbox_dyn = _init_bbox_lineset([0.0, 0.8, 1.0])
        self.sin_hist_bbox_dyn = _init_bbox_lineset([0.0, 0.8, 1.0])
        self.gt_bbox_dyn = _init_bbox_lineset([0.0, 0.8, 1.0])
        self.sout_bbox_dyn = _init_bbox_lineset([0.0, 0.8, 1.0])

        self.vis.add_geometry(self.sin_curr_bbox)
        self.vis.add_geometry(self.sin_hist_bbox)
        self.vis.add_geometry(self.gt_bbox)
        self.vis.add_geometry(self.sout_bbox)
        self.vis.add_geometry(self.sin_curr_bbox_dyn)
        self.vis.add_geometry(self.sin_hist_bbox_dyn)
        self.vis.add_geometry(self.gt_bbox_dyn)
        self.vis.add_geometry(self.sout_bbox_dyn)
        
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
            print(f"Visualize sample {self.curr_sample_idx + 1}/{self.max_visualization} frame {self.curr_frame_idx + 1}/{self.num_frames}")
            self.render_frame(self.data_dict, self.curr_frame_idx)
            
        self.last_next_time = time()

    def reset_visualization(self, vis):
        if abs(time() - self.last_next_time) < 0.02:
            return
        self.curr_frame_idx = 0 
        if self.data_dict is not None:
            print(f"Visualize sample {self.curr_sample_idx + 1}/{self.max_visualization} frame {self.curr_frame_idx + 1}/{self.num_frames}")
            self.render_frame(self.data_dict, self.curr_frame_idx)
        self.vis.reset_view_point(True)
        self.last_next_time = time()

    def render_sample(self, vis):
        if abs(time() - self.last_next_time) < 0.02:
            return
        
        if self.is_first_frame:
            self.data_dict = next(self.train_iter)
            assert len(self.data_dict['partial'][0]) == 1, "Error: batch_size should be 1"
            self.num_frames = self.data_dict['num_frames']
            self.curr_frame_idx = 0
            self.is_first_frame = False
            
        print(f"Visualize sample {self.curr_sample_idx + 1}/{self.max_visualization} frame {self.curr_frame_idx + 1}/{self.num_frames}")
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

        t_p_points_norm_list, t_p_coords_float_list, t_p_coords_voxel_list, _ = voxelization(t_p_points_list, config.resolution)
        _, _, t_c_coords_voxel_list, _ = voxelization(t_c_points_list, config.resolution)
        t_c_coords_float_list = [p * config.resolution for p in t_c_points_list]
        for b in range(len(t_c_coords_float_list)):
            if t_c_coords_float_list[b].shape[0] > 8192:
                idx = torch.randperm(t_c_coords_float_list[b].shape[0], device=t_c_coords_float_list[b].device)[:8192]
                t_c_coords_float_list[b] = t_c_coords_float_list[b][idx]
                t_c_points_list[b] = t_c_points_list[b][idx]
        
        curr_feats_list = compute_feats(
            coords_float_list=t_p_coords_float_list,
            coords_voxel_list=t_p_coords_voxel_list,
            time_encoding=0.0,
            prob=1.0
        )
        
        all_input_coords = []
        all_input_feats = []
        all_input_coords.extend(t_p_coords_voxel_list)
        all_input_feats.extend(curr_feats_list)

        if frame_idx > 0:
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
                bound=0.5
            )
            
            hist_norm_list, hist_float_list, hist_voxel_list, temp_feats_list = voxelization(transformed_points_list, config.resolution, transformed_probs_list) 
            hist_feats_list = compute_feats(
                coords_float_list=hist_float_list,
                coords_voxel_list=hist_voxel_list,
                time_encoding=1.0,
                prob=temp_feats_list
            )
            
            all_input_coords[0] = torch.cat([all_input_coords[0], hist_voxel_list[0]], dim=0)
            all_input_feats[0] = torch.cat([all_input_feats[0], hist_feats_list[0]], dim=0)

        # 构建 Tensor
        batched_input_coords, batched_input_feats = ME.utils.sparse_collate(all_input_coords, all_input_feats, device=device)
        batched_gt_coords = ME.utils.batched_coordinates(t_c_coords_voxel_list).to(device)

        sin_fused = ME.SparseTensor(features=batched_input_feats, coordinates=batched_input_coords, device=device)
        
        cm = sin_fused.coordinate_manager
        in_target_key, _ = cm.insert_and_map(coordinates=batched_gt_coords, string_id="target")

        # 前向传播
        with torch.no_grad():
            out_cls, out_targets, sout, occ_probs = self.net(sin_fused, in_target_key)

        # 后处理与 Loss 计算 (保持不变)
        sout_coords_list, _ = sout.decomposed_coordinates_and_features
        sout_points_norm_list, sout_coords_floats_list = devoxelization(sout_coords_list, res=config.resolution)

        # 切分 occ_prob 并构建特征
        num_points_list = [pts.shape[0] for pts in sout_points_norm_list]
        occ_probs_list = torch.split(occ_probs.detach(), num_points_list)
        self.prev_frame_data = [(coords, probs.unsqueeze(1)) for coords, probs in zip(sout_points_norm_list, occ_probs_list)]

        # Chamfer Distance Loss
        batch_chamfer_loss = 0
        for gt_tensor, pred_tensor in zip(t_c_coords_float_list, sout_coords_floats_list):
            batch_chamfer_loss += self.crit1(pred_tensor, gt_tensor)
        points_reg_loss = batch_chamfer_loss / len(t_c_coords_float_list)

        # Voxel BCE Loss
        batch_bce_loss = 0
        layer_losses = []
        for out_cl, out_target in zip(out_cls, out_targets):
            curr_layer_loss = self.crit2(out_cl.F.squeeze(), out_target.type(out_cl.F.dtype).to(device))
            layer_losses.append(curr_layer_loss.item())
            batch_bce_loss += curr_layer_loss
        voxel_cls_loss = batch_bce_loss / len(out_cls)

        # TV Loss
        batch_tv_loss = 0
        layer_tv_losses = []
        for out_cl in out_cls:
            curr_tv_loss = self.crit3(out_cl)
            layer_tv_losses.append(curr_tv_loss.item())
            batch_tv_loss += curr_tv_loss
        tv_loss = batch_tv_loss / len(out_cls)

        # 总 Loss
        total_loss = self.voxel_coef * voxel_cls_loss + self.chamfer_coef * points_reg_loss + self.tv_coef * tv_loss
        print(f"points_reg_loss: {points_reg_loss}")
        print(f"voxel_cls_loss: {voxel_cls_loss}")
        print(f"tv_loss: {tv_loss} (layer: {layer_tv_losses})")
        print(f"layer_losses: {layer_losses}")
        print(f"total_loss: {total_loss}\n")

        # point cloud
        sin_curr_pc = t_p_points_norm_list[0].cpu().numpy() if t_p_points_norm_list[0].size(0) > 0 else np.empty((0, 3))
        if frame_idx > 0:
            sin_hist_pc = hist_norm_list[0].cpu().numpy()
        else:
            sin_hist_pc = np.empty((0, 3))
        gt_pc = t_c_points_list[0].cpu().numpy()
        sout_pc = sout_points_norm_list[0].cpu().numpy()
        # open3d point cloud
        sin_curr_pcd = PointCloud(sin_curr_pc, color=[1, 0, 0], translate_offset=[-0.5, -0.5, 0], rotate_matrix=self.M) if sin_curr_pc.size > 0 else o3d.geometry.PointCloud()
        sin_hist_pcd = PointCloud(sin_hist_pc, color=[0, 0, 1], translate_offset=[0.5, -0.5, 0], rotate_matrix=self.M) if sin_hist_pc.size > 0 else o3d.geometry.PointCloud()
        gt_pcd = PointCloud(gt_pc, color=[1, 1, 0], translate_offset=[-0.5, 0.5, 0], rotate_matrix=self.M) if gt_pc.size > 0 else o3d.geometry.PointCloud()
        sout_pcd = PointCloud(sout_pc, color=[0, 1, 0], translate_offset=[0.5, 0.5, 0], rotate_matrix=self.M) if sout_pc.size > 0 else o3d.geometry.PointCloud()
        # update render obj
        self.sin_curr_pcd.points = o3d.utility.Vector3dVector(np.asarray(sin_curr_pcd.points))
        self.sin_curr_pcd.colors = o3d.utility.Vector3dVector(np.asarray(sin_curr_pcd.colors))
        self.sin_curr_pcd.normals = o3d.utility.Vector3dVector(np.asarray(sin_curr_pcd.normals))
        self.sin_hist_pcd.points = o3d.utility.Vector3dVector(np.asarray(sin_hist_pcd.points))
        self.sin_hist_pcd.colors = o3d.utility.Vector3dVector(np.asarray(sin_hist_pcd.colors))
        self.sin_hist_pcd.normals = o3d.utility.Vector3dVector(np.asarray(sin_hist_pcd.normals))
        self.gt_pcd.points = o3d.utility.Vector3dVector(np.asarray(gt_pcd.points))
        self.gt_pcd.colors = o3d.utility.Vector3dVector(np.asarray(gt_pcd.colors))
        self.gt_pcd.normals = o3d.utility.Vector3dVector(np.asarray(gt_pcd.normals))
        self.sout_pcd.points = o3d.utility.Vector3dVector(np.asarray(sout_pcd.points))
        self.sout_pcd.colors = o3d.utility.Vector3dVector(np.asarray(sout_pcd.colors))
        self.sout_pcd.normals = o3d.utility.Vector3dVector(np.asarray(sout_pcd.normals))
        self.vis.update_geometry(self.sin_curr_pcd)
        self.vis.update_geometry(self.sin_hist_pcd)
        self.vis.update_geometry(self.gt_pcd)
        self.vis.update_geometry(self.sout_pcd)

        # 更新固定 1x1x1 静态包围盒线框 (归一化坐标 [0,1] 范围)
        def _update_static_bbox(lineset, translate_offset, rotate_matrix):
            verts = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ])
            if translate_offset is not None:
                verts = verts + np.array(translate_offset)
            if rotate_matrix is not None:
                verts = verts @ rotate_matrix.T
            lineset.points = o3d.utility.Vector3dVector(verts)

        # 更新动态点云范围包围盒线框 (体素对齐)
        res = config.resolution
        def _update_dynamic_bbox(lineset, points, translate_offset, rotate_matrix):
            if points.size == 0:
                lineset.points = o3d.utility.Vector3dVector(np.zeros((8, 3)))
                return
            min_bound = np.floor(points.min(axis=0) * res) / res
            max_bound = np.ceil(points.max(axis=0) * res) / res
            max_bound = np.maximum(max_bound, min_bound + 1.0 / res)
            verts = np.array([
                [min_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]],
            ])
            if translate_offset is not None:
                verts = verts + np.array(translate_offset)
            if rotate_matrix is not None:
                verts = verts @ rotate_matrix.T
            lineset.points = o3d.utility.Vector3dVector(verts)

        _update_static_bbox(self.sin_curr_bbox, [-0.5, -0.5, 0], self.M)
        _update_static_bbox(self.sin_hist_bbox, [0.5, -0.5, 0], self.M)
        _update_static_bbox(self.gt_bbox, [-0.5, 0.5, 0], self.M)
        _update_static_bbox(self.sout_bbox, [0.5, 0.5, 0], self.M)
        _update_dynamic_bbox(self.sin_curr_bbox_dyn, sin_curr_pc, [-0.5, -0.5, 0], self.M)
        _update_dynamic_bbox(self.sin_hist_bbox_dyn, sin_hist_pc, [0.5, -0.5, 0], self.M)
        _update_dynamic_bbox(self.gt_bbox_dyn, gt_pc, [-0.5, 0.5, 0], self.M)
        _update_dynamic_bbox(self.sout_bbox_dyn, sout_pc, [0.5, 0.5, 0], self.M)

        self.vis.update_geometry(self.sin_curr_bbox)
        self.vis.update_geometry(self.sin_hist_bbox)
        self.vis.update_geometry(self.gt_bbox)
        self.vis.update_geometry(self.sout_bbox)
        self.vis.update_geometry(self.sin_curr_bbox_dyn)
        self.vis.update_geometry(self.sin_hist_bbox_dyn)
        self.vis.update_geometry(self.gt_bbox_dyn)
        self.vis.update_geometry(self.sout_bbox_dyn)


###############################################################################
# End of utility classes
###############################################################################


###############################################################################
# Network class
###############################################################################

class MinkowskiGlobalContextBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.global_pool = ME.MinkowskiGlobalPooling()
        mid_channels = in_channels // reduction_ratio
        self.fc1 = ME.MinkowskiConvolution(in_channels, mid_channels, kernel_size=1, bias=True, dimension=3)
        self.relu = ME.MinkowskiLeakyReLU()
        self.fc2 = ME.MinkowskiConvolution(mid_channels, in_channels, kernel_size=1, bias=True, dimension=3)
        self.sigmoid = ME.MinkowskiSigmoid()

    def forward(self, x: ME.SparseTensor):
        # x: 输入稀疏张量 [N, C]
        x_global = self.global_pool(x) # [B, C]
        x_global = self.fc1(x_global)
        x_global = self.relu(x_global)
        x_global = self.fc2(x_global)
        x_global = self.sigmoid(x_global) # [B, C]
        x_broadcasted = ME.MinkowskiBroadcast()(x_global, x)
        return x * x_broadcasted


class LidarCompletionNet(nn.Module):
    def __init__(self, resolution, activeF=0.5, 
                 encoder_channels = [16, 32, 64, 128, 256, 512], 
                 decoder_channels = [16, 32, 64, 128, 256, 512]):
        nn.Module.__init__(self)

        # Input sparse tensor must have tensor stride 64
        self.resolution = resolution
        self.activeF = activeF
        # Channels list
        assert len(encoder_channels) == 6, "Encoder Channels list length must equal to 6 !!!"
        assert len(decoder_channels) == 6, "Decoder Channels list length must equal to 6 !!!"
        enc_ch = encoder_channels
        dec_ch = decoder_channels

        # Input features capture layer: 5 channels (Offset 3 + Time 1 + Prob 1)
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(5, enc_ch[0], kernel_size=3, stride=1, dimension=3), # Changed from 4 to 5
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        # Encoder
        # EN B1 64->32
        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
        )

        # EN B2 32->16
        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
        )

        # EN B3 16->8
        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
        )

        # EN B4 8->4
        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
        )

        # EN B4 4->2
        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
        )

        # 全局上下文模块
        self.global_context = MinkowskiGlobalContextBlock(in_channels=enc_ch[5])

        # Decoder
        # DE B1 2-4
        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[5],
                dec_ch[4],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
        )
        
        self.dec_s16_cls = ME.MinkowskiConvolution(
            dec_ch[4], 1, kernel_size=1, bias=True, dimension=3
        )
        
        # DE B2 4->8
        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[4],
                dec_ch[3],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(
            dec_ch[3], 1, kernel_size=1, bias=True, dimension=3
        )

        # DE B3 8->16
        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(
            dec_ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        # DE B4 16->32
        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[2],
                dec_ch[1],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(
            dec_ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        # DE B5 32->64
        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(
            dec_ch[0], 1, kernel_size=1, bias=True, dimension=3
        )

        # Pruning layer
        self.pruning = ME.MinkowskiPruning()

    # GT target_key - Decoder inverse compute
    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target

    # Forward Method
    def forward(self, sin_fused, target_key=None):
        out_cls, targets = [], []

        if self.training:
            assert target_key is not None, "Target Key is required for training to generate labels."

        # Single encoder path using fused input
        enc_s1 = self.enc_block_s1(sin_fused)
        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        
        # 应用全局上下文
        out_s32 = self.global_context(enc_s32)

        # Decoder
        # =========================================
        # Block s32->s16
        # =========================================
        dec_s16 = self.dec_block_s32s16(out_s32)

        # Add encoder features
        dec_s16 = dec_s16 + enc_s16
        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > self.activeF).squeeze()
        out_cls.append(dec_s16_cls)

        if target_key:
            target = self.get_target(dec_s16, target_key)
            targets.append(target)

        if self.training:
            keep_s16 += target

        # Remove voxels s16
        dec_s16 = self.pruning(dec_s16, keep_s16)
        
        # =========================================
        # Block s16->s8
        # =========================================
        dec_s8 = self.dec_block_s16s8(dec_s16)

        # Add encoder features
        dec_s8 = dec_s8 + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)
        keep_s8 = (dec_s8_cls.F > self.activeF).squeeze()
        out_cls.append(dec_s8_cls)

        if target_key:
            target = self.get_target(dec_s8, target_key)
            targets.append(target)

        if self.training:
            keep_s8 += target

        # Remove voxels s8
        dec_s8 = self.pruning(dec_s8, keep_s8)

        # =========================================
        # Block s8->s4
        # =========================================
        dec_s4 = self.dec_block_s8s4(dec_s8)

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)
        keep_s4 = (dec_s4_cls.F > self.activeF).squeeze()
        out_cls.append(dec_s4_cls)

        if target_key:
            target = self.get_target(dec_s4, target_key)
            targets.append(target)

        if self.training:
            keep_s4 += target

        # Remove voxels s4
        dec_s4 = self.pruning(dec_s4, keep_s4)

        # =========================================
        # Block s4->s2
        # =========================================
        dec_s2 = self.dec_block_s4s2(dec_s4)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)
        keep_s2 = (dec_s2_cls.F > self.activeF).squeeze()
        out_cls.append(dec_s2_cls)

        if target_key:
            target = self.get_target(dec_s2, target_key)
            targets.append(target)

        if self.training:
            keep_s2 += target

        # Remove voxels s2
        dec_s2 = self.pruning(dec_s2, keep_s2)

        # =========================================
        # Block s2->s1
        # =========================================
        dec_s1 = self.dec_block_s2s1(dec_s2)

        # Add encoder features
        dec_s1 = dec_s1 + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)
        keep_s1 = (dec_s1_cls.F > self.activeF).squeeze()
        out_cls.append(dec_s1_cls)

        if target_key:
            target = self.get_target(dec_s1, target_key)
            targets.append(target)

        # if self.training:
        #     keep_s1 += target

        # Remove voxels s1
        dec_s1 = self.pruning(dec_s1, keep_s1)
        
        # Occupancy Problities
        occ_probs = torch.sigmoid(dec_s1_cls.F.view(-1)[keep_s1])

        return out_cls, targets, dec_s1, occ_probs


###############################################################################
# End of network class
###############################################################################


###############################################################################
# Train function
###############################################################################

def train(net, dataloader, optimizer, scheduler, start_iter, start_step, device, config):
    # 初始化 SummaryWriter
    timestamp = strftime("%Y%m%d_%H%M%S", localtime())
    run_log_dir = os.path.join(config.log_dir, timestamp)
    os.makedirs(run_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_log_dir)
    print(f"📝 TensorBoard logs saved to: {run_log_dir}")
    
    crit1 = ChamferDistanceLoss(config.chamfer_p_coef, config.chamfer_r_coef).to(device)
    crit2 = nn.BCEWithLogitsLoss().to(device)
    crit3 = MinkowskiTVLoss().to(device)

    net.train()
    train_iter = iter(dataloader)

    voxel_coef = config.voxel_coef
    chamfer_coef = config.chamfer_coef
    tv_coef = config.tv_coef
    
    train_steps = start_step
    data_time = 0
    total_time = 0

    prev_frame_data = [None] * config.batch_size
    beg_iter = start_iter
    end_iter = start_iter + config.max_iter
    for iter_idx in range(beg_iter, end_iter):
        start_time = time()
        data_dict = next(train_iter)
        data_time = time() - start_time
        writer.add_scalar('Time/Iter_Data_Time', data_time, iter_idx)

        step_total_losses = []
        step_cls_losses = []
        step_reg_losses = []
        step_tv_losses = []

        optimizer.zero_grad()

        num_frames = data_dict['num_frames']
        for t in range(num_frames):
            # 数据加载与转换
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

            # 体素化当前帧
            _, t_p_coords_float_list, t_p_coords_voxel_list, _ = voxelization(t_p_points_list, config.resolution)
            _, _, t_c_coords_voxel_list, _ = voxelization(t_c_points_list, config.resolution)
            t_c_coords_float_list = [p * config.resolution for p in t_c_points_list]
            for b in range(len(t_c_coords_float_list)):
                if t_c_coords_float_list[b].shape[0] > 8192:
                    idx = torch.randperm(t_c_coords_float_list[b].shape[0], device=t_c_coords_float_list[b].device)[:8192]
                    t_c_coords_float_list[b] = t_c_coords_float_list[b][idx]
            # 处理当前帧特征
            curr_feats_list = compute_feats(
                coords_float_list=t_p_coords_float_list,
                coords_voxel_list=t_p_coords_voxel_list,
                time_encoding=0.0,
                prob=1.0
            )
            
            all_input_coords = []
            all_input_feats = []
            all_input_coords.extend(t_p_coords_voxel_list)
            all_input_feats.extend(curr_feats_list)

            # 处理历史数据
            if t > 0:
                prev_points_list = [data[0] for data in prev_frame_data if data is not None]
                prev_probs_list = [data[1] for data in prev_frame_data if data is not None]

                transformed_points_list, transformed_probs_list = points_transform_and_normclip(
                    points_prev_list=prev_points_list,
                    pos_curr=torch.stack(t_pos_data_list),
                    quat_curr=torch.stack(t_quat_data_list),
                    pos_prev=torch.stack(_t_pos_data_list),
                    quat_prev=torch.stack(_t_quat_data_list),
                    feats_prev_list=prev_probs_list,
                    scale=3.2,
                    bound=0.5
                )
                
                _, hist_float_list, hist_voxel_list, temp_feats_list = voxelization(transformed_points_list, config.resolution, transformed_probs_list) 
                hist_feats_list = compute_feats(
                    coords_float_list=hist_float_list,
                    coords_voxel_list=hist_voxel_list,
                    time_encoding=1.0,
                    prob=temp_feats_list
                )
                
                for batch_idx in range(config.batch_size):
                    all_input_coords[batch_idx] = torch.cat([all_input_coords[batch_idx], hist_voxel_list[batch_idx]], dim=0)
                    all_input_feats[batch_idx] = torch.cat([all_input_feats[batch_idx], hist_feats_list[batch_idx]], dim=0)

            # 构建稀疏张量
            batched_input_coords, batched_input_feats = ME.utils.sparse_collate(all_input_coords, all_input_feats, device=device)
            batched_gt_coords = ME.utils.batched_coordinates(t_c_coords_voxel_list).to(device)

            sin_fused = ME.SparseTensor(features=batched_input_feats, coordinates=batched_input_coords, device=device)
            
            cm = sin_fused.coordinate_manager
            in_target_key, _ = cm.insert_and_map(coordinates=batched_gt_coords, string_id="target")

            # 前向传播
            out_cls, out_targets, sout, occ_probs = net(sin_fused, in_target_key)

            # 后处理与 Loss 计算 (保持不变)
            sout_coords_list, _ = sout.decomposed_coordinates_and_features
            sout_points_norm_list, sout_coords_floats_list = devoxelization(sout_coords_list, res=config.resolution)

            # 切分 occ_prob 并构建特征
            num_points_list = [pts.shape[0] for pts in sout_points_norm_list]
            occ_probs_list = torch.split(occ_probs.detach(), num_points_list)
            prev_frame_data = [(coords, probs.unsqueeze(1)) for coords, probs in zip(sout_points_norm_list, occ_probs_list)]

            # Chamfer Distance Loss
            batch_chamfer_loss = 0
            for gt_tensor, pred_tensor in zip(t_c_coords_float_list, sout_coords_floats_list):
                batch_chamfer_loss += crit1(pred_tensor, gt_tensor)
            points_reg_loss = batch_chamfer_loss / len(t_c_coords_float_list)
            step_reg_losses.append(points_reg_loss.item())

            # Voxel BCE Loss
            batch_bce_loss = 0
            layer_losses = []
            for out_cl, out_target in zip(out_cls, out_targets):
                curr_layer_loss = crit2(out_cl.F.squeeze(), out_target.type(out_cl.F.dtype).to(device))
                layer_losses.append(curr_layer_loss.item())
                batch_bce_loss += curr_layer_loss
            voxel_cls_loss = batch_bce_loss / len(out_cls)
            step_cls_losses.append(voxel_cls_loss.item())

            # TV Loss
            batch_tv_loss = 0
            layer_tv_losses = []
            for out_cl in out_cls:
                curr_tv_loss = crit3(out_cl)
                layer_tv_losses.append(curr_tv_loss.item())
                batch_tv_loss += curr_tv_loss
            tv_loss = batch_tv_loss / len(out_cls)
            step_tv_losses.append(tv_loss.item())

            # 总 Loss
            total_loss = voxel_coef * voxel_cls_loss + chamfer_coef * points_reg_loss + tv_coef * tv_loss
            (total_loss / num_frames).backward()

            # 记录 Loss
            train_steps += 1
            step_total_losses.append(total_loss.item())
            writer.add_scalar('Loss/Total_Step_Loss', total_loss.item(), train_steps)
            writer.add_scalar('Loss/Points_Reg_Loss', points_reg_loss.item(), train_steps)
            writer.add_scalar('Loss/Voxel_Cls_Loss', voxel_cls_loss.item(), train_steps)
            writer.add_scalar('Loss/TV_Loss', tv_loss.item(), train_steps)
            for layer_idx, loss_val in enumerate(layer_losses):
                writer.add_scalar(f'Loss/Layer{layer_idx+1}_Cls_Loss', loss_val, train_steps)
            for layer_idx, tv_val in enumerate(layer_tv_losses):
                writer.add_scalar(f'Loss/Layer{layer_idx+1}_TV_Loss', tv_val, train_steps)

        # 多帧 loss 累积后统一迭代一次
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config.max_norm)
        optimizer.step()

        # 迭代结束统计
        total_time = time() - start_time
        iter_loss = sum(step_total_losses) / len(step_total_losses)
        writer.add_scalar('Time/Iter_Total_Time', total_time, iter_idx)
        writer.add_scalar('Loss/Iter_Loss', iter_loss, iter_idx)
        writer.add_scalar('Params/LR', scheduler.get_last_lr()[0], iter_idx)

        if iter_idx % config.stat_freq_iter == 0:
            steps_total_losses_str = ", ".join([f"{v:.3e}" for v in step_total_losses])
            step_cls_losses_str = ", ".join([f"{v:.3e}" for v in step_cls_losses])
            step_reg_losses_str = ", ".join([f"{v:.3e}" for v in step_reg_losses])
            step_tv_losses_str = ", ".join([f"{v:.3e}" for v in step_tv_losses])
            logging.info(
                f"Iter: {iter_idx}, Iter Ave Loss: {iter_loss:.3e}, Step: {train_steps}, Data Loading Time: {data_time:.3e}, Total Time: {total_time:.3e}\n"
                f"Step Total Losses: [{steps_total_losses_str}]\n"
                f"Step Cls Losses: [{step_cls_losses_str}]\n"
                f"Step Reg Losses: [{step_reg_losses_str}]\n"
                f"Step TV Losses: [{step_tv_losses_str}]\n"
            )

        if iter_idx % config.save_freq_iter == 0:
            model_save_path = os.path.join(config.save_dir, config.model_name)
            os.makedirs(model_save_path, exist_ok=True)
            model_save_file = os.path.join(model_save_path, f"model_{iter_idx}.pth")
            torch.save({
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "curr_iter": iter_idx,
                "curr_step": train_steps,
            }, model_save_file)
            logging.info(f"LR: {scheduler.get_last_lr()}")
            net.train()

        scheduler.step()


###############################################################################
# End of train function
###############################################################################


###############################################################################
# Visualize function
###############################################################################

# 可视化展现函数
def visualize(net, dataloader, device, config):
    vis_tool = Visualizer(net, dataloader, device, config)
    vis_tool.visualize()


###############################################################################
# End of visualize function
###############################################################################


###############################################################################
# Main Thread
###############################################################################

# 主线程
if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = LidarCompletionNet(
        resolution=config.resolution, 
        activeF=PURNING_THRESHOLD, 
        encoder_channels=ENC_CHANNELS, 
        decoder_channels=DEC_CHANNELS
    ).to(device)
    logging.info(net)
    
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    print("网络参数统计:")
    print(f"  总参数量: {total_params:,} ({total_params / 1e6:.3f}M)")
    print(f"  可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.3f}M)")
    print(f"  不可训练参数量: {non_trainable_params:,} ({non_trainable_params / 1e6:.3f}M)")

    if not config.eval:
        dataloader = make_data_loader(
            phase="train",
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            repeat=True,
            config=config,
            augment_data=True,
            transforms=[
                lambda x: StridedSamplingTransform(x, stride_list=[1]*11 + [2]*7 + [3]*4 + [4]*2 + [5]*1),
                # lambda x: VoxelFilterTransform(x, voxel_size=0.015),
                lambda x: RandomRotationTransform(x, max_angle_deg=1.0),
                lambda x: RandomCylinderCutoutTransform(x, max_radius=0.1, max_cylinders=3),
                lambda x: RandomNoiseTransform(x, noise_std=0.004),
            ]
        )
        optimizer = optim.AdamW(
            net.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.max_iter, 
            eta_min=1e-5
        )
        
        start_iter = 0
        start_step = 0
        if config.resume:
            checkpoint_dir = os.path.join(config.save_dir, config.model_name)
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pth"))
            
            if not checkpoint_files:
                logging.warning(f"No checkpoint found in {checkpoint_dir}. Starting from scratch.")
            else:
                checkpoint_files.sort(key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))
                latest_checkpoint = checkpoint_files[-1]
                logging.info(f"Resuming from latest checkpoint: {latest_checkpoint}")
                
                try:
                    checkpoint = torch.load(latest_checkpoint, map_location=device)
                    net.load_state_dict(checkpoint["state_dict"])
                    if config.load_optimizer:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                        if "scheduler" in checkpoint:
                            scheduler.load_state_dict(checkpoint["scheduler"])

                    filename = os.path.basename(latest_checkpoint)
                    start_iter = checkpoint["curr_iter"] + 1
                    start_step = checkpoint["curr_step"] + 1
                    logging.info(f"Loaded model: {filename}")
                    logging.info(f"Resuming from iteration {start_iter} (based on saved state)")
                except Exception as e:
                    logging.error(f"Failed to load checkpoint: {e}")
                    logging.info("Starting training from scratch.")
        
        train(net, dataloader, optimizer, scheduler, start_iter, start_step, device, config)
        
    else:
        checkpoint_dir_path = os.path.join(config.save_dir, config.model_name)
        if not os.path.exists(checkpoint_dir_path):
            raise FileNotFoundError(f"Model directory not found at: {checkpoint_dir_path}")

        checkpoint_files = glob.glob(os.path.join(checkpoint_dir_path, "model_*.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir_path}.")
        
        checkpoint_files.sort(key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))
        latest_checkpoint_path = checkpoint_files[-1]
        
        try:
            filename = os.path.basename(latest_checkpoint_path)
            latest_iter = int(filename.split('_')[-1].split('.')[0]) 
        except Exception as e:
            logging.warning(f"无法从文件名 {filename} 解析 Iteration，设为 Unknown。Error: {e}")
            latest_iter = -1

        logging.info(f"Found latest checkpoint: {os.path.basename(latest_checkpoint_path)} (Iter: {latest_iter})")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        logging.info("Load weights success.")

        # export model weights
        model_save_path = os.path.join(config.save_dir, config.model_name)
        model_export_path = os.path.join(model_save_path, "export")
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(model_export_path, exist_ok=True)
        model_export_file = os.path.join(model_export_path, f"model.pth")
        torch.save({"state_dict": net.state_dict()}, model_export_file)
        logging.info("Export model weights success.")

        dataloader = make_data_loader(
            phase="train",
            batch_size=1,
            shuffle=True,
            num_workers=0,
            repeat=True,
            config=config,
            augment_data=False,
            transforms=[
                lambda x: StridedSamplingTransform(x, stride_list=[1]*11 + [2]*7 + [3]*4 + [4]*2 + [5]*1),
                # lambda x: VoxelFilterTransform(x, voxel_size=0.015),
                lambda x: RandomRotationTransform(x, max_angle_deg=1.0),
                lambda x: RandomCylinderCutoutTransform(x, max_radius=0.1, max_cylinders=3),
                lambda x: RandomNoiseTransform(x, noise_std=0.004),
            ]
        )

        visualize(net, dataloader, device, config)


###############################################################################
# End of Main Thread
###############################################################################

