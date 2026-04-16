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
from time import time
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

# 可视化时，点云的旋转矩阵，方便正面查看物体状态
M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)

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
parser.add_argument("--resolution",         type=int,       default=128)
parser.add_argument("--force_norm",         type=bool,      default=False)
parser.add_argument("--max_iter",           type=int,       default=5001)
parser.add_argument("--stat_freq_iter",     type=int,       default=50)
parser.add_argument("--lr_step",            type=int,       default=100)
parser.add_argument("--save_freq_iter",     type=int,       default=100)
parser.add_argument("--batch_size",         type=int,       default=4)
parser.add_argument("--lr",                 type=float,     default=1e-3)
parser.add_argument("--alpha",              type=float,     default=0.5)
parser.add_argument("--momentum",           type=float,     default=0.9)
parser.add_argument("--weight_decay",       type=float,     default=1e-4)
parser.add_argument("--num_workers",        type=int,       default=4)
parser.add_argument("--log_dir",            type=str,       default="./output/logs")
parser.add_argument("--save_dir",           type=str,       default="./output/chechpoint")
# lidar_completion, lidar_completion_old, lidar_completion_test, lidar_completion_4layer_v0
parser.add_argument("--model_name",         type=str,       default="lidar_completion_4layer_v0")
parser.add_argument("--load_optimizer",     type=str,       default="true")
parser.add_argument("--max_visualization",  type=int,       default=4)
parser.add_argument("--eval",               action="store_true")

PURNING_THRESHOLD = 0.5
ENC_CHANNELS = [32, 64, 128, 256, 512]
DEC_CHANNELS = [32, 64, 128, 256, 512]

###############################################################################
# End of global configs
###############################################################################


###############################################################################
# Utility functions
###############################################################################

def voxelization(points_list, res: int=1):
    """
    辅助函数: 进行体素尺寸放大，并使用ME量化工具进行去重
    输入:
        points_list: List[torch.Tensor], 包含 Batch 中每个样本的点云坐标 (N_i, 3)。
                     数据类型应为 torch.float32。
        res: float, 体素分辨率
        device: str, 指定量化计算的设备 ('cpu' 或 'cuda')。
                注意：如果 points_list 在 GPU 上，建议设为 'cuda' 以加速。
    返回:
        unique_points_list: List[torch.Tensor], 去重后的原始归一化坐标
        coords_float_list: List[torch.Tensor], 去重后的浮点放大坐标 (用于计算 Offset 特征)
        coords_int_list: List[torch.Tensor], 去重后的整型体素坐标 (用于构建 SparseTensor)
    """
    unique_points_list = []
    coords_float_list = []
    coords_int_list = []
    
    for points in points_list:            
        coords_float = points * res
        discrete_coords, indices = ME.utils.sparse_quantize(
            coordinates=coords_float, 
            return_index=True, 
            quantization_size=1,
            device=device
        )
        
        unique_points_list.append(points[indices])
        coords_float_list.append(coords_float[indices])
        coords_int_list.append(discrete_coords.int())
        
    return unique_points_list, coords_float_list, coords_int_list


def devoxelization(coords_list, offsets_list, res: int=1):
    """
    体素坐标反量化辅助函数：将体素坐标与网络预测的偏移量融合，还原为连续3D点云。
    
    这是一个 "Devoxelization" (去体素化) 过程。输入是整数体素坐标 (coords) 和
    网络预测的亚体素偏移量 (offsets)，输出是精确的 3D 空间坐标。

    Args:
        coords_list (List[torch.Tensor]): 稀疏张量的整型坐标列表。
                                          每个元素形状为 (N_i, 3)，数据类型为 torch.long/int。
                                          通常来自 SparseTensor.C 的空间索引部分。
        offsets_list (List[torch.Tensor]): 网络预测的亚体素偏移量列表。
                                           每个元素形状为 (N_i, 3)，数据类型为 torch.float。
                                           范围通常在 [0, 1] (经过 Sigmoid 后)。
        res (int, optional): 原始体素化的分辨率，用于将坐标还原到 [0, 1] 范围。 
                             默认为 1。

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: 
            - points_norm_list: 归一化到 [0, 1] 范围内的 3D 坐标 (用于计算 Loss)。
                                列表中的每个元素是形状为 (N_i, 3) 的 Tensor。
            - points_list: 未归一化的 3D 坐标 (即体素坐标 + 偏移量)。
                           列表中的每个元素是形状为 (N_i, 3) 的 Tensor。
    """
    points_list = []
    points_norm_list = []
    
    for coords, offsets in zip(coords_list, offsets_list):
        points = coords.float() + offsets 
        points_norm = points / res
        
        points_list.append(points)
        points_norm_list.append(points_norm)
        
    return points_norm_list, points_list


def compute_feats(coords_float_list, coords_voxel_list, time_encoding):
    """
    辅助函数: 计算稀疏张量的特征 (适配 PyTorch Tensor)
    输入:
        coords_float_list: List[torch.Tensor], 浮点放大坐标列表 [(N1, 3), (N2, 3), ...]
        coords_voxel_list: List[torch.Tensor], 整型体素坐标列表 [(N1, 3), (N2, 3), ...]
        time_encoding: float, 时间步编码
    返回:
        feats_list: List[torch.Tensor], 特征矩阵列表 [(N1, 4), (N2, 4), ...]
    """
    feats_list = []
    
    for coords_float, coords_voxel in zip(coords_float_list, coords_voxel_list):
        feats_offset = coords_float - coords_voxel.float()
        feats_temporal = torch.full(
            (coords_float.shape[0], 1), 
            time_encoding, 
            dtype=coords_float.dtype, 
            device=coords_float.device
        )
        
        feats = torch.cat([feats_offset, feats_temporal], dim=1)
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


def make_data_loader(phase, augment_data, batch_size, shuffle, num_workers, repeat, config):
    """
    辅助函数，用于获取 data loader
    """
    data_set = ConstructTerrainDataset(phase=phase, config=config)

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
    
    rot_matrix_curr = matrix_from_quat(quat_curr)                                               # (N, 4) -> (N, 3, 3)

    points_world_prev = transform_points(points_prev * scale, pos_prev, quat_prev)              # (N, P, 3)
    points_world_prev_centered = points_world_prev - pos_curr .unsqueeze(1)                     # (N, P, 3)
    
    points_prev_view = torch.matmul(points_world_prev_centered, rot_matrix_curr)                # (N, P, 3)
    points_prev_view_norm = points_prev_view / scale                                            # (N, P, 3)

    mask_inside = (points_prev_view_norm.abs() < bound).all(dim=2)
    
    has_points = mask_inside.any(dim=1)
    if not has_points.all():
        bad_indices = torch.where(~has_points)[0]
        raise RuntimeError(f"数据异常：以下样本的点云在归一化裁剪后全部丢失 (Mask全为False): {bad_indices.tolist()}。请检查输入点云范围或增大 bound 值。")
    
    inside_lengths = mask_inside.sum(dim=1)
    all_inside_points = points_prev_view_norm[mask_inside]
    points_prev_inside_list = torch.split(all_inside_points, inside_lengths.tolist())

    return points_prev_inside_list


###############################################################################
# End of utility functions
###############################################################################


###############################################################################
# Utility classes
###############################################################################

# 自定义数据集类
class ConstructTerrainDataset(torch.utils.data.Dataset):
    def __init__(self, phase="train", type=None, transform=None, config=None):
        """
        构造函数：按照给定的训练模式，加载全部时序样本的路径列表，并排序
        """
        self.phase = phase                      # "train", "test"
        self.type = type                        # "walk", ...
        self.transform = transform              # 预处理函数
        self.resolution = config.resolution     # 体素分辨率
        self.samples = []                       # 存储配对好的样本信息
        self.cache = {}                         # 样本加载缓存
        self.last_cache_percent = 0             # 样本缓存比例
        self.force_norm = config.force_norm     # 强制归一化
        
        # 修改根目录路径以适应新数据集结构
        self.root = "./DataCollection"  # 根据实际数据集路径调整
        
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
                        'partial_paths': p_fnames,          # 明确表明存储的是文件路径列表
                        'complete_paths': c_fnames,         # 明确表明存储的是文件路径列表
                        'transform_path': transform_path,   # 添加变换文件路径
                        'pos_data': pos_data,               # 位置数据
                        'quat_data': quat_data,             # 姿态数据
                    }
                )
        
        assert len(self.samples) > 0, "No paired samples loaded!"
        logging.info(f"Loaded {len(self.samples)} sequences for phase: {phase}")
        
        # 忽略 Open3D 警告
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
    def __len__(self):
        return len(self.samples)
    
    def _load_pcd_sequence(self, pcd_paths, force_norm=False):
        """
        辅助函数：加载一组 PCD 文件路径，处理并合并为一个样本列表
        """
        data_list = []
        
        for pcd_path in pcd_paths:
            # # 缓存检查
            # if pcd_path in self.cache:
            #     points = self.cache[pcd_path]
            # else:
            # 读取 PCD 文件
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            
            # 检查点云是否为空，确保数据有效性
            if len(points) == 0:
                error_msg = f"Empty point cloud detected in file: {pcd_path}. Data integrity check failed."
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            # 检查是否在 (0, 1) 范围内
            is_norm_type1 = (points.min() > 0.0) and (points.max() < 1.0)
            # 检查是否在 (-0.5, 0.5) 范围内
            is_norm_type2 = (points.min() > -0.51) and (points.max() < 0.51)
            if not is_norm_type1:
                if is_norm_type2:
                    # (-0.5, 0.5) -> (0, 1)
                    points += 0.5
                elif force_norm:
                    min_val = points.min()
                    max_val = points.max()
                    if max_val - min_val < 1e-8:  # 防止除零，处理所有点重合的情况
                        points = np.zeros_like(points)
                    else:
                        points = (points - min_val) / (max_val - min_val)
                else:
                    error_msg = (
                        f"\n--- 数据归一化检查失败 (Data Normalization Check Failed) ---\n"
                        f"File: {pcd_path}\n"
                        f"Coordinate Range: [{points.min():.3f}, {points.max():.3f}]\n"
                        f"Requirement: Must be in the range [0, 1] \n"
                        f"Action: Please normalize your mesh/point cloud data before training.\n"
                        f"--- 建议：请检查预处理脚本是否对 {pcd_path} 执行了归一化 ---"
                    )
                    logging.error(error_msg)
                    assert False, error_msg
                
                # # 存入缓存
                # self.cache[pcd_path] = points

            # 时序性存入列表
            data_list.append(points)
        
        return data_list

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        data_dict = {
            'type': sample_info['type'],
            'seq_idx': sample_info['seq_idx'],
            'num_frames': len(sample_info['partial_paths']),
            'partial': self._load_pcd_sequence(sample_info['partial_paths'], self.force_norm),
            'complete': self._load_pcd_sequence(sample_info['complete_paths'], self.force_norm),
            'pos_data': sample_info['pos_data'],
            'quat_data': sample_info['quat_data'],
        }

        if self.transform:
            data_dict = self.transform(data_dict)
        
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
    def __init__(self):
        super(ChamferDistanceLoss, self).__init__()

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
        # 对两个方向的距离求平均，然后对 Batch 求平均
        loss = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)
        
        return loss


###############################################################################
# End of utility classes
###############################################################################


###############################################################################
# Network class
###############################################################################

class LidarCompletionNet(nn.Module):
    def __init__(self, resolution, activeF=0.5, encoder_channels = [16, 32, 64, 128, 256], decoder_channels = [16, 32, 64, 128, 256]):
        nn.Module.__init__(self)

        # Input sparse tensor must have tensor stride 64
        self.resolution = resolution
        self.activeF = activeF
        # Channels list
        assert len(encoder_channels) == 5, "Encoder Channels list length must equal to 5 !!!"
        assert len(decoder_channels) == 5, "Decoder Channels list length must equal to 5 !!!"
        enc_ch = encoder_channels
        dec_ch = decoder_channels

        # Input features capture layer
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(4, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        # Encoder
        # EN B1
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

        # EN B2
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

        # EN B3
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

        # EN B4
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

        # Fusion layers
        self.fuse_block_s16 = ME.MinkowskiUnion()
        self.fuse_block_s8 = ME.MinkowskiUnion()
        self.fuse_block_s4 = ME.MinkowskiUnion()
        self.fuse_block_s2 = ME.MinkowskiUnion()
        self.fuse_block_s1 = ME.MinkowskiUnion()

        # Decoder
        # DE B1
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

        # DE B2
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

        # DE B3
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

        # DE B4
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
        
        # Output features layer
        self.dec_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], 3, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(3),
            ME.MinkowskiSigmoid(),
        )

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
    def forward(self, sin_current, sin_history, target_key=None):
        out_cls, targets = [], []

        if self.training:
            assert target_key is not None, "Target Key is required for training to generate labels."

        # Encoder
        # Current Input
        enc_curr_s1 = self.enc_block_s1(sin_current)
        enc_curr_s2 = self.enc_block_s1s2(enc_curr_s1)
        enc_curr_s4 = self.enc_block_s2s4(enc_curr_s2)
        enc_curr_s8 = self.enc_block_s4s8(enc_curr_s4)
        enc_curr_s16 = self.enc_block_s8s16(enc_curr_s8)
        # History Input
        enc_hist_s1 = self.enc_block_s1(sin_history)
        enc_hist_s2 = self.enc_block_s1s2(enc_hist_s1)
        enc_hist_s4 = self.enc_block_s2s4(enc_hist_s2)
        enc_hist_s8 = self.enc_block_s4s8(enc_hist_s4)
        enc_hist_s16 = self.enc_block_s8s16(enc_hist_s8)

        # Botleneck fusion layer
        enc_s16 = self.fuse_block_s16(enc_curr_s16, enc_hist_s16)

        # Jump connection fusion layers
        enc_s8 = self.fuse_block_s8(enc_curr_s8, enc_hist_s8)
        enc_s4 = self.fuse_block_s4(enc_curr_s4, enc_hist_s4)
        enc_s2 = self.fuse_block_s2(enc_curr_s2, enc_hist_s2)
        enc_s1 = self.fuse_block_s1(enc_curr_s1, enc_hist_s1)

        # Decoder
        # =========================================
        # Block s16->s8
        # =========================================
        dec_s8 = self.dec_block_s16s8(enc_s16)

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

        # Remove voxels s16
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

        if self.training:
            keep_s1 += target

        # Remove voxels s1
        dec_s1 = self.pruning(dec_s1, keep_s1)
        
        # Ouput
        sout = self.dec_block_s1(dec_s1)

        # predict voxels, gt voxels, output pc(coords + feats) 
        return out_cls, targets, sout


###############################################################################
# End of network class
###############################################################################


###############################################################################
# Train function
###############################################################################

def train(net, dataloader, device, config):
    # 初始化 SummaryWriter
    os.makedirs(config.log_dir, exist_ok=True) 
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # 初始化 SGD 优化器
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    # 初始化 LR 学习率控制器
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    # 初始化损失函数
    crit1 = ChamferDistanceLoss().to(device)
    crit2 = nn.BCEWithLogitsLoss()

    # 网络切换到训练模式
    net.train()
    
    # 获取数据载入器的迭代器
    train_iter = iter(dataloader)
    
    # 记录初始学习率
    current_lr = scheduler.get_lr()[0]
    logging.info(f"LR: {current_lr}")
    writer.add_scalar('Params/LR', current_lr, 0)
    
    # 双损失复合加权系数
    alpha = config.alpha    # 分类损失权重
    beta = 1.0 - alpha      # 回归损失权重
    
    # 初始化计数参数
    train_steps = 0         # 总训练步数
    data_time = 0           # 单次迭代的数据加载耗时
    total_time = 0          # 单次迭代的总耗时
    # 训练周期循环
    for i in range(config.max_iter):
        # 获取小批量训练数据
        start_time = time()
        data_dict = next(train_iter)
        data_time = time() - start_time
        writer.add_scalar('Time/Iter_Data_Time', data_time, i)
        
        # 记录本次 iter 内各 step 的 loss 数值
        step_total_losses = []
        step_cls_losses = []
        step_reg_losses = []

        # 清除梯度
        optimizer.zero_grad()
        
        # 按照时序遍历
        num_frames = data_dict['num_frames']    
        sout_points_norm_list, sout_points_float_list = None, None
        for t in range(num_frames):
            # 获取原始点云序列 (List[np.ndarray])
            t_p_points_list_np = data_dict['partial'][t]
            t_c_points_list_np = data_dict['complete'][t]
            # 获取变换数据
            t_pos_data_list_np = data_dict['pos_data'][t]
            t_quat_data_list_np = data_dict['quat_data'][t]
            _t_pos_data_list_np = data_dict['pos_data'][(t - 1) if t != 0 else 0]
            _t_quat_data_list_np = data_dict['quat_data'][(t - 1) if t != 0 else 0]
            # 将 Numpy List 转换为 Torch List
            t_p_points_list = [
                torch.from_numpy(p).float().to(device) 
                for p in t_p_points_list_np
            ]
            t_c_points_list = [
                torch.from_numpy(p).float().to(device) 
                for p in t_c_points_list_np
            ]
            t_pos_data_list = [
                torch.from_numpy(p).float().to(device) 
                for p in t_pos_data_list_np
            ]
            t_quat_data_list = [
                torch.from_numpy(p).float().to(device) 
                for p in t_quat_data_list_np
            ]
            _t_pos_data_list = [
                torch.from_numpy(p).float().to(device) 
                for p in _t_pos_data_list_np
            ]
            _t_quat_data_list = [
                torch.from_numpy(p).float().to(device) 
                for p in _t_quat_data_list_np
            ]
            # 体素坐标量化，List[np.ndarray]
            _, t_p_coords_float_list, t_p_coords_voxel_list = voxelization(t_p_points_list, config.resolution)
            _, t_c_coords_float_list, t_c_coords_voxel_list = voxelization(t_c_points_list, config.resolution)
            # 计算输入点云坐标对应的特征，List[np.ndarray]
            t_p_feats_list = compute_feats(t_p_coords_float_list, t_p_coords_voxel_list, 0.0)
            # 当前感知点云，稀疏张量聚合
            batched_curr_coords, batched_curr_feats = ME.utils.sparse_collate(t_p_coords_voxel_list, t_p_feats_list, device=device)
            # 真值体素坐标，稀疏张量聚合
            batched_gt_coords = ME.utils.batched_coordinates(t_c_coords_voxel_list).to(device)
            # 历史重建点云，稀疏张量聚合
            with torch.no_grad():
                if t == 0:
                    batched_hist_coords = batched_curr_coords.detach()
                    batched_hist_feats = batched_curr_feats.detach()
                else:
                    hist_points_norm_list = points_transform_and_normclip(sout_points_norm_list,
                                                                        torch.stack(t_pos_data_list), 
                                                                        torch.stack(t_quat_data_list), 
                                                                        torch.stack(_t_pos_data_list), 
                                                                        torch.stack(_t_quat_data_list), 
                                                                        scale=3.2, bound=0.5)
                    _, hist_coords_float_list, hist_coords_voxel_list = voxelization(hist_points_norm_list, config.resolution)
                    hist_feats_list = compute_feats(hist_coords_float_list, hist_coords_voxel_list, 1.0)
                    batched_hist_coords, batched_hist_feats = ME.utils.sparse_collate(hist_coords_voxel_list, hist_feats_list, device=device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 共享坐标管理器
            cm = ME.CoordinateManager(D=3)
            
            # Sparse Input Tensor
            # Current Input Spaser Tensor
            sin_curr = ME.SparseTensor(
                features=batched_curr_feats,
                coordinates=batched_curr_coords,
                coordinate_manager=cm,
                device=device,
            )
            # History Input Spaser Tensor
            sin_hist = ME.SparseTensor(
                features=batched_hist_feats,
                coordinates=batched_hist_coords,
                coordinate_manager=cm,
                device=device,
            )
            # 根据真值计算 target_key
            in_target_key, _ = cm.insert_and_map(
                coordinates=batched_gt_coords,
                string_id="target",
            )
            
            # Forward
            out_cls, out_targets, sout = net(sin_curr, sin_hist, in_target_key)
            
            # reconstruct points
            sout_coords_list, sout_feats_list = sout.decomposed_coordinates_and_features
            detached_sout_coords_list = [coords.detach() for coords in sout_coords_list]
            detached_sout_feats_list = [feats.detach() for feats in sout_feats_list]
            sout_points_norm_list, sout_points_float_list = devoxelization(detached_sout_coords_list, detached_sout_feats_list, res=config.resolution)
        
            # 计算 Chamfer Distance Loss
            num_batchs, batch_chamfer_loss = len(t_c_coords_float_list), 0
            for gt_tensor, pred_tensor in zip(t_c_coords_float_list, sout_points_float_list):
                batch_chamfer_loss += crit1(pred_tensor, gt_tensor)
            points_reg_loss = batch_chamfer_loss / num_batchs
            step_reg_losses.append(points_reg_loss.item())

            # 计算体素交叉熵损失
            num_layers, batch_bce_loss = len(out_cls), 0
            layer_losses = []
            for out_cl, out_target in zip(out_cls, out_targets):
                curr_layer_loss = crit2(out_cl.F.squeeze(), out_target.type(out_cl.F.dtype).to(device))
                layer_losses.append(curr_layer_loss.item())
                batch_bce_loss += curr_layer_loss
            voxel_cls_loss = batch_bce_loss / num_layers
            step_cls_losses.append(voxel_cls_loss.item())
            
            # 计算总损失
            train_steps += 1
            total_loss = alpha * voxel_cls_loss + beta * points_reg_loss
            
            # 反向传播损失，梯度更新网络权重
            total_loss.backward(retain_graph=True)
            optimizer.step()

            # 记录损失
            step_total_losses.append(total_loss.item())
            writer.add_scalar('Loss/Total_Step_Loss', total_loss.item(), train_steps)
            writer.add_scalar('Loss/Points_Reg_Loss', points_reg_loss.item(), train_steps)
            writer.add_scalar('Loss/Voxel_Cls_Loss', voxel_cls_loss.item(), train_steps)
            for layer_idx, loss_val in enumerate(layer_losses):
                writer.add_scalar(f'Loss/Layer{layer_idx+1}_Cls_Loss', loss_val, train_steps)
        
        # 计算并记录本次 Iter 总耗时、平均损失
        total_time = time() - start_time
        iter_loss = sum(step_total_losses)/len(step_total_losses)
        writer.add_scalar('Time/Iter_Total_Time', total_time, i)
        writer.add_scalar('Loss/Iter_Loss', iter_loss, i)
        
        # 打印训练 info
        if i % config.stat_freq_iter == 0:
            steps_total_losses_str = ", ".join([f"{v:.3e}" for v in step_total_losses])
            step_cls_losses_str = ", ".join([f"{v:.3e}" for v in step_cls_losses])
            step_reg_losses_str = ", ".join([f"{v:.3e}" for v in step_reg_losses])
            logging.info(
                f"Iter: {i}, Iter Ave Loss: {iter_loss:.3e}, Step: {train_steps}, Data Loading Time: {data_time:.3e}, Total Time: {total_time:.3e}\n"
                f"Step Total Losses: [{steps_total_losses_str}]\n"
                f"Step Cls Losses: [{step_cls_losses_str}]\n"
                f"Step Reg Losses: [{step_reg_losses_str}]\n"
            )

        # 保存模型和训练断点
        if i % config.save_freq_iter == 0 and i > 0:
            model_save_path = os.path.join(config.save_dir, config.model_name)
            os.makedirs(model_save_path, exist_ok=True)
            model_save_file = os.path.join(model_save_path, f"model_{i}.pth")
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                },
                model_save_file,
            )
            # 切换训练模式
            net.train()
        
        # 学习率控制器更新
        if i % config.lr_step == 0 and i > 0:
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
            logging.info(f"LR: {current_lr}")
            writer.add_scalar('Params/LR', current_lr, i)
            # 切换训练模式
            net.train()


###############################################################################
# End of train function
###############################################################################


###############################################################################
# Visualize function
###############################################################################

# 可视化展现函数
def visualize(net, dataloader, device, config):
    """
    需要设定小批量数据 batch_size=1，以便时序性观察
    切换 shuffle 标志位，选择顺序或随机取样
    """
    # 获取数据载入器的迭代器
    train_iter = iter(dataloader)
    
    # 网络切换评估模式
    net.eval()
    
    # 初始化损失函数
    crit1 = ChamferDistanceLoss().to(device)
    crit2 = nn.BCEWithLogitsLoss()
    
    # 双损失复合加权系数
    alpha = config.alpha    # 分类损失权重
    beta = 1.0 - alpha      # 回归损失权重
    
    print("🔍 可视化布局说明:")
    print("  红: 当前 | 蓝: 历史")
    print("  黄: 真值 | 绿: 输出")

    # 顺序遍历数据加载器，获取小批量样本
    for i in range(config.max_visualization):
        # 获取样本
        data_dict = next(train_iter)
        
        # 批次维度应当为1
        assert len(data_dict['partial'][0]) == 1, "Error: Dataloader batch_size in Visualize(eval mode) should equal to 1 !!!"
        
        # 记录本次 iter 内各 step 的 loss 数值
        step_total_losses = []
        step_cls_losses = []
        step_reg_losses = []
        
        # 按照时序遍历
        num_frames = data_dict['num_frames']
        sout_points_norm_list, sout_points_float_list = None, None
        for t in range(num_frames):
            # 获取原始点云序列 (List[np.ndarray])
            t_p_points_list_np = data_dict['partial'][t]
            t_c_points_list_np = data_dict['complete'][t]
            # 获取变换数据
            t_pos_data_list_np = data_dict['pos_data'][t]
            t_quat_data_list_np = data_dict['quat_data'][t]
            _t_pos_data_list_np = data_dict['pos_data'][(t - 1) if t != 0 else 0]
            _t_quat_data_list_np = data_dict['quat_data'][(t - 1) if t != 0 else 0]
            # 将 Numpy List 转换为 Torch List
            t_p_points_list = [
                torch.from_numpy(p).float().to(device) 
                for p in t_p_points_list_np
            ]
            t_c_points_list = [
                torch.from_numpy(p).float().to(device) 
                for p in t_c_points_list_np
            ]
            t_pos_data_list = [
                torch.from_numpy(p).float().to(device) 
                for p in t_pos_data_list_np
            ]
            t_quat_data_list = [
                torch.from_numpy(p).float().to(device) 
                for p in t_quat_data_list_np
            ]
            _t_pos_data_list = [
                torch.from_numpy(p).float().to(device) 
                for p in _t_pos_data_list_np
            ]
            _t_quat_data_list = [
                torch.from_numpy(p).float().to(device) 
                for p in _t_quat_data_list_np
            ]
            # 体素坐标量化，List[np.ndarray]
            t_p_points_norm_list, t_p_coords_float_list, t_p_coords_voxel_list = voxelization(t_p_points_list, config.resolution)
            t_c_points_norm_list, t_c_coords_float_list, t_c_coords_voxel_list = voxelization(t_c_points_list, config.resolution)
            # 计算输入点云坐标对应的特征，List[np.ndarray]
            t_p_feats_list = compute_feats(t_p_coords_float_list, t_p_coords_voxel_list, 0.0)
            # 当前感知点云，稀疏张量聚合
            batched_curr_coords, batched_curr_feats = ME.utils.sparse_collate(t_p_coords_voxel_list, t_p_feats_list, device=device)
            # 真值体素坐标，稀疏张量聚合
            batched_gt_coords = ME.utils.batched_coordinates(t_c_coords_voxel_list).to(device)
            # 历史重建点云，稀疏张量聚合
            with torch.no_grad():
                if t == 0:
                    batched_hist_coords = batched_curr_coords.detach()
                    batched_hist_feats = batched_curr_feats.detach()
                    hist_points_norm_list = t_p_points_norm_list
                else:
                    hist_points_norm_list = points_transform_and_normclip(sout_points_norm_list,
                                                                        torch.stack(t_pos_data_list), 
                                                                        torch.stack(t_quat_data_list), 
                                                                        torch.stack(_t_pos_data_list), 
                                                                        torch.stack(_t_quat_data_list), 
                                                                        scale=3.2, bound=0.5)
                    _, hist_coords_float_list, hist_coords_voxel_list = voxelization(hist_points_norm_list, config.resolution)
                    hist_feats_list = compute_feats(hist_coords_float_list, hist_coords_voxel_list, 1.0)
                    batched_hist_coords, batched_hist_feats = ME.utils.sparse_collate(hist_coords_voxel_list, hist_feats_list, device=device)
            
            # 共享坐标管理器
            cm = ME.CoordinateManager(D=3)
            
            # Sparse Input Tensor
            # Current Input Spaser Tensor
            sin_curr = ME.SparseTensor(
                features=batched_curr_feats,
                coordinates=batched_curr_coords,
                coordinate_manager=cm,
                device=device,
            )
            # History Input Spaser Tensor
            sin_hist = ME.SparseTensor(
                features=batched_hist_feats,
                coordinates=batched_hist_coords,
                coordinate_manager=cm,
                device=device,
            )
            # 根据真值计算 target_key
            in_target_key, _ = cm.insert_and_map(
                coordinates=batched_gt_coords,
                string_id="target",
            )
            
            # Forward
            out_cls, out_targets, sout = net(sin_curr, sin_hist, in_target_key)
            
            # reconstruct points
            sout_coords_list, sout_feats_list = sout.decomposed_coordinates_and_features
            detached_sout_coords_list = [coords.detach() for coords in sout_coords_list]
            detached_sout_feats_list = [feats.detach() for feats in sout_feats_list]
            sout_points_norm_list, sout_points_float_list = devoxelization(detached_sout_coords_list, detached_sout_feats_list, res=config.resolution)
        
            # 计算 Chamfer Distance Loss
            num_batchs, batch_chamfer_loss = len(t_c_coords_float_list), 0
            for gt_tensor, pred_tensor in zip(t_c_coords_float_list, sout_points_float_list):
                batch_chamfer_loss += crit1(pred_tensor, gt_tensor)
            points_reg_loss = batch_chamfer_loss / num_batchs
            step_reg_losses.append(points_reg_loss.item())

            # 计算体素交叉熵损失
            num_layers, batch_bce_loss = len(out_cls), 0
            layer_losses = []
            for out_cl, out_target in zip(out_cls, out_targets):
                curr_layer_loss = crit2(out_cl.F.squeeze(), out_target.type(out_cl.F.dtype).to(device))
                layer_losses.append(curr_layer_loss.item())
                batch_bce_loss += curr_layer_loss
            voxel_cls_loss = batch_bce_loss / num_layers
            step_cls_losses.append(voxel_cls_loss.item())
            
            # 计算总损失
            total_loss = alpha * voxel_cls_loss + beta * points_reg_loss
            print(f"points_reg_loss: {points_reg_loss}")
            print(f"voxel_cls_loss: {voxel_cls_loss}")
            print(f"layer_losses: {layer_losses}")
            print(f"total_loss: {total_loss}\n")

            # sin curr
            sin_curr_pc = t_p_points_norm_list[0].cpu().numpy()
            # sin hist
            sin_hist_pc = hist_points_norm_list[0].cpu().numpy()
            # gt
            gt_pc = t_c_points_norm_list[0].cpu().numpy()
            # sout
            sout_pc = sout_points_norm_list[0].cpu().detach().numpy()
            
            # visualization
            sin_curr_pcd = PointCloud(sin_curr_pc, color=[1, 0, 0], translate_offset=[-1.0, -1.0, 0], rotate_matrix=M)
            sin_hist_pcd = PointCloud(sin_hist_pc, color=[0, 0, 1], translate_offset=[1.0, -1.0, 0], rotate_matrix=M)
            gt_pcd = PointCloud(gt_pc, color=[1, 1, 0], translate_offset=[-1.0, 1.0, 0], rotate_matrix=M)
            sout_pcd = PointCloud(sout_pc, color=[0, 1, 0], translate_offset=[1.0, 1.0, 0], rotate_matrix=M)
            o3d.visualization.draw_geometries([sin_curr_pcd, sin_hist_pcd, gt_pcd, sout_pcd])


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
            augment_data=True,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            repeat=True,
            config=config,
        )
        
        train(net, dataloader, device, config)
        
    else:
        checkpoint_dir_path = os.path.join(config.save_dir, config.model_name)
        if not os.path.exists(checkpoint_dir_path):
            error_msg = f"Model directory not found at: {checkpoint_dir_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        checkpoint_files = glob.glob(os.path.join(checkpoint_dir_path, "*.pth"))
        if not checkpoint_files:
            error_msg = f"No checkpoint files found in {checkpoint_dir_path}."
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        checkpoint_files.sort(key=lambda x: int(re.search(r'_(\d+)\.pth', os.path.basename(x)).group(1)))
        latest_checkpoint_path = checkpoint_files[-1]
        latest_iter = int(re.search(r'_(\d+)\.pth', os.path.basename(latest_checkpoint_path)).group(1))
        logging.info(f"Found latest checkpoint: {os.path.basename(latest_checkpoint_path)} (Iter: {latest_iter})")
        logging.info(f"Loading weights from {latest_checkpoint_path} ...")

        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        logging.info(f"Load weights success. ")

        dataloader = make_data_loader(
            phase="train",
            augment_data=True,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            repeat=True,
            config=config,
        )

        visualize(net, dataloader, device, config)


###############################################################################
# End of Main Thread
###############################################################################

