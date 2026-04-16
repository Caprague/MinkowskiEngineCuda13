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
parser.add_argument("--save_freq_iter",     type=int,       default=250)
parser.add_argument("--batch_size",         type=int,       default=4)
parser.add_argument("--lr",                 type=float,     default=1e-2)
parser.add_argument("--momentum",           type=float,     default=0.9)
parser.add_argument("--weight_decay",       type=float,     default=1e-4)
parser.add_argument("--num_workers",        type=int,       default=4)
parser.add_argument("--log_dir",            type=str,       default="./output/logs")
parser.add_argument("--save_dir",           type=str,       default="./output/chechpoint")
parser.add_argument("--model_name",         type=str,       default="lidar_completion_old")
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

def voxelize(points_list, res: int=1):
    """
    辅助函数: 进行体素尺寸放大，并使用ME量化工具进行去重
    输入:
        points_list: List[np.ndarray], 包含 Batch 中每个样本的点云坐标 (N_i, 3)
        res: float, 体素分辨率
    返回:
        unique_points_list: List[np.ndarray], 去重后的原始归一化坐标
        coords_float_list: List[np.ndarray], 去重后的浮点放大坐标 (用于计算 Offset 特征)
        coords_int_list: List[np.ndarray], 去重后的整型体素坐标 (用于构建 SparseTensor)
    """
    unique_points_list = []
    coords_float_list = []
    coords_int_list = []
    
    for points in points_list:
        coords_float = (points * res).astype(np.float32)
        coords_int = np.floor(points * res).astype(np.int32)
        
        # sparse_quantize 返回唯一的坐标索引
        _, indices = ME.utils.sparse_quantize(coords_int, return_index=True)
        # 4. 根据索引提取去重后的数据
        unique_points_list.append(points[indices])
        coords_float_list.append(coords_float[indices])
        coords_int_list.append(coords_int[indices])
        
    return unique_points_list, coords_float_list, coords_int_list
    
    
def compute_feats(coords_float_list, coords_voxel_list, time_encoding):
    """
    辅助函数: 计算稀疏张量的特征
    输入:
        coords_float_list: List[np.ndarray], 浮点放大坐标列表 [(N1, 3), (N2, 3), ...]
        coords_voxel_list: List[np.ndarray], 整型体素坐标列表 [(N1, 3), (N2, 3), ...]
        time_encoding: float, 时间步编码
    返回:
        feats_list: List[np.ndarray], 特征矩阵列表 [(N1, 4), (N2, 4), ...]
    """
    feats_list = []
    
    for coords_float, coords_voxel in zip(coords_float_list, coords_voxel_list):
        feats_offset = coords_float - coords_voxel.astype(np.float32)
        feats_temporal = np.full((feats_offset.shape[0], 1), time_encoding, dtype=np.float32)
        feats = np.concatenate([feats_offset, feats_temporal], axis=1)
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
        
        self.root = "./ConstructTerrain"
        
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
        
            # 检查目录是否存在，确保配置的运动类别有效
            if not os.path.exists(partial_root) or not os.path.exists(complete_root):
                error_msg = f"Directory not found for type '{current_type}'. Checked paths:\n  Partial: {partial_root}\n  Complete: {complete_root}"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # 先获取 Partial 下全部序列样本的文件夹列表，并排序 (例如: ['0001', '0002'])
            partial_seq_dirs = sorted(glob.glob(os.path.join(partial_root, "*/")))
            # 遍历 Partial 序列样本文件夹列表
            for partial_seq_path in partial_seq_dirs:
                # 对应构建 Comple 序列样本文件夹路径
                seq_idx_s = os.path.basename(os.path.normpath(partial_seq_path))    # 获取 "0001"
                complete_seq_path = os.path.join(complete_root, seq_idx_s)          # 组合到 complete_root
            
                # 检查 complete 目录是否存在，确保数据配对完整性
                if not os.path.exists(complete_seq_path):
                    error_msg = f"Missing complete data for {seq_idx_s} at path: {complete_seq_path}. Data integrity check failed."
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
            
                # 将这一组时序样本加入列表
                self.samples.append({
                    'type': current_type,
                    'seq_idx': seq_idx_s,
                    'partial_paths': p_fnames,    # 明确表明存储的是文件路径列表
                    'complete_paths': c_fnames    # 明确表明存储的是文件路径列表
                })
        
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
            # 缓存检查
            if pcd_path in self.cache:
                points = self.cache[pcd_path]
            else:
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
                is_norm_type2 = (points.min() > -0.5) and (points.max() < 0.5)
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
            'partial': self._load_pcd_sequence(sample_info['partial_paths'], self.force_norm),
            'complete': self._load_pcd_sequence(sample_info['complete_paths'], self.force_norm),
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
        
        num_frames_tensor = torch.tensor(num_frames_list)
        if torch.all(num_frames_tensor == num_frames_tensor[0]):
            final_num_frames = int(num_frames_tensor[0].item())
            turncated_p_list = partial_list
            turncated_c_list = complete_list
        else:
            min_frames = int(num_frames_tensor.min().item())
            final_num_frames = min_frames
            turncated_p_list = [seq[:min_frames] for seq in partial_list]
            turncated_c_list = [seq[:min_frames] for seq in complete_list]
            
        time_p_list = []
        time_c_list = []
        for t in range(final_num_frames):
            t_p_coords = [sample_seq[t] for sample_seq in turncated_p_list]
            t_c_coords = [sample_seq[t] for sample_seq in turncated_c_list]
            time_p_list.append(t_p_coords)
            time_c_list.append(t_c_coords)

        return {
            'type': type_list,
            'seq_idx': seq_idx_list,
            'num_frames': final_num_frames,
            'partial': time_p_list,
            'complete': time_c_list,
        }


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

        # Last layer does not require adding the target
        # if self.training:
        #     keep_s1 += target

        # Remove voxels s1
        dec_s1 = self.pruning(dec_s1, keep_s1)

        # predict voxels, gt voxels, output pc(coords + feats) 
        return out_cls, targets, dec_s1


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

    # 初始化损失函数，使用 Binary Cross Entropy with Logits Loss，即带 Logits 的二分类交叉熵损失
    crit = nn.BCEWithLogitsLoss()

    # 网络切换到训练模式
    net.train()
    
    # 获取数据载入器的迭代器
    train_iter = iter(dataloader)
    
    # 记录初始学习率
    current_lr = scheduler.get_lr()[0]
    logging.info(f"LR: {current_lr}")
    writer.add_scalar('Params/LR', current_lr, 0)
    
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
        step_losses_in_iter = []

        # 清除梯度
        optimizer.zero_grad()
        
        # 按照时序遍历
        num_frames = data_dict['num_frames']    
        sout = None
        for t in range(num_frames):
            # 原始点云序列，List[np.ndarray]
            t_p_points_list = data_dict['partial'][t]
            t_c_points_list = data_dict['complete'][t]
            # 体素坐标量化，List[np.ndarray]
            _, t_p_coords_float_list, t_p_coords_voxel_list = voxelize(t_p_points_list, config.resolution)
            _, _, t_c_coords_voxel_list = voxelize(t_c_points_list, config.resolution)
            # 计算输入点云坐标对应的特征，List[np.ndarray]
            t_p_feats_list = compute_feats(t_p_coords_float_list, t_p_coords_voxel_list, 0.0)
            # 将松散序列聚合成统一批次的稀疏张量，List[np.ndarray] -> torch.Tensor
            batched_curr_coords, batched_curr_feats = ME.utils.sparse_collate(t_p_coords_voxel_list, t_p_feats_list, device=device)
            batched_labels = ME.utils.batched_coordinates(t_c_coords_voxel_list).to(device)
            if t == 0:
                batched_hist_coords = batched_curr_coords
                batched_hist_feats = batched_curr_feats
            else:
                # 分解获取坐标，torch.Tensor -> List[np.ndarray]
                hist_coords_list = sout.decomposed_coordinates
                hist_np_list = [coords.cpu().numpy() for coords in hist_coords_list]
                # 体素坐标量化，List[np.ndarray]
                _, hist_coords_float_list, hist_coords_voxel_list = voxelize(hist_np_list, 1)
                # 计算输入点云坐标对应的特征，List[np.ndarray]
                hist_feats_list = compute_feats(hist_coords_float_list, hist_coords_voxel_list, 1.0)
                # 将松散序列聚合成统一批次的稀疏张量，List[np.ndarray] -> torch.Tensor
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
                coordinates=batched_labels,
                string_id="target",
            )
            
            # Forward
            out_cls, out_targets, sout = net(sin_curr, sin_hist, in_target_key)
        
            # 计算损失
            num_layers, step_loss = len(out_cls), 0
            layer_losses = []
            for out_cl, out_target in zip(out_cls, out_targets):
                # 计算单层损失
                curr_layer_loss = crit(out_cl.F.squeeze(), out_target.type(out_cl.F.dtype).to(device))
                # 记录单层损失
                layer_losses.append(curr_layer_loss.item())
                # 计算全部层平均损失
                step_loss += curr_layer_loss / num_layers
            # 更新训练步数，计算并记录步数损失
            train_steps += 1
            step_losses_in_iter.append(step_loss.item())
            writer.add_scalar('Loss/Step_Loss', step_loss.item(), train_steps)
            # 分别记录每一层的步数损失
            for layer_idx, loss_val in enumerate(layer_losses):
                writer.add_scalar(f'Loss/Layer{layer_idx+1}_Loss', loss_val, train_steps)
            
            # 反向传播损失，梯度更新网络权重
            step_loss.backward()
            optimizer.step()
        
        # 计算并记录本次 Iter 总耗时、平均损失
        total_time = time() - start_time
        iter_loss = sum(step_losses_in_iter)/len(step_losses_in_iter)
        writer.add_scalar('Time/Iter_Total_Time', total_time, i)
        writer.add_scalar('Loss/Iter_Loss', iter_loss, i)
        
        if i % config.stat_freq_iter == 0:
            steps_losses_str = ", ".join([f"{v:.3e}" for v in step_losses_in_iter])
            logging.info(
                f"Iter: {i}, Step: {train_steps}, Step Losses: [{steps_losses_str}], Iter Ave Loss: {iter_loss:.3e}, Data Loading Time: {data_time:.3e}, Total Time: {total_time:.3e}"
            )

        if i % config.save_freq_iter == 0 and i > 0:
            # 保存模型和训练断点
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
        
        if i % config.lr_step == 0 and i > 0:
            # 学习率控制器更新
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
    crit = nn.BCEWithLogitsLoss()
    
    print("🔍 可视化布局说明:")
    print("  红: 当前 | 蓝: 历史")
    print("  黄: 真值 | 绿: 输出")

    # 顺序遍历数据加载器，获取小批量样本
    for i in range(config.max_visualization):
        # 获取样本
        data_dict = next(train_iter)
        
        # 批次维度应当为1
        assert len(data_dict['partial'][0]) == 1, "Error: Dataloader batch_size in Visualize(eval mode) should equal to 1 !!!"
        
        # 按照时序遍历
        num_frames = data_dict['num_frames']
        for t in range(num_frames):
            # 原始点云序列，List[np.ndarray]
            t_p_points_list = data_dict['partial'][t]
            t_c_points_list = data_dict['complete'][t]
            # 体素坐标量化，List[np.ndarray]
            t_p_points_norm, t_p_coords_float_list, t_p_coords_voxel_list = voxelize(t_p_points_list, config.resolution)
            t_c_points_norm, _, t_c_coords_voxel_list = voxelize(t_c_points_list, config.resolution)
            # 计算输入点云坐标对应的特征，List[np.ndarray]
            t_p_feats_list = compute_feats(t_p_coords_float_list, t_p_coords_voxel_list, 0.0)
            # 将松散序列聚合成统一批次的稀疏张量，List[np.ndarray] -> torch.Tensor
            batched_curr_coords, batched_curr_feats = ME.utils.sparse_collate(t_p_coords_voxel_list, t_p_feats_list, device=device)
            batched_labels = ME.utils.batched_coordinates(t_c_coords_voxel_list).to(device)
            if t == 0:
                batched_hist_coords = batched_curr_coords
                batched_hist_feats = batched_curr_feats
            else:
                # 分解获取坐标，torch.Tensor -> List[np.ndarray]
                hist_coords_list = sout.decomposed_coordinates
                hist_np_list = [coords.cpu().numpy() for coords in hist_coords_list]
                # 体素坐标量化，List[np.ndarray]
                _, hist_coords_float_list, hist_coords_voxel_list = voxelize(hist_np_list, 1)
                # 计算输入点云坐标对应的特征，List[np.ndarray]
                hist_feats_list = compute_feats(hist_coords_float_list, hist_coords_voxel_list, 1.0)
                # 将松散序列聚合成统一批次的稀疏张量，List[np.ndarray] -> torch.Tensor
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
                coordinates=batched_labels,
                string_id="target",
            )
        
            # Forward
            out_cls, out_targets, sout = net(sin_curr, sin_hist, in_target_key)
        
            # 计算损失
            num_layers, step_loss = len(out_cls), 0
            layer_losses = []
            for out_cl, out_target in zip(out_cls, out_targets):
                # 计算单层损失
                curr_layer_loss = crit(out_cl.F.squeeze(), out_target.type(out_cl.F.dtype).to(device))
                # 记录单层损失
                layer_losses.append(curr_layer_loss.item())
                # 计算全部层平均损失
                step_loss += curr_layer_loss / num_layers
            print(f"step_loss: {step_loss} layer_losses: {layer_losses}")

            # sin curr
            sin_curr_pc = t_p_points_norm[0]
            # sin hist
            if t == 0:
                sin_hist_pc = t_p_points_norm[0]
            else:
                sin_hist_pc = hist_np_list[0]/config.resolution
            # gt
            gt_pc = t_c_points_norm[0]
            # sout
            batched_coords = sout.decomposed_coordinates
            sout_pc = batched_coords[0].cpu().numpy()/config.resolution
            
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
