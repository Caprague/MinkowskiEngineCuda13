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

import MinkowskiEngine as ME

from torch.utils.data.sampler import Sampler

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


# 检测数据集目录，若不存在，则执行脚本下载数据集并解压到指定路径
if not os.path.exists("ModelNet40"):
    logging.info("Downloading the pruned ModelNet40 dataset...")
    subprocess.run(["sh", "./examples/download_modelnet40.sh"])


###############################################################################
# Utility classes
###############################################################################

# ModelNet40 专用数据集类，继承自 torch 数据集基类
class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, transform=None, config=None):
        self.phase = phase
        self.files = []
        self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution
        self.last_cache_percent = 0

        # 指定路径，并硬编码仅读取 chair 类的数据
        self.root = "./ModelNet40"
        fnames = glob.glob(os.path.join(self.root, "chair/train/*.off"))            # 相对路径组合，正则化表达式包含所有文件，然后转化绝对路径
        fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])    # 转化成基于 root 路径的相对路径列表，并进行排序
        # 保存获取的数据集文件路径列表
        self.files = fnames
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the subset {phase} from {self.root} with {len(self.files)} files"
        )
        # 设定每个 mesh 三维样本形体的表面采样密度(点数)
        self.density = 30000

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # 获取所有 mesh 样本的总数
    def __len__(self):
        return len(self.files)

    # 按照给定的 idx 索引，返回指定的 mesh 样本数据 
    # （idx 为单个整数，因而每次仅返回 1 个样本）
    def __getitem__(self, idx):
        # 获取指定 mesh 样本的路径
        mesh_file = os.path.join(self.root, self.files[idx])
        # 若已存在于缓存中，则直接从缓存中获取
        if idx in self.cache:
            xyz = self.cache[idx]
        else:
            # Load a mesh, over sample, copy, rotate, voxelization
            # 断言样本路径存在
            assert os.path.exists(mesh_file)
            # 读取样本数据
            pcd = o3d.io.read_triangle_mesh(mesh_file)
            # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
            # 对读取的 mesh 样本顶点列表进行坐标归一化
            vertices = np.asarray(pcd.vertices)
            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            pcd.vertices = o3d.utility.Vector3dVector(
                (vertices - vmin) / (vmax - vmin).max()
            )

            # Oversample points and copy
            # 按照指定密度(点数)，对归一化后的 mesh 样本表面进行点采样
            xyz = resample_mesh(pcd, density=self.density)
            # 表面点云采样结果保存到缓存中
            self.cache[idx] = xyz
            # 计算总样本已缓存的占比
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if (
                cache_percent > 0
                and cache_percent % 10 == 0
                and cache_percent != self.last_cache_percent
            ):
                # 记录总缓存占比增长进度
                logging.info(
                    f"Cached {self.phase}: {len(self.cache)} / {len(self)}: {cache_percent}%"
                )
                self.last_cache_percent = cache_percent

        # Use color or other features if available
        # 按照采样点的长度，生成对应的全 1 特征值（仅表示普通点占据；若需要颜色或其他特征信息，需要自行拓展）
        feats = np.ones((len(xyz), 1))

        # 采样密度(点数)过低，跳过不输出
        if len(xyz) < 1000:
            logging.info(
                f"Skipping {mesh_file}: does not have sufficient CAD sampling density after resampling: {len(xyz)}."
            )
            return None

        # 若提供了样本数据预处理接口，则进行预处理（样本增强、变换、加噪等）
        if self.transform:
            xyz, feats = self.transform(xyz, feats)

        # Get coords
        # 将归一化采样坐标，按照体素网格分辨率，进行尺度放大
        xyz = xyz * self.resolution
        # 放大后的坐标，进行向下取整离散化、体素格内重复点去除，并返回保留的离散点的索引（用于掩码标记对应的原始点坐标）
        coords, inds = ME.utils.sparse_quantize(xyz, return_index=True)

        # 归一化 -> 分辨率放大 -> 坐标下取整离散化后
        # 返回最终离散化坐标列表、对应的原始点坐标列表(经分辨率放大后)、样本索引号
        return (coords, xyz[inds], idx)


# 样本随机采样器，继承自 torch 采样器基类
class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # 构造函数，保存采样数据源、混洗标志位，生成样本采样序列
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    # 生成样本采样序列，后续会然此序列，逐个弹出样本
    def reset_permutation(self):
        # 样本总数
        perm = len(self.data_source)
        # 若开启数据混洗，则随机生成样本序列号列表
        if self.shuffle:
            self._perm = torch.randperm(perm).tolist()
        # 反之，生成顺序样本序列号列表
        else:
            self._perm = list(range(perm))

    # 获取样本迭代器（自身）
    def __iter__(self):
        return self

    # 获取下一样本
    def __next__(self):
        # 若样本生成序列已空，则重新生成
        if len(self._perm) == 0:
            self.reset_permutation()
        # 弹出一个样本
        return self._perm.pop()

    # 获取样本总数
    def __len__(self):
        return len(self.data_source)


# 样本聚合类，用于聚合多 Worker 读取的单样本数据，生成小批量样本数据
class CollationAndTransformation:
    def __init__(self, resolution):
        self.resolution = resolution

    # 样本裁剪函数，以固定的 resolution / 3 对 coords 进行裁剪处理
    def random_crop(self, coords_list):
        crop_coords_list = []
        for coords in coords_list:
            # 注：为什么是截取 [:, 0] ？？？ 为什么使用 coords[:, 0] < self.resolution / 3 作为裁剪掩码？？？
            sel = coords[:, 0] < self.resolution / 3
            crop_coords_list.append(coords[sel])
        return crop_coords_list

    # 回调处理函数，对样本离散坐标进行裁剪，并返回小批量聚合样本
    def __call__(self, list_data):
        coords, feats, labels = list(zip(*list_data))
        # 注：为什么仅仅裁剪 coords ？？？ 为什么不同步裁剪 feats 和 labels ？？？
        coords = self.random_crop(coords)

        # Concatenate all lists
        return {
            "coords": ME.utils.batched_coordinates(coords),
            "xyzs": [torch.from_numpy(feat).float() for feat in feats],
            "cropped_coords": coords,
            "labels": torch.LongTensor(labels),
        }

###############################################################################
# End of utility classes
###############################################################################


###############################################################################
# Utility functions
###############################################################################

# 辅助函数，将输入的 Tensor 点云转化为 Open3D 中的 pcd 点云格式，方便可视化/保存等
def PointCloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 可指定颜色参数
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
    

# 辅助函数，用于获取 data loader
def make_data_loader(
    phase, augment_data, batch_size, shuffle, num_workers, repeat, config
):
    # 获取指定 phase 数据集子集，此处为 “val”
    dset = ModelNet40Dataset(phase, config=config)

    # data loader 的相关参数
    args = {
        "batch_size": batch_size,       # 小批量样本尺寸
        "num_workers": num_workers,     # 样本并行加载线程
        "collate_fn": CollationAndTransformation(config.resolution),        # 小批量样本聚合函数
        "pin_memory": False,            # 内存连续性对齐
        "drop_last": False,             # 是否丢弃不完整的最后一个小批量样本
    }

    if repeat:
        # 样本随机采样器，传入指定的数据子集，并指定 shuffle 数据混洗标志位（此处为启用）
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        # 指定 shuffle 数据混洗标志位（此处为启用）
        args["shuffle"] = shuffle

    # 传入参数：数据子集，及前述相关配置参数
    # 获取并返回 data loader
    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


# 辅助函数，从 mesh 形体上，按照指定密度(点数)均匀采集表面点
def resample_mesh(mesh_cad, density=1):
    """
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud

    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    """
    faces = np.array(mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    vec_cross = np.cross(
        vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
        vertices[faces[:, 1], :] - vertices[faces[:, 2], :],
    )
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)
    # face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Bug fix by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc : acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (
        (1 - np.sqrt(r[:, 0:1])) * A
        + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
        + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    )

    return P

###############################################################################
# End of utility functions
###############################################################################


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
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--max_iter", type=int, default=30000)
parser.add_argument("--val_freq", type=int, default=1000)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
parser.add_argument("--weights", type=str, default="modelnet_completion.pth")
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--max_visualization", type=int, default=4)

###############################################################################
# End of global configs
###############################################################################


class CompletionNet(nn.Module):

    # 编码器和解码器的通道数参数列表
    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self, resolution, in_nchannel=512):
        nn.Module.__init__(self)

        # 保存原始输入的分辨率（三维体素网格边长）
        self.resolution = resolution

        # Input sparse tensor must have tensor stride 128.
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # 注：如下网络层/块后缀的 sxx ，都表示该层/块的输入张量的 tensor_stride
        # 即输入张量的量化分辨率，在本点云补全任务中，表示点体素坐标的量化步长

        # Encoder
        # 原始特征捕获层（Conv、BN、ELU，三层构成一个基本块）
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(1, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        # s1s2，两个基本块构成，0.5分辨率下采样
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

        self.enc_block_s32s64 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[5], enc_ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
        )

        # Decoder
        # s64s32，与编码器层类似，由两个基本块构成，但首个基本块包含TransposeConvolution，进行2倍上采样
        self.dec_block_s64s32 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[6],
                dec_ch[5],
                kernel_size=4,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
        )

        # s32_cls，解码器独有，用于回归预测每个体素块的占有概率，因此ch从N将至1
        # 通过每层分别回归预测体素占有概率，实现与真值target的交叉熵计算，综合多层次编码器的预测损失
        self.dec_s32_cls = ME.MinkowskiConvolution(
            dec_ch[5], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[5],
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

        # pruning
        # 由于解码器为上采样过程，导致其原始Conv输出会变得密集，添加Pruning层继续保持稀疏性
        self.pruning = ME.MinkowskiPruning()

    # 重要方法
    # 通过外部ME的方法接口，从真值点云坐标中计算获取target_key
    # 再输入本方法获取对应tensor_stride下的体素占用情况
    # 其他：具体输入输出数据形式待探明
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

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    # 前向传播方法，其中 partial_in 为转化 ME.SparseTensor 后的点云稀疏张量
    # target_key 为真值点云坐标列表转化后的坐标键，由坐标管理器管理
    def forward(self, partial_in, target_key):
        out_cls, targets = [], []

        # 编码器前向传播
        enc_s1 = self.enc_block_s1(partial_in)
        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)

        ##################################################
        # Decoder 64 -> 32
        ##################################################
        # 编码器层前向传播
        dec_s32 = self.dec_block_s64s32(enc_s64)

        # Add encoder features
        # 跳跃连接编码器特征
        dec_s32 = dec_s32 + enc_s32
        # 预测输出体素占据概率
        dec_s32_cls = self.dec_s32_cls(dec_s32)
        # 获取掩码，表示占据概率大于0的区域
        keep_s32 = (dec_s32_cls.F > 0).squeeze()

        # 获取掩码，从真值点云反向解码对应占据的体素区域
        target = self.get_target(dec_s32, target_key)
        targets.append(target)
        out_cls.append(dec_s32_cls)

        # 在训练模式下，将两种掩码相加
        if self.training:
            keep_s32 += target

        # Remove voxels s32
        # 对输出的原始密集体素，进行掩码裁剪，保持稀疏性
        dec_s32 = self.pruning(dec_s32, keep_s32)

        ##################################################
        # Decoder 32 -> 16
        ##################################################
        dec_s16 = self.dec_block_s32s16(dec_s32)

        # Add encoder features
        dec_s16 = dec_s16 + enc_s16
        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > 0).squeeze()

        target = self.get_target(dec_s16, target_key)
        targets.append(target)
        out_cls.append(dec_s16_cls)

        if self.training:
            keep_s16 += target

        # Remove voxels s16
        dec_s16 = self.pruning(dec_s16, keep_s16)

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s16)

        # Add encoder features
        dec_s8 = dec_s8 + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)

        target = self.get_target(dec_s8, target_key)
        targets.append(target)
        out_cls.append(dec_s8_cls)
        keep_s8 = (dec_s8_cls.F > 0).squeeze()

        if self.training:
            keep_s8 += target

        # Remove voxels s16
        dec_s8 = self.pruning(dec_s8, keep_s8)

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        dec_s4 = self.dec_block_s8s4(dec_s8)

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)

        target = self.get_target(dec_s4, target_key)
        targets.append(target)
        out_cls.append(dec_s4_cls)
        keep_s4 = (dec_s4_cls.F > 0).squeeze()

        if self.training:
            keep_s4 += target

        # Remove voxels s4
        dec_s4 = self.pruning(dec_s4, keep_s4)

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)

        target = self.get_target(dec_s2, target_key)
        targets.append(target)
        out_cls.append(dec_s2_cls)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()

        if self.training:
            keep_s2 += target

        # Remove voxels s2
        dec_s2 = self.pruning(dec_s2, keep_s2)

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)

        # Add encoder features
        dec_s1 = dec_s1 + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        target = self.get_target(dec_s1, target_key)
        targets.append(target)
        out_cls.append(dec_s1_cls)
        keep_s1 = (dec_s1_cls.F > 0).squeeze()

        # Last layer does not require adding the target
        # if self.training:
        #     keep_s1 += target

        # Remove voxels s1
        dec_s1 = self.pruning(dec_s1, keep_s1)

        return out_cls, targets, dec_s1

# 网络训练函数
def train(net, dataloader, device, config):
    # 初始化 SGD 优化器
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    # 初始化 LR 学习率控制器
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    # 初始化损失函数，此处使用 Binary Cross Entropy with Logits Loss，即带 Logits 的二分类交叉熵损失
    crit = nn.BCEWithLogitsLoss()

    # 网络切换到训练模式
    net.train()
    # 获取数据载入器的迭代器
    train_iter = iter(dataloader)
    # val_iter = iter(val_dataloader)
    logging.info(f"LR: {scheduler.get_lr()}")
    
    # 训练周期循环
    for i in range(config.max_iter):

        s = time()
        # 获取小批量训练数据
        # data_dict = train_iter.next()
        data_dict = next(train_iter)
        d = time() - s

        # 清除梯度
        optimizer.zero_grad()

        # 初始化稀疏点云特征，由于仅考虑点云存在性/占据概率，因而只使用全1特征即可
        in_feat = torch.ones((len(data_dict["coords"]), 1))
        # 初始化输入稀疏张量，以稀疏点云坐标列表、稀疏点云特征来构建
        sin = ME.SparseTensor(
            features=in_feat,
            coordinates=data_dict["coords"],
            device=device,
        )

        # Generate target sparse tensor
        # 获取输入稀疏张量的坐标管理器
        cm = sin.coordinate_manager
        # 将真值完整云坐标注册到坐标管理器，便于后续在不同分辨率层级进行坐标匹配和目标掩码生成
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(data_dict["xyzs"]).to(device),
            string_id="target",
        )

        # Generate from a dense tensor
        # 网络前向传播，获取每一层预测的体素占有概率列表、真值体素占据列表、网络预测的输出结果(稀疏张量，由坐标及特征两部分构成)
        out_cls, targets, sout = net(sin, target_key)
        num_layers, loss = len(out_cls), 0
        losses = []
        # 遍历列表
        for out_cl, target in zip(out_cls, targets):
            # 计算单层损失
            curr_loss = crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
            # 记录单层损失
            losses.append(curr_loss.item())
            # 计算全部层平均损失
            loss += curr_loss / num_layers

        # 反向传播损失
        loss.backward()
        # 梯度更新网络权重
        optimizer.step()
        t = time() - s

        # 定期打印训练状态
        if i % config.stat_freq == 0:
            logging.info(
                f"Iter: {i}, Loss: {loss.item():.3e}, Data Loading Time: {d:.3e}, Tot Time: {t:.3e}"
            )

        # 定期保存训练断点
        if i % config.val_freq == 0 and i > 0:
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                },
                config.weights,
            )

            # 学习率控制器更新
            scheduler.step()
            logging.info(f"LR: {scheduler.get_lr()}")

            # 切换训练模式
            net.train()

# 可视化展现函数
def visualize(net, dataloader, device, config):
    # 网络切换评估模式
    net.eval()
    # 初始化训练损失
    crit = nn.BCEWithLogitsLoss()
    n_vis = 0

    # 顺序遍历数据加载器，获取小批量样本
    for data_dict in dataloader:
        # 初始化稀疏点云的稀疏特征，全1
        in_feat = torch.ones((len(data_dict["coords"]), 1))
        # 初始化输入的稀疏张量，使用稀疏点云坐标列表和稀疏点云特征列表构建
        sin = ME.SparseTensor(
            features=in_feat,
            coordinates=data_dict["coords"],
            device=device
        )

        # Generate target sparse tensor
        # 获取坐标管理器
        cm = sin.coordinate_manager
        # target_key = cm.create_coords_key(
        #     ME.utils.batched_coordinates(data_dict["xyzs"]),
        #     force_creation=True,
        #     allow_duplicate_coords=True,
        # )
        
        # 将真值完整云坐标注册到坐标管理器，便于后续在不同分辨率层级进行坐标匹配和目标掩码生成
        target_coords = ME.utils.batched_coordinates(data_dict["xyzs"]).to(device)
        target_key, _ = cm.insert_and_map(
            target_coords,
            string_id="target" 
        )

        # Generate from a dense tensor
        # 网络前向传播
        out_cls, targets, sout = net(sin, target_key)
        num_layers, loss = len(out_cls), 0
        # 计算全部层平均损失
        for out_cl, target in zip(out_cls, targets):
            loss += (
                crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
                / num_layers
            )

        # 核心：可视化点云补全结果，以及原始的残缺点云
        # 按照批次维度，分开获取输出的稀疏点云坐标和特征
        batch_coords, batch_feats = sout.decomposed_coordinates_and_features
        # 遍历输出结果
        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            # 转化稀疏点云坐标为open3d中的点云数据
            pcd = PointCloud(coords.cpu().numpy())
            pcd.estimate_normals()  # 评估法向量，在可视化过程中帮助改善可视化效果
            # 点云平移
            pcd.translate([0.6 * config.resolution, 0, 0])
            # 点云旋转
            pcd.rotate(M, np.array([[0.0], [0.0], [0.0]]))
            # 转化原始残缺点云坐标为open3d中的点云数据
            opcd = PointCloud(data_dict["cropped_coords"][b])
            # 点云平移
            opcd.translate([-0.6 * config.resolution, 0, 0])
            opcd.estimate_normals() # 评估法向量，在可视化过程中帮助改善可视化效果
            # 点云旋转
            opcd.rotate(M, np.array([[0.0], [0.0], [0.0]]))
            # 可视化展现两个点云
            o3d.visualization.draw_geometries([pcd, opcd])

            # 限定可视化次数
            n_vis += 1
            if n_vis > config.max_visualization:
                return

# 主线程
if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取数据加载器
    dataloader = make_data_loader(
        "val",
        augment_data=True,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        config=config,
    )
    
    # 未实际使用的参数，可删除，推测为历史复制遗留代码
    in_nchannel = len(dataloader.dataset)

    # 初始化网络
    net = CompletionNet(config.resolution, in_nchannel=in_nchannel)
    # 转移至GPU
    net.to(device)

    logging.info(net)

    # 训练模式
    if not config.eval:
        train(net, dataloader, device, config)
    # 评估模式
    else:
        # 载入预训练权重
        if not os.path.exists(config.weights):
            logging.info(f"Downloaing pretrained weights. This might take a while...")
            urllib.request.urlretrieve(
                "https://bit.ly/36d9m1n", filename=config.weights
            )

        logging.info(f"Loading weights from {config.weights}")
        checkpoint = torch.load(config.weights)
        net.load_state_dict(checkpoint["state_dict"])

        # 可视化展现点云补全效果
        visualize(net, dataloader, device, config)
