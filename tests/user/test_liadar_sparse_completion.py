#!/usr/bin/env python3
"""
最终修正版：严格对齐原始 usrbin.txt 逻辑
功能：模块化封装 + 正确的 CoordinateManager 共享 + 强制 Train 模式下的 Target Key
"""

import sys
import os
import torch
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d

# 1. 获取项目根目录 (根据实际路径调整)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'examples'))

from lidar_sparse_completion import LidarCompletionNet

# 全局可视化旋转矩阵
M = np.array([
    [0.80656762, -0.5868724, -0.07091862],
    [0.3770505,  0.418344,  0.82632997],
    [-0.45528188, -0.6932309, 0.55870326],
])

# -------------------------------------------------------------------------------------
# 1. 数据生成模块 (严格复刻 usrbin.txt 逻辑)
# -------------------------------------------------------------------------------------
def generate_test_data(resolution=64):
    """
    生成台阶形状真值，并模拟残缺当前帧和历史帧。
    """
    print("\n\n" + "="*60)
    print("阶段 2: 生成模拟测试数据")
    print("="*60)
    print("\n🚀 开始生成模拟测试数据...")
    
    # --- 真值数据：台阶形状 ---
    ground_level = 0.2
    platform_level = 0.7
    transition_x = 0.5

    # 1. 低矮地面 (左侧)
    ground_points = 3000
    ground_coords = np.zeros((ground_points, 3), dtype=np.float32)
    ground_coords[:, 0] = np.random.uniform(0.05, transition_x - 0.02, ground_points)
    ground_coords[:, 1] = np.random.uniform(0.1, 0.9, ground_points)
    ground_coords[:, 2] = np.random.uniform(ground_level - 0.02, ground_level, ground_points)

    # 2. 高台面 (右侧)
    platform_points = 3000
    platform_coords = np.zeros((platform_points, 3), dtype=np.float32)
    platform_coords[:, 0] = np.random.uniform(transition_x + 0.02, 0.95, platform_points)
    platform_coords[:, 1] = np.random.uniform(0.1, 0.9, platform_points)
    platform_coords[:, 2] = np.random.uniform(platform_level - 0.02, platform_level, platform_points)

    # 3. 垂直墙壁 (中间过渡)
    wall_points = 1500
    wall_coords = np.zeros((wall_points, 3), dtype=np.float32)
    wall_coords[:, 0] = np.random.uniform(transition_x - 0.02, transition_x + 0.02, wall_points)
    wall_coords[:, 1] = np.random.uniform(0.1, 0.9, wall_points)
    wall_coords[:, 2] = np.random.uniform(ground_level, platform_level, wall_points)

    coords_gt_norm = np.vstack([ground_coords, platform_coords, wall_coords])
    print(f" 生成真值点云: {coords_gt_norm.shape[0]} 个点")

    # --- 模拟当前帧 (Current Frame) ---
    num_points_current = int(coords_gt_norm.shape[0] * 0.3)
    current_indices = np.random.choice(coords_gt_norm.shape[0], num_points_current, replace=False)
    coords_current_norm = coords_gt_norm[current_indices].copy()
    coords_current_norm += np.random.normal(0, 0.02, coords_current_norm.shape).astype(np.float32)

    # 模拟块状遮挡
    mask_ground = ~((coords_current_norm[:, 0] > 0.1) & (coords_current_norm[:, 0] < 0.4) & 
                   (coords_current_norm[:, 1] > 0.7) & (coords_current_norm[:, 1] < 0.9) & 
                   (coords_current_norm[:, 2] < 0.25))
    coords_current_norm = coords_current_norm[mask_ground]
    
    mask_platform = ~((coords_current_norm[:, 0] > 0.7) & (coords_current_norm[:, 0] < 0.9) & 
                     (coords_current_norm[:, 1] > 0.1) & (coords_current_norm[:, 1] < 0.4) & 
                     (coords_current_norm[:, 2] > 0.65))
    coords_current_norm = coords_current_norm[mask_platform]
    
    mask_wall = ~((coords_current_norm[:, 0] > 0.45) & (coords_current_norm[:, 0] < 0.55) & 
                 (coords_current_norm[:, 1] > 0.4) & (coords_current_norm[:, 1] < 0.6) & 
                 (coords_current_norm[:, 2] > 0.3) & (coords_current_norm[:, 2] < 0.6))
    coords_current_norm = coords_current_norm[mask_wall]
    
    print(f" 生成当前帧点云 (残缺): {coords_current_norm.shape[0]} 个点")

    # --- 模拟历史帧 (History Frame) ---
    crop_mask = coords_gt_norm[:, 1] < 0.57
    coords_history_norm = coords_gt_norm[crop_mask].copy()
    print(f" 生成历史帧点云: {coords_history_norm.shape[0]} 个点")

    # --- 体素化去重 (关键修正：使用 ME.utils.sparse_quantize) ---
    # 这是原始代码中保证坐标唯一性的关键步骤
    def voxelize(points, res):
        coords_float = (points * res).astype(np.float32)
        coords_int = np.floor(points * res).astype(np.int32)
        # sparse_quantize 会返回唯一的坐标索引
        _, indices = ME.utils.sparse_quantize(coords_int, return_index=True)
        return points[indices], coords_float[indices], coords_int[indices]

    coords_gt_norm, coords_gt_float, coords_gt_voxel = voxelize(coords_gt_norm, resolution)
    coords_current_norm, coords_current_float, coords_current_voxel = voxelize(coords_current_norm, resolution)
    coords_history_norm, coords_history_float, coords_history_voxel = voxelize(coords_history_norm, resolution)
    
    print(f"\n真值点云-归一化: {coords_gt_norm}")
    print(f"真值点云-浮点: {coords_gt_float}")
    print(f"真值点云-体素: {coords_gt_voxel}")
    print(f"\n当前帧点云-归一化: {coords_current_norm}")
    print(f"当前帧点云-浮点: {coords_current_float}")
    print(f"当前帧点云-体素: {coords_current_voxel}")
    print(f"\n历史帧点云-归一化: {coords_history_norm}")
    print(f"历史帧点云-浮点: {coords_history_float}")
    print(f"历史帧点云-体素: {coords_history_voxel}")

    return {
        'coords_gt_norm': coords_gt_norm,                   # np.float32
        'coords_gt_float': coords_gt_float,                 # np.float32
        'coords_gt_voxel': coords_gt_voxel,                 # np.int32
        'coords_current_norm': coords_current_norm,         # np.float32
        'coords_current_float': coords_current_float,       # np.float32
        'coords_current_voxel': coords_current_voxel,       # np.int32
        'coords_history_norm': coords_history_norm,         # np.float32
        'coords_history_float': coords_history_float,       # np.float32
        'coords_history_voxel': coords_history_voxel        # np.int32
    }

# -------------------------------------------------------------------------------------
# 2. 张量构建与前向传播模块 (严格复刻 usrbin.txt 逻辑)
# -------------------------------------------------------------------------------------
def build_tensors_and_forward(net, data, resolution=64, device='cpu'):
    """
    包含 CoordinateManager 创建、稀疏张量构建、Target Key 构建及前向传播。
    """
    print("\n\n" + "="*60)
    print("阶段 3: 构建稀疏张量与执行前向传播")
    print("="*60)
    
    # --- 2.1 创建共享的 CoordinateManager ---
    cm = ME.CoordinateManager(D=3)
    print(f"创建共享的 CoordinateManager: {cm}")

    # --- 2.2 辅助函数：构建单个稀疏张量 ---
    def create_sparse_tensor(coords_float, coords_voxel, time_encoding, coordinate_manager, device):
        # features = [offset_x, offset_y, offset_z, temporal_encoding], shape (N, 4)
        feats_offset = coords_float - coords_voxel.float()
        feats_temporal = torch.full((feats_offset.shape[0], 1), time_encoding, dtype=feats_offset.dtype, device=device)
        feats = torch.cat([feats_offset, feats_temporal], dim=1)
        # coordinates = [batch_indices, coords_x, coords_y, coords_z], shape (N, 4)
        batch_indices = torch.zeros(len(coords_float), 1, dtype=torch.int32, device=device)
        coords = torch.cat([batch_indices, coords_voxel], dim=1)
        # return SparseTensor
        return ME.SparseTensor(
            features=feats,
            coordinates=coords,
            coordinate_manager=coordinate_manager, # 关键：使用指定的共享坐标管理器
            device=device
        )

    # --- 2.3 构建当前帧 ---
    sin_current = create_sparse_tensor(
        torch.from_numpy(data['coords_current_float']).to(device),
        torch.from_numpy(data['coords_current_voxel']).to(device), 
        time_encoding=0.0,
        coordinate_manager=cm,
        device=device
    )
    print(f"\n当前帧稀疏张量: {sin_current}")

    # --- 2.4 构建历史帧 ---
    sin_history = create_sparse_tensor(
        torch.from_numpy(data['coords_history_float']).to(device),
        torch.from_numpy(data['coords_history_voxel']).to(device),
        time_encoding=1.0,
        coordinate_manager=cm,
        device=device
    )
    print(f"\n历史帧稀疏张量: {sin_history}")

    # --- 2.5 构建 Target Key (Train 模式强制要求) ---
    batch_indices_gt = torch.zeros(len(data['coords_gt_voxel']), 1, dtype=torch.int32, device=device)
    coords_gt_tensor = torch.cat([batch_indices_gt, torch.from_numpy(data['coords_gt_voxel']).to(device)], dim=1)
    target_key, _ = cm.insert_and_map(coords_gt_tensor, string_id="target")
    print(f"\n真值点云 target_key 已创建: {target_key}\n")

    # --- 2.6 前向传播 (Train 模式) ---
    net.train()
    try:
        out_cls, targets, sout = net(sin_current, sin_history, target_key=target_key)
        print(f"✅ 前向传播成功！输出点云数量: {len(sout.C)}")
        return sout 
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return None 

# -------------------------------------------------------------------------------------
# 3. 可视化模块 (严格复刻 usrbin.txt 逻辑)
# -------------------------------------------------------------------------------------
def visualize_point_clouds(sout, data, resolution=64):
    """
    可视化逻辑，严格对齐原始代码的平移和颜色设置。
    """
    print("\n" + "="*60)
    print("阶段 4: 启动可视化")
    print("="*60)
    
    if sout is None:
        print("无输出数据，仅可视化输入...")
        # 这里保留了原始逻辑中的位移 [-1,-1], [1,-1] 等
        pass

    # --- 3.1 处理输出结果 ---
    batch_coords, _ = sout.decomposed_coordinates_and_features
    output_coords_3d = batch_coords[0].cpu().numpy()
    output_coords_normalized = output_coords_3d / resolution

    # --- 3.2 创建辅助函数 ---
    def create_pcd(points, color, translate_offset):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
        pcd.estimate_normals()
        pcd.translate(translate_offset)
        pcd.rotate(M, center=(0,0,0))
        return pcd

    # --- 3.3 构建所有点云 ---
    # 注意：这里的 coords (如 data['coords_current']) 必须是归一化坐标 [0,1]
    current_pcd = create_pcd(data['coords_current_norm'], [1, 0, 0], [-1.0, -1.0, 0])   # 红色
    history_pcd = create_pcd(data['coords_history_norm'], [0, 0, 1], [1.0, -1.0, 0])    # 蓝色
    gt_pcd = create_pcd(data['coords_gt_norm'], [1, 1, 0], [-1.0, 1.0, 0])              # 黄色
    output_pcd = create_pcd(output_coords_normalized, [0, 1, 0], [1.0, 1.0, 0])         # 绿色

    print("🔍 可视化布局说明:")
    print("  黄: 真值 | 绿: 输出")
    print("  红: 当前 | 蓝: 历史")

    o3d.visualization.draw_geometries([current_pcd, history_pcd, gt_pcd, output_pcd])

# -------------------------------------------------------------------------------------
# 4. 主程序流程
# -------------------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resolution = 64
    activeF = 0.35

    # --- 1. 网络初始化 ---
    print("="*60)
    print("阶段 1: 网络初始化")
    print("="*60)
    net = LidarCompletionNet(resolution=resolution, activeF=activeF).to(device)
    print(f"网络已部署到设备: {device}")

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    print("网络参数统计:")
    print(f"  总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"  不可训练参数量: {non_trainable_params:,} ({non_trainable_params / 1e6:.2f}M)")

    # --- 2. 数据生成 ---
    data = generate_test_data(resolution)

    # --- 3. 张量构建与前向传播 ---
    sout = build_tensors_and_forward(net, data, resolution, device)

    # --- 4. 可视化 ---
    if sout is not None:
        visualize_point_clouds(sout, data, resolution)
    else:
        print("前向传播失败，跳过可视化。")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    main()

