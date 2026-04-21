#!/usr/bin/env python3
"""
数据集预处理脚本：对 Meta 下的时序点云数据集进行连续裁剪分割。

按 seq_len 帧一组，连续切分原序列，不足 seq_len 的尾部丢弃。
处理后的数据保存到 DataCollection/Processed/，目录结构与 Meta 一致。

用法示例：
  python scripts/preprocess_dataset.py --seq_len 20
"""

import os
import glob
import shutil
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def get_sorted_pcd_paths(directory):
    """获取目录下所有 .pcd 文件，按文件名排序返回列表"""
    paths = sorted(glob.glob(os.path.join(directory, "*.pcd")))
    return paths


def slice_transform(src_npz_path, indices, dst_npz_path):
    """
    从源 npz 中按 indices 切片 pos/quat，保存到目标 npz。
    indices: 要保留的帧索引列表（基于原始序列的0-indexed位置）
    """
    data = np.load(src_npz_path)
    pos = data["pos"]    # (N, 3)
    quat = data["quat"]  # (N, 4)

    new_pos = pos[indices]
    new_quat = quat[indices]

    np.savez(dst_npz_path, pos=new_pos, quat=new_quat)


def process_continuous(src_root, dst_root, seq_len):
    """
    对每个原始序列，按 seq_len 帧一组连续切分，不足 seq_len 的尾部丢弃。
    """
    logging.info(f"[seq_len={seq_len}] 开始连续裁剪...")
    total_new_seqs = 0

    # 遍历 type/phase 层级
    for type_dir in sorted(glob.glob(os.path.join(src_root, "*/"))):
        type_name = os.path.basename(os.path.normpath(type_dir))
        out_type_name = f"{type_name}_continuous"
        for phase_dir in sorted(glob.glob(os.path.join(type_dir, "*/"))):
            phase_name = os.path.basename(os.path.normpath(phase_dir))

            partial_root = os.path.join(phase_dir, "partial")
            complete_root = os.path.join(phase_dir, "complete")
            transform_root = os.path.join(phase_dir, "transform")

            if not os.path.exists(partial_root):
                continue

            # 遍历每个原始序列
            for seq_dir in sorted(glob.glob(os.path.join(partial_root, "*/"))):
                seq_name = os.path.basename(os.path.normpath(seq_dir))

                p_pcds = get_sorted_pcd_paths(os.path.join(partial_root, seq_name))
                c_pcds = get_sorted_pcd_paths(os.path.join(complete_root, seq_name))
                t_npz = os.path.join(transform_root, f"{seq_name}.npz")

                num_frames = len(p_pcds)
                if num_frames < seq_len:
                    logging.warning(
                        f"跳过 {type_name}/{phase_name}/{seq_name}: "
                        f"帧数 {num_frames} < seq_len {seq_len}"
                    )
                    continue

                # 连续切分
                num_chunks = num_frames // seq_len
                for chunk_idx in range(num_chunks):
                    start = chunk_idx * seq_len
                    end = start + seq_len
                    indices = list(range(start, end))

                    new_seq_name = f"{total_new_seqs:04d}"

                    # 创建目标目录
                    dst_partial_seq = os.path.join(
                        dst_root, out_type_name, phase_name, "partial", new_seq_name
                    )
                    dst_complete_seq = os.path.join(
                        dst_root, out_type_name, phase_name, "complete", new_seq_name
                    )
                    dst_transform_dir = os.path.join(
                        dst_root, out_type_name, phase_name, "transform"
                    )
                    os.makedirs(dst_partial_seq, exist_ok=True)
                    os.makedirs(dst_complete_seq, exist_ok=True)
                    os.makedirs(dst_transform_dir, exist_ok=True)

                    # 复制 pcd 文件（重新编号 000~）
                    for new_idx, orig_idx in enumerate(indices):
                        # partial
                        src_pcd = p_pcds[orig_idx]
                        dst_pcd = os.path.join(dst_partial_seq, f"{new_idx:03d}.pcd")
                        shutil.copy2(src_pcd, dst_pcd)
                        # complete
                        src_pcd = c_pcds[orig_idx]
                        dst_pcd = os.path.join(dst_complete_seq, f"{new_idx:03d}.pcd")
                        shutil.copy2(src_pcd, dst_pcd)

                    # 切片 transform
                    dst_npz = os.path.join(dst_transform_dir, f"{new_seq_name}.npz")
                    slice_transform(t_npz, indices, dst_npz)

                    total_new_seqs += 1

            logging.info(
                f"  {type_name}/{phase_name}: 已生成 {total_new_seqs} 个新序列"
            )

    logging.info(f"完成，共生成 {total_new_seqs} 个新序列")
    return total_new_seqs


def main():
    parser = argparse.ArgumentParser(
        description="时序点云数据集预处理：连续裁剪分割"
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default="./DataCollection/Meta",
        help="源数据集根目录",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="./DataCollection/Processed",
        help="输出数据集根目录",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=20,
        help="新序列长度（每组帧数）",
    )

    args = parser.parse_args()

    if not os.path.exists(args.src_root):
        raise FileNotFoundError(f"源数据集目录不存在: {args.src_root}")

    # 创建输出根目录
    os.makedirs(args.dst_root, exist_ok=True)

    logging.info(f"源目录: {args.src_root}")
    logging.info(f"输出目录: {args.dst_root}")
    logging.info(f"参数: seq_len={args.seq_len}")

    process_continuous(args.src_root, args.dst_root, args.seq_len)

    logging.info("全部处理完成！")


if __name__ == "__main__":
    main()
