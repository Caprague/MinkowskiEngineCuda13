05/10 11:30:05 设备: cuda
05/10 11:30:05 测试帧数: 500, 预热帧数: 20
05/10 11:30:05 加载补全网络...
05/10 11:30:06 补全网络参数量: 14,653,189 (14.65M)
05/10 11:30:06 加载高度图采样网络...
05/10 11:30:06 HeightMapSampler JIT compiled.
05/10 11:30:06 高度图采样网络参数量: 1,197,601 (1.198M)
05/10 11:30:06 加载数据集...
05/10 11:30:06 Scanning dataset types: ['walk_block_continuous', 'walk_mix_continuous', 'walk_stair_continuous']
05/10 11:30:07 Loaded 750 sequences for phase: train
05/10 11:30:07 开始基准测试 (20 预热 + 500 测试)...
======================================================================
  [warmup] frame 1/520 | total: 567.7ms | completion: 218.9ms | heightmap: 267.1ms | in_pts: 2635 | out_pts: 3995
  [warmup] frame 11/520 | total: 157.6ms | completion: 56.7ms | heightmap: 94.2ms | in_pts: 2371 | out_pts: 4296
  [test] frame 21/520 | total: 122.6ms | completion: 52.2ms | heightmap: 58.5ms | in_pts: 6653 | out_pts: 4330
  [test] frame 31/520 | total: 137.4ms | completion: 57.3ms | heightmap: 71.7ms | in_pts: 3275 | out_pts: 5627
  [test] frame 41/520 | total: 130.5ms | completion: 55.0ms | heightmap: 65.0ms | in_pts: 8896 | out_pts: 5946
  [test] frame 51/520 | total: 126.4ms | completion: 53.4ms | heightmap: 66.4ms | in_pts: 2707 | out_pts: 4124
  [test] frame 61/520 | total: 133.1ms | completion: 59.0ms | heightmap: 67.7ms | in_pts: 2105 | out_pts: 4836
  [test] frame 71/520 | total: 130.0ms | completion: 57.8ms | heightmap: 65.2ms | in_pts: 2742 | out_pts: 4208
  [test] frame 81/520 | total: 130.1ms | completion: 52.8ms | heightmap: 65.6ms | in_pts: 6726 | out_pts: 4273
  [test] frame 91/520 | total: 126.7ms | completion: 54.7ms | heightmap: 65.4ms | in_pts: 2612 | out_pts: 4777
  [test] frame 101/520 | total: 122.9ms | completion: 51.5ms | heightmap: 64.8ms | in_pts: 2725 | out_pts: 4300
  [test] frame 111/520 | total: 133.7ms | completion: 56.5ms | heightmap: 64.8ms | in_pts: 6422 | out_pts: 4298
  [test] frame 121/520 | total: 125.4ms | completion: 53.4ms | heightmap: 64.2ms | in_pts: 2696 | out_pts: 4242
  [test] frame 131/520 | total: 137.8ms | completion: 58.4ms | heightmap: 72.6ms | in_pts: 2862 | out_pts: 4347
  [test] frame 141/520 | total: 142.0ms | completion: 64.1ms | heightmap: 66.6ms | in_pts: 7409 | out_pts: 4522
  [test] frame 151/520 | total: 125.7ms | completion: 48.7ms | heightmap: 64.0ms | in_pts: 6379 | out_pts: 4235
  [test] frame 161/520 | total: 137.3ms | completion: 58.0ms | heightmap: 66.1ms | in_pts: 6713 | out_pts: 4337
  [test] frame 171/520 | total: 139.0ms | completion: 54.5ms | heightmap: 66.5ms | in_pts: 6882 | out_pts: 4302
  [test] frame 181/520 | total: 124.9ms | completion: 51.9ms | heightmap: 62.8ms | in_pts: 7690 | out_pts: 5302
  [test] frame 191/520 | total: 145.8ms | completion: 59.5ms | heightmap: 72.1ms | in_pts: 7175 | out_pts: 4995
  [test] frame 201/520 | total: 120.4ms | completion: 47.6ms | heightmap: 60.9ms | in_pts: 6688 | out_pts: 4647
  [test] frame 211/520 | total: 118.9ms | completion: 48.1ms | heightmap: 59.4ms | in_pts: 6734 | out_pts: 4587
  [test] frame 221/520 | total: 131.0ms | completion: 53.0ms | heightmap: 65.4ms | in_pts: 7295 | out_pts: 4632
  [test] frame 231/520 | total: 130.2ms | completion: 53.8ms | heightmap: 66.2ms | in_pts: 7045 | out_pts: 4642
  [test] frame 241/520 | total: 130.8ms | completion: 52.4ms | heightmap: 67.5ms | in_pts: 6754 | out_pts: 4292
  [test] frame 251/520 | total: 134.9ms | completion: 51.4ms | heightmap: 66.7ms | in_pts: 6859 | out_pts: 4387
  [test] frame 261/520 | total: 152.8ms | completion: 62.5ms | heightmap: 82.0ms | in_pts: 2746 | out_pts: 4291
  [test] frame 271/520 | total: 125.9ms | completion: 53.4ms | heightmap: 62.6ms | in_pts: 2678 | out_pts: 4128
  [test] frame 281/520 | total: 140.2ms | completion: 58.2ms | heightmap: 74.9ms | in_pts: 2351 | out_pts: 5705
  [test] frame 291/520 | total: 142.0ms | completion: 60.0ms | heightmap: 65.4ms | in_pts: 8081 | out_pts: 5688
  [test] frame 301/520 | total: 132.5ms | completion: 58.7ms | heightmap: 66.6ms | in_pts: 2768 | out_pts: 4073
  [test] frame 311/520 | total: 129.5ms | completion: 53.8ms | heightmap: 65.0ms | in_pts: 7056 | out_pts: 4686
  [test] frame 321/520 | total: 138.0ms | completion: 56.2ms | heightmap: 75.3ms | in_pts: 2372 | out_pts: 4447
  [test] frame 331/520 | total: 147.1ms | completion: 59.7ms | heightmap: 74.9ms | in_pts: 6280 | out_pts: 4342
  [test] frame 341/520 | total: 107.9ms | completion: 42.6ms | heightmap: 53.7ms | in_pts: 6293 | out_pts: 4301
  [test] frame 351/520 | total: 128.6ms | completion: 54.8ms | heightmap: 62.2ms | in_pts: 7851 | out_pts: 5026
  [test] frame 361/520 | total: 127.5ms | completion: 52.2ms | heightmap: 63.1ms | in_pts: 8586 | out_pts: 5419
  [test] frame 371/520 | total: 131.7ms | completion: 51.1ms | heightmap: 68.9ms | in_pts: 6405 | out_pts: 4230
  [test] frame 381/520 | total: 125.0ms | completion: 50.8ms | heightmap: 63.7ms | in_pts: 7337 | out_pts: 4936
  [test] frame 391/520 | total: 119.7ms | completion: 54.0ms | heightmap: 55.5ms | in_pts: 8804 | out_pts: 6517
  [test] frame 401/520 | total: 124.8ms | completion: 50.0ms | heightmap: 63.8ms | in_pts: 6621 | out_pts: 4385
  [test] frame 411/520 | total: 133.3ms | completion: 54.9ms | heightmap: 66.5ms | in_pts: 6556 | out_pts: 4582
  [test] frame 421/520 | total: 142.2ms | completion: 58.0ms | heightmap: 73.5ms | in_pts: 10510 | out_pts: 8399
  [test] frame 431/520 | total: 123.3ms | completion: 50.1ms | heightmap: 62.2ms | in_pts: 6873 | out_pts: 4902
  [test] frame 441/520 | total: 128.7ms | completion: 47.7ms | heightmap: 69.9ms | in_pts: 6393 | out_pts: 4115
  [test] frame 451/520 | total: 131.0ms | completion: 53.1ms | heightmap: 65.6ms | in_pts: 6912 | out_pts: 4304
  [test] frame 461/520 | total: 145.4ms | completion: 55.5ms | heightmap: 78.4ms | in_pts: 6619 | out_pts: 4136
  [test] frame 471/520 | total: 121.7ms | completion: 47.6ms | heightmap: 62.7ms | in_pts: 6826 | out_pts: 4152
  [test] frame 481/520 | total: 134.9ms | completion: 56.2ms | heightmap: 67.6ms | in_pts: 6542 | out_pts: 4661
  [test] frame 491/520 | total: 142.5ms | completion: 52.7ms | heightmap: 78.7ms | in_pts: 6575 | out_pts: 4737
  [test] frame 501/520 | total: 144.7ms | completion: 61.9ms | heightmap: 70.7ms | in_pts: 7157 | out_pts: 5100
  [test] frame 511/520 | total: 120.9ms | completion: 55.2ms | heightmap: 53.2ms | in_pts: 8474 | out_pts: 5824
  [test] frame 520/520 | total: 124.5ms | completion: 51.7ms | heightmap: 64.0ms | in_pts: 2393 | out_pts: 4271

======================================================================
  实时性能基准测试结果
======================================================================

  测试环境:
    设备:           Orin
    PyTorch:        2.1.0a0+32f93b1
    CUDA:           12.2
    批大小:         1
    高度图采样次数: 10/帧
    总测试帧数:     500
    总样本数:       40

----------------------------------------------------------------------
  各阶段耗时统计
----------------------------------------------------------------------

  [数据预处理 (体素化 + 历史帧融合)]
    样本数:   500
    均值:     10.23 ms
    中位数:   10.19 ms
    最小值:   5.10 ms
    最大值:   21.63 ms
    P95:      12.62 ms
    P99:      16.88 ms
    标准差:   1.91 ms
    平均频率: 97.75 FPS

  [补全网络推理 (LidarCompletionNet)]
    样本数:   500
    均值:     55.86 ms
    中位数:   54.53 ms
    最小值:   40.58 ms
    最大值:   84.57 ms
    P95:      69.94 ms
    P99:      78.90 ms
    标准差:   7.03 ms
    平均频率: 17.90 FPS

  [补全后处理 (反体素化 + 缓存更新)]
    样本数:   500
    均值:     1.09 ms
    中位数:   1.05 ms
    最小值:   0.83 ms
    最大值:   4.66 ms
    P95:      1.27 ms
    P99:      1.64 ms
    标准差:   0.25 ms
    平均频率: 920.65 FPS

  [高度图网络推理 (HeightMapSampler)]
    样本数:   500
    均值:     65.44 ms
    中位数:   65.08 ms
    最小值:   48.93 ms
    最大值:   84.23 ms
    P95:      75.12 ms
    P99:      79.99 ms
    标准差:   5.42 ms
    平均频率: 15.28 FPS

  [单帧总耗时 (端到端)]
    样本数:   500
    均值:     132.73 ms
    中位数:   131.46 ms
    最小值:   105.72 ms
    最大值:   188.69 ms
    P95:      152.88 ms
    P99:      166.28 ms
    标准差:   11.19 ms
    平均频率: 7.53 FPS

----------------------------------------------------------------------
  点云规模统计
----------------------------------------------------------------------

  [输入点数 (体素化后)]
    均值: 6767
    最小: 1725
    最大: 10609

  [补全输出点数]
    均值: 4759
    最小: 3618
    最大: 8399

----------------------------------------------------------------------
  综合频率
----------------------------------------------------------------------
    端到端 (含预处理):       7.53 FPS (mean 132.73 ms)
    纯推理 (补全+高度图):    8.24 FPS (mean 121.30 ms)
    补全网络单独:            17.90 FPS (mean 55.86 ms)
    高度图网络单独:          15.28 FPS (mean 65.44 ms)

----------------------------------------------------------------------
  GPU 显存使用
----------------------------------------------------------------------
    当前分配:  141.4 MB
    峰值分配:  199.8 MB
    当前缓存:  256.0 MB
    峰值缓存:  256.0 MB

======================================================================
  测试完成
======================================================================

