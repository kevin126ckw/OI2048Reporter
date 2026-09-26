# OI2048Reporter

OI-2048 游戏的 GPU 并行求解器与自动提交工具。使用 CUDA C++ 在 GPU 上并行运行数千万局游戏（**支持多 GPU 并行**），通过启发式评估函数搜索高分策略，并自动提交到在线排行榜。

## 游戏简介

OI-2048 是经典 2048 的变体，增加了**负数倍率方块**：

- 正数方块：2, 4, 8, 16, ..., 65536
- 负数方块：-1, -2（约 13% 概率生成）
- **倍率合并**：正数 × 负数 = 结果取正（如 4 × (-2) = -8，计为 8 分）
- ≤-8 的负数方块可跳过空格与非相邻正数合并

## 工作原理

1. **GPU 并行搜索**：每批 262,144 个线程（可扩展到多张 GPU），每局独立模拟
2. **4 种策略**：分别将最大数推向 4 个角（左上、右上、左下、右下）
3. **启发式评估**：综合考虑位置权重、空格数量、单调性、平滑度、合并潜力
4. **选出最佳**：取所有模拟中得分最高的一局
5. **Host 回放**：用相同种子重现最优对局，生成完整历史记录
6. **JSON 输出**：写入 `submit_output.json`
7. **提交排行**：通过 `submit.py` 验证并 POST 到服务器

## 环境要求

- **CUDA Toolkit**（NVCC 编译器）
- **CMake ≥ 3.18**
- **GCC**（支持 C++17）
- NVIDIA GPU（Compute Capability ≥ 5.0），多张 GPU 可选（自动全部使用）

Python 提交脚本额外需要：
- Python 3
- `requests` 库

## 编译

```bash
cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

若使用 CLion，直接用 IDE 打开即可自动配置 CMake。

## 使用

### 1. 搜索并生成记录

```bash
./cmake-build-release/OI2048Reporter [目标分数] [搜索批数] [选项]
```

- `目标分数`：达到此分数后提前终止搜索
- `搜索批数`：搜索批数（默认 1024），每批固定 262,144 局游戏，由所有 GPU 共同分担
- 省略位置参数则进入交互模式；结果写入 `submit_output.json`

输出示例（双卡，128 SM + 82 SM）：

```
检测到 2 个 CUDA 设备
目标分数: 551144
搜索批数: 1024
使用 2 个 GPU 切片并行搜索 (2 张卡), 共 262144 个线程:
  [0] GPU 0 NVIDIA GeForce RTX 4090 (128 SM) → 624 块 / 159744 线程
  [1] GPU 1 NVIDIA GeForce RTX 3090 (82 SM) → 400 块 / 102400 线程
随机种子: 1758888888
GPU 内存分配完成
启动批次 0/1024...
批次 0: 本批最高 482300 分, 累计最高 482300 分
批次 1: 本批最高 524100 分, 累计最高 524100 分
...
批次 5: 达到目标分 551144 [已达标!], 本批最高 552000 分, 累计最高 552000 分
搜索完成: 6 批次, 共 1572864 局游戏, 用时 3.60 秒 (436906 局/秒)

========== 生成结果 ==========
分数: 552000
步数: 1198
最大方块 log2: 16
最终棋盘:
    _    4    2    2
    2   16   64   32
    4 2048  128    8
 1024  256 4096 65536

完整 JSON 已写入 submit_output.json (xxx 字节)
```

### 2. 多 GPU 并行

**默认自动使用所有可见 GPU**（`CUDA_VISIBLE_DEVICES` 依然生效）：

```bash
./cmake-build-release/OI2048Reporter 551144 1024              # 全部 GPU
./cmake-build-release/OI2048Reporter 551144 1024 --gpus 0,1   # 只用 0、1 号卡
./cmake-build-release/OI2048Reporter 551144 1024 --gpus 0-3   # 区间写法
./cmake-build-release/OI2048Reporter --list-devices           # 查看可见设备后退出
```

工作原理：

- 每批的 262,144 个线程按**线程块**切分给各 GPU，各卡同时跑同一批次的不同线程段，双缓冲流水线照旧；
- 分配比例按 **SM 数量**加权（最大余数法，纯整数运算，结果确定）；两张 58 SM 的卡会把 1024 块均分为 512 / 512；
- 线程的随机种子和策略**只由全局线程号 `tid` 决定**，因此 1 张卡与 N 张卡搜索到的逐线程结果**完全一致**（见下方测试），多卡只是更快；
- 某张卡初始化或内核启动失败时自动跳过该切片，其余 GPU 继续搜索，并在输出中提示。

常用选项：

| 选项                    | 说明                                                             |
|-------------------------|------------------------------------------------------------------|
| `-g, --gpus <列表>`     | 选择 GPU，如 `0,1`、`0-2`、`all`（默认 `all`）                   |
| `--logical-devices <n>` | 把每张卡当作 n 个逻辑切片（默认 1），单卡机器上验证多卡调度用    |
| `--list-devices`        | 列出可见 GPU（序号、计算能力、SM 数、显存）后退出                |
| `--seed <整数>`         | 固定基准随机种子，便于复现和对比测试                             |
| `--dump-scores <文件>`  | 导出每个批次全部线程的分数（二进制，测试用）                     |
| `-h, --help`            | 帮助                                                             |

> **提示**：CMake 使用 `-arch=native`，会为**编译时可见的所有 GPU** 生成代码。如果构建机器上看不到某张卡（例如换机器运行、或在 CI 里构建），请显式指定架构，例如
> `cmake -B build -DCMAKE_CUDA_ARCHITECTURES="75;86;89"`。否则缺少内核镜像的卡会在启动时报错并被自动跳过（其余 GPU 继续工作）。

### 3. 提交到排行榜

```bash
pip install requests        # 首次需要安装依赖
python3 submit.py
```

脚本会验证游戏记录的合法性，确认无误后提交到服务器。

### 4. 调整搜索强度

- **搜索批数**：命令行第二个参数（默认 `DEFAULT_SEARCH_BATCHES` = 1024，定义在 `game_logic.cuh`）；每批固定 `NUM_THREADS` = 262,144 局游戏
- **每批线程数**：`NUM_BLOCKS` / `THREADS_PER_BLOCK`（`game_logic.cuh`），多卡时自动按 SM 比例分摊
- **单局最大步数**：`MAX_STEPS`（`game_logic.cuh`）

> **注意**：单卡 58 SM 大约 43 万局/秒（约 0.6 秒/批），多卡接近线性加速。

## 测试

```bash
cmake --build cmake-build-release
cd cmake-build-release && ctest --output-on-failure
```

- `gpu_plan`：纯 CPU 单元测试（不需要 GPU），验证 GPU 选择器解析与线程块划分的完整性、确定性、负载均衡性质
- `gpu_equivalence`：多卡等价性测试——固定种子下，「1 个切片」与「2/3/4 个逻辑切片」的逐线程分数必须**逐字节一致**；无 GPU 环境自动跳过

单卡机器上验证多卡调度路径：

```bash
./cmake-build-release/OI2048Reporter 551144 64 --logical-devices 4
```

## 文件结构

```
├── main.cu              # 入口：命令行/交互输入 → 多 GPU 双缓冲流水线搜索 → Top-K expectimax → JSON
├── kernel.cuh/.cu       # GPU 内核：单个 GPU 切片的并行模拟
├── gpu_plan.cuh/.cu     # 多 GPU 计划：设备枚举、--gpus 解析、线程块加权划分（.cuh 为纯 C++，可单测）
├── game_logic.cuh       # 游戏规则与常量（Device + Host）
├── evaluate.cuh         # 启发式评估函数
├── replay.cuh/.cu       # Host 端回放与 expectimax 深搜
├── submit_format.cuh/.cu# 提交 JSON 生成
├── tests/               # 单元测试 + 多 GPU 等价性测试
├── submit.py            # Python 提交脚本：验证记录并 POST 到排行榜
├── CMakeLists.txt       # CMake 构建配置
├── submit_output.json   # 生成的 JSON 输出（.gitignore 已忽略）
└── README.md
```

## 策略细节

评估函数考虑的维度：

| 维度         | 权重             | 说明                             |
|--------------|------------------|----------------------------------|
| 位置权重     | 指数衰减         | 越靠近目标角的正数方块得分越高   |
| 空格奖励     | 24 + log2(max)×3 | 空格越多越好，后期每个空格更珍贵 |
| 行/列单调性  | ±log2×0.5        | 方块值沿目标角方向单调递增加分   |
| 平滑度       | -                | Δlog2                            |×0.8 | 相邻方块值差距过大扣分 |
| 倍率合并潜力 | log2(result)×6~8 | 负倍率方块邻近正数时的潜在收益   |
| 普通合并潜力 | log2×2           | 相邻正数相等时的潜在收益         |

## License

见LICENSE文件
