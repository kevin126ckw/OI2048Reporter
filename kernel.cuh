#pragma once

#include <cstdint>

// ─── GPU 内核结果 ────────────────────────────────────────────────────
struct SimResult {
    int score;  // 仅传分数，final_grid/steps 由 Host 回放生成
};

// ─── GPU 内核：并行模拟（单个 GPU 切片）────────────────────────────────
// base_seed      : 本批次的基准种子（各切片相同）
// target_score   : 达到该分数即可提前结束本局
// tid_offset     : 本切片第一个线程的全局线程号（多 GPU 时各切片连续且不重叠）
// thread_count   : 本切片的线程数（= 本切片线程块数 × blockDim.x）
// results        : 本切片的结果数组，按切片内局部下标 results[local] 存放
//
// 全局线程号 tid = tid_offset + 局部下标，策略与随机种子都只依赖 tid，
// 因此「一张卡跑全部线程」与「多张卡分摊同样的线程」结果完全一致。
__global__ void simulate_games(uint64_t base_seed, int target_score,
                               int tid_offset, int thread_count,
                               SimResult *results);
