#pragma once

#include <cstdint>

// ─── GPU 内核结果 ────────────────────────────────────────────────────
struct SimResult {
    int score;  // 仅传分数，final_grid/steps 由 Host 回放生成
};

// ─── GPU 内核：并行模拟 ────────────────────────────────────────────────
__global__ void simulate_games(uint64_t base_seed, int target_score,
                               SimResult *results);
