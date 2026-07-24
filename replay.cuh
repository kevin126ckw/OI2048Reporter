#pragma once

#include <vector>

#include "game_logic.cuh"

// ─── Host 端回放数据结构 ──────────────────────────────────────────────
struct HistoryEntry {
    int before[16];
    int after[16];
};

struct HostSimResult {
    int score{};
    int steps{};
    std::vector<HistoryEntry> history;
    int final_grid[16]{};
};

// ─── Host 端回放函数 ──────────────────────────────────────────────────
HostSimResult replay_game(uint32_t rng_seed, int strategy);
HostSimResult replay_game_expectimax(uint32_t rng_seed, int strategy);
