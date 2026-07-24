#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ─── 常量 ────────────────────────────────────────────────────────────
constexpr int MAX_STEPS = 2048;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int NUM_BLOCKS = 1024;
constexpr int NUM_THREADS = THREADS_PER_BLOCK * NUM_BLOCKS; // 65536
constexpr int DEFAULT_SEARCH_BATCHES = 1024;

// ─── RNG ────────────────────────────────────────────────────────────
__host__ __device__ static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// ─── 棋盘快照 ────────────────────────────────────────────────────────
__host__ __device__ static void copy_grid(const int *src, int *dst) {
#pragma unroll
    for (int i = 0; i < 16; i++) dst[i] = src[i];
}

__host__ __device__ static bool grid_equal(const int *a, const int *b) {
#pragma unroll
    for (int i = 0; i < 16; i++) if (a[i] != b[i]) return false;
    return true;
}

__host__ __device__ static int count_empty(const int *grid) {
    int cnt = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) if (grid[i] == 0) cnt++;
    return cnt;
}

__host__ __device__ static int max_value_log2(const int *grid) {
    int mx = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) if (grid[i] > mx) mx = grid[i];
    if (mx <= 0) return 0;
    int log = 0;
    while (mx > 1) { mx >>= 1; log++; }
    return log;
}

// ─── 移动核心逻辑 ─────────────────────────────────────────────────────
// 按原版 2048 规则逐格处理一行：先滑到最远，再尝试合并
// 关键规则：
//   - 同值合并（≤32768, ≥-2）不需要邻接检查
//   - 倍增方块合并（正×负）必须原始位置相邻（is_adj）
// 返回本次行合并获得的分数
__host__ __device__ static int slide_line(int *arr) {
    int score = 0;
    bool merged[4] = {false, false, false, false};

#pragma unroll
    for (int i = 0; i < 4; i++) {
        if (arr[i] == 0) continue;

        int farthest = i;
        while (farthest > 0 && arr[farthest - 1] == 0) {
            farthest--;
        }

        const int prev = farthest - 1;
        bool did_merge = false;

        if (prev >= 0 && arr[prev] != 0 && !merged[prev]) {
            const int a = arr[prev];

            // 同值合并（不检查邻接）
            if (const int b = arr[i]; a == b && a <= 32768 && a >= -2) {
                arr[prev] = a * 2;
                score += a * 2;
                arr[i] = 0;
                merged[prev] = true;
                did_merge = true;
            }
            // 倍增方块合并：服务端 is_adj 豁免 <= -8 的方块
            else if (i == prev + 1 || a <= -8 || b <= -8) {
                if (a > 0 && b <= -1 && -b * a <= 65536) {
                    arr[prev] = -b * a;
                    score += arr[prev];
                    arr[i] = 0;
                    merged[prev] = true;
                    did_merge = true;
                } else if (a <= -1 && b > 0 && -a * b <= 65536) {
                    arr[prev] = -a * b;
                    score += arr[prev];
                    arr[i] = 0;
                    merged[prev] = true;
                    did_merge = true;
                }
            }
        }

        if (!did_merge && farthest != i) {
            arr[farthest] = arr[i];
            arr[i] = 0;
        }
    }

    return score;
}

// ─── 朝指定方向移动整个棋盘（原地修改），返回本次得分 ──────────────────
// dir: 0=上, 1=右, 2=下, 3=左
__host__ __device__ static int apply_move(int *grid, const int direction, bool *moved_out = nullptr) {
    int score = 0;
    int backup[16];
    copy_grid(grid, backup);

    if (direction == 0) {
        for (int col = 0; col < 4; col++) {
            int col_arr[4];
            for (int row = 0; row < 4; row++) col_arr[row] = grid[row * 4 + col];
            score += slide_line(col_arr);
            for (int row = 0; row < 4; row++) grid[row * 4 + col] = col_arr[row];
        }
    } else if (direction == 2) {
        for (int col = 0; col < 4; col++) {
            int col_arr[4];
            for (int row = 0; row < 4; row++) col_arr[3 - row] = grid[row * 4 + col];
            score += slide_line(col_arr);
            for (int row = 0; row < 4; row++) grid[row * 4 + col] = col_arr[3 - row];
        }
    } else if (direction == 1) {
        for (int row = 0; row < 4; row++) {
            int row_arr[4];
            for (int col = 0; col < 4; col++) row_arr[3 - col] = grid[row * 4 + col];
            score += slide_line(row_arr);
            for (int col = 0; col < 4; col++) grid[row * 4 + col] = row_arr[3 - col];
        }
    } else { // direction == 3
        for (int row = 0; row < 4; row++) {
            int row_arr[4];
            for (int col = 0; col < 4; col++) row_arr[col] = grid[row * 4 + col];
            score += slide_line(row_arr);
            for (int col = 0; col < 4; col++) grid[row * 4 + col] = row_arr[col];
        }
    }

    if (moved_out) *moved_out = !grid_equal(backup, grid);
    return score;
}

// ─── 检查是否还能走 ────────────────────────────────────────────────────
__host__ __device__ static bool can_move(const int *grid) {
    if (count_empty(grid) > 0) return true;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 3; c++) {
            const int a = grid[r * 4 + c];
            const int b = grid[r * 4 + c + 1];
            if (a <= -1 && b > 0 && -a * b <= 65536) return true;
            if (a > 0 && b <= -1 && -b * a <= 65536) return true;
            if (a == b && a <= 32768 && a >= -2) return true;
        }
    }
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 3; r++) {
            const int a = grid[r * 4 + c];
            const int b = grid[(r + 1) * 4 + c];
            if (a <= -1 && b > 0 && -a * b <= 65536) return true;
            if (a > 0 && b <= -1 && -b * a <= 65536) return true;
            if (a == b && a <= 32768 && a >= -2) return true;
        }
    }
    // 检查非相邻的倍率合并（<= -8 豁免规则）
    for (int r = 0; r < 4; r++) {
        for (int c1 = 0; c1 < 4; c1++) {
            const int a = grid[r * 4 + c1];
            if (a == 0) continue;
            for (int c2 = c1 + 2; c2 < 4; c2++) {
                const int b = grid[r * 4 + c2];
                if (b == 0) continue;
                if (a <= -1 && a <= -8 && b > 0 && -a * b <= 65536) return true;
                if (b <= -1 && b <= -8 && a > 0 && -b * a <= 65536) return true;
            }
        }
    }
    for (int c = 0; c < 4; c++) {
        for (int r1 = 0; r1 < 4; r1++) {
            const int a = grid[r1 * 4 + c];
            if (a == 0) continue;
            for (int r2 = r1 + 2; r2 < 4; r2++) {
                const int b = grid[r2 * 4 + c];
                if (b == 0) continue;
                if (a <= -1 && a <= -8 && b > 0 && -a * b <= 65536) return true;
                if (b <= -1 && b <= -8 && a > 0 && -b * a <= 65536) return true;
            }
        }
    }
    return false;
}

// ─── 随机生成新方块 ────────────────────────────────────────────────────
__host__ __device__ static void add_random_tile(int *grid, uint32_t *rng, const bool is_start = false) {
    int empty[16];
    int cnt = 0;
    for (int i = 0; i < 16; i++) if (grid[i] == 0) empty[cnt++] = i;
    if (cnt == 0) return;
    const int pos = empty[xorshift32(rng) % cnt];

    if (const uint32_t r = xorshift32(rng) % 100; is_start || r < 87) {
        grid[pos] = xorshift32(rng) % 10 < 9 ? 2 : 4;
    } else {
        grid[pos] = xorshift32(rng) % 100 < 86 ? -1 : -2;
    }
}

// ─── log2 快速计算（GPU 用硬件 __clz，CPU 回退到循环）─────────────────────
__host__ __device__ static int ilog2(int v) {
#if defined(__CUDA_ARCH__)
    return 31 - __clz((unsigned int)v);
#else
    int log = 0;
    while (v > 1) { v >>= 1; log++; }
    return log;
#endif
}
