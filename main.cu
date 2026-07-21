#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <random>

//constexpr int GRID_SIZE = 4;
constexpr int MAX_STEPS = 2048;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int NUM_BLOCKS = 1024;
constexpr int NUM_THREADS = THREADS_PER_BLOCK * NUM_BLOCKS; // 65536
constexpr int DEFAULT_SEARCH_BATCHES = 1024;

using std::string;

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

        int prev = farthest - 1;
        bool did_merge = false;

        if (prev >= 0 && arr[prev] != 0 && !merged[prev]) {
            int a = arr[prev];
            int b = arr[i];

            // 同值合并（不检查邻接）
            if (a == b && a <= 32768 && a >= -2) {
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
__host__ __device__ static int apply_move(int *grid, int direction, bool *moved_out = nullptr) {
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
            int a = grid[r * 4 + c];
            int b = grid[r * 4 + c + 1];
            if (a <= -1 && b > 0 && (-a) * b <= 65536) return true;
            if (a > 0 && b <= -1 && (-b) * a <= 65536) return true;
            if (a == b && a <= 32768 && a >= -2) return true;
        }
    }
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 3; r++) {
            int a = grid[r * 4 + c];
            int b = grid[(r + 1) * 4 + c];
            if (a <= -1 && b > 0 && (-a) * b <= 65536) return true;
            if (a > 0 && b <= -1 && (-b) * a <= 65536) return true;
            if (a == b && a <= 32768 && a >= -2) return true;
        }
    }
    // 检查非相邻的倍率合并（<= -8 豁免规则）
    for (int r = 0; r < 4; r++) {
        for (int c1 = 0; c1 < 4; c1++) {
            int a = grid[r * 4 + c1];
            if (a == 0) continue;
            for (int c2 = c1 + 2; c2 < 4; c2++) {
                int b = grid[r * 4 + c2];
                if (b == 0) continue;
                if (a <= -1 && a <= -8 && b > 0 && (-a) * b <= 65536) return true;
                if (b <= -1 && b <= -8 && a > 0 && (-b) * a <= 65536) return true;
            }
        }
    }
    for (int c = 0; c < 4; c++) {
        for (int r1 = 0; r1 < 4; r1++) {
            int a = grid[r1 * 4 + c];
            if (a == 0) continue;
            for (int r2 = r1 + 2; r2 < 4; r2++) {
                int b = grid[r2 * 4 + c];
                if (b == 0) continue;
                if (a <= -1 && a <= -8 && b > 0 && (-a) * b <= 65536) return true;
                if (b <= -1 && b <= -8 && a > 0 && (-b) * a <= 65536) return true;
            }
        }
    }
    return false;
}

// ─── 随机生成新方块 ────────────────────────────────────────────────────
__host__ __device__ static void add_random_tile(int *grid, uint32_t *rng, bool is_start = false) {
    int empty[16];
    int cnt = 0;
    for (int i = 0; i < 16; i++) if (grid[i] == 0) empty[cnt++] = i;
    if (cnt == 0) return;
    int pos = empty[xorshift32(rng) % cnt];

    uint32_t r = xorshift32(rng) % 100;
    if (is_start || r < 87) {
        grid[pos] = (xorshift32(rng) % 10 < 9) ? 2 : 4;
    } else {
        grid[pos] = (xorshift32(rng) % 100 < 86) ? -1 : -2;
    }
}

// ─── log2 快速计算（GPU 用硬件 __clz，CPU 回退到循环）─────────────────────
__host__ __device__ static inline int ilog2(int v) {
#if defined(__CUDA_ARCH__)
    return 31 - __clz((unsigned int)v);
#else
    int log = 0;
    while (v > 1) { v >>= 1; log++; }
    return log;
#endif
}

// ─── 整数评估常量（×1024 定点数，确保 GPU/CPU 结果完全一致）─────────────
enum : int {
    EVAL_NEG_W8      = 10240,  // 10.0
    EVAL_NEG_W_OTHER  = 6144,   // 6.0
    EVAL_SMOOTH       = 819,    // 0.8
    EVAL_MONO         = 512,    // 0.5
    EVAL_MERGE        = 2048,   // 2.0
    EVAL_MONO_EXTRA   = 82,     // 0.08
    EVAL_EMPTY_BASE   = 24576,  // 24.0
    EVAL_EMPTY_COEF   = 358,    // 0.35
    EVAL_CORNER_MUL   = 2048,   // 2.0
    EVAL_DEAD_PENALTY = 30720,  // 30.0
};
// dist_w 定点数: {12.0, 7.44, 4.6128, 2.859936, 1.77316032, 1.09935936, 0.68160288} * 1024
__host__ __device__ static const int dist_w_int[7] = {12288, 7619, 4724, 2929, 1816, 1126, 698};

// ─── OI-2048 纯整数评估（GPU/CPU 结果完全一致）────────────────────────
__host__ __device__ static int evaluate(const int *grid, const int *pos_w, int cr, int cc, int = 0) {
    int log_grid[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int v = grid[i];
        log_grid[i] = (v != 0) ? ((v > 0) ? ilog2(v) : ilog2(-v)) : 0;
    }

    int score = 0;
    int empty_cnt = 0;
    int max_val = 0;
    int max_log = 0;
    int max_pos_r = -1;
    int max_pos_c = -1;
    int row_sign = (cc == 3) ? -1 : 1;
    int col_sign = (cr == 3) ? -1 : 1;

#pragma unroll
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int c = 0; c < 4; c++) {
            int idx = r * 4 + c;
            int v = grid[idx];
            int log_v = log_grid[idx];
            bool v_pos = false;

            if (v == 0) {
                empty_cnt++;
            } else {
                v_pos = (v > 0);
                if (v_pos) {
                    score += log_v * pos_w[idx];
                    if (v > max_val) { max_val = v; max_log = log_v; max_pos_r = r; max_pos_c = c; }
                } else {
                    int best_nbr = 0;
                    if (r > 0 && grid[idx - 4] > 0) best_nbr = max(best_nbr, grid[idx - 4]);
                    if (r < 3 && grid[idx + 4] > 0) best_nbr = max(best_nbr, grid[idx + 4]);
                    if (c > 0 && grid[idx - 1] > 0) best_nbr = max(best_nbr, grid[idx - 1]);
                    if (c < 3 && grid[idx + 1] > 0) best_nbr = max(best_nbr, grid[idx + 1]);
                    if (v <= -8) {
                        for (int c2 = 0; c2 < 4; c2++) {
                            int nbr = grid[r * 4 + c2];
                            if (nbr > 0 && nbr > best_nbr) best_nbr = nbr;
                        }
                        for (int r2 = 0; r2 < 4; r2++) {
                            int nbr = grid[r2 * 4 + c];
                            if (nbr > 0 && nbr > best_nbr) best_nbr = nbr;
                        }
                    }
                    if (best_nbr > 0) {
                        if (v <= -8 && (-v) * best_nbr > 65536) { /* skip */ }
                        else {
                            int w = (v <= -8) ? EVAL_NEG_W8 : EVAL_NEG_W_OTHER;
                            score += (log_v + ilog2(best_nbr)) * w;
                        }
                    }
                }
            }

            if (c < 3) {
                int b = grid[idx + 1];
                if (v_pos && b > 0) {
                    int lb = log_grid[idx + 1];
                    int diff = log_v - lb;
                    if (diff < 0) diff = -diff;
                    score -= diff * EVAL_SMOOTH;
                    score += (v >= b ? 1 : -1) * row_sign * log_v * EVAL_MONO;
                    if (v == b) score += log_v * EVAL_MERGE;
                }
            }

            if (r < 3) {
                int b = grid[idx + 4];
                if (v_pos && b > 0) {
                    int lb = log_grid[idx + 4];
                    int diff = log_v - lb;
                    if (diff < 0) diff = -diff;
                    score -= diff * EVAL_SMOOTH;
                    score += (v >= b ? 1 : -1) * col_sign * log_v * EVAL_MONO;
                    if (v == b) score += log_v * EVAL_MERGE;
                }
            }
        }
    }

    int mono_extra = (max_log - 8) * EVAL_MONO_EXTRA;
    if (mono_extra > 0) {
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 3; c++) {
                int idx = r * 4 + c;
                int a = grid[idx], b = grid[idx + 1];
                if (a > 0 && b > 0)
                    score += (a >= b ? 1 : -1) * row_sign * log_grid[idx] * mono_extra;
            }
        }
        for (int c = 0; c < 4; c++) {
            for (int r = 0; r < 3; r++) {
                int idx = r * 4 + c;
                int a = grid[idx], b = grid[idx + 4];
                if (a > 0 && b > 0)
                    score += (a >= b ? 1 : -1) * col_sign * log_grid[idx] * mono_extra;
            }
        }
    }

    int empty_bonus = EVAL_EMPTY_BASE;
    if (max_val > 0) empty_bonus += max_log * max_log * EVAL_EMPTY_COEF;
    score += empty_cnt * empty_bonus;

    if (max_pos_r >= 0 && (max_pos_r != cr || max_pos_c != cc)) {
        int dist = abs(max_pos_r - cr) + abs(max_pos_c - cc);
        score -= max_log * dist * EVAL_CORNER_MUL;
    }

    if (empty_cnt <= 3 && max_log >= 10) {
        bool has_merge = false;
        for (int i = 0; i < 16 && !has_merge; i++)
            if (grid[i] <= -8) has_merge = true;
        for (int r = 0; r < 4 && !has_merge; r++)
            for (int c = 0; c < 3 && !has_merge; c++) {
                int a = grid[r * 4 + c], b = grid[r * 4 + c + 1];
                if (a > 0 && a == b) has_merge = true;
                if (a > 0 && b < 0 && (-b) * a <= 65536) has_merge = true;
                if (a < 0 && b > 0 && (-a) * b <= 65536) has_merge = true;
            }
        for (int c = 0; c < 4 && !has_merge; c++)
            for (int r = 0; r < 3 && !has_merge; r++) {
                int a = grid[r * 4 + c], b = grid[(r + 1) * 4 + c];
                if (a > 0 && a == b) has_merge = true;
                if (a > 0 && b < 0 && (-b) * a <= 65536) has_merge = true;
                if (a < 0 && b > 0 && (-a) * b <= 65536) has_merge = true;
            }
        if (!has_merge) score -= max_log * EVAL_DEAD_PENALTY;
    }

    return score;
}

namespace {
    // ─── GPU 内核：并行模拟 ────────────────────────────────────────────────
    struct SimResult {
        int score;  // 仅传分数，final_grid/steps 由 Host 回放生成
    };
}

__global__ __launch_bounds__(256, 2) static void simulate_games(uint64_t base_seed,
        int target_score, SimResult *results) {
    int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= NUM_THREADS) return;

    int strategy = tid & 3;
    int cr = (strategy >= 2) ? 3 : 0;
    int cc = (strategy == 1 || strategy == 3) ? 3 : 0;
    int pos_w[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int r = i >> 2, c = i & 3;
        pos_w[i] = dist_w_int[abs(r - cr) + abs(c - cc)];
    }

    auto rng = static_cast<uint32_t>(base_seed + tid * 2654435761ULL);
    int grid[16] = {0};

    add_random_tile(grid, &rng, true);
    add_random_tile(grid, &rng, true);

    int score = 0;
    int steps = 0;

    while (true) {
        int best_eval = -2000000000;
        int best_dir = -1;
        int best_temp[16];
        int best_move_score = 0;

#pragma unroll
        for (int dir = 0; dir < 4; dir++) {
            int temp[16];
            copy_grid(grid, temp);
            bool moved;
            int move_score = apply_move(temp, dir, &moved);
            if (!moved) continue;
            int e = evaluate(temp, pos_w, cr, cc);
            if (e > best_eval) {
                best_eval = e;
                best_dir = dir;
                copy_grid(temp, best_temp);
                best_move_score = move_score;
            }
        }

        if (best_dir < 0) break;
        if (steps >= MAX_STEPS) break;

        copy_grid(best_temp, grid);
        score += best_move_score;
        add_random_tile(grid, &rng);
        steps++;

        if (score >= target_score) break;
    }

    results[tid].score = score;
}

namespace {
    // ─── Host 端回放函数 ──────────────────────────────────────────────────
    struct HistoryEntry {
        int before[16];
        int after[16];
    };
}

namespace {
    struct HostSimResult {
        int score{};
        int steps{};
        std::vector<HistoryEntry> history;
        int final_grid[16]{};
    };
}

static HostSimResult replay_game(uint32_t rng_seed, int strategy, int target_score = -1) {
    HostSimResult res;
    res.score = 0;
    res.steps = 0;
    res.history.clear();

    int cr = (strategy >= 2) ? 3 : 0;
    int cc = (strategy == 1 || strategy == 3) ? 3 : 0;
    int pos_w[16];
    for (int i = 0; i < 16; i++) {
        int r = i >> 2, c = i & 3;
        pos_w[i] = dist_w_int[abs(r - cr) + abs(c - cc)];
    }

    uint32_t rng = rng_seed;
    int grid[16] = {0};

    // 初始状态: before=全空, after=初始两方块
    HistoryEntry init{};
    for (int & i : init.before) i = 0;
    add_random_tile(grid, &rng, true);
    add_random_tile(grid, &rng, true);
    copy_grid(grid, init.after);
    res.history.push_back(init);

    while (can_move(grid) && res.steps < MAX_STEPS) {
        int best_eval = -2000000000;
        int best_dir = -1;

        for (int dir = 0; dir < 4; dir++) {
            int temp[16];
            copy_grid(grid, temp);
            bool moved;
            apply_move(temp, dir, &moved);
            if (!moved) continue;
            int e = evaluate(temp, pos_w, cr, cc);
            if (e > best_eval) {
                best_eval = e;
                best_dir = dir;
            }
        }

        if (best_dir < 0) break;

        HistoryEntry entry{};
        res.score += apply_move(grid, best_dir);
        copy_grid(grid, entry.before);  // 移动后、新方块前
        add_random_tile(grid, &rng);
        copy_grid(grid, entry.after);   // 移动后 + 新方块
        res.history.push_back(entry);
        res.steps++;

        // GPU 模拟用 target_score 找到达标的种子，
        // 但 Host 回放需要走到自然结束（游戏规则要求通关或无合法移动才能提交）
    }

    copy_grid(grid, res.final_grid);
    return res;
}

// ─── CPU expectimax 深搜（1-ply 前向搜索，采样空格 + 加权平均）─────────
// 限制采样数以保持速度，使用随机采样避免位置偏差
static int choose_move_expectimax(const int *grid, int cr, int cc, const int *pos_w) {
    int best_expected = -2000000000;
    int best_dir = -1;

    for (int dir = 0; dir < 4; dir++) {
        int temp[16];
        copy_grid(grid, temp);
        bool moved;
        apply_move(temp, dir, &moved);
        if (!moved) continue;

        // 找到移动后的空格
        int empty[16];
        int empty_cnt = 0;
        for (int i = 0; i < 16; i++) if (temp[i] == 0) empty[empty_cnt++] = i;

        if (empty_cnt == 0) {
            int e = evaluate(temp, pos_w, cr, cc);
            if (e > best_expected) { best_expected = e; best_dir = dir; }
            continue;
        }

        // 跳跃采样（步长 ≥1，确保覆盖棋盘各区域，最多 5 个样本）
        int step = (empty_cnt <= 5) ? 1 : (empty_cnt / 5);
        int samples = 0;
        int expected_sum = 0;

        for (int si = 0; si < empty_cnt && samples < 5; si += step) {
            int after_tile[16];
            copy_grid(temp, after_tile);
            after_tile[empty[si]] = 2; // 最常见生成 (~78%)

            // 玩家最佳应对（随机方块放置后的最佳移动）
            int best_response = -2000000000;
            for (int d2 = 0; d2 < 4; d2++) {
                int temp2[16];
                copy_grid(after_tile, temp2);
                bool moved2;
                apply_move(temp2, d2, &moved2);
                if (!moved2) continue;
                int e = evaluate(temp2, pos_w, cr, cc);
                if (e > best_response) best_response = e;
            }
            if (best_response < -1900000000)
                best_response = evaluate(after_tile, pos_w, cr, cc);
            expected_sum += best_response;
            samples++;
        }
        int expected = expected_sum / samples;

        if (expected > best_expected) {
            best_expected = expected;
            best_dir = dir;
        }
    }

    return best_dir;
}

static HostSimResult replay_game_expectimax(uint32_t rng_seed, int strategy, int target_score = -1) {
    HostSimResult res;
    res.score = 0;
    res.steps = 0;
    res.history.clear();

    int cr = (strategy >= 2) ? 3 : 0;
    int cc = (strategy == 1 || strategy == 3) ? 3 : 0;
    int pos_w[16];
    for (int i = 0; i < 16; i++) {
        int r = i >> 2, c = i & 3;
        pos_w[i] = dist_w_int[abs(r - cr) + abs(c - cc)];
    }

    uint32_t rng = rng_seed;
    int grid[16] = {0};

    HistoryEntry init{};
    for (int &i : init.before) i = 0;
    add_random_tile(grid, &rng, true);
    add_random_tile(grid, &rng, true);
    copy_grid(grid, init.after);
    res.history.push_back(init);

    while (can_move(grid) && res.steps < MAX_STEPS) {
        int best_dir = choose_move_expectimax(grid, cr, cc, pos_w);

        if (best_dir < 0) break;

        HistoryEntry entry{};
        res.score += apply_move(grid, best_dir);
        copy_grid(grid, entry.before);
        add_random_tile(grid, &rng);
        copy_grid(grid, entry.after);
        res.history.push_back(entry);
        res.steps++;
    }

    copy_grid(grid, res.final_grid);
    return res;
}

// ─── 随机字符串生成 ────────────────────────────────────────────────────
static string generate_gameid() {
    static constexpr char charset[] = "0123456789qwertyuiopasdfghjklzxcvbnm";
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_int_distribution<int> dist(0, 35);
    string id;
    for (int i = 0; i < 16; i++) id += charset[dist(rng)];
    return id;
}

// ─── JSON 格式化 ──────────────────────────────────────────────────────
static string grid_to_json(const int *grid) {
    string s = "[";
    for (int r = 0; r < 4; r++) {
        s += '[';
        for (int c = 0; c < 4; c++) {
            int v = grid[r * 4 + c];
            if (v == 0) s += "null";
            else s += std::to_string(v);
            if (c < 3) s += ',';
        }
        s += ']';
        if (r < 3) s += ',';
    }
    s += ']';
    return s;
}

static string history_to_json(const std::vector<HistoryEntry> &history) {
    string s = "[";
    for (size_t i = 0; i < history.size(); i++) {
        s += '[';
        s += grid_to_json(history[i].before);
        s += ',';
        s += grid_to_json(history[i].after);
        s += ']';
        if (i + 1 < history.size()) s += ',';
    }
    s += ']';
    return s;
}

static string generate_submit_data(const HostSimResult &result, bool cheated = false) {
    string gameid = generate_gameid();
    int mv = max_value_log2(result.final_grid);
    string fg_json = grid_to_json(result.final_grid);

    string data = "{";
    data += R"("gameid":")" + gameid + "\",";
    data += R"("finalGrid":")" + fg_json + "\",";
    data += "\"maxValueLog\":" + std::to_string(mv) + ",";
    data += "\"steps\":" + std::to_string(result.steps) + ",";
    data += R"("history":")" + history_to_json(result.history) + "\",";
    data += "\"score\":" + std::to_string(result.score) + ",";
    data += R"("scoreStringForReference":")" + std::to_string(result.score) + "\",";
    data += "\"cheated\":" + string(cheated ? "true" : "false");
    data += '}';
    return data;
}

// ─── main ────────────────────────────────────────────────────────────
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA 错误 %s:%d: %s (调用: %s)\n", __FILE__, __LINE__, cudaGetErrorString(err), #call); \
            return 1; \
        } \
    } while(0)

int main(int argc, char *argv[]) {
    int cuda_devices;
    cudaError_t err = cudaGetDeviceCount(&cuda_devices);
    if (err != cudaSuccess) {
        printf("错误: 无法检测 CUDA 设备!\n");
        printf("CUDA 错误: %s (code %d)\n", cudaGetErrorString(err), err);
        printf("\n可能原因:\n");
        printf("  1. GPU 驱动未正确加载\n");
        printf("  2. WDDM TDR 超时（GPU 在之前的计算中崩溃，驱动需要重置）\n");
        printf("  3. CUDA 版本与 GPU 不匹配\n");
        printf("  4. GPU 正在被其他进程占用\n");
        printf("\n建议: 重启电脑后重试\n");
        return 1;
    }
    printf("检测到 %d 个 CUDA 设备\n", cuda_devices);

    int target_score = 0;
    int search_batches = DEFAULT_SEARCH_BATCHES;

    if (argc >= 2) {
        // 命令行模式（向后兼容）
        target_score = static_cast<int>(strtol(argv[1], nullptr, 10));
        if (argc >= 3) {
            search_batches = static_cast<int>(strtol(argv[2], nullptr, 10));
            if (search_batches <= 0) search_batches = DEFAULT_SEARCH_BATCHES;
        }
    } else {
        // 交互模式
        printf("\n=== OI2048 Reporter ===\n\n");

        // 目标分数（必填）
        while (target_score <= 0) {
            printf("请输入目标分数: ");
            fflush(stdout);
            char line[256];
            if (fgets(line, sizeof(line), stdin) == nullptr) {
                printf("读取输入失败，程序退出。\n");
                return 1;
            }
            size_t len = strlen(line);
            if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';
            if (line[0] == '\0') {
                printf("目标分数为必填项，请重新输入。\n");
                continue;
            }
            char *endptr;
            long val = strtol(line, &endptr, 10);
            if (endptr == line || *endptr != '\0' || val <= 0) {
                printf("请输入有效的正整数。\n");
                continue;
            }
            target_score = static_cast<int>(val);
        }

        // 搜索批数（可选，留空使用默认值）
        printf("请输入搜索批数 (默认 %d，直接回车跳过): ", DEFAULT_SEARCH_BATCHES);
        fflush(stdout);
        char line[256];
        if (fgets(line, sizeof(line), stdin) != nullptr) {
            size_t len = strlen(line);
            if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';
            if (line[0] != '\0') {
                char *endptr;
                long val = strtol(line, &endptr, 10);
                if (endptr != line && *endptr == '\0' && val > 0) {
                    search_batches = static_cast<int>(val);
                } else {
                    printf("输入无效，使用默认值 %d。\n", DEFAULT_SEARCH_BATCHES);
                }
            }
        }
    }

    printf("目标分数: %d\n", target_score);
    printf("搜索批数: %d\n", search_batches);
    printf("启动 %d 个线程在 GPU 上并行搜索...\n", NUM_THREADS);

    SimResult *d_results[2];
    SimResult *h_results[2];
    cudaStream_t stream[2];

    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaMalloc(&d_results[i], NUM_THREADS * sizeof(SimResult)));
        CUDA_CHECK(cudaMemset(d_results[i], 0, NUM_THREADS * sizeof(SimResult)));
        CUDA_CHECK(cudaHostAlloc(&h_results[i], NUM_THREADS * sizeof(SimResult),
                      cudaHostAllocDefault));
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }
    printf("GPU 内存分配完成\n");

    auto base_seed = static_cast<uint64_t>(time(nullptr));

    bool found = false;
    int best_score = 0;
    int best_tid = -1;
    int best_batch = -1;

    // Top-K 种子追踪（用于 CPU 深搜）
    constexpr int TOP_K = 24;
    struct TopSeed { int score; int tid; int batch; };
    TopSeed top_seeds[TOP_K] = {};
    int top_count = 0;

    auto update_top_k = [&](int score, int tid, int batch) {
        int pos = top_count;
        while (pos > 0 && top_seeds[pos - 1].score < score) pos--;
        if (pos >= TOP_K) return;
        int limit = (top_count < TOP_K) ? top_count : TOP_K - 1;
        for (int i = limit; i > pos; i--)
            top_seeds[i] = top_seeds[i - 1];
        top_seeds[pos] = {score, tid, batch};
        if (top_count < TOP_K) top_count++;
    };

    auto process_batch = [&](int slot, int batch) {
        for (int i = 0; i < NUM_THREADS; i++) {
            int s = h_results[slot][i].score;
            if (s > best_score) {
                best_score = s;
                best_tid = i;
                best_batch = batch;
            }
            if (s > 0) update_top_k(s, i, batch);
            if (s >= target_score) {
                best_score = s;
                best_tid = i;
                best_batch = batch;
                found = true;
                printf("批次 %d: 最高 %d 分 [已达标!]\n", batch, best_score);
                return;
            }
        }
        printf("批次 %d: 最高 %d 分\n", batch, best_score);
    };

    for (int batch = 0; batch < search_batches && !found; batch++) {
        int cur = batch % 2;

        if (batch >= 2) {
            cudaStreamSynchronize(stream[cur]);
            cudaError_t sync_err = cudaGetLastError();
            if (sync_err != cudaSuccess) {
                printf("CUDA 内核错误 批次%d: %s\n", batch - 2, cudaGetErrorString(sync_err));
                // 继续处理已有结果
            }
            process_batch(cur, batch - 2);
        }

        if (batch % 100 == 0) {
            printf("启动批次 %d/%d...\n", batch, search_batches);
        }

        simulate_games<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, stream[cur]>>>(
            base_seed + batch * NUM_THREADS, target_score, d_results[cur]);
        
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            printf("内核启动失败 批次%d: %s\n", batch, cudaGetErrorString(launch_err));
            break;
        }
        
        CUDA_CHECK(cudaMemcpyAsync(h_results[cur], d_results[cur],
                        NUM_THREADS * sizeof(SimResult),
                        cudaMemcpyDeviceToHost, stream[cur]));
    }

    printf("同步所有流...\n");
    for (auto & i : stream) {
        CUDA_CHECK(cudaStreamSynchronize(i));
    }

    int first_unprocessed = (search_batches >= 2) ? (search_batches - 2) : 0;
    for (int batch = first_unprocessed; batch < search_batches && !found; batch++)
        process_batch(batch % 2, batch);

    printf("释放 GPU 内存...\n");
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaFree(d_results[i]));
        CUDA_CHECK(cudaFreeHost(h_results[i]));
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }
    printf("GPU 内存释放完成\n");

    if (!found) {
        printf("未达到目标分数，最终最佳: %d 分 (线程%d, 批次%d)，将使用此结果。\n", best_score, best_tid, best_batch);
    }

    // Host 端回放生成完整 history
    if (best_tid < 0 || best_batch < 0) {
        printf("错误: 没有找到任何有效结果 (best_tid=%d, best_batch=%d)\n", best_tid, best_batch);
        return 1;
    }
    uint64_t batch_seed = base_seed + best_batch * NUM_THREADS;
    auto rng_seed = static_cast<uint32_t>(batch_seed + best_tid * 2654435761ULL);
    printf("开始回放...\n");
    HostSimResult best = replay_game(rng_seed, best_tid % 4, target_score);
    printf("GPU 分数: %d  |  CPU 贪婪回放: %d\n", best_score, best.score);

    // ── CPU expectimax 深搜（Top-K 种子） ──
    if (top_count > 0) printf("\n对 Top-%d 种子进行 CPU expectimax 深搜...\n", top_count);
    for (int k = 0; k < top_count; k++) {
        uint64_t bs = base_seed + static_cast<uint64_t>(top_seeds[k].batch) * NUM_THREADS;
        auto rs = static_cast<uint32_t>(bs + top_seeds[k].tid * 2654435761ULL);
        HostSimResult result = replay_game_expectimax(rs, top_seeds[k].tid % 4, target_score);
        if (result.score > best.score) {
            printf("  种子 #%d: GPU=%d → CPU expectimax=%d ⬆ 提升!\n", k, top_seeds[k].score, result.score);
            best = result;
        }
    }

    // ── 额外：独立 CPU expectimax 游戏（fresh seeds，不受 GPU 种子限制）──
    {
        auto fresh_seed = static_cast<uint32_t>(time(nullptr) ^ 0xDEADBEEF);
        for (int i = 0; i < 4; i++) {
            int strat = i & 3;
            HostSimResult result = replay_game_expectimax(fresh_seed + i * 999983, strat, target_score);
            if (result.score > best.score) {
                printf("  独立 expectimax #%d (策略%d): %d ⬆ 提升!\n", i, strat, result.score);
                best = result;
            }
        }
    }

    printf("\n========== 生成结果 ==========\n");
    printf("分数: %d\n", best.score);
    printf("步数: %d\n", best.steps);
    printf("最大方块 log2: %d\n", max_value_log2(best.final_grid));
    printf("最终棋盘:\n");
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int v = best.final_grid[r * 4 + c];
            if (v == 0) printf("    _");
            else printf("%5d", v);
        }
        printf("\n");
    }

    const string json = generate_submit_data(best, false);
    if (FILE *fout = fopen("submit_output.json", "w")) {
        fprintf(fout, "%s\n", json.c_str());
        fclose(fout);
        printf("\n完整 JSON 已写入 submit_output.json (%zu 字节)\n", json.size());
    } else {
        printf("\n无法写入文件！\n");
    }

    return 0;
}
