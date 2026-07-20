#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <random>

constexpr int GRID_SIZE = 4;
constexpr int MAX_STEPS = 2048;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int NUM_BLOCKS = 1024;
constexpr int NUM_THREADS = THREADS_PER_BLOCK * NUM_BLOCKS; // 65536
constexpr int SEARCH_BATCHES = 1024;

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

// ─── OI-2048 得分驱动评估（单趟遍历 + log_grid 预计算消除重复 __clz）─────
__host__ __device__ static float evaluate(const int *grid, const float *pos_w, int cr, int cc) {
    // 预计算全部 16 格 log2，后续行/列相邻分析直接读取，消除 ~24 次重复 __clz
    int log_grid[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int v = grid[i];
        log_grid[i] = (v != 0) ? ((v > 0) ? ilog2(v) : ilog2(-v)) : 0;
    }

    float score = 0.0f;
    int empty_cnt = 0;
    int max_val = 0;
    int max_log = 0;
    int row_sign = (cc == 3) ? -1 : 1;
    int col_sign = (cr == 3) ? -1 : 1;

    // 单趟遍历 16 格同时完成全部评估维度
#pragma unroll
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int c = 0; c < 4; c++) {
            int idx = r * 4 + c;
            int v = grid[idx];
            int log_v = log_grid[idx];
            bool v_pos = false;

            // ── 单格分析 ──
            if (v == 0) {
                empty_cnt++;
            } else {
                v_pos = (v > 0);
                if (v_pos) {
                    score += static_cast<float>(log_v) * pos_w[idx];
                    if (v > max_val) { max_val = v; max_log = log_v; }
                } else {
                    // 倍率合并潜力：检查四邻域
                    int best_nbr = 0;
                    if (r > 0 && grid[idx - 4] > 0) best_nbr = max(best_nbr, grid[idx - 4]);
                    if (r < 3 && grid[idx + 4] > 0) best_nbr = max(best_nbr, grid[idx + 4]);
                    if (c > 0 && grid[idx - 1] > 0) best_nbr = max(best_nbr, grid[idx - 1]);
                    if (c < 3 && grid[idx + 1] > 0) best_nbr = max(best_nbr, grid[idx + 1]);
                    if (best_nbr > 0) {
                        if (v <= -8 && (-v) * best_nbr > 65536) { /* skip */ }
                        else {
                            float w = (v <= -8) ? 8.0f : 6.0f;
                            // |v| 和 best_nbr 都是 2 的幂: log2(|v|*nbr) = log2(|v|) + log2(nbr)
                            score += static_cast<float>(log_v + ilog2(best_nbr)) * w;
                        }
                    }
                }
            }

            // ── 行相邻对（当前格 vs 右侧），lb 直接取自 log_grid ──
            if (c < 3) {
                int b = grid[idx + 1];
                if (v_pos && b > 0) {
                    int lb = log_grid[idx + 1];
                    int diff = log_v - lb;
                    if (diff < 0) diff = -diff;
                    score -= static_cast<float>(diff) * 0.8f;
                    score += (v >= b ? 1 : -1) * row_sign * static_cast<float>(log_v) * 0.5f;
                    if (v == b) score += static_cast<float>(log_v) * 2.0f;
                }
            }

            // ── 列相邻对（当前格 vs 下方），lb 直接取自 log_grid ──
            if (r < 3) {
                int b = grid[idx + 4];
                if (v_pos && b > 0) {
                    int lb = log_grid[idx + 4];
                    int diff = log_v - lb;
                    if (diff < 0) diff = -diff;
                    score -= static_cast<float>(diff) * 0.8f;
                    score += (v >= b ? 1 : -1) * col_sign * static_cast<float>(log_v) * 0.5f;
                    if (v == b) score += static_cast<float>(log_v) * 2.0f;
                }
            }
        }
    }

    // 自适应空格奖励（max_log 已在遍历中记录，避免额外 ilog2）
    float empty_bonus = 24.0f;
    if (max_val > 0) empty_bonus += static_cast<float>(max_log) * 3.0f;
    score += static_cast<float>(empty_cnt) * empty_bonus;

    return score;
}

namespace {
    // ─── GPU 内核：并行模拟 ────────────────────────────────────────────────
    struct SimResult {
        int score;  // 仅传分数，final_grid/steps 由 Host 回放生成
    };
}

__global__ __launch_bounds__(256, 4) static void simulate_games(uint64_t base_seed,
        int target_score, SimResult *results) {
    int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= NUM_THREADS) return;

    int strategy = tid & 3;
    int cr = (strategy >= 2) ? 3 : 0;
    int cc = (strategy == 1 || strategy == 3) ? 3 : 0;
    float pos_w[16];
    // powf(0.62f, dist) * 12.0f 预计算, dist = 0..6
    const float dist_w[7] = {12.0f, 7.44f, 4.6128f, 2.859936f, 1.77316032f, 1.09935936f, 0.68160288f};
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int r = i >> 2, c = i & 3;
        pos_w[i] = dist_w[abs(r - cr) + abs(c - cc)];
    }

    auto rng = static_cast<uint32_t>(base_seed + tid * 2654435761ULL);
    int grid[16] = {0};

    add_random_tile(grid, &rng, true);
    add_random_tile(grid, &rng, true);

    int score = 0;
    int steps = 0;

    while (true) {
        float best_eval = -1e9f;
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
            float e = evaluate(temp, pos_w, cr, cc);
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
    float pos_w[16];
    const float dist_w[7] = {12.0f, 7.44f, 4.6128f, 2.859936f, 1.77316032f, 1.09935936f, 0.68160288f};
    for (int i = 0; i < 16; i++) {
        int r = i >> 2, c = i & 3;
        pos_w[i] = dist_w[abs(r - cr) + abs(c - cc)];
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
        float best_eval = -1e9f;
        int best_dir = -1;

        for (int dir = 0; dir < 4; dir++) {
            int temp[16];
            copy_grid(grid, temp);
            bool moved;
            apply_move(temp, dir, &moved);
            if (!moved) continue;
            float e = evaluate(temp, pos_w, cr, cc);
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
int main(int argc, char *argv[]) {
    int cuda_devices;
    cudaGetDeviceCount(&cuda_devices);
    if (cuda_devices == 0) {
        printf("错误: 未检测到 CUDA 设备！\n");
        return 1;
    }

    int target_score = 1042766;
    if (argc >= 2) target_score = static_cast<int>(strtol(argv[1], nullptr, 10));
    printf("目标分数: %d\n", target_score);
    printf("启动 %d 个线程在 GPU 上并行搜索...\n", NUM_THREADS);

    SimResult *d_results[2];
    SimResult *h_results[2];
    cudaStream_t stream[2];

    for (int i = 0; i < 2; i++) {
        cudaMalloc(&d_results[i], NUM_THREADS * sizeof(SimResult));
        cudaMemset(d_results[i], 0, NUM_THREADS * sizeof(SimResult));
        cudaHostAlloc(&h_results[i], NUM_THREADS * sizeof(SimResult),
                      cudaHostAllocDefault);
        cudaStreamCreate(&stream[i]);
    }

    auto base_seed = static_cast<uint64_t>(time(nullptr));

    bool found = false;
    int best_score = 0;
    int best_tid = -1;
    int best_batch = -1;

    auto process_batch = [&](int slot, int batch) {
        for (int i = 0; i < NUM_THREADS; i++) {
            if (h_results[slot][i].score > best_score) {
                best_score = h_results[slot][i].score;
                best_tid = i;
                best_batch = batch;
            }
            if (h_results[slot][i].score >= target_score) {
                best_score = h_results[slot][i].score;
                best_tid = i;
                best_batch = batch;
                found = true;
                printf("批次 %d: 最高 %d 分 [已达标!]\n", batch, best_score);
                return;
            }
        }
        printf("批次 %d: 最高 %d 分\n", batch, best_score);
    };

    for (int batch = 0; batch < SEARCH_BATCHES && !found; batch++) {
        int cur = batch % 2;

        if (batch >= 2) {
            cudaStreamSynchronize(stream[cur]);
            process_batch(cur, batch - 2);
        }

        simulate_games<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, stream[cur]>>>(
            base_seed + batch * NUM_THREADS, target_score, d_results[cur]);
        cudaMemcpyAsync(h_results[cur], d_results[cur],
                        NUM_THREADS * sizeof(SimResult),
                        cudaMemcpyDeviceToHost, stream[cur]);
    }

    for (auto & i : stream)
        cudaStreamSynchronize(i);

    int first_unprocessed = (SEARCH_BATCHES >= 2) ? (SEARCH_BATCHES - 2) : 0;
    for (int batch = first_unprocessed; batch < SEARCH_BATCHES && !found; batch++)
        process_batch(batch % 2, batch);

    for (int i = 0; i < 2; i++) {
        cudaFree(d_results[i]);
        cudaFreeHost(h_results[i]);
        cudaStreamDestroy(stream[i]);
    }

    if (!found) {
        printf("未达到目标分数，最终最佳: %d 分，将使用此结果。\n", best_score);
    }

    // Host 端回放生成完整 history
    uint64_t batch_seed = base_seed + best_batch * NUM_THREADS;
    auto rng_seed = static_cast<uint32_t>(batch_seed + best_tid * 2654435761ULL);
    HostSimResult best = replay_game(rng_seed, best_tid % 4, target_score);

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
