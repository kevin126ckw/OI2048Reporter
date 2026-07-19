#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

constexpr int GRID_SIZE = 4;
constexpr int MAX_STEPS = 2048;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int NUM_BLOCKS = 1024;
constexpr int NUM_THREADS = THREADS_PER_BLOCK * NUM_BLOCKS; // 65536
constexpr int SEARCH_BATCHES = 32;

using std::string;

// ─── RNG ────────────────────────────────────────────────────────────
__host__ __device__ uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// ─── 棋盘快照 ────────────────────────────────────────────────────────
__host__ __device__ void copy_grid(const int *src, int *dst) {
    for (int i = 0; i < 16; i++) dst[i] = src[i];
}

__host__ __device__ bool grid_equal(const int *a, const int *b) {
    for (int i = 0; i < 16; i++) if (a[i] != b[i]) return false;
    return true;
}

__host__ __device__ int count_empty(const int *grid) {
    int cnt = 0;
    for (int i = 0; i < 16; i++) if (grid[i] == 0) cnt++;
    return cnt;
}

__host__ __device__ int max_value_log2(const int *grid) {
    int mx = 0;
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
__host__ __device__ int slide_line(int *arr) {
    int score = 0;
    bool merged[4] = {false, false, false, false};

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
            // 倍增方块合并：必须 i == prev + 1（原始相邻）
            else if (i == prev + 1) {
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
__host__ __device__ int apply_move(int *grid, int direction, bool *moved_out = nullptr) {
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
__host__ __device__ bool can_move(const int *grid) {
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
    return false;
}

// ─── 随机生成新方块 ────────────────────────────────────────────────────
__host__ __device__ void add_random_tile(int *grid, uint32_t *rng, bool is_start = false) {
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

// ─── 简单启发式评估 ────────────────────────────────────────────────────
__host__ __device__ float evaluate(const int *grid) {
    int empty_cnt = count_empty(grid);
    float score = (float)(empty_cnt) * 16.0f;

    int corners[4] = {grid[0], grid[3], grid[12], grid[15]};
    for (int & corner : corners) {
        if (corner > 0) score += (float)corner * 0.005f;
    }

    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 3; c++) {
            int a = grid[r * 4 + c], b = grid[r * 4 + c + 1];
            if (a > 0 && b > 0) {
                if (a >= b) score += 1.0f;
                else score -= 2.0f;
            }
        }
    }
    return score;
}

// ─── GPU 内核：并行模拟 ────────────────────────────────────────────────
struct SimResult {
    int score;
    int steps;
    int final_grid[16];
};

__global__ void simulate_games(uint64_t base_seed, int target_score,
                               SimResult *results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_THREADS) return;

    uint32_t rng = (uint32_t)(base_seed + tid * 2654435761ULL);
    int grid[16] = {0};

    add_random_tile(grid, &rng, true);
    add_random_tile(grid, &rng, true);

    int score = 0;
    int steps = 0;

    while (can_move(grid) && steps < MAX_STEPS) {
        float best_eval = -1e9f;
        int best_dir = -1;

        for (int dir = 0; dir < 4; dir++) {
            int temp[16];
            copy_grid(grid, temp);
            bool moved;
            apply_move(temp, dir, &moved);
            if (!moved) continue;
            float e = evaluate(temp);
            if (e > best_eval) {
                best_eval = e;
                best_dir = dir;
            }
        }

        if (best_dir < 0) break;

        score += apply_move(grid, best_dir);
        add_random_tile(grid, &rng);
        steps++;

        if (score >= target_score) break;
    }

    results[tid].score = score;
    results[tid].steps = steps;
    copy_grid(grid, results[tid].final_grid);
}

// ─── Host 端回放函数 ──────────────────────────────────────────────────
struct HistoryEntry {
    int before[16];
    int after[16];
};

struct HostSimResult {
    int score;
    int steps;
    std::vector<HistoryEntry> history;
    int final_grid[16];
};

HostSimResult replay_game(uint32_t rng_seed, int target_score = -1) {
    HostSimResult res;
    res.score = 0;
    res.steps = 0;
    res.history.clear();

    uint32_t rng = rng_seed;
    int grid[16] = {0};

    // 初始状态: before=全空, after=初始两方块
    HistoryEntry init;
    for (int i = 0; i < 16; i++) init.before[i] = 0;
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
            float e = evaluate(temp);
            if (e > best_eval) {
                best_eval = e;
                best_dir = dir;
            }
        }

        if (best_dir < 0) break;

        HistoryEntry entry;
        res.score += apply_move(grid, best_dir);
        copy_grid(grid, entry.before);  // 移动后、新方块前
        add_random_tile(grid, &rng);
        copy_grid(grid, entry.after);   // 移动后 + 新方块
        res.history.push_back(entry);
        res.steps++;

        if (target_score > 0 && res.score >= target_score) break;
    }

    copy_grid(grid, res.final_grid);
    return res;
}

// ─── 随机字符串生成 ────────────────────────────────────────────────────
string generate_gameid() {
    static const char charset[] = "0123456789qwertyuiopasdfghjklzxcvbnm";
    string id;
    for (int i = 0; i < 16; i++) id += charset[rand() % 36];
    return id;
}

// ─── JSON 格式化 ──────────────────────────────────────────────────────
string grid_to_json(const int *grid) {
    string s = "[";
    for (int r = 0; r < 4; r++) {
        s += "[";
        for (int c = 0; c < 4; c++) {
            int v = grid[r * 4 + c];
            if (v == 0) s += "null";
            else s += std::to_string(v);
            if (c < 3) s += ",";
        }
        s += "]";
        if (r < 3) s += ",";
    }
    s += "]";
    return s;
}

string history_to_json(const std::vector<HistoryEntry> &history) {
    string s = "[";
    for (size_t i = 0; i < history.size(); i++) {
        s += "[";
        s += grid_to_json(history[i].before);
        s += ",";
        s += grid_to_json(history[i].after);
        s += "]";
        if (i + 1 < history.size()) s += ",";
    }
    s += "]";
    return s;
}

string generate_submit_data(const HostSimResult &result, bool cheated = false) {
    string gameid = generate_gameid();
    int mv = max_value_log2(result.final_grid);
    string fg_json = grid_to_json(result.final_grid);

    string data = "{";
    data += "\"gameid\":\"" + gameid + "\",";
    data += "\"finalGrid\":\"" + fg_json + "\",";
    data += "\"maxValueLog\":" + std::to_string(mv) + ",";
    data += "\"steps\":" + std::to_string(result.steps) + ",";
    data += "\"history\":\"" + history_to_json(result.history) + "\",";
    data += "\"score\":" + std::to_string(result.score) + ",";
    data += "\"scoreStringForReference\":\"" + std::to_string(result.score) + "\",";
    data += "\"cheated\":" + string(cheated ? "true" : "false");
    data += "}";
    return data;
}

// ─── main ────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    srand(time(nullptr));

    int cuda_devices;
    cudaGetDeviceCount(&cuda_devices);
    if (cuda_devices == 0) {
        printf("错误: 未检测到 CUDA 设备！\n");
        return 1;
    }

    int target_score = 100000;
    if (argc >= 2) target_score = atoi(argv[1]);
    printf("目标分数: %d\n", target_score);
    printf("启动 %d 个线程在 GPU 上并行搜索...\n", NUM_THREADS);

    SimResult *d_results;
    cudaMalloc(&d_results, NUM_THREADS * sizeof(SimResult));
    cudaMemset(d_results, 0, NUM_THREADS * sizeof(SimResult));

    uint64_t base_seed = (uint64_t)time(nullptr);

    bool found = false;
    int best_score = 0;
    int best_tid = -1;
    int best_batch = -1;

    for (int batch = 0; batch < SEARCH_BATCHES && !found; batch++) {
        simulate_games<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            base_seed + batch * NUM_THREADS, target_score, d_results);
        cudaDeviceSynchronize();

        SimResult *h_results = new SimResult[NUM_THREADS];
        cudaMemcpy(h_results, d_results, NUM_THREADS * sizeof(SimResult),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < NUM_THREADS; i++) {
            if (h_results[i].score > best_score) {
                best_score = h_results[i].score;
                best_tid = i;
                best_batch = batch;
            }
            if (h_results[i].score >= target_score) {
                best_score = h_results[i].score;
                best_tid = i;
                best_batch = batch;
                found = true;
                break;
            }
        }
        delete[] h_results;

        printf("批次 %d: 当前最高 %d 分%s\n", batch, best_score,
               found ? " [已达标!]" : "");
    }

    if (!found) {
        printf("未达到目标分数，最终最佳: %d 分，将使用此结果。\n", best_score);
    }

    // Host 端回放生成完整 history
    uint64_t batch_seed = base_seed + best_batch * NUM_THREADS;
    uint32_t rng_seed = (uint32_t)(batch_seed + best_tid * 2654435761ULL);
    HostSimResult best = replay_game(rng_seed, target_score);

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

    string json = generate_submit_data(best, false);
    FILE *fout = fopen("submit_output.json", "w");
    if (fout) {
        fprintf(fout, "%s\n", json.c_str());
        fclose(fout);
        printf("\n完整 JSON 已写入 submit_output.json (%zu 字节)\n", json.size());
    } else {
        printf("\n无法写入文件！\n");
    }

    cudaFree(d_results);
    return 0;
}
