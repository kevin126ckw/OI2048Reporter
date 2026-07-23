#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <cstring>
#include <string>

#include "kernel.cuh"
#include "replay.cuh"
#include "submit_format.cuh"

using std::string;

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

    #pragma unroll
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
