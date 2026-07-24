#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cstdint>
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
            std::cout << "CUDA 错误 " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << " (调用: " << #call << ")" << std::endl; \
            return 1; \
        } \
    } while(0)

int main(int argc, char *argv[]) {
    int cuda_devices;
    cudaError_t err = cudaGetDeviceCount(&cuda_devices);
    if (err != cudaSuccess) {
        std::cout << "错误: 无法检测 CUDA 设备!" << std::endl;
        std::cout << "CUDA 错误: " << cudaGetErrorString(err) << " (code " << err << ")" << std::endl;
        std::cout << std::endl << "可能原因:" << std::endl;
        std::cout << "  1. GPU 驱动未正确加载" << std::endl;
        std::cout << "  2. WDDM TDR 超时（GPU 在之前的计算中崩溃，驱动需要重置）" << std::endl;
        std::cout << "  3. CUDA 版本与 GPU 不匹配" << std::endl;
        std::cout << "  4. GPU 正在被其他进程占用" << std::endl;
        std::cout << std::endl << "建议: 重启电脑后重试" << std::endl;
        return 1;
    }
    std::cout << "检测到 " << cuda_devices << " 个 CUDA 设备" << std::endl;

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
        std::cout << std::endl << "=== OI2048 Reporter ===" << std::endl << std::endl;

        // 目标分数（必填）
        while (target_score <= 0) {
            std::cout << "请输入目标分数: " << std::flush;
            char line[256];
            if (!std::cin.getline(line, sizeof(line))) {
                std::cout << "读取输入失败，程序退出。" << std::endl;
                return 1;
            }
            if (line[0] == '\0') {
                std::cout << "目标分数为必填项，请重新输入。" << std::endl;
                continue;
            }
            char *endptr;
            long val = strtol(line, &endptr, 10);
            if (endptr == line || *endptr != '\0' || val <= 0) {
                std::cout << "请输入有效的正整数。" << std::endl;
                continue;
            }
            target_score = static_cast<int>(val);
        }

        // 搜索批数（可选，留空使用默认值）
        std::cout << "请输入搜索批数 (默认 " << DEFAULT_SEARCH_BATCHES << "，直接回车跳过): " << std::flush;
        char line[256];
        if (std::cin.getline(line, sizeof(line))) {
            if (line[0] != '\0') {
                char *endptr;
                long val = strtol(line, &endptr, 10);
                if (endptr != line && *endptr == '\0' && val > 0) {
                    search_batches = static_cast<int>(val);
                } else {
                    std::cout << "输入无效，使用默认值 " << DEFAULT_SEARCH_BATCHES << "。" << std::endl;
                }
            }
        }
    }

    std::cout << "目标分数: " << target_score << std::endl;
    std::cout << "搜索批数: " << search_batches << std::endl;
    std::cout << "启动 " << NUM_THREADS << " 个线程在 GPU 上并行搜索..." << std::endl;

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
    std::cout << "GPU 内存分配完成" << std::endl;

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
                std::cout << "批次 " << batch << ": 最高 " << best_score << " 分 [已达标!]" << std::endl;
                return;
            }
        }
        std::cout << "批次 " << batch << ": 最高 " << best_score << " 分" << std::endl;
    };

    for (int batch = 0; batch < search_batches && !found; batch++) {
        int cur = batch % 2;

        if (batch >= 2) {
            cudaStreamSynchronize(stream[cur]);
            cudaError_t sync_err = cudaGetLastError();
            if (sync_err != cudaSuccess) {
                std::cout << "CUDA 内核错误 批次" << (batch - 2) << ": " << cudaGetErrorString(sync_err) << std::endl;
                // 继续处理已有结果
            }
            process_batch(cur, batch - 2);
        }

        if (batch % 100 == 0) {
            std::cout << "启动批次 " << batch << "/" << search_batches << "..." << std::endl;
        }

        simulate_games<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, stream[cur]>>>(
            base_seed + batch * NUM_THREADS, target_score, d_results[cur]);

        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            std::cout << "内核启动失败 批次" << batch << ": " << cudaGetErrorString(launch_err) << std::endl;
            break;
        }

        CUDA_CHECK(cudaMemcpyAsync(h_results[cur], d_results[cur],
                        NUM_THREADS * sizeof(SimResult),
                        cudaMemcpyDeviceToHost, stream[cur]));
    }

    std::cout << "同步所有流..." << std::endl;
    for (auto & i : stream) {
        CUDA_CHECK(cudaStreamSynchronize(i));
    }

    int first_unprocessed = (search_batches >= 2) ? (search_batches - 2) : 0;
    for (int batch = first_unprocessed; batch < search_batches && !found; batch++)
        process_batch(batch % 2, batch);

    std::cout << "释放 GPU 内存..." << std::endl;
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaFree(d_results[i]));
        CUDA_CHECK(cudaFreeHost(h_results[i]));
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }
    std::cout << "GPU 内存释放完成" << std::endl;

    if (!found) {
        std::cout << "未达到目标分数，最终最佳: " << best_score << " 分 (线程" << best_tid << ", 批次" << best_batch << ")，将使用此结果。" << std::endl;
    }

    // Host 端回放生成完整 history
    if (best_tid < 0 || best_batch < 0) {
        std::cout << "错误: 没有找到任何有效结果 (best_tid=" << best_tid << ", best_batch=" << best_batch << ")" << std::endl;
        return 1;
    }
    uint64_t batch_seed = base_seed + best_batch * NUM_THREADS;
    auto rng_seed = static_cast<uint32_t>(batch_seed + best_tid * 2654435761ULL);
    std::cout << "开始回放..." << std::endl;
    HostSimResult best = replay_game(rng_seed, best_tid % 4, target_score);
    std::cout << "GPU 分数: " << best_score << "  |  CPU 贪婪回放: " << best.score << std::endl;

    // ── CPU expectimax 深搜（Top-K 种子） ──
    if (top_count > 0) std::cout << std::endl << "对 Top-" << top_count << " 种子进行 CPU expectimax 深搜..." << std::endl;
    for (int k = 0; k < top_count; k++) {
        uint64_t bs = base_seed + static_cast<uint64_t>(top_seeds[k].batch) * NUM_THREADS;
        auto rs = static_cast<uint32_t>(bs + top_seeds[k].tid * 2654435761ULL);
        HostSimResult result = replay_game_expectimax(rs, top_seeds[k].tid % 4, target_score);
        if (result.score > best.score) {
            std::cout << "  种子 #" << k << ": GPU=" << top_seeds[k].score << " → CPU expectimax=" << result.score << " ⬆ 提升!" << std::endl;
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
                std::cout << "  独立 expectimax #" << i << " (策略" << strat << "): " << result.score << " ⬆ 提升!" << std::endl;
                best = result;
            }
        }
    }

    std::cout << std::endl << "========== 生成结果 ==========" << std::endl;
    std::cout << "分数: " << best.score << std::endl;
    std::cout << "步数: " << best.steps << std::endl;
    std::cout << "最大方块 log2: " << max_value_log2(best.final_grid) << std::endl;
    std::cout << "最终棋盘:" << std::endl;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int v = best.final_grid[r * 4 + c];
            if (v == 0) std::cout << "    _";
            else std::cout << std::setw(5) << v;
        }
        std::cout << std::endl;
    }

    const string json = generate_submit_data(best, false);
    if (std::ofstream fout("submit_output.json"); fout.is_open()) {
        fout << json << std::endl;
        fout.close();
        std::cout << std::endl << "完整 JSON 已写入 submit_output.json (" << json.size() << " 字节)" << std::endl;
    } else {
        std::cout << std::endl << "无法写入文件！" << std::endl;
    }

    return 0;
}
