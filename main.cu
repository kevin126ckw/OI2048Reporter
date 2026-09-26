#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "gpu_plan.cuh"
#include "kernel.cuh"
#include "replay.cuh"
#include "submit_format.cuh"

using std::string;

namespace {

// ─── 命令行选项 ──────────────────────────────────────────────────────
struct Options {
    std::vector<string> positional;   // [目标分数] [搜索批数]（与旧版命令行兼容）
    string gpu_selector = "all";      // --gpus
    int logical_per_device = 1;       // --logical-devices（单卡模拟多卡，用于验证调度）
    bool list_devices = false;        // --list-devices
    bool help = false;                // --help
    bool has_seed = false;            // --seed
    uint64_t seed = 0;
    string dump_scores_path;          // --dump-scores（测试用：导出每批次全部线程分数）
};

// ─── 单个 GPU 切片的运行时资源 ────────────────────────────────────────
// 每个切片（物理卡或逻辑切片）拥有独立的双缓冲：2 个 stream + 2 组显存/锁页内存
struct SliceRuntime {
    GpuSlice plan;
    cudaStream_t streams[2]{};
    SimResult *d_results[2]{};   // 显存：本切片 threads 个结果
    SimResult *h_results[2]{};   // 锁页主机内存：回传目标
    bool active = false;         // 是否还会继续下发新批次
    bool pending[2]{};           // 该槽位上是否有已下发、尚未读取的批次结果
    bool readable[2]{};          // 该槽位的结果是否已成功同步（可以读）
};

std::string slice_label(const SliceRuntime &rt) {
    std::string label = "GPU " + std::to_string(rt.plan.ordinal);
    if (rt.plan.logical_count > 1) {
        label += " 逻辑切片 " + std::to_string(rt.plan.logical_index + 1) + "/" +
                 std::to_string(rt.plan.logical_count);
    }
    return label;
}

// 读取并清空线程级「最后错误」状态。
// cudaGetLastError() 不会因为后续调用成功而复位，所以在用 cudaGetLastError() 判断
// 某次操作是否失败之前必须先清一次，否则某个切片上一次失败留下的状态会被误判成
// 本次失败，把健康的切片一起踢掉。
void clear_cuda_error_state() {
    cudaGetLastError();
}

void print_usage(const char *prog) {
    std::cout << "用法: " << prog << " [目标分数] [搜索批数] [选项]" << std::endl;
    std::cout << std::endl;
    std::cout << "位置参数（省略则进入交互模式）:" << std::endl;
    std::cout << "  目标分数              达到该分数后提前停止搜索" << std::endl;
    std::cout << "  搜索批数              搜索批次数（默认 " << DEFAULT_SEARCH_BATCHES
              << "，每批 " << NUM_THREADS << " 局游戏）" << std::endl;
    std::cout << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -g, --gpus <列表>     使用的 GPU，如 \"0,1\"、\"0-2\"、\"all\"（默认 all：全部可见 GPU）"
              << std::endl;
    std::cout << "      --logical-devices <n>" << std::endl;
    std::cout << "                        把每张卡当作 n 个逻辑切片（默认 1；用于单卡验证多卡调度）"
              << std::endl;
    std::cout << "      --list-devices    列出所有可见 GPU 后退出" << std::endl;
    std::cout << "      --seed <整数>     固定基准随机种子（便于复现和对比测试）" << std::endl;
    std::cout << "      --dump-scores <文件>" << std::endl;
    std::cout << "                        导出每个已处理批次的全部线程分数（二进制: 批次号、线程数、分数数组，"
                 "测试用）" << std::endl;
    std::cout << "  -h, --help            显示本帮助" << std::endl;
}

bool parse_long_token(const string &token, long long *out) {
    if (token.empty()) return false;
    char *end = nullptr;
    const long long value = strtoll(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0') return false;
    *out = value;
    return true;
}

bool parse_u64_token(const string &token, uint64_t *out) {
    // 拒绝负号：strtoull 会把 "-1" 静默回绕成一个巨大的数
    if (token.empty() || token[0] == '-') return false;
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0') return false;
    if (errno == ERANGE) return false;   // 溢出（超过 uint64 上限）
    *out = static_cast<uint64_t>(value);
    return true;
}

bool parse_options(const int argc, char **argv, Options *opt, string *error) {
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        string value;
        bool has_value = false;

        if (arg.rfind("--", 0) == 0) {
            if (const std::size_t eq = arg.find('='); eq != string::npos) {
                value = arg.substr(eq + 1);
                arg = arg.substr(0, eq);
                has_value = true;
            }
        }

        const auto need_value = [&](string *dst) {
            if (has_value) { *dst = value; return true; }
            if (i + 1 < argc) { *dst = argv[++i]; return true; }
            *error = "选项 " + arg + " 缺少参数";
            return false;
        };

        if (arg == "-h" || arg == "--help") {
            opt->help = true;
        } else if (arg == "--list-devices") {
            opt->list_devices = true;
        } else if (arg == "-g" || arg == "--gpus") {
            string text;
            if (!need_value(&text)) return false;
            opt->gpu_selector = text;
        } else if (arg == "--logical-devices") {
            string text;
            long long count = 0;
            if (!need_value(&text)) return false;
            if (!parse_long_token(text, &count) || count <= 0 || count > 64) {
                *error = "--logical-devices 需要 1..64 之间的整数";
                return false;
            }
            opt->logical_per_device = static_cast<int>(count);
        } else if (arg == "--seed") {
            string text;
            uint64_t seed = 0;
            if (!need_value(&text)) return false;
            if (!parse_u64_token(text, &seed)) {
                *error = "--seed 需要非负整数";
                return false;
            }
            opt->has_seed = true;
            opt->seed = seed;
        } else if (arg == "--dump-scores") {
            string path;
            if (!need_value(&path)) return false;
            if (path.empty()) {
                *error = "--dump-scores 需要一个文件路径";
                return false;
            }
            opt->dump_scores_path = path;
        } else if (arg.size() > 1 && arg[0] == '-' && (arg[1] < '0' || arg[1] > '9')) {
            *error = "未知选项: " + arg;
            return false;
        } else {
            opt->positional.push_back(arg);
        }
    }
    return true;
}

void print_device_table(const std::vector<GpuDeviceInfo> &devices) {
    std::cout << std::endl << "可见 CUDA 设备:" << std::endl;
    for (const GpuDeviceInfo &device : devices) {
        std::cout << "  GPU " << device.ordinal << ": " << device.name
                  << "  计算能力 " << device.major << "." << device.minor
                  << "  SM " << device.sm_count
                  << "  显存 " << std::fixed << std::setprecision(1)
                  << static_cast<double>(device.memory_bytes) / (1024.0 * 1024.0 * 1024.0)
                  << " GiB" << std::defaultfloat << std::endl;
    }
}

// ─── 初始化 / 释放单个切片的显存、锁页内存与 stream ────────────────────
bool init_slice_runtime(SliceRuntime &rt, string *error) {
    const int ordinal = rt.plan.ordinal;
    const auto label = "GPU " + std::to_string(ordinal);

    cudaError_t err = cudaSetDevice(ordinal);
    if (err != cudaSuccess) {
        *error = label + " cudaSetDevice 失败: " + cudaGetErrorString(err);
        return false;
    }

    const std::size_t bytes = static_cast<std::size_t>(rt.plan.threads) * sizeof(SimResult);
    for (int i = 0; i < 2; ++i) {
        if ((err = cudaMalloc(reinterpret_cast<void **>(&rt.d_results[i]), bytes)) != cudaSuccess) {
            *error = label + " cudaMalloc 失败: " + cudaGetErrorString(err);
            return false;
        }
        if ((err = cudaMemset(rt.d_results[i], 0, bytes)) != cudaSuccess) {
            *error = label + " cudaMemset 失败: " + cudaGetErrorString(err);
            return false;
        }
        if ((err = cudaHostAlloc(reinterpret_cast<void **>(&rt.h_results[i]), bytes,
                                 cudaHostAllocDefault)) != cudaSuccess) {
            *error = label + " cudaHostAlloc 失败: " + cudaGetErrorString(err);
            return false;
        }
        if ((err = cudaStreamCreate(&rt.streams[i])) != cudaSuccess) {
            *error = label + " cudaStreamCreate 失败: " + cudaGetErrorString(err);
            return false;
        }
    }

    rt.active = true;
    return true;
}

void release_slice_runtime(SliceRuntime &rt) {
    if (!rt.active && rt.streams[0] == nullptr && rt.d_results[0] == nullptr) return;

    cudaSetDevice(rt.plan.ordinal);
    for (int i = 0; i < 2; ++i) {
        if (rt.streams[i] != nullptr) {
            cudaStreamSynchronize(rt.streams[i]);
            cudaStreamDestroy(rt.streams[i]);
            rt.streams[i] = nullptr;
        }
        if (rt.d_results[i] != nullptr) {
            cudaFree(rt.d_results[i]);
            rt.d_results[i] = nullptr;
        }
        if (rt.h_results[i] != nullptr) {
            cudaFreeHost(rt.h_results[i]);
            rt.h_results[i] = nullptr;
        }
    }
    cudaGetLastError(); // 清理释放过程中的错误状态
    rt.active = false;
}

} // namespace

// ─── main ────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    Options opt;
    string error;
    if (!parse_options(argc, argv, &opt, &error)) {
        std::cout << "参数错误: " << error << std::endl << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    if (opt.help) {
        print_usage(argv[0]);
        return 0;
    }

    int cuda_devices = 0;
    if (cudaError_t err = cudaGetDeviceCount(&cuda_devices); err != cudaSuccess) {
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

    std::vector<GpuDeviceInfo> devices = enumerate_gpu_devices(&error);
    if (devices.empty()) {
        std::cout << "错误: 无法获取 CUDA 设备信息";
        if (!error.empty()) std::cout << ": " << error;
        std::cout << std::endl;
        return 1;
    }

    if (opt.list_devices) {
        print_device_table(devices);
        return 0;
    }

    // ── 选择 GPU 并划分线程块 ──
    std::vector<int> selected_ordinals;
    if (!parse_gpu_selector(opt.gpu_selector, cuda_devices, &selected_ordinals, &error)) {
        std::cout << "错误: " << error << std::endl;
        return 1;
    }
    if (selected_ordinals.empty()) {
        std::cout << "错误: 没有选择任何 GPU (--gpus " << opt.gpu_selector << ")" << std::endl;
        return 1;
    }

    std::vector<GpuDeviceInfo> selected_devices;
    for (const int ordinal : selected_ordinals) {
        for (const GpuDeviceInfo &device : devices) {
            if (device.ordinal == ordinal) selected_devices.push_back(device);
        }
    }
    if (selected_devices.empty()) {
        std::cout << "错误: 选中的 GPU 均不可用" << std::endl;
        return 1;
    }

    std::vector<GpuSlice> slices = compute_gpu_slices(selected_devices, NUM_BLOCKS,
                                                      THREADS_PER_BLOCK,
                                                      opt.logical_per_device, &error);
    if (slices.empty()) {
        std::cout << "错误: 无法生成 GPU 任务划分: " << error << std::endl;
        return 1;
    }

    // ── 目标分数与搜索批数 ──
    int target_score = 0;
    int search_batches = DEFAULT_SEARCH_BATCHES;

    if (!opt.positional.empty()) {
        // 命令行模式（向后兼容：<目标分数> [搜索批数]）
        constexpr long long kMaxInt = std::numeric_limits<int>::max();
        long long value = 0;
        if (!parse_long_token(opt.positional[0], &value) || value <= 0 || value > kMaxInt) {
            std::cout << "错误: 目标分数必须是 1.." << kMaxInt << " 之间的整数，收到 \""
                      << opt.positional[0] << "\"" << std::endl;
            return 1;
        }
        target_score = static_cast<int>(value);
        if (opt.positional.size() >= 2) {
            if (!parse_long_token(opt.positional[1], &value) || value <= 0 || value > kMaxInt) {
                std::cout << "错误: 搜索批数必须是 1.." << kMaxInt << " 之间的整数，收到 \""
                          << opt.positional[1] << "\"" << std::endl;
                return 1;
            }
            search_batches = static_cast<int>(value);
        }
        if (opt.positional.size() > 2) {
            std::cout << "警告: 忽略多余的位置参数" << std::endl;
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
                if (long val = strtol(line, &endptr, 10); endptr != line && *endptr == '\0' && val > 0) {
                    search_batches = static_cast<int>(val);
                } else {
                    std::cout << "输入无效，使用默认值 " << DEFAULT_SEARCH_BATCHES << "。" << std::endl;
                }
            }
        }
    }

    std::cout << "目标分数: " << target_score << std::endl;
    std::cout << "搜索批数: " << search_batches << std::endl;

    // ── 输出任务划分 ──
    long long planned_threads = 0;
    for (const GpuSlice &slice : slices) planned_threads += slice.threads;

    std::cout << "使用 " << slices.size() << " 个 GPU 切片并行搜索";
    if (opt.logical_per_device > 1)
        std::cout << " (每卡 " << opt.logical_per_device << " 个逻辑切片，测试模式)";
    else if (selected_devices.size() > 1)
        std::cout << " (" << selected_devices.size() << " 张卡)";
    std::cout << ", 共 " << planned_threads << " 个线程:" << std::endl;

    for (std::size_t i = 0; i < slices.size(); ++i) {
        const GpuSlice &slice = slices[i];
        std::cout << "  [" << i << "] GPU " << slice.ordinal << " " << slice.name
                  << " (" << slice.sm_count << " SM)";
        if (slice.logical_count > 1)
            std::cout << " 逻辑切片 " << (slice.logical_index + 1) << "/" << slice.logical_count;
        std::cout << " → " << slice.blocks << " 块 / " << slice.threads << " 线程";
        if (slice.blocks <= 0) std::cout << "（跳过: 线程块数不足）";
        std::cout << std::endl;
    }

    if (planned_threads != NUM_THREADS) {
        std::cout << "错误: 任务划分不完整 (" << planned_threads << " != " << NUM_THREADS << ")" << std::endl;
        return 1;
    }

    // ── 随机种子与（可选的）结果导出 ──
    const uint64_t base_seed = opt.has_seed ? opt.seed : static_cast<uint64_t>(time(nullptr));
    std::cout << "随机种子: " << base_seed << std::endl;

    std::ofstream dump;
    if (!opt.dump_scores_path.empty()) {
        dump.open(opt.dump_scores_path, std::ios::binary | std::ios::trunc);
        if (!dump.is_open()) {
            std::cout << "错误: 无法写入 " << opt.dump_scores_path << std::endl;
            return 1;
        }
        std::cout << "线程分数导出: " << opt.dump_scores_path << std::endl;
    }

    // ── 初始化各切片的运行时资源 ──
    std::vector<SliceRuntime> runtimes;
    runtimes.reserve(slices.size());
    int skipped_slices = 0;
    for (const GpuSlice &slice : slices) {
        if (slice.blocks <= 0) {
            ++skipped_slices;
            continue;
        }
        SliceRuntime rt;
        rt.plan = slice;
        if (!init_slice_runtime(rt, &error)) {
            std::cout << "警告: " << error << "，该切片不参与计算" << std::endl;
            release_slice_runtime(rt);
            ++skipped_slices;
            continue;
        }
        runtimes.push_back(rt);
    }
    if (runtimes.empty()) {
        std::cout << "错误: 没有任何可用的 GPU 切片" << std::endl;
        return 1;
    }
    if (skipped_slices > 0)
        std::cout << "注意: " << skipped_slices << " 个切片未参与计算，实际搜索线程数减少" << std::endl;
    std::cout << "GPU 内存分配完成" << std::endl;

    bool found = false;
    int best_score = 0;
    int best_tid = -1;
    int best_batch = -1;

    // Top-K 种子追踪（用于 CPU 深搜）
    constexpr int TOP_K = 24;
    struct TopSeed { int score; int tid; int batch; };
    TopSeed top_seeds[TOP_K] = {};
    int top_count = 0;

    auto update_top_k = [&](const int score, const int tid, const int batch) {
        int pos = top_count;
        while (pos > 0 && top_seeds[pos - 1].score < score) pos--;
        if (pos >= TOP_K) return;
        const int limit = top_count < TOP_K ? top_count : TOP_K - 1;
        for (int i = limit; i > pos; i--)
            top_seeds[i] = top_seeds[i - 1];
        top_seeds[pos] = {.score = score, .tid = tid, .batch = batch};
        if (top_count < TOP_K) top_count++;
    };

    // 同步某个槽位：把该槽位上已下发批次的回传拷贝等完，并标记结果是否可读。
    // 即使切片已经退出计算（active=false），只要还有未读取的结果，也要同步并读取，
    // 否则一次故障会白白丢掉已经算好的批次（可能正是最高分那一批）。
    auto sync_slot = [&](const int slot) {
        for (SliceRuntime &rt : runtimes) {
            if (!rt.pending[slot]) continue;
            rt.readable[slot] = false;

            if (cudaError_t dev_err = cudaSetDevice(rt.plan.ordinal); dev_err != cudaSuccess) {
                std::cout << "警告: " << slice_label(rt) << " 无法切换设备: "
                          << cudaGetErrorString(dev_err) << "，该切片退出计算" << std::endl;
                rt.active = false;
                clear_cuda_error_state();
                continue;
            }

            clear_cuda_error_state();   // 只统计本次同步产生的错误
            const cudaError_t sync_err = cudaStreamSynchronize(rt.streams[slot]);
            const cudaError_t kernel_err = cudaGetLastError();
            const cudaError_t err = sync_err != cudaSuccess ? sync_err : kernel_err;
            if (err != cudaSuccess) {
                std::cout << "警告: " << slice_label(rt) << " 同步/内核错误: "
                          << cudaGetErrorString(err) << "，该切片退出计算（本批次结果作废）" << std::endl;
                rt.active = false;
                continue;
            }
            rt.readable[slot] = true;
        }
    };

    // 导出本批次全部线程分数（按全局线程号排列，用于对比不同设备划分是否等价）
    std::vector<int> batch_scores;
    if (dump.is_open()) batch_scores.assign(NUM_THREADS, 0);

    auto process_batch = [&](const int slot, const int batch) {
        if (dump.is_open()) {
            std::fill(batch_scores.begin(), batch_scores.end(), 0);
            for (const SliceRuntime &rt : runtimes) {
                if (!rt.pending[slot] || !rt.readable[slot]) continue;
                for (int i = 0; i < rt.plan.threads; ++i)
                    batch_scores[rt.plan.tid_offset + i] = rt.h_results[slot][i].score;
            }
            const int32_t header[2] = {batch, static_cast<int32_t>(batch_scores.size())};
            dump.write(reinterpret_cast<const char *>(header), sizeof(header));
            dump.write(reinterpret_cast<const char *>(batch_scores.data()),
                       static_cast<std::streamsize>(batch_scores.size() * sizeof(int)));
        }

        int batch_best = 0;
        bool target_hit = false;
        for (SliceRuntime &rt : runtimes) {
            const bool readable = rt.pending[slot] && rt.readable[slot];
            rt.pending[slot] = false;
            rt.readable[slot] = false;
            if (!readable) continue;

            // 达标后不中断扫描：这些结果已经算好并回传到主机，扫完可以把更好的分数/种子留下，
            // 中断只会白白丢掉本批次里更高的分（代价仅为几十万次整数比较）
            for (int i = 0; i < rt.plan.threads; ++i) {
                const int s = rt.h_results[slot][i].score;
                const int tid = rt.plan.tid_offset + i;
                if (s > batch_best) batch_best = s;
                if (s > best_score) {          // 累计最高分只增不减
                    best_score = s;
                    best_tid = tid;
                    best_batch = batch;
                }
                if (s > 0) update_top_k(s, tid, batch);
                if (s >= target_score) {
                    found = true;              // 只停止下发新批次，不丢弃已算好的结果
                    target_hit = true;
                }
            }
        }

        if (target_hit) {
            std::cout << "批次 " << batch << ": 达到目标分 " << target_score << " [已达标!], 本批最高 "
                      << batch_best << " 分, 累计最高 " << best_score << " 分" << std::endl;
            return;
        }
        std::cout << "批次 " << batch << ": 本批最高 " << batch_best << " 分, 累计最高 "
                  << best_score << " 分" << std::endl;
    };

    // ── 搜索主循环：全部 GPU 同时跑同一个批次的不同线程段 ──
    const auto search_start = std::chrono::steady_clock::now();
    int launched = 0;              // 已启动的批次数
    int processed = 0;             // 已处理（读取结果）的批次数
    long long searched_games = 0;  // 实际下发的模拟局数（切片故障时会少于 批次数×NUM_THREADS）

    for (int batch = 0; batch < search_batches && !found; ++batch) {
        const int slot = batch % 2;

        if (batch >= 2) {
            sync_slot(slot);                 // 该槽位上是批次 batch-2 的结果
            process_batch(slot, batch - 2);
            processed = batch - 1;
            if (found) break;                // 已达标：不再启动新批次
        }

        if (batch % 100 == 0) {
            std::cout << "启动批次 " << batch << "/" << search_batches << "..." << std::endl;
        }

        bool launched_any = false;
        for (SliceRuntime &rt : runtimes) {
            if (!rt.active) continue;

            if (cudaError_t dev_err = cudaSetDevice(rt.plan.ordinal); dev_err != cudaSuccess) {
                std::cout << "警告: " << slice_label(rt) << " 无法切换设备: "
                          << cudaGetErrorString(dev_err) << "，该切片退出计算" << std::endl;
                rt.active = false;
                clear_cuda_error_state();
                continue;
            }

            clear_cuda_error_state();        // 只让本次启动的错误影响下面的判断
            simulate_games<<<rt.plan.blocks, THREADS_PER_BLOCK, 0, rt.streams[slot]>>>(
                base_seed + static_cast<uint64_t>(batch) * NUM_THREADS, target_score,
                rt.plan.tid_offset, rt.plan.threads, rt.d_results[slot]);

            if (cudaError_t launch_err = cudaGetLastError(); launch_err != cudaSuccess) {
                std::cout << "内核启动失败 " << slice_label(rt) << " 批次" << batch << ": "
                          << cudaGetErrorString(launch_err) << "，该切片退出计算" << std::endl;
                rt.active = false;
                continue;
            }
            if (cudaError_t copy_err = cudaMemcpyAsync(
                    rt.h_results[slot], rt.d_results[slot],
                    static_cast<std::size_t>(rt.plan.threads) * sizeof(SimResult),
                    cudaMemcpyDeviceToHost, rt.streams[slot]); copy_err != cudaSuccess) {
                std::cout << "结果回传失败 " << slice_label(rt) << " 批次" << batch << ": "
                          << cudaGetErrorString(copy_err) << "，该切片退出计算" << std::endl;
                rt.active = false;
                clear_cuda_error_state();    // 不让本次失败污染后面切片的健康检查
                continue;
            }

            rt.pending[slot] = true;
            rt.readable[slot] = false;
            searched_games += rt.plan.threads;
            launched_any = true;
        }

        if (!launched_any) {
            std::cout << "错误: 所有 GPU 切片均不可用，中止搜索" << std::endl;
            break;
        }
        ++launched;
    }

    // ── 收尾：处理仍在流水线中的最后 1~2 个批次 ──
    for (int batch = processed; batch < launched; ++batch) {
        const int slot = batch % 2;
        sync_slot(slot);                     // 槽位上只可能有批次 batch（batch+2 未启动）
        process_batch(slot, batch);
    }
    sync_slot(0);
    sync_slot(1);

    const double search_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - search_start).count();
    std::cout << "搜索完成: " << launched << " 批次, 共 " << searched_games << " 局游戏, 用时 "
              << std::fixed << std::setprecision(2) << search_seconds << " 秒";
    if (search_seconds > 0.0)
        std::cout << " (" << static_cast<long long>(static_cast<double>(searched_games) / search_seconds)
                  << " 局/秒)";
    std::cout << std::defaultfloat << std::endl;

    std::cout << "释放 GPU 内存..." << std::endl;
    for (SliceRuntime &rt : runtimes) release_slice_runtime(rt);
    std::cout << "GPU 内存释放完成" << std::endl;

    // 导出文件必须完整落盘，否则等价性测试会误判为通过
    if (dump.is_open()) {
        dump.flush();
        if (!dump.good()) {
            std::cout << "错误: 线程分数导出失败 (" << opt.dump_scores_path << ")" << std::endl;
            return 1;
        }
        dump.close();
    }

    if (!found) {
        std::cout << "未达到目标分数，最终最佳: " << best_score << " 分 (线程" << best_tid
                  << ", 批次" << best_batch << ")，将使用此结果。" << std::endl;
    }

    // Host 端回放生成完整 history
    if (best_tid < 0 || best_batch < 0) {
        std::cout << "错误: 没有找到任何有效结果 (best_tid=" << best_tid << ", best_batch="
                  << best_batch << ")" << std::endl;
        return 1;
    }
    const uint64_t batch_seed = base_seed + static_cast<uint64_t>(best_batch) * NUM_THREADS;
    auto rng_seed = static_cast<uint32_t>(batch_seed + best_tid * 2654435761ULL);
    std::cout << "开始回放..." << std::endl;
    HostSimResult best = replay_game(rng_seed, best_tid % 4);
    std::cout << "GPU 分数: " << best_score << "  |  CPU 贪婪回放: " << best.score;
    if (found)
        std::cout << "  (GPU 达标后本局提前结束, 回放走到自然结束, 提交以回放结果为准)";
    std::cout << std::endl;

    // ── CPU expectimax 深搜（Top-K 种子） ──
    if (top_count > 0)
        std::cout << std::endl << "对 Top-" << top_count << " 种子进行 CPU expectimax 深搜..." << std::endl;
    for (int k = 0; k < top_count; k++) {
        uint64_t bs = base_seed + static_cast<uint64_t>(top_seeds[k].batch) * NUM_THREADS;
        auto rs = static_cast<uint32_t>(bs + top_seeds[k].tid * 2654435761ULL);
        if (HostSimResult result = replay_game_expectimax(rs, top_seeds[k].tid % 4);
            result.score > best.score) {
            std::cout << "  种子 #" << k << ": GPU=" << top_seeds[k].score
                      << " → CPU expectimax=" << result.score << " ⬆ 提升!" << std::endl;
            best = result;
        }
    }

    // ── 额外：独立 CPU expectimax 游戏（fresh seeds，不受 GPU 种子限制）──
    {
        // 指定 --seed 时也由此派生，保证整个运行（含最终 JSON）可复现
        auto fresh_seed = static_cast<uint32_t>(
            (opt.has_seed ? base_seed : static_cast<uint64_t>(time(nullptr))) ^ 0xDEADBEEF);
        for (int i = 0; i < 4; i++) {
            int strat = i & 3;
            if (HostSimResult result = replay_game_expectimax(fresh_seed + i * 999983, strat);
                result.score > best.score) {
                std::cout << "  独立 expectimax #" << i << " (策略" << strat << "): " << result.score
                          << " ⬆ 提升!" << std::endl;
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
            if (int v = best.final_grid[r * 4 + c]; v == 0) std::cout << "    _";
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
