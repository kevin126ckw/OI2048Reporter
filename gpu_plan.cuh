#pragma once

// ─── 多 GPU 计划：设备信息、设备选择、线程块划分 ──────────────────────
// 本文件是纯 C++（不包含 CUDA 头文件），因此可以在没有 GPU 的机器上做单元测试：
//   tests/test_gpu_plan.cpp
// 只有设备枚举 enumerate_gpu_devices() 需要 CUDA 运行时（实现在 gpu_plan.cu）。

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

// ─── 单个物理 GPU 的信息 ──────────────────────────────────────────────
struct GpuDeviceInfo {
    int ordinal = 0;            // CUDA 设备序号（受 CUDA_VISIBLE_DEVICES 影响）
    int sm_count = 0;           // SM 数量，作为负载权重
    int major = 0;              // 计算能力主版本
    int minor = 0;              // 计算能力次版本
    std::size_t memory_bytes = 0;
    std::string name;
};

// ─── 一份线程块切片（一个物理 GPU，或它的第 k 份逻辑切片）─────────────
// 所有切片合起来恰好覆盖 [0, NUM_THREADS) 的全部线程（全局线程号 tid）
struct GpuSlice {
    int ordinal = 0;          // CUDA 设备序号
    int logical_index = 0;    // 该物理设备上的第几份逻辑切片
    int logical_count = 1;    // 该物理设备被切成几份（>1 仅用于单卡模拟多卡）
    int blocks = 0;           // 分配的线程块数（0 表示该切片不参与计算）
    int tid_offset = 0;       // 全局线程号起点
    int threads = 0;          // 线程数 = blocks * threads_per_block
    int sm_count = 0;         // 物理设备 SM 数
    std::string name;         // 设备名（仅用于输出）
};

// ─── 解析 GPU 选择器 ──────────────────────────────────────────────────
// 支持 "all"/""/"*"（全部设备）、"0"、"0,2"、"1,0"（保留顺序、自动去重）、"0-2"（区间）
// 序号是 CUDA 运行时看到的序号（即 CUDA_VISIBLE_DEVICES 生效之后的序号）。
inline bool parse_gpu_selector(const std::string &selector, int device_count,
                               std::vector<int> *out_ordinals, std::string *error);

// ─── 把 total_blocks 个线程块分配到各个切片 ───────────────────────────
// 规则：
//   1. 先给每个切片保底 1 块，剩余块按 SM 数量加权用最大余数法分配（整数运算，结果确定）；
//   2. 切片数 > total_blocks 时，前 total_blocks 个切片各 1 块，其余 0 块（调用方需跳过）；
//   3. 切片顺序为「设备优先」，同设备内按逻辑切片下标排列，tid_offset 连续且从 0 开始。
inline std::vector<GpuSlice> compute_gpu_slices(const std::vector<GpuDeviceInfo> &devices,
                                                int total_blocks, int threads_per_block,
                                                int logical_per_device, std::string *error);

// ─── 枚举 CUDA 设备（需要 CUDA 运行时，实现在 gpu_plan.cu）────────────
std::vector<GpuDeviceInfo> enumerate_gpu_devices(std::string *error);

// ─────────────────────────────────────────────────────────────────────
// 以下是纯 C++ 的内联实现
// ─────────────────────────────────────────────────────────────────────

namespace gpu_plan_detail {

inline std::string trim_copy(const std::string &s) {
    const char *ws = " \t\r\n";
    const std::size_t begin = s.find_first_not_of(ws);
    if (begin == std::string::npos) return {};
    const std::size_t end = s.find_last_not_of(ws);
    return s.substr(begin, end - begin + 1);
}

inline bool parse_long_token(const std::string &token, long long *out) {
    if (token.empty()) return false;
    char *end = nullptr;
    const long long value = std::strtoll(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0') return false;
    *out = value;
    return true;
}

} // namespace gpu_plan_detail

inline bool parse_gpu_selector(const std::string &selector, const int device_count,
                               std::vector<int> *out_ordinals, std::string *error) {
    using namespace gpu_plan_detail;
    out_ordinals->clear();
    const std::string text = trim_copy(selector);

    if (text.empty() || text == "all" || text == "ALL" || text == "*") {
        for (int i = 0; i < device_count; ++i) out_ordinals->push_back(i);
        return true;
    }

    std::vector<int> requested;
    const auto out_of_range = [&](const long long id) {
        if (error) {
            *error = "GPU 序号 " + std::to_string(id) + " 越界（当前可见设备数: " +
                     std::to_string(device_count) + "）";
        }
        return false;
    };

    std::size_t pos = 0;
    while (true) {
        const std::size_t comma = text.find(',', pos);
        const std::string token = trim_copy(
            text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));
        if (token.empty()) {
            if (error) *error = "GPU 列表中存在空项: \"" + selector + "\"";
            return false;
        }

        const std::size_t dash = token.find('-');
        if (dash != std::string::npos) {
            long long first = 0;
            long long last = 0;
            if (!parse_long_token(trim_copy(token.substr(0, dash)), &first) ||
                !parse_long_token(trim_copy(token.substr(dash + 1)), &last)) {
                if (error) *error = "无法解析 GPU 区间: \"" + token + "\"";
                return false;
            }
            if (first > last) {
                if (error) *error = "GPU 区间起止颠倒: \"" + token + "\"";
                return false;
            }
            // 先校验范围再展开，避免 "--gpus 0-999999999" 之类的输入撑爆内存
            if (first < 0) return out_of_range(first);
            if (last >= device_count) return out_of_range(last);
            for (long long id = first; id <= last; ++id) requested.push_back(static_cast<int>(id));
        } else {
            long long id = 0;
            if (!parse_long_token(token, &id)) {
                if (error) *error = "无法解析 GPU 序号: \"" + token + "\"";
                return false;
            }
            if (id < 0 || id >= device_count) return out_of_range(id);
            requested.push_back(static_cast<int>(id));
        }

        if (comma == std::string::npos) break;
        pos = comma + 1;
    }

    for (const int id : requested) {
        bool duplicate = false;
        for (const int kept : *out_ordinals) if (kept == id) { duplicate = true; break; }
        if (!duplicate) out_ordinals->push_back(id);
    }
    return true;
}

inline std::vector<GpuSlice> compute_gpu_slices(const std::vector<GpuDeviceInfo> &devices,
                                                const int total_blocks,
                                                const int threads_per_block,
                                                const int logical_per_device,
                                                std::string *error) {
    std::vector<GpuSlice> slices;

    const auto fail = [error](const std::string &message) {
        if (error) *error = message;
        return std::vector<GpuSlice>{};
    };

    if (devices.empty()) return fail("没有可用的 GPU 设备");
    if (total_blocks <= 0) return fail("线程块总数必须为正数");
    if (threads_per_block <= 0) return fail("每块线程数必须为正数");
    if (logical_per_device <= 0) return fail("每卡逻辑切片数必须为正数");

    // 展开逻辑切片：同一物理设备复制 logical_per_device 份（单卡模拟多卡）
    slices.reserve(devices.size() * static_cast<std::size_t>(logical_per_device));
    for (const GpuDeviceInfo &device : devices) {
        for (int k = 0; k < logical_per_device; ++k) {
            GpuSlice slice;
            slice.ordinal = device.ordinal;
            slice.logical_index = k;
            slice.logical_count = logical_per_device;
            slice.sm_count = device.sm_count;
            slice.name = device.name;
            slices.push_back(slice);
        }
    }

    if (static_cast<int>(slices.size()) > total_blocks) {
        // 块数不足以覆盖所有切片：前面的切片各 1 块，其余置 0，调用方需跳过 0 块切片
        for (std::size_t i = 0; i < slices.size(); ++i)
            slices[i].blocks = static_cast<int>(i) < total_blocks ? 1 : 0;
    } else {
        const long long count = static_cast<long long>(slices.size());
        const long long remaining = total_blocks - count; // 保底 1 块之后的剩余块数

        long long total_weight = 0;
        for (const GpuSlice &slice : slices) {
            total_weight += slice.sm_count > 0 ? slice.sm_count : 1;
        }

        std::vector<long long> remainder(slices.size(), 0);
        long long assigned = 0;
        for (std::size_t i = 0; i < slices.size(); ++i) {
            const long long weight = slices[i].sm_count > 0 ? slices[i].sm_count : 1;
            const long long numerator = remaining * weight;
            const long long share = numerator / total_weight;
            remainder[i] = numerator % total_weight;
            slices[i].blocks = 1 + static_cast<int>(share);
            assigned += share;
        }

        // 最大余数法补齐（相同余数按下标先后，保证结果确定）
        long long leftover = remaining - assigned;
        if (leftover > 0) {
            std::vector<std::size_t> order(slices.size());
            for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::stable_sort(order.begin(), order.end(),
                             [&remainder](const std::size_t a, const std::size_t b) {
                                 if (remainder[a] != remainder[b]) return remainder[a] > remainder[b];
                                 return a < b;
                             });
            for (long long k = 0; k < leftover; ++k)
                slices[order[static_cast<std::size_t>(k) % order.size()]].blocks += 1;
        }
    }

    // 计算全局线程号偏移
    int offset = 0;
    for (GpuSlice &slice : slices) {
        slice.tid_offset = offset;
        slice.threads = slice.blocks * threads_per_block;
        offset += slice.threads;
    }
    return slices;
}
