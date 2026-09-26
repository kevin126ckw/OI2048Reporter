// ─── gpu_plan 单元测试（纯 CPU，不需要 GPU）────────────────────────────
// 验证：设备选择器解析、线程块划分的完整性与负载均衡性质。
// 运行: ctest -R gpu_plan  或直接执行 ./test_gpu_plan

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "gpu_plan.cuh"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::cout << "  [失败] " << __FILE__ << ":" << __LINE__ << "  "     \
                      << #cond;                                                 \
            std::cout << "  ";                                                  \
            std::printf(__VA_ARGS__);                                           \
            std::cout << std::endl;                                             \
        }                                                                       \
    } while (false)

GpuDeviceInfo make_device(const int ordinal, const int sm_count) {
    GpuDeviceInfo info;
    info.ordinal = ordinal;
    info.sm_count = sm_count;
    info.major = 8;
    info.minor = 6;
    info.memory_bytes = 16ULL * 1024 * 1024 * 1024;
    info.name = "Test GPU " + std::to_string(ordinal);
    return info;
}

// ─── 选择器解析 ──────────────────────────────────────────────────────
void test_selector_all_forms() {
    std::cout << "选择器: 基本形式" << std::endl;
    std::string error;
    std::vector<int> out;

    for (const std::string text : {"all", "ALL", "*", "", "  "}) {
        out.clear();
        CHECK(parse_gpu_selector(text, 3, &out, &error), "selector=%s", text.c_str());
        CHECK(out.size() == 3, "全部设备应有 3 个, 得到 %zu", out.size());
        CHECK(out[0] == 0 && out[1] == 1 && out[2] == 2, "序号应为 0,1,2");
    }

    out.clear();
    CHECK(parse_gpu_selector("0", 3, &out, &error), "单设备");
    CHECK(out.size() == 1 && out[0] == 0, "应为 {0}");

    out.clear();
    CHECK(parse_gpu_selector("0,2", 3, &out, &error), "列表");
    CHECK(out.size() == 2 && out[0] == 0 && out[1] == 2, "应为 {0,2}");

    out.clear();
    CHECK(parse_gpu_selector("2,0", 3, &out, &error), "列表保序");
    CHECK(out.size() == 2 && out[0] == 2 && out[1] == 0, "应为 {2,0}");

    out.clear();
    CHECK(parse_gpu_selector("0-2", 4, &out, &error), "区间");
    CHECK(out.size() == 3 && out[0] == 0 && out[1] == 1 && out[2] == 2, "应为 {0,1,2}");

    out.clear();
    CHECK(parse_gpu_selector(" 1 , 2 ", 3, &out, &error), "空白");
    CHECK(out.size() == 2 && out[0] == 1 && out[1] == 2, "应为 {1,2}");

    out.clear();
    CHECK(parse_gpu_selector("1,1,0", 2, &out, &error), "去重");
    CHECK(out.size() == 2 && out[0] == 1 && out[1] == 0, "应为 {1,0}");
}

void test_selector_errors() {
    std::cout << "选择器: 错误输入" << std::endl;
    std::string error;
    std::vector<int> out;

    const std::vector<std::string> bad = {"5", "a", "1,,2", "-1", "2-1", "1-", "0,", "x-y", "1.5"};
    for (const std::string &text : bad) {
        out.clear();
        error.clear();
        const bool ok = parse_gpu_selector(text, 2, &out, &error);
        CHECK(!ok, "应拒绝 \"%s\"", text.c_str());
        CHECK(!error.empty(), "应给出错误信息 \"%s\"", text.c_str());
    }

    out.clear();
    error.clear();
    CHECK(!parse_gpu_selector("0", 0, &out, &error), "设备数为 0 时序号越界");

    // 超大区间必须先校验范围再展开（否则会试图分配上亿个元素）
    out.clear();
    error.clear();
    CHECK(!parse_gpu_selector("0-999999999", 2, &out, &error), "超大区间应直接报错");
    CHECK(out.empty(), "超大区间不应产生任何序号");

    out.clear();
    error.clear();
    CHECK(!parse_gpu_selector("0-1-2", 3, &out, &error), "多段区间非法");
}

// ─── 线程块划分的通用不变量 ──────────────────────────────────────────
void check_partition_invariants(const std::vector<GpuDeviceInfo> &devices,
                                const int total_blocks, const int threads_per_block,
                                const int logical_per_device, const char *label) {
    std::string error;
    const std::vector<GpuSlice> slices =
        compute_gpu_slices(devices, total_blocks, threads_per_block, logical_per_device, &error);

    CHECK(!slices.empty(), "%s: 应能生成切片", label);
    if (slices.empty()) return;
    CHECK(error.empty(), "%s: 不应有错误信息", label);

    const std::size_t expected_slices = devices.size() * static_cast<std::size_t>(logical_per_device);
    CHECK(slices.size() == expected_slices, "%s: 切片数 %zu != %zu", label, slices.size(), expected_slices);

    int blocks_sum = 0;
    int next_offset = 0;
    for (std::size_t i = 0; i < slices.size(); ++i) {
        const GpuSlice &slice = slices[i];
        CHECK(slice.blocks >= 0, "%s: 块数不能为负", label);
        CHECK(slice.tid_offset == next_offset, "%s: 切片 %zu 偏移应为 %d, 得到 %d", label, i, next_offset,
              slice.tid_offset);
        CHECK(slice.threads == slice.blocks * threads_per_block, "%s: 切片 %zu 线程数不一致", label, i);
        CHECK(slice.logical_count == logical_per_device, "%s: 逻辑切片数不一致", label);
        CHECK(slice.ordinal == devices[i / static_cast<std::size_t>(logical_per_device)].ordinal,
              "%s: 切片 %zu 的设备序号错误", label, i);
        CHECK(slice.sm_count == devices[i / static_cast<std::size_t>(logical_per_device)].sm_count,
              "%s: 切片 %zu 的 SM 数错误", label, i);
        if (static_cast<int>(slices.size()) <= total_blocks) {
            CHECK(slice.blocks >= 1, "%s: 切片 %zu 至少应有 1 块", label, i);
        }
        blocks_sum += slice.blocks;
        next_offset += slice.threads;
    }

    CHECK(blocks_sum == total_blocks, "%s: 块数总和 %d != %d", label, blocks_sum, total_blocks);
    CHECK(next_offset == total_blocks * threads_per_block, "%s: 线程总数 %d != %d", label, next_offset,
          total_blocks * threads_per_block);

    // 确定性：同样输入必须得到同样结果
    const std::vector<GpuSlice> again =
        compute_gpu_slices(devices, total_blocks, threads_per_block, logical_per_device, nullptr);
    bool identical = again.size() == slices.size();
    for (std::size_t i = 0; identical && i < slices.size(); ++i) {
        identical = again[i].blocks == slices[i].blocks &&
                    again[i].tid_offset == slices[i].tid_offset &&
                    again[i].ordinal == slices[i].ordinal;
    }
    CHECK(identical, "%s: 划分结果不确定", label);
}

void test_partition_matrix() {
    std::cout << "划分: 各种设备与块数组合" << std::endl;
    const int tpb = 256;

    const std::vector<std::vector<GpuDeviceInfo>> device_sets = {
        {make_device(0, 58)},
        {make_device(0, 58), make_device(1, 58)},
        {make_device(0, 58), make_device(1, 16)},
        {make_device(0, 132), make_device(1, 58), make_device(2, 20)},
        {make_device(0, 0)}, // SM 数缺失时按权重 1 处理
        {make_device(0, 0), make_device(1, 0)},
        {make_device(0, 80), make_device(1, 80), make_device(2, 80), make_device(3, 80)},
    };
    const std::vector<int> block_counts = {1, 2, 3, 4, 7, 16, 100, 256, 1024};

    for (const auto &devices : device_sets) {
        for (const int blocks : block_counts) {
            for (const int logical : {1, 2, 3, 4}) {
                const std::string label = "devices=" + std::to_string(devices.size()) +
                                          " blocks=" + std::to_string(blocks) +
                                          " logical=" + std::to_string(logical);
                check_partition_invariants(devices, blocks, tpb, logical, label.c_str());
            }
        }
    }
}

void test_partition_balance() {
    std::cout << "划分: 负载均衡" << std::endl;
    const int tpb = 256;
    const int blocks = 1024;

    // 单卡单切片：应独占全部线程块
    {
        std::string error;
        const auto slices = compute_gpu_slices({make_device(0, 58)}, blocks, tpb, 1, &error);
        CHECK(slices.size() == 1, "单卡应只有 1 个切片");
        if (slices.size() == 1) {
            CHECK(slices[0].blocks == blocks, "单卡应拿到全部 %d 块", blocks);
            CHECK(slices[0].threads == blocks * tpb, "单卡线程数应为 %d", blocks * tpb);
            CHECK(slices[0].tid_offset == 0, "单卡偏移应为 0");
        }
    }

    // 同构双卡：应严格均分
    {
        std::string error;
        const auto slices = compute_gpu_slices({make_device(0, 58), make_device(1, 58)}, blocks, tpb, 1, &error);
        CHECK(slices.size() == 2, "双卡应有 2 个切片");
        if (slices.size() == 2) {
            CHECK(slices[0].blocks == 512 && slices[1].blocks == 512, "同构双卡应各 512 块, 得到 %d/%d",
                  slices[0].blocks, slices[1].blocks);
            CHECK(slices[1].tid_offset == 512 * tpb, "第二个切片偏移错误");
        }
    }

    // 异构双卡：块数比例应接近 SM 比例（58 : 16）
    {
        std::string error;
        const auto slices = compute_gpu_slices({make_device(0, 58), make_device(1, 16)}, blocks, tpb, 1, &error);
        CHECK(slices.size() == 2, "异构双卡应有 2 个切片");
        if (slices.size() == 2) {
            CHECK(slices[0].blocks > slices[1].blocks, "SM 多的卡应分到更多块");
            const double actual = static_cast<double>(slices[0].blocks) / slices[1].blocks;
            const double expected = 58.0 / 16.0;
            CHECK(std::fabs(actual - expected) / expected < 0.05,
                  "块数比例 %.3f 应接近 SM 比例 %.3f", actual, expected);
        }
    }

    // 同卡多逻辑切片：应均分（相差不超过 1 块）
    {
        std::string error;
        const auto slices = compute_gpu_slices({make_device(0, 58)}, blocks, tpb, 4, &error);
        CHECK(slices.size() == 4, "4 个逻辑切片");
        int min_blocks = blocks;
        int max_blocks = 0;
        for (const auto &slice : slices) {
            min_blocks = std::min(min_blocks, slice.blocks);
            max_blocks = std::max(max_blocks, slice.blocks);
        }
        CHECK(max_blocks - min_blocks <= 1, "逻辑切片应均分, 得到 %d..%d", min_blocks, max_blocks);
    }
}

void test_partition_more_slices_than_blocks() {
    std::cout << "划分: 切片数多于块数" << std::endl;
    const int tpb = 256;
    const std::vector<GpuDeviceInfo> devices = {make_device(0, 58), make_device(1, 58)};

    std::string error;
    const auto slices = compute_gpu_slices(devices, 3, tpb, 2, &error); // 4 个切片, 只有 3 块
    CHECK(slices.size() == 4, "应有 4 个切片");
    if (slices.size() == 4) {
        CHECK(slices[0].blocks == 1 && slices[1].blocks == 1 && slices[2].blocks == 1,
              "前 3 个切片各 1 块");
        CHECK(slices[3].blocks == 0, "最后一个切片应为 0 块");
        CHECK(slices[3].threads == 0, "0 块切片线程数应为 0");
    }
}

void test_partition_invalid_input() {
    std::cout << "划分: 非法输入" << std::endl;
    std::string error;
    std::vector<GpuSlice> slices;

    slices = compute_gpu_slices({}, 1024, 256, 1, &error);
    CHECK(slices.empty() && !error.empty(), "无设备应报错");

    slices = compute_gpu_slices({make_device(0, 58)}, 0, 256, 1, &error);
    CHECK(slices.empty() && !error.empty(), "块数为 0 应报错");

    slices = compute_gpu_slices({make_device(0, 58)}, 1024, 0, 1, &error);
    CHECK(slices.empty() && !error.empty(), "线程块大小为 0 应报错");

    slices = compute_gpu_slices({make_device(0, 58)}, 1024, 256, 0, &error);
    CHECK(slices.empty() && !error.empty(), "逻辑切片数为 0 应报错");
}

} // namespace

int main() {
    test_selector_all_forms();
    test_selector_errors();
    test_partition_matrix();
    test_partition_balance();
    test_partition_more_slices_than_blocks();
    test_partition_invalid_input();

    std::cout << std::endl << (g_failures == 0 ? "全部通过" : "存在失败") << ": " << g_checks
              << " 项检查, " << g_failures << " 项失败" << std::endl;
    return g_failures == 0 ? 0 : 1;
}
