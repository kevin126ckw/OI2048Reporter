#include "gpu_plan.cuh"

#include <cuda_runtime.h>

// ─── 枚举 CUDA 设备 ───────────────────────────────────────────────────
// 序号是 CUDA 运行时看到的序号（CUDA_VISIBLE_DEVICES 之后）。
// 某个设备属性查询失败时跳过该设备，而不是整体失败。
std::vector<GpuDeviceInfo> enumerate_gpu_devices(std::string *error) {
    std::vector<GpuDeviceInfo> devices;

    int count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err != cudaSuccess) {
        if (error) *error = cudaGetErrorString(count_err);
        return devices;
    }

    for (int ordinal = 0; ordinal < count; ++ordinal) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess) {
            cudaGetLastError(); // 清理错误状态，继续枚举其它设备
            continue;
        }
        GpuDeviceInfo info;
        info.ordinal = ordinal;
        info.sm_count = prop.multiProcessorCount;
        info.major = prop.major;
        info.minor = prop.minor;
        info.memory_bytes = prop.totalGlobalMem;
        info.name = prop.name;
        devices.push_back(info);
    }

    if (devices.empty() && error && error->empty())
        *error = "所有 CUDA 设备属性查询均失败";
    return devices;
}
