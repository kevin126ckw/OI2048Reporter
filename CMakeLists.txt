cmake_minimum_required(VERSION 3.18)
project(OI2048Reporter CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 自动检测 GPU 架构
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(OI2048Reporter
    main.cu
    kernel.cu
    gpu_plan.cu
    replay.cu
    submit_format.cu
)

set_target_properties(OI2048Reporter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(OI2048Reporter PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -diag-suppress=1835,20091>
)

# ─── 测试 ────────────────────────────────────────────────────────────
# test_gpu_plan 是纯 C++ 单元测试（不需要 GPU）：验证设备选择与线程块划分
option(OI2048_BUILD_TESTS "构建单元测试" ON)

if (OI2048_BUILD_TESTS)
    enable_testing()

    add_executable(test_gpu_plan tests/test_gpu_plan.cpp)
    target_include_directories(test_gpu_plan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME gpu_plan COMMAND test_gpu_plan)

    # 多 GPU 等价性测试：需要能访问 GPU，无 GPU 时脚本自行跳过
    add_test(NAME gpu_equivalence
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_gpu_equivalence.sh
                     $<TARGET_FILE:OI2048Reporter>)
endif()
