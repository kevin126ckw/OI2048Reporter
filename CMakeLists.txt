cmake_minimum_required(VERSION 3.18)
project(OI2048Reporter CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 自动检测 GPU 架构
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(OI2048Reporter main.cu)

set_target_properties(OI2048Reporter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(OI2048Reporter PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)
