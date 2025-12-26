cmake_minimum_required(VERSION 3.10)
project(md_nesc)

set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
cmake_policy(SET CMP0015 NEW)

#SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_BUILD_TYPE "Debug")

IF (CMAKE_BUILD_TYPE STREQUAL Debug) 
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()

SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall")


# LibTorch配置
# 请根据实际安装路径修改TORCH_PATH
set(TORCH_PATH "/home/pkg/libtorch_cu128/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
find_package(Torch REQUIRED)

# 强制移除可能存在的 ABI=0 标志
string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=0" "-D_GLIBCXX_USE_CXX11_ABI=1" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
# 并且全局定义 ABI=1
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)


# 主程序
add_executable(mds_main main_gpu.cpp)
target_compile_options(mds_main PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(mds_main PRIVATE "${TORCH_LIBRARIES}")

