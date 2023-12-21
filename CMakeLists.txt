cmake_minimum_required(VERSION 3.24.0)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_STANDARD 20)

project(scalGraph LANGUAGES CXX)

set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wdouble-promotion -Wshadow -Wuninitialized -Wmissing-declarations -Woverloaded-virtual")
set(DISABLED_WARNINGS "-Wno-c++20-extensions -Wno-unknown-cuda-version -Wno-unused -Wno-unused-parameter")

set(OPT_FLAGS "-march=native -mtune=native -Ofast -fomit-frame-pointer")

SET(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -sycl-std=2020 -fsycl-unnamed-lambda")

set(CMAKE_CXX_FLAGS_RELEASE "${OPT_FLAGS} ${WARNING_FLAGS} ${DISABLED_WARNINGS}")
set(CMAKE_CXX_FLAGS_DEBUG " ${WARNING_FLAGS}  ${DISABLED_WARNINGS} -g3 -Og")

add_subdirectory(src)
