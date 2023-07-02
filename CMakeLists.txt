cmake_minimum_required(VERSION 3.24.0)
project(scalGraph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic")

# set(SYCL_FLAGS 
#       "-fsycl" 
#       "-fsycl-targets=nvptx64-nvidia-cuda"
#       "-Wno-unknown-cuda-version"
#       "-Wno-linker-warnings")

add_subdirectory(src)
