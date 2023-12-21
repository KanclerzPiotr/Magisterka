add_subdirectory(containers)
add_subdirectory(algorithms)

add_executable(scalGraph main.cpp)
target_link_libraries(scalGraph PUBLIC containers algorithms)
target_include_directories(scalGraph PUBLIC ${PROJECT_SOURCE_DIR}/include)
