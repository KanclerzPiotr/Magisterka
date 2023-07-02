add_subdirectory(logger)

add_executable(scalGraph main.cpp)
target_link_libraries(scalGraph PUBLIC logger)
target_include_directories(scalGraph PUBLIC ${PROJECT_SOURCE_DIR}/include)
