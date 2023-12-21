
set(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include/algorithms)

set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/delta.cpp
)

set(HEADERS
    ${INCLUDE_PATH}/delta.hpp
)

find_package(oneDPL REQUIRED)

add_library(algorithms ${SRC} ${HEADERS})
target_include_directories(algorithms PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(algorithms PUBLIC containers oneDPL)