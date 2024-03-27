
set(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include/containers)

set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/AdjeacenacyMatrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CSR.cpp
)

set(HEADERS
    ${INCLUDE_PATH}/AdjeacenacyMatrix.hpp
    ${INCLUDE_PATH}/CSR.hpp
)

add_library(containers ${SRC} ${HEADERS})
target_include_directories(containers PUBLIC ${PROJECT_SOURCE_DIR}/include)