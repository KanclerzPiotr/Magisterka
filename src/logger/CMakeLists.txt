add_library(logger "")

target_include_directories(logger PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_sources(logger 
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/Logger.cpp
        ${CMAKE_CURRENT_LIST_DIR}/LogFileWriter.cpp     
        ${CMAKE_CURRENT_LIST_DIR}/LogContainer.cpp     
)

