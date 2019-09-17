cmake_minimum_required(VERSION 2.8)
project(athena CXX CUDA)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)

add_library(athena SHARED)

target_sources(athena
PRIVATE
    src/utils/frame.cu
    src/fullSystem/fullSystem.cu
    src/mapViewer/mapViewer.cu
    src/denseTracker/denseTracker.cu
    src/denseTracker/cudaImageProc.cu
    src/globalMapper/ceresSolver.cpp
    src/globalMapper/globalMap.cpp
    src/localMapper/localMapper.cu
    src/localMapper/denseMap.cu
)

target_compile_features(athena
PUBLIC
    cxx_std_11
)

target_include_directories(athena
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(athena
PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${Pangolin_LIBRARIES}
)

add_subdirectory(test)