cmake_minimum_required(VERSION 2.8)
project(athena CXX CUDA)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Ceres 2.0 REQUIRED)

add_library(athena SHARED)

target_sources(athena
PRIVATE
    src/utils/map.cpp
    src/utils/frame.cpp
    src/utils/mapPoint.cpp
    src/fullSystem/fullSystem.cu
    src/mapViewer/mapViewer.cpp
    src/denseTracker/denseTracker.cu
    src/denseTracker/cudaImageProc.cu
    src/optimizer/localOptimizer.cpp
    src/optimizer/featureMatcher.cpp
    src/localMapper/denseMap.cu
    src/localMapper/localMapper.cpp
    src/localMapper/depthFusion.cu
    src/localMapper/meshScene.cu
    src/localMapper/raytracing.cu
    src/loopClosure/loopCloser.cpp
    src/localizer/localizer.cpp
)

target_compile_features(athena
PUBLIC
    cxx_std_11
)

set_target_properties(athena
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 11
)

target_include_directories(athena
PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}  
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(athena
PUBLIC
    Eigen3::Eigen
    ${CUDA_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Pangolin_LIBRARIES}
)

add_subdirectory(test)