cmake_minimum_required(VERSION 2.8)
project(athena CXX CUDA)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Ceres 2.0 REQUIRED)

add_library(athena SHARED)

target_sources(athena
PRIVATE
    src/dataStruct/map.cpp
    src/dataStruct/frame.cpp
    src/dataStruct/mapPoint.cpp
    src/dataStruct/keyFrame.cpp

    src/utils/mapCUDA.cu

    src/fullSystem/fullSystem.cpp

    src/mapViewer/mapViewer.cpp

    src/loopCloser/loopCloser.cpp

    src/denseTracker/denseTracker.cu
    src/denseTracker/cudaImageProc.cu

    src/localMapper/localizer.cpp
    src/localMapper/localMapper.cpp
    src/localMapper/featureMatcher.cpp

    src/denseMapper/denseMap.cu
    src/denseMapper/denseMapper.cpp
    src/denseMapper/depthFusion.cu
    src/denseMapper/meshScene.cu
    src/denseMapper/raytracing.cu
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