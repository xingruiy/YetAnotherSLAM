cmake_minimum_required(VERSION 2.8)
project(slam CXX CUDA)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin 0.5 REQUIRED)
find_package(g2o 1.0.0 REQUIRED)

add_library(${PROJECT_NAME} SHARED
src/_core/Map.cpp
src/_core/Frame.cpp
src/_core/MapDrawer.cpp
src/_core/MapPoint.cpp
src/_core/KeyFrame.cpp
src/_core/LocalMapping.cpp
src/_core/System.cpp
src/_core/Viewer.cpp
src/_core/Tracking.cpp
src/_core/Optimizer.cpp
src/_core/ORBMatcher.cpp
src/_core/GlobalDef.cpp
src/_core/LoopClosing.cpp
src/_core/KeyFrameDatabase.cpp
src/_core/ORBextractor.cpp
src/_core/MapManager.cpp
src/_core/PoseSolver.cpp
src/_cuda/ImageProc.cu
src/_cuda/VoxelMap.cu
src/_cuda/CoarseTracking.cu
src/_cuda/MeshEngine.cu
src/_cuda/RayTraceEngine.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES 
CUDA_SEPARABLE_COMPILATION ON
POSITION_INDEPENDENT_CODE ON
CUDA_STANDARD 11)

target_compile_features(${PROJECT_NAME} PUBLIC
cxx_std_11)

target_include_directories(${PROJECT_NAME} PUBLIC
${CUDA_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
${Pangolin_INCLUDE_DIRS}  
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/3rdparty/
${PROJECT_SOURCE_DIR}/src/_core
${PROJECT_SOURCE_DIR}/src/_cuda
${PROJECT_SOURCE_DIR}/src/_utils)

target_link_libraries(${PROJECT_NAME} PUBLIC
Eigen3::Eigen 
${OpenCV_LIBS}
${CUDA_LIBRARIES}
g2o_core g2o_types_sba
g2o_types_sim3
${Pangolin_LIBRARIES} 
${PROJECT_SOURCE_DIR}/3rdparty/DBoW2/lib/libDBoW2.so)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  
${CMAKE_SOURCE_DIR}/bin/)

add_executable(live 
src/main_ni2.cpp)
target_include_directories(live PRIVATE 
3rdparty/OpenNI2/include)
target_link_libraries(live PRIVATE 
${PROJECT_NAME} 
${PROJECT_SOURCE_DIR}/3rdparty/OpenNI/lib/libOpenNI.so)

add_executable(tum 
src/main_tum.cpp)
target_link_libraries(tum PRIVATE 
${PROJECT_NAME})