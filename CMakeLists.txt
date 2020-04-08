cmake_minimum_required(VERSION 2.8)
project(slam CXX CUDA)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin 0.5 REQUIRED)
find_package(g2o 1.0.0 REQUIRED)

add_library(${PROJECT_NAME} SHARED
FullSystem/src/Map.cpp
FullSystem/src/Frame.cpp
FullSystem/src/MapDrawer.cpp
FullSystem/src/MapPoint.cpp
FullSystem/src/KeyFrame.cpp
FullSystem/src/LocalMapping.cpp
FullSystem/src/System.cpp
FullSystem/src/Viewer.cpp
FullSystem/src/Tracking.cpp
FullSystem/src/Optimizer.cpp
FullSystem/src/ORBMatcher.cpp
FullSystem/src/GlobalDef.cpp
FullSystem/src/LoopClosing.cpp
FullSystem/src/KeyFrameDatabase.cpp
FullSystem/src/ORBextractor.cpp
FullSystem/src/MapManager.cpp
FullSystem/src/PoseSolver.cpp

RGBDFusion/src/ImageProc.cu
RGBDFusion/src/VoxelMap.cu
RGBDFusion/src/CoarseTracking.cu
RGBDFusion/src/MeshEngine.cu
RGBDFusion/src/RayTraceEngine.cu
utils/cudaDeviceFuncs.cu)

set_target_properties(${PROJECT_NAME}
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 11
)

target_compile_features(${PROJECT_NAME}
PUBLIC
    cxx_std_11
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}  
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/3rdparty/
    ${PROJECT_SOURCE_DIR}/RGBDFusion/include
    ${PROJECT_SOURCE_DIR}/FullSystem/include
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    Eigen3::Eigen 
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    g2o_core g2o_types_sba g2o_types_sim3
    ${Pangolin_LIBRARIES} 
    ${PROJECT_SOURCE_DIR}/3rdparty/DBoW2/lib/libDBoW2.so
)

# Build Examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_SOURCE_DIR}/bin/)
add_executable(NI2Live Examples/OpenNI2/NI2Live.cc)
target_include_directories(NI2Live PRIVATE 3rdparty/OpenNI2/include)
target_link_libraries(NI2Live PRIVATE ${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/3rdparty/OpenNI/lib/libOpenNI.so)

add_executable(NI2Record Examples/OpenNI2/NI2Record.cc)
target_link_libraries(NI2Record PRIVATE  ${PROJECT_NAME})

find_package(PythonLibs 3 REQUIRED)
add_executable(TUM ${PROJECT_SOURCE_DIR}/Examples/TUM-RGBD/TUM.cc)
target_link_libraries(TUM PRIVATE ${PROJECT_NAME} ${PYTHON_LIBRARIES})

# Test python bindings
# add_subdirectory(pybind11)
# pybind11_add_module(pyslam ${PROJECT_SOURCE_DIR}/Examples/PythonBindings.cpp)
# target_link_libraries(pyslam PRIVATE ${PROJECT_NAME})
