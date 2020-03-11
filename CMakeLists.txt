cmake_minimum_required(VERSION 2.8)
project(YetAnotherSLAM CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin 0.5 REQUIRED)
find_package(g2o 1.0.0 REQUIRED)

add_library(${PROJECT_NAME} SHARED "")

target_sources(${PROJECT_NAME}
PRIVATE
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
    FullSystem/src/PoseSolver.cpp

    RGBDFusion/src/ImageProc.cu
    RGBDFusion/src/VoxelMap.cu
    RGBDFusion/src/RGBDTracking.cu
    RGBDFusion/src/MeshEngine.cu
    RGBDFusion/src/RayTraceEngine.cu
)

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
    ${PROJECT_SOURCE_DIR}/Thirdparty/
    ${PROJECT_SOURCE_DIR}/RGBDFusion/utils
    ${PROJECT_SOURCE_DIR}/OptBackend/include
    ${PROJECT_SOURCE_DIR}/RGBDFusion/include
    ${PROJECT_SOURCE_DIR}/FullSystem/include
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    Eigen3::Eigen
    g2o_core 
    g2o_types_sba
    g2o_types_sim3
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
)

# Build Examples
# Build Live Viewer
add_executable(asus_rgbd
    ${PROJECT_SOURCE_DIR}/Examples/asus_rgbd.cpp
)
target_include_directories(asus_rgbd 
PRIVATE
    ${PROJECT_SOURCE_DIR}/Thirdparty/OpenNI2/include
)
target_link_libraries(asus_rgbd 
PRIVATE 
    ${PROJECT_NAME} 
    ${PROJECT_SOURCE_DIR}/Thirdparty/OpenNI2/lib/libOpenNI2.so
)

add_executable(asus_recorded 
    ${PROJECT_SOURCE_DIR}/Examples/asus_recorded.cpp
)
target_link_libraries(asus_recorded
PRIVATE 
    ${PROJECT_NAME} 
)

# Build TUM Viewer
add_executable(tum_rgbd
    ${PROJECT_SOURCE_DIR}/Examples/tum_rgbd.cpp
)
target_link_libraries(tum_rgbd 
PRIVATE 
    ${PROJECT_NAME} 
)

# Build Vocabulary Converter
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Vocabulary)
add_executable(Text2Bin 
    Vocabulary/Text2Bin.cpp
)
target_include_directories(Text2Bin
PRIVATE
    ${PROJECT_SOURCE_DIR}/FullSystem/include
    ${PROJECT_SOURCE_DIR}/Thirdparty/
)
target_compile_features(Text2Bin
PRIVATE
    cxx_std_11
)
target_link_libraries(Text2Bin 
PRIVATE
    ${OpenCV_LIBS}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
)