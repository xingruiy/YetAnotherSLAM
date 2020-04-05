cmake_minimum_required(VERSION 2.8)
project(YetAnotherSLAM CXX CUDA)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin 0.5 REQUIRED)
find_package(g2o 1.0.0 REQUIRED)

add_library(${PROJECT_NAME} SHARED "")

target_sources(${PROJECT_NAME}
PRIVATE
    src/CoreSystem/Map.cpp
    src/CoreSystem/Frame.cpp
    src/CoreSystem/MapDrawer.cpp
    src/CoreSystem/MapPoint.cpp
    src/CoreSystem/KeyFrame.cpp
    src/CoreSystem/LocalMapping.cpp
    src/CoreSystem/CoreSystem.cc
    src/CoreSystem/Viewer.cpp
    src/CoreSystem/Tracking.cpp
    src/CoreSystem/Optimizer.cpp
    src/CoreSystem/ORBMatcher.cpp
    src/CoreSystem/GlobalDef.cpp
    src/CoreSystem/LoopClosing.cpp
    src/CoreSystem/KeyFrameDatabase.cpp
    src/CoreSystem/ORBextractor.cpp
    src/CoreSystem/MapManager.cpp
    src/CoreSystem/PoseSolver.cpp

    src/RGBDOdometry//ImageProc.cu
    src/RGBDOdometry//VoxelMap.cu
    src/RGBDOdometry//CoarseTracking.cu
    src/RGBDOdometry//MeshEngine.cu
    src/RGBDOdometry//RayTraceEngine.cu

    src/IOSystem/PangolinViewer.cc

    src/utils/GlobalSettings.cc
    src/utils/cudaDeviceFuncs.cu
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
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rdparty/
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
    ${PROJECT_SOURCE_DIR}/3rdparty/DBoW2/lib/libDBoW2.so
)

add_executable(liveRecorded
    ${PROJECT_SOURCE_DIR}/src/Examples/OpenNI2/liveRecordings.cc
)
target_link_libraries(liveRecorded 
PRIVATE 
    ${PROJECT_NAME} 
    ${PROJECT_SOURCE_DIR}/3rdparty/OpenNI/lib/libOpenNI.so
)

# Build Examples
# Build Live Viewer
# add_executable(asus_rgbd
#     ${PROJECT_SOURCE_DIR}/Examples/asus_rgbd.cpp
# )
# target_include_directories(asus_rgbd 
# PRIVATE
#     ${PROJECT_SOURCE_DIR}/3rdparty/OpenNI2/include
# )
# target_link_libraries(asus_rgbd 
# PRIVATE 
#     ${PROJECT_NAME} 
#     ${PROJECT_SOURCE_DIR}/3rdparty/OpenNI/lib/libOpenNI.so
# )

# add_executable(asus_recorded 
#     ${PROJECT_SOURCE_DIR}/Examples/asus_recorded.cpp
# )

# target_link_libraries(asus_recorded
# PRIVATE 
#     ${PROJECT_NAME} 
# )

# Build TUM Viewer
# add_executable(tum_rgbd
#     ${PROJECT_SOURCE_DIR}/Examples/tum_rgbd.cpp
# )
# target_link_libraries(tum_rgbd 
# PRIVATE 
#     ${PROJECT_NAME} 
# )

# Build Vocabulary Converter
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Vocabulary)
# add_executable(Text2Bin 
#     Vocabulary/Text2Bin.cpp
# )
# target_include_directories(Text2Bin
# PRIVATE
#     ${PROJECT_SOURCE_DIR}/FullSystem/include
#     ${PROJECT_SOURCE_DIR}/3rdparty/
# )
# target_compile_features(Text2Bin
# PRIVATE
#     cxx_std_11
# )
# target_link_libraries(Text2Bin 
# PRIVATE
#     ${OpenCV_LIBS}
#     ${PROJECT_SOURCE_DIR}/3rdparty/DBoW2/lib/libDBoW2.so
# )