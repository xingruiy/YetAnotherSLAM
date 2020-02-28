cmake_minimum_required(VERSION 2.8)
project(YetAnotherSLAM CXX)

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
    FullSystem/src/Bundler.cpp
    FullSystem/src/ORBMatcher.cpp
    FullSystem/src/GlobalDef.cpp
    FullSystem/src/LoopClosing.cpp
    FullSystem/src/KeyFrameDatabase.cpp
    FullSystem/src/ORBextractor.cpp
    FullSystem/src/Converter.cc
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
    ${PROJECT_SOURCE_DIR}/FullSystem/include
    ${PROJECT_SOURCE_DIR}/Thirdparty/RGBDSLAM/include
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    Eigen3::Eigen
    g2o_core 
    g2o_types_sba
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/RGBDSLAM/lib/libRGBDSLAM.so
)

# Build Examples
# Build Live Viewer
add_executable(liveDemo 
    ${PROJECT_SOURCE_DIR}/Examples/LiveDemo.cpp
)
target_include_directories(liveDemo 
PRIVATE
    ${PROJECT_SOURCE_DIR}/Thirdparty/OpenNI2/include
)
target_link_libraries(liveDemo 
PRIVATE 
    ${PROJECT_NAME} 
    ${PROJECT_SOURCE_DIR}/Thirdparty/OpenNI2/lib/libOpenNI2.so
)

# Build TUM Viewer
add_executable(tumDemo 
    ${PROJECT_SOURCE_DIR}/Examples/TUMDemo.cpp
)
target_link_libraries(tumDemo 
PRIVATE 
    ${PROJECT_NAME} 
)
