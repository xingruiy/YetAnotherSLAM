cmake_minimum_required(VERSION 2.8)
project(athena CXX)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)

add_library(${PROJECT_NAME} SHARED "")

target_sources(${PROJECT_NAME}
PRIVATE
    src/Map.cpp
    src/Frame.cpp
    src/MapPoint.cpp
    src/KeyFrame.cpp
    src/Mapping.cpp
    src/System.cpp
    src/Viewer.cpp
    src/Tracking.cpp

    src/ORBextractor.cpp
    src/Bundler.cpp
    src/Converter.cpp
    src/ORBmatcher.cpp
)

target_compile_features(${PROJECT_NAME}
PUBLIC
    cxx_std_11
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}  
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/tools/
    ${PROJECT_SOURCE_DIR}/tools/CameraOpenNI/include
    ${PROJECT_SOURCE_DIR}/tools/DENSE/include
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    Eigen3::Eigen
    ${g2o_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/tools/DENSE/lib/libDENSE.so
    ${PROJECT_SOURCE_DIR}/tools/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/tools/g2o/lib/libg2o.so
)

# Build Examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_executable(liveDemo ${PROJECT_SOURCE_DIR}/demo/LiveDemo.cpp)
target_link_libraries(liveDemo 
PRIVATE 
    athena 
    ${PROJECT_SOURCE_DIR}/tools/CameraOpenNI/lib/libCameraOpenNI.so
)