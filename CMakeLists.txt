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
    src/LocalMapping.cpp
    src/FullSystem.cpp
    src/Viewer.cpp
    src/Tracking.cpp

    src/ORBextractor.cpp
    src/Optimizer.cpp
    src/Converter.cpp
    src/ORBmatcher.cpp
)

target_compile_features(${PROJECT_NAME}
PUBLIC
    cxx_std_11
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}  
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party/
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    Eigen3::Eigen
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/third_party/DENSE/lib/libDENSE.so
    ${PROJECT_SOURCE_DIR}/third_party/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/third_party/g2o/lib/libg2o.so
)

# Build Examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_executable(liveDemo ${PROJECT_SOURCE_DIR}/demo/LiveDemo.cpp)
target_link_libraries(liveDemo 
PRIVATE 
    athena 
    ${PROJECT_SOURCE_DIR}/third_party/ONI_Camera/lib/libOniCamera.so
)

add_executable(testRun ${PROJECT_SOURCE_DIR}/demo/TestRun.cpp)
target_link_libraries(testRun 
PRIVATE 
    athena 
)