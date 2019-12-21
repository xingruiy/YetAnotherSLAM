cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} 
SHARED
    src/ORBextractor.cc
)

target_include_directories(${PROJECT_NAME} 
PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
)
