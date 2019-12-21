cmake_minimum_required(VERSION 2.8)
project(OniCamera)

add_library(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
PRIVATE
    OniCamera.cpp
)

target_compile_features(${PROJECT_NAME}
PUBLIC
    cxx_std_11
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    /usr/local/lib/OpenNI2/libOpenNI2.so
)