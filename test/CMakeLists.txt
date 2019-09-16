project(testAPI CXX CUDA)

add_executable(liveViewer)

target_sources(liveViewer 
PRIVATE
    liveViewer/liveViewer.cu
    liveViewer/oniCamera.cpp
)

target_link_libraries(liveViewer
PRIVATE
    athena
    OpenNI2
)