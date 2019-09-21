project(testAPI CXX CUDA)

add_executable(liveViewer)

target_sources(liveViewer 
PRIVATE
    liveViewer/liveViewer.cpp
    liveViewer/oniCamera.cpp
)

target_include_directories(liveViewer
PRIVATE
    /usr/local/include/OpenNI2
)

target_link_libraries(liveViewer
PRIVATE
    athena
    OpenNI2
)

# tum viewer

add_executable(tumViewer)

target_sources(tumViewer 
PRIVATE
    tumViewer/tumViewer.cpp
)

target_link_libraries(tumViewer
PRIVATE
    athena
)