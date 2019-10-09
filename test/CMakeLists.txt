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

# unit test

add_executable(unitTest)

target_sources(unitTest 
PRIVATE
    unitTest/testSophusCeresInteraction.cpp
)

target_link_libraries(unitTest
PRIVATE
    athena
)

# draw cam

add_executable(drawCam)

target_sources(drawCam 
PRIVATE
    unitTest/simpleDrawingPangolin.cpp
)

target_link_libraries(drawCam
PRIVATE
    athena
)