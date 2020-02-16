#!/bin/bash

BuildType=RELEASE
RED='\033[0;31m'
NC='\033[0m'

if [ -n "$1" ]; then
    if [ "$1" -eq 1 ]; then
        BuildType=DEBUG
    fi
fi

echo -e "CMAKE_BUILD_TYPE is set to: ${RED}$BuildType${NC}"
echo "Configuring and building tools/CameraOpenNI ..."

cd ./tools/CameraOpenNI
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

echo "Configuring and building tools/DatasetLoader ..."

cd ../../DatasetLoader/
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

echo "Configuring and building tools/CudaUtilities ..."

cd ../../CudaUtilities/
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

cd ../../ORB_SLAM2
echo "Configuring and building tools/ORB_SLAM2/Thirdparty/DBoW2 ..."

cd ./Thirdparty/DBoW2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

echo "Configuring and building tools/ORB_SLAM2 ..."

cd ../../../
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

echo "Uncompress vocabulary ..."

cd ../../../vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "building SLAM..."

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=$BuildType
make -j
