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
echo "Configuring and building Thirdparty/CameraOpenNI ..."

cd ./Thirdparty/CameraOpenNI
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

echo "Configuring and building Thirdparty/RGBDSLAM ..."

cd ../../RGBDSLAM/
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

echo "Configuring and building Thirdparty/ORB_SLAM2/Thirdparty/DBoW2 ..."

cd ../../DBoW2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

echo "Uncompress vocabulary ..."

cd ../../../Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "building SLAM..."

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=$BuildType
make -j
