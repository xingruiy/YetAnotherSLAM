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
echo "building third party..."

cd ./third_party/DENSE/
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

cd ../../ONI_Camera
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

cd ../../DBoW2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

cd ../../g2o
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=$BuildType
make -j

echo "Uncompress vocabulary ..."

cd ../../../vocabulary
tar -xf ORBvoc.txt.tar.gz

echo "building FSLAM..."

cd ../
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=$BuildType
make -j
