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
echo "Configuring and Building 3rdparty/OpenNI2 ..."

cd ./3rdparty/OpenNI
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

echo "Configuring and Building 3rdparty/ORB_SLAM2/3rdparty/DBoW2 ..."

cd ../../DBoW2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

# echo "Uncompress Vocabulary ..."

# cd ../../../Vocabulary
# tar -xf ORBvoc.txt.tar.gz
cd ../../../

echo "Building SLAM ..."

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=$BuildType
make -j

# echo "Converting Vocabulary Files to Binary Version ..."
# cd ../Vocabulary
# ./Text2Bin
# cd ..
