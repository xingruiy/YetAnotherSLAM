echo "building third party..."

cd ./module/DENSE/
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=DEBUG
make -j

cd ../../OniCamera
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=DEBUG
make -j

cd ../../DBoW2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=DEBUG
make -j

cd ../../ORB_SLAM2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=DEBUG
make -j

echo "Uncompress vocabulary ..."

cd ../../../vocabulary
tar -xf ORBvoc.txt.tar.gz

echo "building FSLAM..."

cd ../
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=DEBUG
make -j