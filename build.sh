echo "building third party..."

cd ./module/DENSE/
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
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j

echo "Uncompress vocabulary ..."

cd ../../../vocabulary
tar -xf ORBvoc.txt.tar.gz

echo "building FSLAM..."

cd ../
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j