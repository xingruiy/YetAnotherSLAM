# An SLAM system in development
## Dependancies
1. [OpenCV](https://github.com/opencv/opencv) >= 3.4.5
2. [Eigen3](https://github.com/eigenteam/eigen-git-mirror) >= 3.3.6
3. [Pangolin](https://github.com/stevenlovegrove/Pangolin)
4. [g2o](https://github.com/RainerKuemmerle/g2o) >= 1.0
5. [Sophus](https://github.com/xingruiy/Sophus)
## How to build
We provide a simple script ```build.sh``` to build the system and all modules needed. The script accepts an argument, which determines if this is going to be a ```release``` build or a ```debug``` build. 
## Use GUI
The GUI part hasn't been done yet, so only map points and some basic input can be seen from the window.