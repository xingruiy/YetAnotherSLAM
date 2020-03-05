# Yet Another SLAM

## Change Log

+ _2020-03-05_:
    + Merged ```dev``` branch to ```master``` branch.
    + Implemented Map fusion function.
    + Solved problems with global bundle adjustment.

+ _2020-03-05_:
    + Merged ```dev2``` branch to ```master``` branch.
    + Solved problems with ```OptimizeEssentialGraph```
    + Added dense map structure for keyframes

+ _2020-03-02_: 
    + Added ```LoopClosing``` from ```ORB-SLAM2```.
    + Had to change the sim3 optimization to se3.   
    + The ```OptimizeEssentialGraph``` function gives a segment fault for unknown reason.

+ _2020-02-24_: 
    + Changed ```README.md```, added the dependencies list.
    + Refined the file structure.
    + Try to add a copy of g2o but failed.
    + Removed ```ImageProc``` class, now all functions are in the global scope.

## Dependancies

+ [OpenCV](https://github.com/opencv/opencv) >= 3.4.5
+ [Eigen3](https://github.com/eigenteam/eigen-git-mirror) >= 3.3.6
+ [Pangolin](https://github.com/stevenlovegrove/Pangolin)
+ [g2o](https://github.com/RainerKuemmerle/g2o) >= 1.0
+ [Sophus](https://github.com/xingruiy/Sophus)
