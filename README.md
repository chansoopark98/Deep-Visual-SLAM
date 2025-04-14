# Deep-Visual-SLAM

[![Github All Releases](https://img.shields.io/github/downloads/chansoopark98/Deep-Visual-SLAM/total.svg)]() 


<p align="center">
 <img src="https://img.shields.io/github/issues/chansoopark98/Deep-Visual-SLAM">
 <img src="https://img.shields.io/github/forks/chansoopark98/Deep-Visual-SLAM">
 <img src="https://img.shields.io/github/stars/chansoopark98/Deep-Visual-SLAM">
 <img src="https://img.shields.io/github/license/chansoopark98/Deep-Visual-SLAM">
 </p>

<p align="center">
 <img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Tensorflow-FF6F00.svg?&style=for-the-badge&logo=Tensorflow&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Keras-D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/>
 <img src ="https://img.shields.io/badge/OpenCV-5C3EE8.svg?&style=for-the-badge&logo=OpenCV&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Numpy-013243.svg?&style=for-the-badge&logo=Numpy&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Pandas-150458.svg?&style=for-the-badge&logo=Pandas&logoColor=white"/>
 <br>
</p>


<p align="center">
  <a href="https://youtu.be/in82U0r8gM0?si=tQRSAcHJxwvZIJG9" target="_blank">
   <img src="output_concat_video.gif" alt="Deep-Visual-SLAM Demo GIF" width="640" height="480">
  </a> 
</p>


This repository implements a deep learning-based Monocular Visual SLAM. The key idea is to integrate a pre-trained depth estimation model and a camera optical flow estimation algorithm to construct the Visual SLAM Front-end. Using self-supervised learning, we enable camera ego-motion estimation, similar to Monodepth2. In the future, once the training results of each module (depth, flow, pose) are stabilized, we aim to integrate SLAM Backend algorithms as well.

## Features
- **Deep-based Visual SLAM (VO/VIO)**
  - Monocular Depth Estimation
  - Optical Flow Estimation
  - Visual Odometry (with inertial sensors)

## Monocular Depth Estimation
[Train and Evaluate Mono-Depth](https://github.com/chansoopark98/Deep-Visual-SLAM/tree/main/depth)


## Optical Flow Estimation

Future work


## Visual Odometry

http://redwood-data.org/indoor_lidar_rgbd/download.html

The focal length is 525 for both axes and the principal point is (319.5, 239.5)