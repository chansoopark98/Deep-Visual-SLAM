# Deep-Visual-SLAM-Monodepth2

[![Github All Releases](https://img.shields.io/github/downloads/chansoopark98/Deep-Visual-SLAM-Monodepth2/total.svg)]()

<p align="center">
 <img src="https://img.shields.io/github/issues/chansoopark98/Deep-Visual-SLAM-Monodepth2">
 <img src="https://img.shields.io/github/forks/chansoopark98/Deep-Visual-SLAM-Monodepth2">
 <img src="https://img.shields.io/github/stars/chansoopark98/Deep-Visual-SLAM-Monodepth2">
 <img src="https://img.shields.io/github/license/chansoopark98/Deep-Visual-SLAM-Monodepth2">
</p>

<p align="center">
 <img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
 <img src ="https://img.shields.io/badge/PyTorch-EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white"/>
 <img src ="https://img.shields.io/badge/OpenCV-5C3EE8.svg?&style=for-the-badge&logo=OpenCV&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Numpy-013243.svg?&style=for-the-badge&logo=Numpy&logoColor=white"/>
 <img src ="https://img.shields.io/badge/CUDA-76B900.svg?&style=for-the-badge&logo=NVIDIA&logoColor=white"/>
</p>

<p align="center">
  <a href="https://youtu.be/in82U0r8gM0?si=tQRSAcHJxwvZIJG9" target="_blank">
   <img src="output_concat_video.gif" alt="Deep-Visual-SLAM Demo GIF" width="640" height="480">
  </a>
</p>

This repository implements a **deep learning-based Monocular Visual SLAM (Visual Odometry)** system based on [Monodepth2](https://github.com/nianticlabs/monodepth2). The system integrates a self-supervised depth estimation network and a camera pose estimation network to enable ego-motion estimation from monocular video sequences. The key innovation is the joint training of depth and pose networks using photometric consistency loss without requiring ground truth depth or pose labels.

## Features

- **Self-Supervised Learning**: Joint training of depth and pose networks using photometric loss (SSIM + L1)
- **Monocular Depth Estimation**: Single-image depth prediction using ResNet-18 encoder-decoder architecture
- **Visual Odometry**: Camera ego-motion estimation from image pairs
- **3D Reconstruction**: Real-time point cloud generation and trajectory visualization
- **Multi-Dataset Support**: Redwood RGBD, NYU Depth V2, and custom datasets

## Project Structure

```
Deep-Visual-SLAM-Monodepth2/
├── depth/                      # Depth estimation module
│   ├── train.py               # Depth model training
│   ├── eval.py                # Depth evaluation
│   ├── config.yaml            # Depth training configuration
│   └── dataset/               # Dataset loaders (NYU, Redwood, Custom)
│
├── vo/                         # Visual Odometry module
│   ├── train.py               # VO training (depth + pose)
│   ├── predict.py             # VO inference and visualization
│   ├── eval_traj.py           # Trajectory evaluation
│   ├── config.yaml            # VO training configuration
│   └── dataset/               # VO dataset loaders
│
├── model/                      # Neural network architectures
│   ├── depthnet.py            # Depth decoder network
│   ├── posenet.py             # Pose estimation network
│   ├── resnet_encoder.py      # ResNet backbone encoder
│   └── layers.py              # Common layers
│
├── slam/                       # SLAM backend components
│   ├── MonoVO.py              # Monocular VO implementation
│   ├── frontend.py            # Feature extraction & matching
│   └── optimizer.py           # Backend optimization
│
├── weights/                    # Pre-trained model weights
│   ├── depth/
│   └── vo/
│
└── requirements.txt
```

## Environment Setup

### System Requirements
- Ubuntu 22.04
- CUDA 12.8
- Python 3.12
- PyTorch 2.8.0+

### Installation

1. **Create conda environment**
```bash
conda create -n vslam python=3.12
conda activate vslam
```

2. **Install PyTorch with CUDA 12.8**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies
- `torch`, `torchvision` - Deep learning framework
- `opencv-python` - Image processing
- `pyvista` - 3D visualization
- `pytransform3d` - 3D transformations
- `tensorboard` - Training visualization
- `imageio[ffmpeg]` - Video I/O
- `scipy`, `numpy`, `pandas` - Scientific computing

## Datasets

### Supported Datasets

| Dataset | Description | Usage |
|---------|-------------|-------|
| [Redwood RGBD](http://redwood-data.org/indoor_lidar_rgbd/download.html) | Indoor RGB-D sequences with camera poses | VO Training/Evaluation |
| [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) | Indoor RGB-D pairs | Depth Training |
| Custom | User-defined RGB-D data | Both |

### Redwood Dataset Preparation

1. **Download** the dataset from [Redwood Indoor LiDAR-RGBD](http://redwood-data.org/indoor_lidar_rgbd/download.html)

2. **Organize** the dataset with the following structure:
```
vo/data/redwood/
├── train/
│   └── {scene_name}/
│       ├── image/              # RGB images (*.jpg)
│       ├── depth/              # Depth maps (*.png, 16-bit, millimeters)
│       └── {scene_name}.json   # Camera poses (Open3D format)
└── test/
    └── {scene_name}/
        ├── image/
        ├── depth/
        └── {scene_name}.json
```

3. **Camera intrinsics** (default for Redwood):
   - Focal length: 525 (both fx and fy)
   - Principal point: (319.5, 239.5)
   - Resolution: 640 x 480

### NYU Depth V2 Preparation

```
depth/data/nyu_depth_v2_raw/
├── train/
│   ├── rgb/     # RGB images (*.png)
│   └── depth/   # Depth maps (*.png)
└── valid/
    ├── rgb/
    └── depth/
```

### Custom Dataset

Place your RGB-D data in the following structure:
```
depth/data/custom/
└── train/
    ├── rgb/     # RGB images
    └── depth/   # Depth maps (16-bit PNG, values in millimeters)
```

## Demo Execution

### Running Visual Odometry Prediction

The `vo/predict.py` script performs inference using pre-trained depth and pose networks, generating 3D point clouds and camera trajectories.

**Prerequisites:**
- Pre-trained weights in `./weights/vo/` directory:
  - `depth_net_epoch_30.pth`
  - `pose_net_epoch_30.pth`
- Test dataset configured in `vo/config.yaml`

**Run the demo:**
```bash
cd /path/to/Deep-Visual-SLAM-Monodepth2
python vo/predict.py
```

**What it does:**
1. Loads pre-trained DepthNet and PoseNet (ResNet-18 backbone)
2. Processes test sequences from the configured dataset
3. Estimates depth maps from single images
4. Estimates relative camera poses from image pairs
5. Accumulates poses to build global trajectory
6. Generates 3D point cloud visualization
7. Records output video with trajectory overlay

**Output:**
- Real-time 3D visualization window (1920x1080)
- Output video file (e.g., `0414_traj.mp4`) containing:
  - Colored point clouds
  - Camera trajectory (red line)
  - Camera model visualization

### Configuration Options

Edit `vo/config.yaml` to customize:
```yaml
Train:
  img_w: 640          # Image width
  img_h: 480          # Image height
  min_depth: 0.1      # Minimum depth (meters)
  max_depth: 10.0     # Maximum depth (meters)
  batch_size: 1       # Batch size for inference
```

## Training

### Depth Network Training
```bash
python depth/train.py
```

### Visual Odometry Training
```bash
python vo/train.py
```

### Training Configuration

Both training scripts use YAML configuration files:
- `depth/config.yaml` - Depth training settings
- `vo/config.yaml` - VO training settings

Key parameters:
| Parameter | Depth | VO | Description |
|-----------|-------|-----|-------------|
| `batch_size` | 64 | 16 | Training batch size |
| `epoch` | 31 | 31 | Number of epochs |
| `init_lr` | 0.0001 | 0.0001 | Initial learning rate |
| `use_amp` | True | False | Automatic mixed precision |
| `use_compile` | True | False | PyTorch 2.0 compile |

## Model Architecture

### Depth Network (Monodepth2-based)
```
Input Image [B, 3, H, W]
       ↓
ResNet-18 Encoder
       ↓
Multi-scale Decoder (4 scales)
       ↓
Disparity Maps → Depth = 1/Disparity
```

### Pose Network
```
Image Pair [B, 6, H, W]
       ↓
Shared ResNet-18 Encoders
       ↓
Pose Decoder
       ↓
Axis-angle (3D) + Translation (3D)
```

## Evaluation

### Trajectory Evaluation
```bash
python vo/eval_traj.py
```

### Depth Evaluation
```bash
python depth/eval.py
```

Evaluation metrics:
- **Depth**: AbsRel, SqRel, RMSE, RMSE_log, δ < 1.25^n
- **Trajectory**: ATE (Absolute Trajectory Error), RPE (Relative Pose Error)

## References

- [Monodepth2](https://github.com/nianticlabs/monodepth2) - Digging into Self-Supervised Monocular Depth Prediction
- [Redwood Indoor Dataset](http://redwood-data.org/indoor_lidar_rgbd/)
- [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
