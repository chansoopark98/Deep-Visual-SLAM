# Monocular Depth Estimation

Deep Learning-Based Monocular Depth Estimation TensorFlow Implementation

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Usage](#usage)
   - [Prepare Dataset](#prepare-dataset)
   - [Training](#training)
   - [Evaluation](#evaluation)
3. [Modules](#modules)
   - [Monocular Depth Estimation](#monocular-depth-estimation)
   - [Optical Flow Estimation](#optical-flow-estimation)
   - [Visual Odometry (with Inertial)](#visual-odometry-with-inertial)
4. [Results](#results)
5. [Future Work](#future-work)

---

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/your_username/deep-visual-slam.git

# Navigate to the project directory
cd deep-visual-slam

conda create -n vio python=3.11
conda activate vio

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Prepare datasets

1. Nyu_Depth_V2

```bash
python script/depth/tfds_nyu_converter_tfrecord.py
```

2. DIODE

Download DIODE Official Dataset(https://diode-dataset.org/)

```bash
cd Deep-Visual-SLAM/depth
mkdir data && cd data

wget http://diode-dataset.s3.amazonaws.com/train.tar.gz
wget http://diode-dataset.s3.amazonaws.com/val.tar.gz

unzip train.tar.gz
unzip val.tar.gz

python script/depth/diode_converter_tfrecord.py --root_dir ./
```
3. DIML

Download indoor raw files using Naver Cloud(http://naver.me/xsYr9HL4).

Move all scene into the diml folder

```bash
cd Deep-Visual-SLAM/depth
mkdir data && cd data

mkdir diml

# and then downloads all diml data from link

mv download_link/* diml

python script/depth/diml_converter_tfrecord.py
```

4. Hypersim

ml-hypersim(https://github.com/apple/ml-hypersim/tree/main) You can download all hypersim scenes by using script. But, you have to check your computer's drive storage because Hypersim dataset is very huge(1.9T B).

```bash
cd Deep-Visual-SLAM/depth
mkdir data && cd data
mkdir hypersim && cd hypersim
mkdir hypersim_raw && hypersim

python script/depth/dataset_download_image.py --downloads_dir ./hypersim_raw --decompress_dir ./hypersim

# And then, raw dataset convert to the type of monocular depth dataset.

cd Deep-Visual-SLAM
python script/depth/hypersim_preprocess.py --split_csv ./script/depth/metadata_images_split_scene_v1.csv --dataset_dir ./depth/data/hypersim/hypersim/ --output_dir ./depth/data/hypersim/hypersim_output/

# raw -> preprocess -> tfrecord dataset
python script/depth/hypersim_converter --root_dir ./depth/data/hypersim/hypersim_output/ --save_dir ./depth/data/
