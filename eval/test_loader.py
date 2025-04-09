import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import glob
import numpy as np
import numpy.linalg
import tensorflow as tf, tf_keras

def umeyama_alignment(src, dst, with_scale=True):
    """
    Aligns `src` to `dst` using Umeyama's method.
    Args:
        src (Nx3): estimated camera trajectory (e.g., from Monodepth2)
        dst (Nx3): ground truth trajectory (e.g., from Redwood .log)
        with_scale (bool): estimate scale (Sim(3)) or not (SE(3))
    Returns:
        s: scale factor (float)
        R: rotation matrix (3x3)
        t: translation vector (3x1)
        T: full transformation matrix (4x4)
    """
    assert src.shape == dst.shape
    N = src.shape[0]

    # Mean
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    # Centered vectors
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    # Covariance matrix
    cov_matrix = np.dot(dst_centered.T, src_centered) / N

    # SVD
    U, D, Vt = np.linalg.svd(cov_matrix)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    # Scale
    if with_scale:
        var_src = np.var(src_centered, axis=0).sum()
        s = np.trace(np.dot(np.diag(D), S)) / var_src
    else:
        s = 1.0

    # Translation
    t = mu_dst - s * R @ mu_src

    # Full transformation matrix
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t

    return s, R, t, T

class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat
    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)

def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape = (4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype = float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj

def write_trajectory(traj, filename):
    with open(filename, 'w') as f:
        for x in traj:
            p = x.pose.tolist()
            f.write(' '.join(map(str, x.metadata)) + '\n')
            f.write('\n'.join(' '.join(map('{0:.12f}'.format, p[i])) for i in range(4)))
            f.write('\n')

class RedwoodDataLoader(object):
    def __init__(self, config):
        self.config = config
        self.root_dir = '/media/park-ubuntu/park_cs/slam_data/redwood'
        self.image_size = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.num_source = self.config['Train']['num_source'] # 1
        self.imu_seq_len = self.config['Train']['imu_seq_len'] # 10
        self.original_image_size = (480, 640)

        self.intrinsic = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
        self.test_dir = os.path.join(self.root_dir, 'test_2')
        self.test_data = self.generate_datasets(fold_dir=self.test_dir)

    def get_dataset_size(self, fold_dir):
        """데이터셋의 총 샘플 수를 계산합니다."""
        total_samples = 0
        scene_files = sorted(glob.glob(os.path.join(fold_dir, '*')))
        
        for scene in scene_files:
            if os.path.isdir(scene):
                rgb_files = sorted(glob.glob(os.path.join(scene, 'image', '*.jpg')))
                # 연속된 두 프레임씩 사용하므로 이미지 수 - 1이 샘플 수
                total_samples += len(rgb_files) - 1
        
        return total_samples
    
    def align_scale(self, pred_translations, gt_translations):
        """스케일 보정을 위한 함수"""
        # 최적의 스케일 계수 계산
        scale = np.sum(pred_translations * gt_translations) / np.sum(pred_translations * pred_translations)
        
        # 예측값에 스케일 적용
        pred_scaled = pred_translations * scale
        return pred_scaled, scale

    def calc_relative_pose(self, target_pose, right_pose):
        # Convert to numpy arrays
        target_pose = np.array(target_pose.pose)
        right_pose = np.array(right_pose.pose)

        # Calculate relative pose
        target_inv = np.linalg.inv(target_pose)
        rel_pose = np.matmul(target_inv, right_pose)

        # Extract translation and rotation
        rotation = rel_pose[:3, :3]
        translation = rel_pose[:3, 3]
        return rel_pose
    
    @tf.function()
    def read_image(self, rgb_path):
        rgb_image = tf.io.read_file(rgb_path)
        rgb_image = tf.io.decode_png(rgb_image, channels=3)
        rgb_image = tf.image.resize(rgb_image, self.image_size)
        rgb_image = tf.cast(rgb_image, tf.uint8)
        return rgb_image
    
    @tf.function()
    def _read_depth(self, depth_path):
        depth_image = tf.io.read_file(depth_path)
        depth_image = tf.io.decode_image(depth_image, channels=1, dtype=tf.uint16)
        depth_image /= 1000
        depth_image = tf.ensure_shape(depth_image, (480, 640, 1))
        depth_image = tf.image.resize(depth_image, self.image_size, method='nearest')
        depth_image = tf.cast(depth_image, tf.float32)
        return depth_image 
    
    @tf.function(jit_compile=True)
    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image

    @tf.function(jit_compile=True)
    def denormalize_image(self, image):
        image *= 255.0
        image = tf.cast(image, tf.uint8)
        return image
    
    def _process(self, scene_dir: str):
        # load camera metadata
        rgb_files = sorted(glob.glob(os.path.join(scene_dir, 'image', '*.jpg')))
        depth_files = sorted(glob.glob(os.path.join(scene_dir, 'depth', '*.png')))
        length = len(rgb_files)

        pose_lists = read_trajectory(os.path.join(scene_dir, 'apartment.log'))

        assert len(pose_lists) == length, "length of pose lists and rgb files are not equal"

        # make generator
        for i in range(length -1):
            target_image = self.read_image(rgb_files[i])
            target_depth = self._read_depth(depth_files[i])
            target_image = self.normalize_image(target_image)

            right_image = self.read_image(rgb_files[i+1])
            right_depth = self._read_depth(depth_files[i+1])
            right_image = self.normalize_image(right_image)
            

            target_pose = pose_lists[i] # global pose
            right_pose = pose_lists[i+1] # global pose
            
            rel_pose = self.calc_relative_pose(target_pose, right_pose)

            sample = {
                'target_image': target_image,
                'right_image': right_image,
                'target_depth': target_depth,
                'right_depth': right_depth,
                'rel_pose': rel_pose,
                
            }
            yield sample

    def generate_datasets(self, fold_dir):
        scene_files = sorted(glob.glob(os.path.join(fold_dir, '*')))
        scene = scene_files[0]
        if os.path.isdir(scene):  # 디렉토리인 경우만 처리
            print(f"Processing scene: {os.path.basename(scene)}")
            yield from self._process(scene)

if __name__ == '__main__':
    import yaml

    # load config
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset = RedwoodDataLoader(config)
    for sample in dataset.generate_datasets(dataset.test_dir):
        target_image = sample['target_image']
        right_image = sample['right_image']
        