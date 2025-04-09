import tensorflow as tf, tf_keras
import numpy as np
import pandas as pd
import yaml
import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from test_loader import RedwoodDataLoader, umeyama_alignment
from model.pose_net import PoseNet
from model.depth_net import DispNet
from vo.utils.projection_utils import pose_axis_angle_vec2mat, pose_vec2mat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as SciRot
import time

def extract_camera_centers(pose_matrices):
    return np.array([-pose[:3, :3].T @ pose[:3, 3] for pose in pose_matrices])

def compute_ate(gt, pred):
    errors = np.linalg.norm(gt - pred, axis=1)
    return np.sqrt(np.mean(errors**2))

redwood_to_mono_rot = np.array([
    [0, -1, 0],
    [0,  0, -1],
    [1,  0, 0]
])

def compute_relative_rotation_error(gt_poses, pred_poses, delta=1):
    rot_errors = []
    for i in range(len(gt_poses) - delta):
        R1_gt = gt_poses[i][:3, :3]
        R2_gt = gt_poses[i + delta][:3, :3]
        R1_pred = pred_poses[i][:3, :3]
        R2_pred = pred_poses[i + delta][:3, :3]

        dR_gt = R1_gt.T @ R2_gt
        dR_pred = R1_pred.T @ R2_pred
        dR_err = dR_gt.T @ dR_pred

        angle = np.linalg.norm(SciRot.from_matrix(dR_err).as_rotvec()) * 180 / np.pi
        rot_errors.append(angle)
    return rot_errors

def compute_translation_errors(gt_poses, pred_poses):
    """
    Calculate Euclidean distance between GT and predicted camera centers
    """
    gt_centers = extract_camera_centers(gt_poses)
    pred_centers = extract_camera_centers(pred_poses)
    trans_errors = np.linalg.norm(gt_centers - pred_centers, axis=1)
    return trans_errors

def compute_relative_translation_error(gt_poses, pred_poses, delta=1):
    errors = []
    for i in range(len(gt_poses) - delta):
        # GT 이동
        gt_t1 = extract_camera_centers([gt_poses[i]])[0]
        gt_t2 = extract_camera_centers([gt_poses[i + delta]])[0]
        gt_disp = gt_t2 - gt_t1

        # Predicted 이동
        pred_t1 = extract_camera_centers([pred_poses[i]])[0]
        pred_t2 = extract_camera_centers([pred_poses[i + delta]])[0]
        pred_disp = pred_t2 - pred_t1

        # 상대 이동 벡터 차이 (drift)
        error = np.linalg.norm(gt_disp - pred_disp)
        errors.append(error)
    return errors

def disp_to_depth( disp, min_depth, max_depth):
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth
    scaled_disp = tf.cast(min_disp, tf.float32) + tf.cast(max_disp - min_disp, tf.float32) * disp
    depth = tf.cast(1., tf.float32) / scaled_disp
    return depth

def transform_redwood_pose(pose_list):
    transformed = []
    for pose in pose_list:
        R = pose[:3, :3]
        t = pose[:3, 3]
        R_new = redwood_to_mono_rot @ R
        t_new = redwood_to_mono_rot @ t
        T_new = np.eye(4)
        T_new[:3, :3] = R_new
        T_new[:3, 3] = t_new
        transformed.append(T_new)
    return np.stack(transformed)

if __name__ == '__main__':
    with open('./vo/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

    with tf.device('/gpu:0'):
        dataset = RedwoodDataLoader(config)
        total_samples = dataset.get_dataset_size(dataset.test_dir)

        # load model
        batch_size = config['Train']['batch_size']
        image_shape = (config['Train']['img_h'], config['Train']['img_w'])

        pose_net = PoseNet(image_shape=image_shape, batch_size=batch_size, prefix='mono_posenet')
        posenet_input_shape = (batch_size, *image_shape, 6)
        pose_net.build(posenet_input_shape)
        pose_net.load_weights('./weights/vo/mode=axisAngle_res=(480, 640)_ep=31_bs=16_initLR=0.0001_endLR=1e-05_prefix=Monodepth2-resnet18-Posenet-onlyRedwood/pose_net_epoch_4_model.weights.h5')
        pose_net.trainable = False
        # dummy
        dummy_input = tf.zeros((batch_size, *image_shape, 6), dtype=tf.float32)
        _ = pose_net(dummy_input, training=False)

        depth_net = DispNet(image_shape=image_shape, batch_size=batch_size, prefix='mono_depthnet')
        depth_net.build((batch_size, *image_shape, 3))
        depth_net.load_weights('./weights/vo/mode=axisAngle_res=(480, 640)_ep=31_bs=16_initLR=0.0001_endLR=1e-05_prefix=Monodepth2-resnet18-Posenet-onlyRedwood/depth_net_epoch_4_model.weights.h5')
        depth_net.trainable = False
        # dummy
        dummy_input = tf.zeros((batch_size, *image_shape, 3), dtype=tf.float32)
        _ = depth_net(dummy_input, training=False)
        
        gt_rel_pose_list = [] # 4x4 transform matrix
        pred_rel_pose_list = [] # 4x4 transform matrix
        duration_list = []

        idx = 0
        for sample in tqdm.tqdm(dataset.generate_datasets(dataset.test_dir), 
                            desc="Processing samples", 
                            unit="sample",
                            total=total_samples):
            target_image = sample['target_image']
            target_depth = sample['target_depth']
            right_image = sample['right_image']
            gt_rel_pose = sample['rel_pose']


            target_image = tf.convert_to_tensor(target_image, dtype=tf.float32)
            right_image = tf.convert_to_tensor(right_image, dtype=tf.float32)
            
            disps = depth_net(tf.expand_dims(target_image, axis=0), training=False)
            depth = disp_to_depth(disps[0], 0.1, 10.0)

            concat_image = tf.concat([target_image, right_image], axis=-1)
            concat_image = tf.expand_dims(concat_image, axis=0)

            start_time = time.time()
            pred_pose = pose_net(concat_image, training=False)
            duration = time.time() - start_time
            duration_list.append(duration)

            pred_pose_mat = pose_axis_angle_vec2mat(vec=pred_pose, depth=depth, invert=True)
            pred_pose_mat = tf.squeeze(pred_pose_mat, axis=0)
            pred_rel_pose_list.append(pred_pose_mat.numpy())
            gt_rel_pose_list.append(gt_rel_pose)

            idx += 1
            if idx > 100:
                break
        
        # rel poses to global poses
        gt_rel_pose_list = np.array(gt_rel_pose_list)
        pred_rel_pose_list = np.array(pred_rel_pose_list)
        gt_global_pose_list = np.zeros((len(gt_rel_pose_list), 4, 4))
        pred_global_pose_list = np.zeros((len(pred_rel_pose_list), 4, 4))
        gt_global_pose_list[0] = np.eye(4)
        pred_global_pose_list[0] = np.eye(4)
        for i in range(1, len(gt_rel_pose_list)):
            # gt_global_pose_list[i] = np.dot(gt_global_pose_list[i-1], gt_rel_pose_list[i])
            gt_global_pose_list[i] = gt_rel_pose_list[i] @ gt_global_pose_list[i-1]
            # pred_global_pose_list[i] = np.dot(pred_global_pose_list[i-1], pred_rel_pose_list[i])
            pred_global_pose_list[i] = np.dot(pred_rel_pose_list[i], pred_global_pose_list[i-1])

        gt_global_pose_list = transform_redwood_pose(gt_global_pose_list)

        # 2. 카메라 중심 위치만 추출 (Nx3)
        src = extract_camera_centers(pred_global_pose_list)  # 예측 경로
        dst = extract_camera_centers(gt_global_pose_list)     # GT 경로

        # 3. 정렬
        s, R, t, T = umeyama_alignment(src, dst)

        # 4. pose 정렬 적용
        aligned_monodepth2_poses = [T @ pose for pose in pred_global_pose_list]
        
        # 5. 정렬된 포즈에서 camera center 추출
        aligned_centers = extract_camera_centers(aligned_monodepth2_poses)

        ate = compute_ate(dst, aligned_centers)
        print(f"ATE (RMSE): {ate:.4f} m")

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(*dst.T, label='GT', color='green')
        ax.plot(*src.T, label='Before Alignment', color='red', linestyle='--')
        ax.plot(*aligned_centers.T, label='After Alignment', color='blue')
        ax.legend()
        plt.show()

        # pred_global_pose_list → Umeyama 정렬 후
        aligned_rot_errors = compute_relative_rotation_error(gt_global_pose_list, pred_global_pose_list)
        aligned_trans_errors = compute_relative_translation_error(gt_global_pose_list, pred_global_pose_list)

        print(f"[Aligned] 평균 회전 오차: {np.mean(aligned_rot_errors):.3f}°")
        print(f"[Aligned] 평균 이동 오차: {np.mean(aligned_trans_errors):.3f} m")
        print(f"[Aligned] 평균 이동 오차: {np.mean(aligned_trans_errors) * 100:.3f} cm")
        avg_duration_sec = np.mean(duration_list)
        avg_duration_ms = avg_duration_sec * 1000
        print(f"평균 처리 시간: {avg_duration_sec:.4f} 초 ({avg_duration_ms:.2f} 밀리초)")


        # 프레임 수를 기준으로 데이터 정렬
        frame_count = len(aligned_rot_errors)

        data = {
            "Frame": list(range(frame_count)),
            "Rotation Error (degrees)": aligned_rot_errors,
            "Translation Error (Meter)": aligned_trans_errors,
            "Durations (sec)": duration_list[:frame_count],
        }

        df = pd.DataFrame(data)
        df.to_csv("./eval/pose_error_log.csv", index=False)
        print("✅ 오차 로그가 pose_error_log.csv에 저장되었습니다.")