import os
import glob
import numpy as np
import time

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import math

_EPS = np.finfo(float).eps * 4.0

def remove_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

# matplotlib.use('TkAgg') 
def plot_camera_poses(Rs, ts):
    """
     Rs: 3x3 회전 행렬 리스트
     ts: 3x1 변환 벡터 리스트
    """
    # plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 화살표 길이 설정
    length = 0.5

    for R, t in zip(Rs, ts):
        # Plot camera center
        ax.scatter([t[0]], [t[1]], [t[2]], c='r', marker='o')

        # 카메라 축을 나타내는 화살표 끝점 계산
        end_points = np.dot(R, np.array([
            [length, 0, 0], 
            [0, length, 0], 
            [0, 0, length]
        ]).T).T + t.reshape(-1)

        # Plot each axis
        colors = ['r', 'g', 'b']
        for i, color in enumerate(colors):
            ax.quiver(t[0], t[1], t[2], end_points[i, 0]-t[0], end_points[i, 1]-t[1], end_points[i, 2]-t[2], color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def isRotationMatrix(R):
    '''
    정규화된 회전 행렬인지 확인하는 함수
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def euler_from_matrix(matrix):
    '''
    회전 행렬에서 오일러각 추출 함수
    '''
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
    ay = math.atan2(-M[2, 0], cy)
    if ay < -math.pi / 2 + _EPS and ay > -math.pi / 2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2(-M[1, 2], -M[0, 2])
    elif ay < math.pi / 2 + _EPS and ay > math.pi / 2 - _EPS:
        ax = 0
        az = math.atan2(M[1, 2], M[0, 2])
    else:
        ax = math.atan2(M[2, 1], M[2, 2])
        az = math.atan2(M[1, 0], M[0, 0])
    return np.array([ax, ay, az])

def get_relative_pose(Rt1, Rt2):
    '''
    두 포즈 행렬 사이 4x4 포즈 행렬을 계산하는 함수
    '''
    Rt1_inv = np.linalg.inv(Rt1)
    Rt_rel = Rt1_inv @ Rt2
    return Rt_rel

def get_relative_pose_6DoF(Rt1, Rt2):
    '''
    두 포즈 행렬 사이 상대 회전 및 이동 변화량을 계산하는 함수
    '''
    
    # Calculate the relative transformation Rt_rel
    Rt_rel = get_relative_pose(Rt1, Rt2)

    R_rel = Rt_rel[:3, :3]
    t_rel = Rt_rel[:3, 3]

    # Extract the Eular angle from the relative rotation matrix
    x, y, z = euler_from_matrix(R_rel)
    theta = [x, y, z]

    pose_rel = np.concatenate((theta, t_rel))
    return pose_rel

def rotationError(Rt1, Rt2):
    '''
    두 포즈 행렬 간의 회전 차이를 계산하는 함수
    '''
    pose_error = get_relative_pose(Rt1, Rt2)
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))

def translationError(Rt1, Rt2):
    '''
    두 포즈 행렬 간의 변환 차이를 계산하는 함수
    '''
    pose_error = get_relative_pose(Rt1, Rt2)
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    return np.sqrt(dx**2 + dy**2 + dz**2)

def eulerAnglesToRotationMatrix(theta):
    '''
    오일러각roll, yaw, pitch)으로 회전 행렬을 계산하는 함수
    '''
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def normalize_angle_delta(angle):
    '''
    회전각이 -pi 에서 pi 사이에 있도록 제한하는 함수
    '''
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

def pose_6DoF_to_matrix(pose):
    '''
    오일러각과 이동 벡터로부터 3x4 변환 행렬을 계산함 
    '''
    R = eulerAnglesToRotationMatrix(pose[:3])
    t = pose[3:].reshape(3, 1)
    R = np.concatenate((R, t), 1)
    R = np.concatenate((R, np.array([[0, 0, 0, 1]])), 0)
    return R

def pose_accu(Rt_pre, R_rel):
    '''
    최근 포즈와 상대 회전 및 이동 거리로부터 누적된 포즈를 계산하는 함수
    '''
    Rt_rel = pose_6DoF_to_matrix(R_rel)
    return Rt_pre @ Rt_rel

def path_accu(pose):
    '''
    상대 포즈로부터 전역 포즈를 계산하는 함수
    '''
    answer = [np.eye(4)]
    for index in range(pose.shape[0]):
        pose_ = pose_accu(answer[-1], pose[index, :])
        answer.append(pose_)
    return answer

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def rmse_err_cal(pose_est, pose_gt):
    '''
    상대 이동 및 회전의 RMSE(Root Mean Square Error) 계산하는 함수
    '''
    t_rmse = np.sqrt(np.mean(np.sum((pose_est[:, 3:] - pose_gt[:, 3:])**2, -1)))
    r_rmse = np.sqrt(np.mean(np.sum((pose_est[:, :3] - pose_gt[:, :3])**2, -1)))
    return t_rmse, r_rmse

def trajectoryDistances(poses):
    '''
    각 프레임별 거리와 속도 계산 함수
    '''
    dist = [0]
    speed = [0]
    for i in range(len(poses) - 1):
        cur_frame_idx = i
        next_frame_idx = cur_frame_idx + 1
        P1 = poses[cur_frame_idx]
        P2 = poses[next_frame_idx]
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dz = P1[2, 3] - P2[2, 3]
        dist.append(dist[i] + np.sqrt(dx**2 + dy**2 + dz**2))
        speed.append(np.sqrt(dx**2 + dy**2 + dz**2) * 10)
    return dist, speed

def lastFrameFromSegmentLength(dist, first_frame, len_):
    for i in range(first_frame, len(dist), 1):
        if dist[i] > (dist[first_frame] + len_):
            return i
    return -1

def computeOverallErr(seq_err):
    t_err = 0
    r_err = 0
    seq_len = len(seq_err)

    for item in seq_err:
        r_err += item[1]
        t_err += item[2]
    ave_t_err = t_err / seq_len
    ave_r_err = r_err / seq_len
    return ave_t_err, ave_r_err

def read_pose(line):
    '''
    Reading 4x4 pose matrix from .txt files
    input: a line of 12 parameters
    output: 4x4 numpy matrix
    '''
    values= np.reshape(np.array([float(value) for value in line.split(' ')]), (3, 4))
    Rt = np.concatenate((values, np.array([[0, 0, 0, 1]])), 0)
    return Rt
    
def read_pose_from_text(path):
    with open(path) as f:
        lines = [line.split('\n')[0] for line in f.readlines()]
        poses_rel, poses_abs = [], []
        values_p = read_pose(lines[0])
        poses_abs.append(values_p)            
        for i in range(1, len(lines)):
            values = read_pose(lines[i])
            poses_rel.append(get_relative_pose_6DoF(values_p, values)) 
            values_p = values.copy()
            poses_abs.append(values) 
        poses_abs = np.array(poses_abs)
        poses_rel = np.array(poses_rel)
        
    return poses_abs, poses_rel

def saveSequence(poses, file_name):
    with open(file_name, 'w') as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            f.write(' '.join([str(r) for r in pose]))
            f.write('\n')

def kitti_err_cal(pose_est_mat, pose_gt_mat):

    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    num_lengths = len(lengths)

    err = []
    dist, speed = trajectoryDistances(pose_gt_mat)
    step_size = 10  # 10Hz

    for first_frame in range(0, len(pose_gt_mat), step_size):

        for i in range(num_lengths):
            len_ = lengths[i]
            last_frame = lastFrameFromSegmentLength(dist, first_frame, len_)
            # Continue if sequence not long enough
            if last_frame == -1 or last_frame >= len(pose_est_mat) or first_frame >= len(pose_est_mat):
                continue

            pose_delta_gt = np.dot(np.linalg.inv(pose_gt_mat[first_frame]), pose_gt_mat[last_frame])
            pose_delta_result = np.dot(np.linalg.inv(pose_est_mat[first_frame]), pose_est_mat[last_frame])

            r_err = rotationError(pose_delta_result, pose_delta_gt)
            t_err = translationError(pose_delta_result, pose_delta_gt)

            err.append([first_frame, r_err / len_, t_err / len_, len_])

    t_rel, r_rel = computeOverallErr(err)
    return err, t_rel, r_rel, np.asarray(speed)

def kitti_eval(pose_est, dec_est, pose_gt):
    
    # First decision is always true
    dec_est = np.insert(dec_est, 0, 1)
    
    # Calculate the translational and rotational RMSE
    t_rmse, r_rmse = rmse_err_cal(pose_est, pose_gt)

    # Transfer to 3x4 pose matrix
    pose_est_mat = path_accu(pose_est)
    pose_gt_mat = path_accu(pose_gt)

    # Using KITTI metric
    err_list, t_rel, r_rel, speed = kitti_err_cal(pose_est_mat, pose_gt_mat)
    
    t_rel = t_rel * 100
    r_rel = r_rel / np.pi * 180 * 100
    r_rmse = r_rmse / np.pi * 180
    usage = np.mean(dec_est) * 100

    return pose_est_mat, pose_gt_mat, t_rel, r_rel, t_rmse, r_rmse, usage, speed

def plot_results(seq, poses_gt_mat, poses_est_mat, plot_path_dir, decision, speed, window_size):
    
    if decision is not None:
        # Apply smoothing to the decision
        decision = np.insert(decision, 0, 1)
        decision = moving_average(decision, window_size)

    fontsize_ = 10
    plot_keys = ["GT", "Pred"]
    start_point = [0, 0]
    style_pred = 'b-'
    style_gt = 'r-'
    style_O = 'ko'

    # get the value
    x_gt = np.asarray([pose[0, 3] for pose in poses_gt_mat])
    y_gt = np.asarray([pose[1, 3] for pose in poses_gt_mat])
    z_gt = np.asarray([pose[2, 3] for pose in poses_gt_mat])

    x_pred = np.asarray([pose[0, 3] for pose in poses_est_mat])
    y_pred = np.asarray([pose[1, 3] for pose in poses_est_mat])
    z_pred = np.asarray([pose[2, 3] for pose in poses_est_mat])

    # Plot 2d trajectory estimation map
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.gca()
    plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
    plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
    plt.plot(start_point[0], start_point[1], style_O, label='Start point')
    plt.legend(loc="upper right", prop={'size': fontsize_})
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    # set the range of x and y
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean))
                       for lim in lims])
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    plt.title('2D path')
    png_title = "{}_path_2d".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Plot decision hearmap
    if decision is not None:
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = plt.gca()
        cout = np.insert(decision, 0, 0) * 100
        cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
        max_usage = max(cout)
        min_usage = min(cout)
        ticks = np.floor(np.linspace(min_usage, max_usage, num=5))
        cbar = fig.colorbar(cax, ticks=ticks)
        cbar.ax.set_yticklabels([str(i) + '%' for i in ticks])

        plt.title('decision heatmap : moving average filter {}'.format(window_size))
        png_title = "{}_decision_smoothed".format(seq)
        plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # Plot the speed map
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.gca()
    cout = speed
    cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('z (m)', fontsize=fontsize_)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
    max_speed = max(cout)
    min_speed = min(cout)
    ticks = np.floor(np.linspace(min_speed, max_speed, num=5))
    cbar = fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels([str(i) + 'm/s' for i in ticks])

    plt.title('speed heatmap')
    png_title = "{}_speed".format(seq)
    plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    plt.close()
