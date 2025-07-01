import tensorflow as tf
import numpy as np
import sys
sys.path.append('/home/park-ubuntu/park/Deep-Visual-SLAM')
from vo.utils.projection_utils import pose_axis_angle_vec2mat

def matrix_to_axis_angle_vectorized(matrix, depth=None):
    """
    4x4 변환 행렬을 6DoF axis-angle 벡터로 변환
    pose_axis_angle_vec2mat의 역변환
    
    Args:
        matrix: [B, 4, 4] 변환 행렬
        depth: [B, H, W, 1] depth map (선택사항, 스케일 정규화용)
    
    Returns:
        [B, 6] axis-angle + translation 벡터
    """
    batch_size = tf.shape(matrix)[0]
    
    # Translation 추출
    translation = matrix[:, :3, 3]  # [B, 3]
    
    # Rotation matrix 추출
    R = matrix[:, :3, :3]  # [B, 3, 3]
    
    # Depth 스케일링 역적용 (있는 경우)
    if depth is not None:
        inv_depth = 1.0 / (depth + 1e-6)
        mean_inv_depth = tf.reduce_mean(inv_depth, axis=[1, 2, 3])  # [B]
        mean_inv_depth = tf.reshape(mean_inv_depth, [batch_size, 1])
        # 역스케일링 적용
        translation = translation / (mean_inv_depth + 1e-8)
    
    # Rotation matrix to axis-angle
    # 1. Trace를 이용한 angle 계산
    trace = tf.linalg.trace(R)  # [B]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = tf.clip_by_value(cos_angle, -1.0, 1.0)
    angle = tf.acos(cos_angle)  # [B]
    
    # 2. Axis 계산
    # 작은 각도 처리
    eps = 1e-7
    is_small_angle = tf.less(tf.abs(angle), eps)
    
    # Skew-symmetric 부분에서 axis 추출
    # axis = (R - R^T) / (2 * sin(angle))
    R_transpose = tf.transpose(R, [0, 2, 1])
    R_skew = R - R_transpose
    
    sin_angle = tf.sin(angle)
    sin_angle_safe = tf.where(is_small_angle, tf.ones_like(sin_angle), sin_angle)
    
    axis_x = R_skew[:, 2, 1] / (2.0 * sin_angle_safe + eps)
    axis_y = R_skew[:, 0, 2] / (2.0 * sin_angle_safe + eps)  
    axis_z = R_skew[:, 1, 0] / (2.0 * sin_angle_safe + eps)
    
    # 배치 처리를 위한 조건부 axis 계산
    axis_default = tf.stack([axis_x, axis_y, axis_z], axis=1)  # [B, 3]
    axis_small = tf.zeros([batch_size, 3])  # 작은 각도일 때는 0 벡터
    
    # 최종 axis 선택
    axis = tf.where(tf.expand_dims(is_small_angle, 1), axis_small, axis_default)
    
    # Axis 정규화
    axis_norm = tf.norm(axis, axis=1, keepdims=True)
    axis_normalized = tf.where(
        tf.expand_dims(is_small_angle, 1),
        axis,  # 작은 각도일 때는 정규화 안 함
        axis / (axis_norm + eps)
    )
    
    # Axis-angle 벡터 생성
    angle_expanded = tf.expand_dims(angle, 1)  # [B, 1]
    axis_angle = axis_normalized * angle_expanded  # [B, 3]
    
    # 작은 각도의 경우 0 벡터로 설정
    axis_angle = tf.where(tf.expand_dims(is_small_angle, 1), 
                          tf.zeros_like(axis_angle), 
                          axis_angle)
    
    # Translation과 결합
    pose_vec = tf.concat([axis_angle, translation], axis=1)  # [B, 6]
    
    return pose_vec

def create_test_matrix(batch_size=4):
    """테스트용 4x4 변환 행렬 생성"""
    # 회전 부분: 작은 회전 각도로 생성
    angle = tf.random.normal([batch_size, 1]) * 0.1  # 작은 각도
    axis = tf.random.normal([batch_size, 3])
    axis = axis / tf.norm(axis, axis=1, keepdims=True)  # 정규화
    
    # Rodrigues 공식으로 회전 행렬 생성
    cos_a = tf.cos(angle)[:, 0]  # [B]
    sin_a = tf.sin(angle)[:, 0]  # [B]
    
    # 더 간단한 방법: 각 요소를 직접 지정
    zeros = tf.zeros([batch_size])
    
    # Skew-symmetric matrix K
    row0 = tf.stack([zeros, -axis[:, 2], axis[:, 1]], axis=1)  # [B, 3]
    row1 = tf.stack([axis[:, 2], zeros, -axis[:, 0]], axis=1)  # [B, 3]
    row2 = tf.stack([-axis[:, 1], axis[:, 0], zeros], axis=1)  # [B, 3]
    K = tf.stack([row0, row1, row2], axis=1)  # [B, 3, 3]
    
    I = tf.eye(3, batch_shape=[batch_size])
    
    # axis outer product
    axis_expanded_1 = tf.expand_dims(axis, 2)  # [B, 3, 1]
    axis_expanded_2 = tf.expand_dims(axis, 1)  # [B, 1, 3]
    axis_outer = tf.matmul(axis_expanded_1, axis_expanded_2)  # [B, 3, 3]
    
    # Rodrigues formula: R = cos(θ)I + sin(θ)K + (1-cos(θ))aa^T
    cos_a_expanded = tf.reshape(cos_a, [batch_size, 1, 1])
    sin_a_expanded = tf.reshape(sin_a, [batch_size, 1, 1])
    one_minus_cos_a = tf.reshape(1 - cos_a, [batch_size, 1, 1])
    
    R = cos_a_expanded * I + sin_a_expanded * K + one_minus_cos_a * axis_outer
    
    # Translation 부분
    t = tf.random.normal([batch_size, 3]) * 0.5
    
    # 4x4 행렬 조립
    zeros_row = tf.zeros([batch_size, 1, 3])
    ones_corner = tf.ones([batch_size, 1, 1])
    
    # 상위 3x4 부분 생성
    upper = tf.concat([R, tf.expand_dims(t, 2)], axis=2)  # [B, 3, 4]
    
    # 하위 1x4 부분 생성
    lower = tf.concat([zeros_row, ones_corner], axis=2)  # [B, 1, 4]
    
    # 최종 4x4 행렬
    matrix = tf.concat([upper, lower], axis=1)  # [B, 4, 4]
    
    return matrix


def test_conversion_consistency():
    """변환 함수들의 일관성을 테스트"""
    print("=" * 60)
    print("Testing 4x4 Matrix <-> 6DoF Pose Vector Conversion")
    print("=" * 60)
    
    batch_size = 8
    
    # Test 1: Pose vector -> Matrix -> Pose vector
    print("\nTest 1: Pose Vector -> Matrix -> Pose Vector")
    print("-" * 40)
    
    # 다양한 크기의 회전 테스트
    test_cases = [
        ("Small rotations", 0.01),
        ("Medium rotations", 0.5),
        ("Large rotations", 1.5)
    ]
    
    for case_name, rotation_scale in test_cases:
        print(f"\n{case_name} (scale={rotation_scale}):")
        
        # 테스트용 pose vector 생성
        rotation_part = tf.random.normal([batch_size, 3]) * rotation_scale
        translation_part = tf.random.normal([batch_size, 3]) * 1.0
        original_pose_vec = tf.concat([rotation_part, translation_part], axis=1)
        
        # Forward: pose vector -> matrix
        matrix_from_vec = pose_axis_angle_vec2mat(original_pose_vec, depth=None, invert=False)
        
        # Backward: matrix -> pose vector
        recovered_pose_vec = matrix_to_axis_angle_vectorized(matrix_from_vec, depth=None)
        
        # 오차 계산
        error = tf.reduce_mean(tf.abs(original_pose_vec - recovered_pose_vec))
        max_error = tf.reduce_max(tf.abs(original_pose_vec - recovered_pose_vec))
        
        print(f"  Original pose vec (sample): {original_pose_vec[0].numpy()}")
        print(f"  Recovered pose vec (sample): {recovered_pose_vec[0].numpy()}")
        print(f"  Average error: {error:.8f}")
        print(f"  Max error: {max_error:.8f}")
    
    # Test 2: Matrix -> Pose vector -> Matrix
    print("\n\nTest 2: Matrix -> Pose Vector -> Matrix")
    print("-" * 40)
    
    # 테스트용 4x4 matrix 생성
    original_matrix = create_test_matrix(batch_size)
    
    # Forward: matrix -> pose vector
    pose_vec_from_matrix = matrix_to_axis_angle_vectorized(original_matrix, depth=None)
    
    # Backward: pose vector -> matrix
    recovered_matrix = pose_axis_angle_vec2mat(pose_vec_from_matrix, depth=None, invert=False)
    
    # 오차 계산
    matrix_error = tf.reduce_mean(tf.abs(original_matrix - recovered_matrix))
    max_matrix_error = tf.reduce_max(tf.abs(original_matrix - recovered_matrix))
    
    print(f"  Average matrix error: {matrix_error:.8f}")
    print(f"  Max matrix error: {max_matrix_error:.8f}")
    
    # Rotation 부분만 별도 확인
    R_original = original_matrix[:, :3, :3]
    R_recovered = recovered_matrix[:, :3, :3]
    R_error = tf.reduce_mean(tf.abs(R_original - R_recovered))
    print(f"  Rotation matrix error: {R_error:.8f}")
    
    # Translation 부분만 별도 확인
    t_original = original_matrix[:, :3, 3]
    t_recovered = recovered_matrix[:, :3, 3]
    t_error = tf.reduce_mean(tf.abs(t_original - t_recovered))
    print(f"  Translation error: {t_error:.8f}")
    
    # Test 3: With depth scaling
    print("\n\nTest 3: With Depth Scaling")
    print("-" * 40)
    
    # 테스트용 depth map 생성
    test_depth = tf.random.uniform([batch_size, 192, 640, 1], 0.1, 10.0)
    test_vec_depth = tf.random.normal([batch_size, 6]) * 0.1
    
    # Forward with depth
    matrix_with_depth = pose_axis_angle_vec2mat(test_vec_depth, depth=test_depth, invert=False)
    
    # Backward with depth
    recovered_vec_depth = matrix_to_axis_angle_vectorized(matrix_with_depth, depth=test_depth)
    
    error_depth = tf.reduce_mean(tf.abs(test_vec_depth - recovered_vec_depth))
    print(f"  Average error with depth scaling: {error_depth:.8f}")
    print(f"  Max error with depth scaling: {tf.reduce_max(tf.abs(test_vec_depth - recovered_vec_depth)):.8f}")
    
    # Test 4: Invert option
    print("\n\nTest 4: Invert Option Test")
    print("-" * 40)
    
    test_vec_inv = tf.random.normal([batch_size, 6]) * 0.2
    
    # Forward with invert=False
    matrix_no_inv = pose_axis_angle_vec2mat(test_vec_inv, depth=None, invert=False)
    
    # Forward with invert=True
    matrix_inv = pose_axis_angle_vec2mat(test_vec_inv, depth=None, invert=True)
    
    # Check if they are inverses
    identity_check = tf.matmul(matrix_inv, matrix_no_inv)
    identity_error = tf.reduce_mean(tf.abs(identity_check - tf.eye(4, batch_shape=[batch_size])))
    print(f"  Identity check error (M_inv * M): {identity_error:.8f}")
    
    # Test 5: Stereo transformation
    print("\n\nTest 5: Stereo Transformation Simulation")
    print("-" * 40)
    
    # 스테레오 변환: 작은 회전, x축 translation만
    stereo_R = tf.eye(3, batch_shape=[batch_size])
    stereo_t = tf.constant([[0.1, 0.0, 0.0]], dtype=tf.float32) * tf.ones([batch_size, 1])
    stereo_t = tf.reshape(stereo_t, [batch_size, 3])
    
    # 작은 회전 추가
    small_angle = tf.random.normal([batch_size, 3]) * 0.001
    small_rotation_matrix = pose_axis_angle_vec2mat(
        tf.concat([small_angle, tf.zeros([batch_size, 3])], axis=1), 
        depth=None, 
        invert=False
    )
    stereo_R = tf.matmul(stereo_R, small_rotation_matrix[:, :3, :3])
    
    # 4x4 행렬 조립
    zeros = tf.zeros([batch_size, 1, 3])
    ones = tf.ones([batch_size, 1, 1])
    upper = tf.concat([stereo_R, tf.expand_dims(stereo_t, 2)], axis=2)
    lower = tf.concat([zeros, ones], axis=2)
    stereo_matrix = tf.concat([upper, lower], axis=1)
    
    # Convert to pose vector and back
    stereo_pose_vec = matrix_to_axis_angle_vectorized(stereo_matrix, depth=None)
    recovered_stereo_matrix = pose_axis_angle_vec2mat(stereo_pose_vec, depth=None, invert=False)
    
    stereo_error = tf.reduce_mean(tf.abs(stereo_matrix - recovered_stereo_matrix))
    print(f"  Stereo transformation error: {stereo_error:.8f}")
    print(f"  Stereo pose vector (sample): {stereo_pose_vec[0].numpy()}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    all_errors = [error, matrix_error, error_depth, identity_error, stereo_error]
    print(f"  All tests passed: {all(e < 1e-5 for e in all_errors)}")
    print(f"  Max error across all tests: {max(e for e in all_errors):.8f}")
    print("=" * 60)

if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Run tests
    test_conversion_consistency()