import numpy as np

# ------------------------------------------------------------------------------
# 도우미 함수들
# ------------------------------------------------------------------------------
def normalize_quaternion(q):
    """Quaternion을 정규화하는 간단한 함수."""
    return q / np.linalg.norm(q)

def quaternion_to_rotation_matrix(q):
    """
    Quaternion -> 3x3 회전 행렬 변환.
    q = [qw, qx, qy, qz]
    """
    qw, qx, qy, qz = q
    # 회전 행렬 공식
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qw*qz,     2*qx*qz + 2*qw*qy    ],
        [2*qx*qy + 2*qw*qz,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx    ],
        [2*qx*qz - 2*qw*qy,     2*qy*qz + 2*qw*qx,     1 - 2*qx**2 - 2*qy**2]
    ], dtype=np.float32)
    return R

def rotation_matrix_to_quaternion(R):
    """
    3x3 회전행렬 -> Quaternion. (W, X, Y, Z) 순서.
    """
    # 수치 안정성 고려해 eps 추가
    eps = 1e-8
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0 + eps)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        # 대각원소 중 최댓값을 기준으로 계산
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(max(eps, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(max(eps, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(eps, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    return normalize_quaternion(q)

def build_transformation_matrix(pos, quat):
    """
    위치 pos(3,)와 쿼터니언 quat(4,) -> 4x4 변환행렬.
    """
    R = quaternion_to_rotation_matrix(quat)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T

def decompose_transformation_matrix(T):
    """
    4x4 변환행렬 -> (pos(3,), quat(4,)) 분해.
    """
    R = T[:3, :3]
    pos = T[:3, 3]
    quat = rotation_matrix_to_quaternion(R)
    return pos, quat

# ------------------------------------------------------------------------------
# 간단한 확장 칼만 필터(EKF) 클래스 예시
# ------------------------------------------------------------------------------
class SimpleEKF:
    def __init__(self):
        # 상태 벡터: [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
        # 초기값(정지, 단위쿼터니언)
        self.state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)

        # 공분산 행렬
        self.P = np.eye(10, dtype=np.float32) * 1e-3

        # 프로세스 노이즈, 측정 노이즈 가정
        self.Q = np.eye(10, dtype=np.float32) * 1e-3   # 시스템(과정) 노이즈
        self.R = np.eye(6, dtype=np.float32) * 1e-2    # 측정(포즈) 노이즈

    def predict(self, imu_data, dt):
        """
        imu_data: (gx, gy, gz, ax, ay, az) 단일 샘플이라고 가정.
        dt: 샘플 간격(sec).
        """
        # 상태 벡터 파싱
        px, py, pz, vx, vy, vz, qw, qx, qy, qz = self.state
        gyro = imu_data[:3]  # (gx, gy, gz)
        acc  = imu_data[3:]  # (ax, ay, az)

        # 현재 쿼터니언 -> 회전행렬
        R = quaternion_to_rotation_matrix([qw, qx, qy, qz])
        # 예측 단계(가장 간단한 버전):
        #   1) 오리엔테이션 갱신(소각 근사 혹은 쿼터니언 미분)
        #   2) 속도 갱신
        #   3) 위치 갱신

        # 1) Gyro 기반 Orientation 업데이트(아주 단순화)
        # 실제로는 쿼터니언 미분 또는 소각 근사: delta_q ~ 1 + 0.5*gyro*dt
        angle_axis = gyro * dt
        # 소각 근사: 회전 벡터가 (rx, ry, rz)일 때,
        # dq ~= [1, 0.5*rx, 0.5*ry, 0.5*rz]
        dq = np.array([
            1.0,
            0.5 * angle_axis[0],
            0.5 * angle_axis[1],
            0.5 * angle_axis[2]
        ], dtype=np.float32)
        q_new = quaternion_multiply([qw, qx, qy, qz], dq)
        q_new = normalize_quaternion(q_new)

        # 2) 가속도 기반 속도 업데이트 (중력 상쇄 등을 단순화한다고 가정)
        # R @ acc => 바디좌표계 측정 a를 월드좌표계로 변환
        acc_world = R @ acc
        vx_new = vx + acc_world[0] * dt
        vy_new = vy + acc_world[1] * dt
        vz_new = vz + acc_world[2] * dt

        # 3) 위치 업데이트
        px_new = px + vx_new * dt
        py_new = py + vy_new * dt
        pz_new = pz + vz_new * dt

        # state 갱신
        self.state = np.array([
            px_new, py_new, pz_new,
            vx_new, vy_new, vz_new,
            q_new[0], q_new[1], q_new[2], q_new[3]
        ], dtype=np.float32)

        # 공분산 P도 예측. (선형 근사 자코비안 필요 -> 단순화하여 P += Q)
        self.P = self.P + self.Q

    def update(self, measured_transform):
        """
        딥러닝 모델이 추정한 4x4 변환행렬로부터 위치, 오리엔테이션을 측정값으로 사용한다.
        측정값 z: [px, py, pz, roll, pitch, yaw] (단순 6D로 사용 가능) 등등
        
        여기서는 4x4를 pos, quat으로 분해 후, 
        (px, py, pz, qx, qy, qz)만 쓰는 식으로 단순화.
        """
        z_pos, z_quat = decompose_transformation_matrix(measured_transform)

        # 내부적으로는 [px, py, pz, qw, qx, qy, qz]를 사용하지만,
        # roll/pitch/yaw 형태로 측정 업데이트를 할 수도 있음. 
        # 여기서는 (px, py, pz, qx, qy, qz) 형태로 단순화된 측정값으로 사용.
        # 측정 벡터 z: 6차원 (pos 3 + quat( x, y, z ) => qw는 별도로?)
        # EKF에서는 오리엔테이션 측정이 약간 까다롭지만, 간단히 처리하겠습니다.
        
        # 상태에서 추출한 예측값
        px, py, pz, vx, vy, vz, qw, qx, qy, qz = self.state
        
        # 측정 z
        z_vec = np.concatenate([z_pos, z_quat[1:]], axis=0)  # (6,)
        
        # 예측값 h(x)
        # 위치 3개 + quat( x, y, z ) 3개만 측정과 비교
        h_pos = np.array([px, py, pz], dtype=np.float32)
        h_quat = np.array([qw, qx, qy, qz], dtype=np.float32)
        h_vec = np.concatenate([h_pos, h_quat[1:]], axis=0)  # (6,)

        # 잔차
        y = z_vec - h_vec  # (6,)

        # 측정행렬 H (선형 근사). 여기서는 단위 행렬 일부를 발췌한 형태로 단순 가정
        # 상태는 10차원 -> 측정은 6차원
        # [ px py pz vx vy vz qw qx qy qz ]
        # 측정은 [ px py pz qx qy qz ]
        H = np.zeros((6, 10), dtype=np.float32)
        # 위치 부분
        H[0, 0] = 1.0  # px
        H[1, 1] = 1.0  # py
        H[2, 2] = 1.0  # pz
        # quat 중 x,y,z 부분만
        H[3, 8] = 1.0  # qy => 실제로 인덱스 맞춤 필요
        # 주의: 우리가 state에 qw,qx,qy,qz=(6,7,8,9)이므로
        #       z_vec의 [3,4,5]가 (qx,qy,qz)에 해당
        #       state의 [7,8,9]가 (qx,qy,qz)
        #       잘 맞춰야 함.
        H[3, 7] = 1.0  # qx
        H[4, 8] = 1.0  # qy
        H[5, 9] = 1.0  # qz

        # 칼만 이득 K = P H^T (H P H^T + R)^-1
        S = H @ self.P @ H.T + self.R  # (6,6)
        K = self.P @ H.T @ np.linalg.inv(S)  # (10,6)
        
        # 상태 보정
        x_update = self.state + K @ y  # (10,) 
        self.state = x_update
        
        # P 보정: (I - K H) P
        I = np.eye(10, dtype=np.float32)
        self.P = (I - K @ H) @ self.P
        
        # 마지막으로 quaternion 정규화
        qw_new, qx_new, qy_new, qz_new = self.state[6:10]
        q_norm = np.linalg.norm(self.state[6:10])
        if q_norm > 1e-8:
            self.state[6:10] /= q_norm

    def get_state(self):
        """현재 추정 상태 (pos, vel, quat)."""
        px, py, pz, vx, vy, vz, qw, qx, qy, qz = self.state
        return (np.array([px, py, pz], dtype=np.float32),
                np.array([vx, vy, vz], dtype=np.float32),
                np.array([qw, qx, qy, qz], dtype=np.float32))

    def get_transformation_matrix(self):
        """현재 추정 pos, quat -> 4x4 행렬 반환."""
        pos, vel, quat = self.get_state()
        return build_transformation_matrix(pos, quat)

def quaternion_multiply(q1, q2):
    """q1, q2: [qw, qx, qy, qz] 형태."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z], dtype=np.float32)