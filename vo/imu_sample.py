import numpy as np
from scipy.spatial.transform import Rotation as R

def remove_gravity_dynamic(acc_measured, orientation):
    """
    acc_measured: 3D 가속도 센서 측정값 (m/s^2), 센서 좌표계 기준.
    orientation:  (roll, pitch, yaw) in radians, 센서가 바라보는 자세
                  예: (0.0, 0.0, 0.0) 이면, z축이 위쪽으로 향한다고 가정(기본 자세).
                  or 쿼터니언으로 주어질 수도 있음(그럼 코드 일부 변경).
    return:       중력 벡터를 제거한 실제 선형 가속도 (센서 좌표계 기준)
    """
    # 월드 좌표계에서 중력 벡터 (z+ 방향이 위라고 가정)
    g_world = np.array([0, 0, 9.81])
    
    # orientation(roll, pitch, yaw)를 회전 행렬로 변환
    rot = R.from_euler('xyz', orientation, degrees=False)  
    # 센서 좌표계에서 보았을 때의 중력 벡터
    g_sensor = rot.apply(g_world)
    
    # 측정된 가속도에서 중력을 뺀다
    acc_linear = acc_measured - g_sensor
    
    return acc_linear

# 예시 시뮬레이션
if __name__ == "__main__":
    # 예를 들어, 5회 측정치가 있고, 매번 기기 자세가 조금씩 변한다고 하자
    acc_data = [
        np.array([0.0, 0.0, 9.8]),   # 처음엔 z축이 위, 실제 선형가속도는 0
        np.array([1.0, 0.0, 9.7]),   # 가로로 조금 움직임
        np.array([9.8, 0.1, 0.2]),   # 기기를 90도 돌려 X축이 위로 간 상태
        np.array([7.0, 7.0, 4.0]),   # 45도 기울여서 중력이 분산
        np.array([-0.5, 0.3, 9.9])   # 또 다른 자세
    ]
    
    # roll, pitch, yaw (rad)로 가정(단순 예시)
    orientation_data = [
        (0.0, 0.0, 0.0),          # z축 위
        (0.0, 0.1, 0.0),          # 살짝 pitch
        (0.0, np.pi/2, 0.0),      # 90도 회전
        (np.pi/4, np.pi/4, 0.0),  # 45도 기울임
        (0.05, 0.02, 0.1),        # 약간 복합 회전
    ]
    
    for i, (acc, ori) in enumerate(zip(acc_data, orientation_data)):
        acc_lin = remove_gravity_dynamic(acc, ori)
        print(f"측정 {i}: orientation={ori}, acc={acc}, 중력제거-> {acc_lin}")
