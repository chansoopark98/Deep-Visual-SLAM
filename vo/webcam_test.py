import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from model.monodepth2 import build_posenet
from vo.utils.projection_utils import pose_vec2mat

# 카메라 궤적을 저장할 리스트
trajectory = []

# 초기 변환 행렬 설정 (단위 행렬)
current_transform = np.eye(4, dtype=np.float32)

def capture_webcam_video():
    global current_transform
    
    model = build_posenet(image_shape=(432, 768), batch_size=1)
    image_buffer = []
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        global current_transform
        
        ret, raw_frame = cap.read()
        if not ret:
            return

        frame = raw_frame.copy()[:, :, ::-1]
        frame = tf.convert_to_tensor(frame, dtype=tf.uint8)
        frame = tf.image.resize(frame, size=(432, 768))
        frame = tf.expand_dims(frame, axis=0)
        frame = tf.keras.applications.imagenet_utils.preprocess_input(frame, mode='torch')

        image_buffer.append(frame)

        if len(image_buffer) >= 2:
            input_frames = tf.concat([image_buffer[0], image_buffer[1]], axis=-1)

            poses = model(input_frames, training=False)[0] # 1, 1, 6 -> 6
            print(poses)
            transform_matrix = pose_vec2mat(poses)[0]
            
            # 누적 변환 행렬 업데이트
            current_transform = np.matmul(current_transform, transform_matrix)

            # 현재 위치 계산
            position = current_transform[:3, 3]
            x, y, z = position[0], position[1], position[2]
            
            # 궤적에 추가
            trajectory.append((x, y, z))
            
            # 3D 그래프 업데이트
            ax.clear()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            if len(trajectory) > 0:
                data = np.array(trajectory)
                ax.plot(data[:, 0], data[:, 1], data[:, 2], color='b')

            # 프레임을 화면에 출력
            cv2.imshow('Webcam Video', raw_frame)

            image_buffer.pop(0)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            plt.close(fig)
            cap.release()
            cv2.destroyAllWindows()

    ani = FuncAnimation(fig, animate, interval=100)
    plt.show()

if __name__ == "__main__":
    capture_webcam_video()
