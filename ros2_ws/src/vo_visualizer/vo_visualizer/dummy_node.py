# 파일: vo_visualizer/vo_visualizer/dummy_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
import tf2_ros
import numpy as np

def create_dummy_pointcloud(num_points=100):
    # 1) 랜덤 XYZ 생성
    xyz = np.random.uniform(-1.0, 1.0, size=(num_points, 3)).astype(np.float32)
    # 2) RGB 흰색으로 고정
    rgb = np.ones((num_points, 3), dtype=np.uint8) * 255

    # 3) PointField 정의 (키워드 인자 사용)
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]

    # 4) RGB packing
    rgb_packed = (
        (rgb[:, 0].astype(np.uint32) << 16) |
        (rgb[:, 1].astype(np.uint32) << 8) |
         rgb[:, 2].astype(np.uint32)
    )

    # 5) structured array 생성
    pc_data = np.zeros(num_points, dtype=[('x','f4'),('y','f4'),('z','f4'),('rgb','u4')])
    pc_data['x'], pc_data['y'], pc_data['z'], pc_data['rgb'] = (
        xyz[:,0], xyz[:,1], xyz[:,2], rgb_packed
    )

    # 6) PointCloud2 메시지 생성
    msg = PointCloud2()
    msg.header = Header()            # 빈 Header 생성
    msg.header.frame_id = 'world'    # 프레임 설정
    msg.height = 1
    msg.width = num_points
    msg.fields = fields
    msg.is_dense = False
    msg.is_bigendian = False
    msg.point_step = 16              # 4 floats + 1 uint32 = 16 bytes
    msg.row_step = msg.point_step * msg.width
    msg.data = pc_data.tobytes()
    return msg

class DummyVisualizer(Node):
    def __init__(self):
        super().__init__('dummy_visualizer')
        # PointCloud2 퍼블리셔
        self.pc_pub = self.create_publisher(PointCloud2, 'dummy/points', 10)
        # TF 브로드캐스터
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # 1Hz 타이머
        self.timer = self.create_timer(1.0, self.on_timer)
        self.get_logger().info('DummyVisualizer node started.')

    def on_timer(self):
        # 포인트클라우드 생성 및 퍼블리시
        pc2 = create_dummy_pointcloud(200)
        pc2.header.stamp = self.get_clock().now().to_msg()
        self.pc_pub.publish(pc2)
        self.get_logger().debug('Published dummy PointCloud2')

        # 정적 TF 브로드캐스트 (world → camera)
        t = TransformStamped()
        t.header.stamp = pc2.header.stamp
        t.header.frame_id = 'world'
        t.child_frame_id = 'camera'
        t.transform.translation.x = 0.5
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.2
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().debug('Broadcasted dummy TF')

def main(args=None):
    rclpy.init(args=args)
    node = DummyVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
