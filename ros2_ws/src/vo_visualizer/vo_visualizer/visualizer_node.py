import os
import sys
import yaml
# 프로젝트 루트 경로를 sys.path에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
import tf2_ros
import numpy as np
import torch

from model.depthnet import DepthNet
from model.posenet_single import PoseNet
from vo.utils.utils import remove_prefix_from_state_dict
from vo.learner_func import disp_to_depth, transformation_from_parameters
from vo.dataset.vo_loader import VoDataLoader

# Helper to convert XYZ and color arrays into a PointCloud2 message
def create_pointcloud2(points, colors, frame_id='world'):
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),  # float32로 변경
    ]
    # pack RGB into uint32
    rgb_uint32 = (
        (colors[:, 0].astype(np.uint32) << 16) |
        (colors[:, 1].astype(np.uint32) << 8) |
         colors[:, 2].astype(np.uint32)
    )
    rgb_packed = rgb_uint32.view(np.float32)  # uint32를 float32로 reinterpret

    pc_data = np.zeros(points.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4'),('rgb','f4')])
    pc_data['x'], pc_data['y'], pc_data['z'], pc_data['rgb'] = (
        points[:,0], points[:,1], points[:,2], rgb_packed
    )
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = fields
    msg.is_dense = False
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.data = pc_data.tobytes()
    return msg

class VisualizerNode(Node):
    def __init__(self):
        super().__init__('vo_visualizer')
        # Publishers and broadcasters
        self.pc_pub = self.create_publisher(PointCloud2, 'camera/points', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Load config
        pkg_share = get_package_share_directory('vo_visualizer')
        config_path = os.path.join(pkg_share, 'config.yaml')  # place your config.yaml here
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load PoseNet
        self.pose_net = PoseNet(
            num_layers=18,
            pretrained=True,
            num_input_images=2
        ).to(self.device)
        pose_w = os.path.join(pkg_share, 'weights', 'vo', 'pose_net_epoch_18.pth')
        pd = torch.load(pose_w)
        pd = remove_prefix_from_state_dict(pd, prefix='_orig_mod.')
        self.pose_net.load_state_dict(pd)
        self.pose_net.eval()

        # Load DepthNet
        self.depth_net = DepthNet(
            num_layers=18,
            pretrained=True,
            num_input_images=1,
            scales=range(4),
            num_output_channels=1,
            use_skips=True
        ).to(self.device)
        depth_w = os.path.join(pkg_share, 'weights', 'vo', 'depth_net_epoch_18.pth')
        dd = torch.load(depth_w)
        dd = remove_prefix_from_state_dict(dd, prefix='_orig_mod.')
        self.depth_net.load_state_dict(dd)
        self.depth_net.eval()

        # DataLoader
        self.config['Train']['batch_size'] = 1
        self.data_loader = VoDataLoader(config=self.config)
        self.test_iter = iter(self.data_loader.test_mono_loader)

        # Initial world pose
        self.world_pose = np.eye(4, dtype=np.float32)

        # Timer at configured fps
        fps = self.config.get('Visualization', {}).get('fps', 30)
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)
        self.get_logger().info('VO Visualizer node initialized')

    def timer_callback(self):
        try:
            sample = next(self.test_iter)
        except StopIteration:
            self.get_logger().info('All test data processed, resetting iterator')
            self.test_iter = iter(self.data_loader.test_mono_loader)
            sample = next(self.test_iter)

        # Move tensors to device
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.to(self.device)

        # Intrinsics
        K = sample[('K', 0)][0].detach().cpu().numpy()
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]

        # Pose inference: concat target and source
        tgt = sample[('target_image', 0)]
        src = sample[('source_right', 0)]
        inp = torch.cat([tgt, src], dim=1)
        axis, trans = self.pose_net(inp)
        T = transformation_from_parameters(axis[:,0], trans[:,0], invert=False)
        T = T.detach().cpu().numpy()[0]

        # Depth inference
        disp = self.depth_net(tgt)[('disp', 0)]
        _, depth_map = disp_to_depth(disp, 
            min_depth=self.config['Train']['min_depth'],
            max_depth=self.config['Train']['max_depth']
        )
        depth_map = depth_map.detach().cpu().numpy()[0,0]

        # Update world pose
        self.world_pose = self.world_pose @ T

        # Generate point cloud
        H, W = depth_map.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth_map.flatten()
        x = (u.flatten() - cx) * z / fx
        y = (v.flatten() - cy) * z / fy
        pts = np.vstack((x, y, z)).T

        # Color from image
        img_tensor = sample[('target_image', 0)][0]  # (C, H, W)
        img_tensor = self.data_loader.denormalize_image(img_tensor.unsqueeze(0))[0]  # (C, H, W)
        img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        img = (img * 255).astype(np.uint8)
        colors = img.reshape(-1, 3)


        # Publish point cloud
        pc2 = create_pointcloud2(pts, colors)
        pc2.header.stamp = self.get_clock().now().to_msg()
        self.pc_pub.publish(pc2)

        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = pc2.header.stamp
        t.header.frame_id = 'world'
        t.child_frame_id = 'camera'
        t.transform.translation.x = float(self.world_pose[0,3])
        t.transform.translation.y = float(self.world_pose[1,3])
        t.transform.translation.z = float(self.world_pose[2,3])
        # quaternion from rotation matrix
        # placeholder: compute quaternion qx,qy,qz,qw
        rot = self.world_pose[:3,:3]
        qw = np.sqrt(1 + rot[0,0] + rot[1,1] + rot[2,2]) / 2
        qx = (rot[2,1] - rot[1,2]) / (4*qw)
        qy = (rot[0,2] - rot[2,0]) / (4*qw)
        qz = (rot[1,0] - rot[0,1]) / (4*qw)
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)
        self.tf_broadcaster.sendTransform(t)

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
