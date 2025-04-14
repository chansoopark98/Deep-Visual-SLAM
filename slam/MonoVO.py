from optimizer import Map
from frontend import Frame, Point, match_frame_kps
from network import Networks
import numpy as np

class MonoVO:
	def __init__(self, intrinsic):
		self.intrinsic = intrinsic
		self.mp = Map()
		self.nn = Networks(depth_weight_path='./assets/weights/vo/0414_marsOnly/depth_net_epoch_30_model.weights.h5',
							pose_weight_path='./assets/weights/vo/0414_marsOnly/pose_net_epoch_30_model.weights.h5',
							image_shape=(480, 640))

		
	def process_frame(self, frame, optimize=True):
		"""Process a single frame with D3VO."""
		# Run DepthNet to get depth map
		depth= self.nn.depth(frame)
		uncertainty = np.zeros_like(depth)
		
		if len(self.mp.frames) == 0:
			# Set first frame pose to identity 
			pose, a, b = np.eye(4), 1, 0
		else:
			# Pass PoseNet the two most recent frames 
			pose = self.nn.pose(self.mp.frames[-1].image, frame, depth=depth)
			a = 1
			b = 0
			
		# Run frontend tracking with added uncertainty and brightness params
		if not self.frontend(frame, depth, uncertainty, pose, (a, b)):
			return
			
		# Run backend optimization
		if optimize:
			self.mp.optimize(self.intrinsic)
		
		return depth, uncertainty, self.mp.frames[-1].pose, a, b

	def frontend(self, frame, depth, uncertainty, pose, brightness_params):
		"""Run frontend tracking on the given frame. Extract keypoints, match them keypoints in the preceding 
        frame, add Frame to the map, and possibly make the Frame a keyframe. Return true to run backend 
        optimization after this function returns."""
		# create frame and add it to the map
		f = Frame(self.mp, frame, depth, uncertainty, pose, brightness_params)

		# cannot match initial frame to any previous frames (but make it a keyframe)
		if f.id == 0:
			self.mp.check_add_key_frame(f)
			return False

		# Process f and the preceeding frame with a feature matcher. Iterate over match indices
		prev_f = self.mp.frames[-2]
		l1, l2 = match_frame_kps(f, prev_f)

		# Store matches
		for idx1, idx2 in zip(l1, l2):
			if idx2 in prev_f.pts:
				# Point already exists in prev_f
				prev_f.pts[idx2].add_observation(f, idx1)
			else:
				# New point
				pt = Point(self.mp)
				pt.add_observation(f, idx1)
				pt.add_observation(prev_f, idx2)

		# Check if this new frame should be a keyframe
		if self.mp.check_add_key_frame(f):
			# Keyframe has been added, run backend optimization
			return True

		return False