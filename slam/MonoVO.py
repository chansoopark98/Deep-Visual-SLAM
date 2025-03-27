from optimizer import Map
from frontend import Frame, Point, match_frame_kps
from network import Networks
import numpy as np

class MonoVO:
	def __init__(self, intrinsic):
		self.intrinsic = intrinsic
		self.mp = Map()
		self.nn = Networks((480, 640))

	def process_frame(self, frame, optimize=True):
		"""Process a single frame with D3VO."""
		# Run DepthNet to get depth map
		depth = self.nn.depth(frame)
		
		uncertainty = depth * 0.05  # 5%의 상대 불확실성
		
		if len(self.mp.frames) == 0:
			# Set first frame pose to identity 
			pose = np.eye(4)
			# Default brightness params
			a, b = 1.0, 0.0
		else:
			# Pass PoseNet the two most recent frames 
			pose = self.nn.pose(self.mp.frames[-1].image, frame, depth=depth)
			# Default brightness params
			a, b = 1.0, 0.0
		
		# Run frontend tracking with added uncertainty and brightness params
		if not self.frontend(frame, depth, uncertainty, pose, (a, b)):
			return
			
		# Run backend optimization
		if optimize:
			self.mp.optimize(self.intrinsic)

	def frontend(self, frame, depth, uncertainty, pose, brightness_params):
		"""Run frontend tracking on the given frame."""
		# Create frame with all parameters
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