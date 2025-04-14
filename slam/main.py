import cv2
import os
import numpy as np
import tensorflow as tf, tf_keras
from MonoVO import MonoVO
import numpy as np
import cv2
import time
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from display import display_trajectory
# from vo.utils.visualization import Visualizer


def calc_avg_matches(frame, out_frame, show_correspondence=False):
    """Return the average number of matches each keypoint in the specified frame has. 
    Visualize these matches if show_correspondence=True"""
    n_match = 0		# avg. number of matches of keypoints in the current frame
    for idx in frame.pts:
        # red line to connect current keypoint with Point location in other frames
        pt = [int(i) for i in frame.kps[idx]]
        if show_correspondence:
            for f, f_idx in zip(frame.pts[idx].frames, frame.pts[idx].idxs):
                cv2.line(out_frame, pt, [int(i) for i in f.kps[f_idx]], (0, 0, 255), thickness=2)
        n_match += len(frame.pts[idx].frames)
    if len(frame.pts) > 0:
        n_match /= len(frame.pts)
    return n_match, out_frame


class OfflineRunner:
	def __init__(self,
                 video_path: str,
                 camera_poses: np.ndarray,
                 intrinsic: np.ndarray,
                 image_size: tuple = (480, 640)):
		self.video_path = video_path
		self.image_size = image_size
		self.cap = cv2.VideoCapture(self.video_path)
		self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		self.camera_poses = camera_poses
		self.intrinsic = intrinsic
		self.mono_vo = MonoVO(self.intrinsic)
		# self.visualizer = Visualizer(draw_plane=False, is_record=False, video_fps=30, video_name="visualization.mp4")
		self.flip_transform = np.diag([1, -1, -1, 1])

	def run(self):
		current_idx = 0
		current_pose = np.eye(4)

		while self.cap.isOpened():
			ret, frame = self.cap.read()
			if not ret:
				break
			print("\n*** frame %d/%d ***" % (current_idx, CNT))
			frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))

			# return depth, uncertainty, self.mp.frames[-1].pose, a, b
			_ = self.mono_vo.process_frame(frame, optimize=True)

			if DEBUG:
				# plot all poses (invert poses so they move in correct direction)
				display_trajectory([f.pose for f in self.mono_vo.mp.frames])

				# show keypoints with matches in this frame
				for pidx, p in enumerate(self.mono_vo.mp.frames[-1].kps):
					if pidx in self.mono_vo.mp.frames[-1].pts:
						# green for matched keypoints 
						cv2.circle(frame, [int(i) for i in p], color=(0, 255, 0), radius=3)
					else:
						# black for unmatched keypoint in this frame
						cv2.circle(frame, [int(i) for i in p], color=(0, 0, 0), radius=3)

				# Calculate the average number of frames each point in the last frame is also visible in
				n_match, frame = calc_avg_matches(self.mono_vo.mp.frames[-1], frame, show_correspondence=False)
				print("Matches: %d / %d (%f)" % (len(self.mono_vo.mp.frames[-1].pts), len(self.mono_vo.mp.frames[-1].kps), n_match))

				
				cv2.imshow('d3vo', frame)
				if cv2.waitKey(1) == 27:     # Stop if ESC is pressed
					break

			current_idx += 1
			# self.visualizer.render()
        
		self.cap.release()
		cv2.destroyAllWindows()
        
		save_path = os.path.join('./output_pose.npy')
		np.save(save_path, self.mono_vo.mp.relative_to_global())
		print("-> Predictions saved to", save_path)


if __name__ == "__main__":
	with tf.device('/gpu:0'):

		video_path = '/media/park-ubuntu/park_cs/slam_data/mars_logger/test/2025_01_06_18_21_32/movie.mp4'
		cap = cv2.VideoCapture(video_path)
		DEBUG = False
		PER_FRAME_ERROR = True
		W = 640
		H = 480
		intrinsic = np.array([[429.6226669926592, 0.0, 313.09860139546925],
						[0.0, 429.12834359066613, 251.87390881347434],
                                        [0.,   0.,   1.]])
		CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		OfflineRunner(video_path=video_path,
				camera_poses=None,
				intrinsic=intrinsic,
				image_size=(H, W)).run()
		#  (video_path=video_path,
		#            camera_poses=None,
		#            intrinsic=np.array([[427.0528736,   0., 328.9062192],
		#                                [0., 427.0528736, 230.6455664],
		# 							   [0.       ,   0.       ,   1.       ]])