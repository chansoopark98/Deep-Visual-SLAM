import cv2
import os
import numpy as np
from MonoVO import MonoVO

DEBUG = True
PER_FRAME_ERROR = True


def offline_vo(cap, save_path):
	"""Run D3VO on offline video"""
	intrinsic = np.array([[427.0528736,   0.       , 328.9062192],
	   [  0.       , 427.0528736, 230.6455664],
	   [  0.       ,   0.       ,   1.       ]])
	monoVO = MonoVO(intrinsic)

	# Run D3VO offline with prerecorded video
	i = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:				
			frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			frame = cv2.resize(frame, (W, H))
			print("\n*** frame %d/%d ***" % (i, CNT))
			monoVO.process_frame(frame)

		else:
			break
		i += 1

		if DEBUG:
			cv2.imshow('d3vo', frame)
			if cv2.waitKey(1) == 27:     # Stop if ESC is pressed
				break
	

	# Store pose predictions to a file (do not save identity pose of first frame)
	save_path = os.path.join(save_path)
	np.save(save_path, monoVO.mp.relative_to_global())
	print("-> Predictions saved to", save_path)


if __name__ == "__main__":
	video_path = '/home/park-ubuntu/park/Deep-Visual-SLAM/vo/data/mars_logger/test/2025_01_06_18_21_32/movie.mp4'
	cap = cv2.VideoCapture(video_path)
	W = 640
	H = 480
	CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


	# run offline visual odometry on provided video
	offline_vo(cap, './')