import numpy as np
from marker_detection_using_dlib import marker_detector
from camera_calibration import camera_calibration


class Camera:
    def __init__(self, K, R, T, dist, camera_pos):
        self.K = K
        self.R = R
        self.T = T
        self.dist = dist
        self.camera_pos = camera_pos


checkerboard_image_path = "/home/piljae98/data_set/checkerboard/camera1"
K, R, T, dist, camera_pos = camera_calibration(checkerboard_image_path)
camera1 = Camera(K, R, T, dist, camera_pos)

checkerboard_image_path = "/home/piljae98/data_set/checkerboard/camera2"
K, R, T, dist, camera_pos = camera_calibration(checkerboard_image_path)
camera2 = Camera(K, R, T, dist, camera_pos)

detected_point_camera1 = marker_detector("image0_camera1.png")
print(detected_point_camera1)

detected_point_camera2 = marker_detector("image0_camera2.png")
print(detected_point_camera2)