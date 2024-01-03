import cv2
import numpy as np
import os
import glob
from process_raw import DngFile

# 체커보드의 차원 정의
CHECKERBOARD = (4,5) # 체커보드 행과 열당 내부 코너 수
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 각 체커보드 이미지에 대한 3D 점 벡터를 저장할 벡터 생성
objpoints = []
# 각 체커보드 이미지에 대한 2D 점 벡터를 저장할 벡터 생성
imgpoints = [] 
# 3D 점의 세계 좌표 정의
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
# 주어진 디렉터리에 저장된 개별 이미지의 경로 추출
images = glob.glob('/home/piljae98/data_set/checkerboard/*.DNG')

# 이미지 변수를 초기화
img = None
gray = None
mtx = None

for fname in images:
    dng = DngFile.read(fname)
    img = dng.postprocess()
    img = img[:, :, ::-1]
    if img is not None:  # 이미지가 성공적으로 읽어왔을 때만 처리
        # 그레이 스케일로 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray is not None:
            # 체커보드 코너 찾기
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # 원하는 개수의 코너가 감지되면,
            # 픽셀 좌표 미세조정 -> 체커보드 이미지 표시
            if ret == True:
                objpoints.append(objp)
                # 주어진 2D 점에 대한 픽셀 좌표 미세조정
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                # 코너 그리기 및 표시
                img = cv2.drawChessboardCorners(gray, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)

cv2.destroyAllWindows()

if img is not None:  # 이미지가 성공적으로 읽어왔을 때만 처리
    h, w = img.shape[:2]  # 마지막 이미지의 높이와 너비 가져옴
# 알려진 3D 점(objpoints) 값과 감지된 코너의 해당 픽셀 좌표(imgpoints) 전달, 카메라 캘리브레이션 수행
if gray is not None:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n") # 내부 카메라 행렬
print(mtx)