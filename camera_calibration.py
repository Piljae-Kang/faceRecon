import cv2
import numpy as np
import os
import glob
from process_raw import DngFile
import argparse

def draw_point_in_image(image, w_points, K, R, T, image_point, dist, image_num):


    RT = np.hstack((R, T.reshape(-1, 1)))
    M = np.dot(K, RT)

    # print("R : ")
    # print(R)
    # print("T : ")
    # print(T)
    # print("RT : ")
    # print(RT)

    count = 0
    error_x = 0
    error_y = 0

    for i, w_p in enumerate(w_points):

        count += 1

        w_p = np.array([w_p[0], w_p[1], w_p[2], 1])

        i_p = np.dot(M, w_p)

        x = i_p[0]/i_p[2]
        y = i_p[1]/i_p[2]

        # print("Result calcuated by calibration -> x : ", x , " y : ", y)
        # print("Result of corner detection -> x : ", image_point[i][0][0], " y : ", image_point[i][0][1])
        # print()

        error_x += abs(x - image_point[i][0][0])
        error_y += abs(y - image_point[i][0][1])
        for i in range(-2, 2):
            for j in range(-2, 2):
                image[int(y) + i, int(x) + j] = (0, 0, 255)
    
    image_name = f'image{image_num}.jpg'
    cv2.imwrite(image_name, image)

    print("image ", image_num,"avg error of pixel x : ", round(error_x/count, 4), " | avg error of pixel y : ", round(error_y/count, 4))
    

# argument parsing part
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input',default='/home/piljae98/data_set/checkerboard/', help='input file path')
parser.add_argument('-o', '--output', help='output file path')
args = parser.parse_args()

# checkerboard dimension
CHECKERBOARD = (4,5) # inner checkboard cols and rows
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# vector of  checkerboard 3D world coordinate 
objpoints = []
# vector of  checkerboard 2D image corrdinate
imgpoints = [] 
# 3D world coordinates point
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# image path
input_path = args.input + "/*.DNG"
#images = glob.glob('/home/piljae98/data_set/checkerboard/*.DNG')
images = glob.glob(input_path)

# initialize image variables
img = None
gray = None
mtx = None

for i, fname in enumerate(images):
    dng = DngFile.read(fname)
    img = dng.postprocess()
    img = img[:, :, ::-1]
    if img is not None:  # image read success
        # convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray is not None:
            # find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # 원하는 개수의 코너가 감지되면,
            # 픽셀 좌표 미세조정 -> 체커보드 이미지 표시
            if ret == True:
                objpoints.append(objp)
                # 주어진 2D 점에 대한 픽셀 좌표 미세조정
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                # draw corners
                img = cv2.drawChessboardCorners(gray, CHECKERBOARD, corners2, ret)
            cv2.imwrite(f'img{i}.jpg', img)



if img is not None:  # 이미지가 성공적으로 읽어왔을 때만 처리
    h, w = img.shape[:2]  # 마지막 이미지의 높이와 너비 가져옴
# 알려진 3D 점(objpoints) 값과 감지된 코너의 해당 픽셀 좌표(imgpoints) 전달, 카메라 캘리브레이션 수행
if gray is not None:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n") # 내부 카메라 행렬
print(mtx)

print("distortion coefficient:")
print(dist)

R_list = []
T_list = []

for i in range(len(rvecs)):

    R = cv2.Rodrigues(rvecs[i])[0]
    R_list.append(R)
    
    T = tvecs[i]
    T_list.append(T)

print("--------------------------------------------------")
print("\nthis is for verification of camera calibration\n")
print("--------------------------------------------------")
print()

for i, fname in enumerate(images):
    dng = DngFile.read(fname)
    img = dng.postprocess()
    img = img[:, :, ::-1]
    if img is not None:  # image read success
        # convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray is not None:
            # find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret == True:
                objpoints.append(objp)
                # 주어진 2D 점에 대한 픽셀 좌표 미세조정
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                #corners2와 직접 KRT 계산한 결과 비교
                draw_point_in_image(img, objp[0], mtx, R_list[i], T_list[i], imgpoints[i], dist, i)