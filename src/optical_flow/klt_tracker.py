# Import library modules
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
# PIL is the Python Imaging Library
from PIL import Image
import dlib

NUMBER_OF_IMAGES = 50  
DISPLAY_RADIUS = 3
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

# Gaussian kernel for smoothing
def linear_filter(img, kernel):
    conv_output = convolve2d(img, kernel, mode='same', boundary='symm')
    return conv_output

def readImages(filecount, image_folder_path):
  print("In function readImages")
  allImages = []
  for i in range(filecount):
    print (f'reading image {i:02}')
    imagetmp = cv2.imread(image_folder_path + "/image" + f'{i}' + ".png", cv2.IMREAD_GRAYSCALE)
    imagetmp = cv2.rotate(imagetmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #imagetmp = cv2.imread(image_folder_path + "/hotel.seq" + f'{i:02d}' + ".png", cv2.COLOR_BGR2GRAY)
    allImages.append(imagetmp)
  return allImages

def harris_corner_detector(img, k, window_size, threshold):
    
    # Harris corner detection
    corners = cv2.cornerHarris(img, blockSize=window_size, ksize=3, k=k)

    # Thresholding
    corners = np.argwhere(corners > 0.1 * corners.max())

    # print(corners.max())

    return corners

def dlib_face_feature_detector(img):

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()

    # cv2.imshow("img",img)
    # cv2.waitKey(0)

    faces = detector(img, 1)

    keypoints = []
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for face in faces:
        landmarks = predictor(img, face)

        for i in range(68):
            x, y = landmarks.part(i).x , landmarks.part(i).y
            cv2.circle(img_color, (x, y), DISPLAY_RADIUS, GREEN)
            keypoints.append([x, y])

    # cv2.imshow("facial feature", img_color)
    # cv2.waitKey(0)

    return np.array(keypoints)

def getKeypoints(img):

    corners = harris_corner_detector(img, 0.04, 5, 100)

    if corners is None:
        print("no keypoints were detected")
        return
    
    return corners


def getNextPoints(img1, img2, keypoints, movedOutFlag):

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    Iy, Ix = np.gradient(img1)
    next_keypoints = np.copy(keypoints).astype(float)

    for i, point in enumerate(keypoints):

        patch_x = cv2.getRectSubPix(Ix, (15, 15), (int(point[0]), int(point[1])))
        patch_y = cv2.getRectSubPix(Iy, (15, 15), (int(point[0]), int(point[1])))
        A = np.array([[np.sum(patch_x * patch_x), np.sum(patch_x * patch_y)], [np.sum(patch_x * patch_y), np.sum(patch_y * patch_y)]])

        # for cnt in range(25):
        patch_t = cv2.getRectSubPix(img2, (15,15), (int(next_keypoints[i,0]),int(next_keypoints[i,1]))) - cv2.getRectSubPix(img1, (15, 15), (int(point[0]), int(point[1])))
        B = -1* np.array([[np.sum(patch_x*patch_t)],[np.sum(patch_y*patch_t)]])
        disp = np.matmul(np.linalg.pinv(A), B)

        u = disp[0]
        v = disp[1]

        next_keypoints[i, 0] += float(u)
        next_keypoints[i, 1] += float(v)


            # if np.hypot(u, v) <= 0.01:
            #     break
        
        H, W = img1.shape

        # Setting the movedOutFlag to 1 if the new pixels are out of bounds      
        if next_keypoints[i,0] >= W or next_keypoints[i,0] < 0 or next_keypoints[i,1] >= H or next_keypoints[i,1] < 0:
            movedOutFlag[i] = 1
            print(f'next_keypoints[i] : {next_keypoints[i]}, u : {u}, v : {v}')

    return (next_keypoints, movedOutFlag)

def drawPaths(img, keypoints):
    print ("In function drawPaths")

    # Using cv2.circle to draw dots for all new points in xyt, by setting radius to 0
    for point in keypoints:
        img = cv2.circle(img,(round(point[0]),round(point[1])), radius=0, color=(0, 255, 255), thickness=1)

    print ("FINISHED: here are the paths of the tracked keypoints")
    cv2.imshow("image tracker",img)
    cv2.waitKey(0)

def trackPoints(keypoints, imageSequence):
    print ("In function trackPoints")
    print (f'length of imageSequence = {len(imageSequence)}')
    movedOutFlag = np.zeros(keypoints.shape[0])

    keypoints_list = []

    for i in range(0, len(imageSequence)-1): # predict for all images except first in sequence
        print (f't = {i}; predicting for t = {i+1}') 
        keypoints, movedOutFlag = getNextPoints(imageSequence[i], imageSequence[i+1], keypoints, movedOutFlag)

        print(f'keypoints size : {len(keypoints)}')

        for point in keypoints:
            keypoints_list.append(point)

        # for selected instants in time, display the latest image with highlighted keypoints 
        if ((i == 0) or (i == 10) or (i == 20) or (i == 30) or (i == 40) or (i == 49)):
            img_color = cv2.cvtColor(imageSequence[i+1], cv2.COLOR_GRAY2BGR)
            corners = np.intp(np.round(keypoints))              
            
            for c in range(0, corners.shape[0]):

                if movedOutFlag[c] == False:
                    x = corners[c][0]
                    y = corners[c][1]
                    cv2.circle(img_color, (x, y), DISPLAY_RADIUS, GREEN)


            dlib_feacture = dlib_face_feature_detector(imageSequence[i+1])

            for point in dlib_feacture:
                cv2.circle(img_color, (point[0], point[1]), DISPLAY_RADIUS, RED)

            cv2.imshow("image", img_color)
            cv2.waitKey(0)
        
    return keypoints_list


def main():
    
    face_path = "/home/piljae98/VScode/faceRecon/data/dng2png/myface"
    #house_path = "/home/piljae98/VScode/faceRecon/code/src/optical_flow/Kanade-Lucas-Tomasi-KLT-feature-tracker/hotel_images"

    images = readImages(NUMBER_OF_IMAGES, face_path)
    print (f'number of images that were read = {len(images)}')

    image_0th = images[0] # first image
    # keypoints = getKeypoints(image_0th) # harris cornor detection
    keypoints = dlib_face_feature_detector(image_0th) # dlib method

    if keypoints is None:
        print("points weren't detected")
        return
    print(f"{keypoints.shape[0]} points are detected")

    image_0th_color = cv2.cvtColor(image_0th, cv2.COLOR_GRAY2BGR)
    corners = np.intp(np.round(keypoints))

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image_0th_color, (x, y), DISPLAY_RADIUS, GREEN)

    
    keypoints_list = trackPoints(keypoints, images)

    drawPaths(image_0th_color, keypoints_list)

    return

if __name__ == "__main__":

    main()