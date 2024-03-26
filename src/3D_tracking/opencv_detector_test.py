import cv2
import numpy as np

def distance(p1, p2):
    
    return np.sqrt((p1.pt[0] - p2.pt[0])**2 + (p1.pt[1] - p2.pt[1])**2)

def detect_circle(keypoints, threshold):

    visited = np.zeros(len(keypoints), dtype = bool)
    circle_keypoints = []

    for i in range(len(keypoints)):

        if visited[i] == True:
            continue
        
        visited[i] = True
        near_points = []
        near_points.append([keypoints[i].pt[0], keypoints[i].pt[1]])

        for index, point in enumerate(keypoints):

            if visited[index] == True:
                continue

            if distance(keypoints[i], point) < threshold:
                near_points.append([point.pt[0], point.pt[1]])
                visited[index] = True

        near_points = np.array(near_points)
        print(f'len of near points : {len(near_points)}')
        if len(near_points) > 1:
            circle_keypoints.append([np.mean(near_points[:, 0]), np.mean(near_points[:, 1])])



    return circle_keypoints

def draw_circles(img, points):

    for point in points:
        print(point)
        cv2.circle(img, (int(point[0]), int(point[1])), 30, (0, 0, 255))

    return img


img = cv2.imread("image0.png", 0)

# # # fast algorithm
# fast = cv2.FastFeatureDetector_create(threshold=50)
# keypoints = fast.detect(img, None)

# # SIFT algorithm
# sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=100)  # SIFT 객체 생성
# keypoints, descriptors = sift.detectAndCompute(img, None)

# new_keypoints = detect_circle(keypoints, 50)
# print(len(keypoints))
# print(len(new_keypoints))

# img2 = cv2.drawKeypoints(img, keypoints, None, color=(0, 0, 255))
# new_img2 = draw_circles(img, new_keypoints)

# h, w, c = img2.shape
# img2 = cv2.resize(img2, (int(w/2), int(h/2)))
# new_img2 = cv2.resize(new_img2, (int(w/2), int(h/2)))

# cv2.imshow("result of fast detecton image", img2)
# cv2.imshow("result of fast detecton new image", new_img2)
# cv2.waitKey(0)


img = cv2.imread("image0.png")

blurred_image = cv2.GaussianBlur(img, (15, 15), 0)

gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area < 100:
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)


cv2.imshow("image", img)
cv2.imshow("binary image", binary)
cv2.waitKey(0)