import cv2
import dlib
import numpy as np

def marker_detector(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    face_area_img = np.zeros_like(gray)

    result_img = np.full_like(img, 255)

    for face in faces:
        
        landmarks = predictor(gray, face)

        points = []
        right_eye_points = []
        left_eye_points = []
        mouth_points = []

        for n in range(0, 68):
            point = (landmarks.part(n).x, landmarks.part(n).y)
            points.append(point)

            if n >= 36 and n < 42:
                right_eye_points.append(point)

            if n >= 42 and n < 48:
                left_eye_points.append(point)            

            if n >= 48 and n < 68:
                mouth_points.append(point)
        
        points = np.array(points, np.int32)
        convexhull = cv2.convexHull(points)

        # print(convexhull)

        cv2.fillConvexPoly(face_area_img, convexhull, (255))

    indeices = np.where(face_area_img == 255)
    result_img[indeices] = img[indeices]

    right_eye_points = np.array(right_eye_points, np.int32)
    right_eye_points_convexhull = cv2.convexHull(right_eye_points)
    cv2.fillConvexPoly(result_img, right_eye_points_convexhull, (255, 255, 255))

    left_eye_points = np.array(left_eye_points, np.int32)
    left_eye_points_convexhull = cv2.convexHull(left_eye_points)
    cv2.fillConvexPoly(result_img, left_eye_points_convexhull, (255, 255, 255))

    mouth_points = np.array(mouth_points, np.int32)
    mouth_points_convexhull = cv2.convexHull(mouth_points)
    cv2.fillConvexPoly(result_img, mouth_points_convexhull, (255, 255, 255))


    # cv2.imshow("face_area_img.png", result_img)
    # cv2.waitKey(0)


    gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("gray", gray)
    # cv2.waitKey(0)

    gray_array = np.array(gray)

    indeices = np.where(gray_array != 255)

    # Sobel 연산자를 사용하여 그라디언트 계산
    grad_x = cv2.Sobel(gray_array, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_array, cv2.CV_64F, 0, 1, ksize=3)

    # 그라디언트 크기 계산
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    _, binary_image = cv2.threshold(grad_magnitude, 100, 255, cv2.THRESH_BINARY)

    cv2.imwrite("/detection_result/gradient_binary_image.png", binary_image)

    # gamma = 0.5

    # corrected = np.power(gray_array[indeices] / 255.0, gamma)
    # gray_array[indeices] = np.uint8(np.round(corrected * 255))

    # cv2.imshow("gamma_corrected", gray_array)
    # cv2.waitKey(0)

    # _, binary = cv2.threshold(gray_array, 150, 255, cv2.THRESH_BINARY)

    # cv2.imshow("binary", binary)
    # cv2.waitKey(0)

    # kernel = np.ones((3,3),np.uint8)
    # opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("opening", opening)
    # cv2.waitKey(0)

    contours, _ = cv2.findContours(np.uint8(binary_image), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_point = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 20 and area < 250:
            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 3)
            points = np.array(cnt).reshape(-1, 2)
            mean_point = (np.mean(points[:, 0]), np.mean(points[:, 1]))
            distance = (points[:,0] - mean_point[0]) ** 2 + (points[:,1] - mean_point[1]) ** 2
            variances = np.var(distance)

            if variances <= 10000:
                detected_point.append(mean_point)
                cv2.circle(img, (round(mean_point[0]), round(mean_point[1])), 1, (0, 0, 255), 2)



    # cv2.imshow("/detection_result/result_img.png", result_img)
    # cv2.waitKey(0)

    cv2.imwrite("/detection_result/img.png", img)

    return detected_point

# detected_point = marker_detector("image0_camera2.png")
# print(detected_point)