import rawpy
import numpy as np
import argparse
import glob
import cv2
from process_raw import DngFile

def read_dng_file(file_path):
    # DNG 파일 열기
    raw = rawpy.imread(file_path)

    # RGB 이미지로 디모자이싱
    rgb = raw.postprocess()

    return rgb

def thresholding(rgb_image, threshold):

    rgb_avg = np.mean(rgb_image, axis = 2, keepdims=True)
    
    reshaped_image = rgb_image.reshape([-1,3])
    thresholded_reshaped_image = np.where(reshaped_image < threshold, 255, reshaped_image)

    rgb_min = np.min ( thresholded_reshaped_image, axis = 0)
    rgb_max = np.max ( rgb_image.reshape([-1,3]), axis = 0)

    index = rgb_avg < threshold

    red_image = rgb_image[:,:,0:1]
    green_image = rgb_image[:,:,1:2]
    blue_image = rgb_image[:,:,2:3]

    red_image [index] = 0
    green_image [index]= 0
    blue_image[index] = 0
    rgb_image = np.concatenate( [red_image, green_image, blue_image], axis = 2)

    return rgb_image , rgb_max, rgb_min

def stretching_image(image, max, min):

    red_image = image[:,:,0:1]
    green_image = image[:,:,1:2]
    blue_image = image[:,:,2:3]

    index_r = red_image > min[0]
    index_g = green_image > min[1]
    index_b = blue_image > min[2]

    red_image = red_image.astype(np.float64)
    red_image[index_r] = (red_image[index_r] - min[0]) * 255 / (max[0] - min[0])
    red_image = red_image.astype(np.uint8)

    green_image = green_image.astype(np.float64)
    green_image[index_g] = (green_image[index_g] - min[1]) * 255 / (max[1] - min[1])
    green_image = green_image.astype(np.uint8)

    blue_image = blue_image.astype(np.float64)
    blue_image[index_b] = (blue_image[index_b] - min[2]) * 255 / (max[2] - min[2])
    blue_image = blue_image.astype(np.uint8)
    
    image = np.concatenate([red_image, green_image, blue_image], axis = 2)
            
    return image

# argument parsing part
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input file path')
parser.add_argument('-o', '--output', help='output file path')
args = parser.parse_args()

input_path = args.input + "/*.DNG"
images = glob.glob(input_path)

for i, frame in enumerate(images):

    image_array = DngFile.read(frame)
    print(image_array.sh)
    # 결과 확인
    # print(image_array.shape)  # 이미지 배열의 형태 출력

    # gray_image_array, max, min = rgb_to_gray(image_array, 30)
    # image = cv2.cvtColor(gray_image_array, cv2.COLOR_GRAY2BGR)  # OpenCV는 BGR 순서를 사용하므로 흑백을 BGR로 변환
    # cv2.imshow("thresholded Image", image)
    # cv2.waitKey(0)

    # stretched_image_array = stretching_image(gray_image_array, max, min)
    # image = cv2.cvtColor(stretched_image_array, cv2.COLOR_GRAY2BGR)  # OpenCV는 BGR 순서를 사용하므로 흑백을 BGR로 변환
    # cv2.imshow("stretched Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    thresholded_image_array, max, min = thresholding(image_array, 70)
    image = cv2.cvtColor(thresholded_image_array, cv2.COLOR_RGB2BGR)
    cv2.imshow(f"thresholded Image{i}", image)
    cv2.imwrite(f"thresholded_Image{i}.png", image)
    cv2.waitKey(0)

    stretched_image_array = stretching_image(thresholded_image_array, max, min)
    image = cv2.cvtColor(stretched_image_array, cv2.COLOR_RGB2BGR)
    cv2.imshow(f"stretched Image{i}", image)
    cv2.imwrite(f"stretched_Image{i}.png", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()