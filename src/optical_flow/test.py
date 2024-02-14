import cv2
import numpy as np

# 이미지 읽기
image = cv2.imread("image0.png", cv2.IMREAD_GRAYSCALE)

# 이미지 데이터 타입 변환 (float64에서 uint8로)
image = image.astype(np.float32)
Iy, Ix = np.gradient(image)

# 추출할 영역의 중심 좌표 지정
center_point = (100, 100)

# 추출할 영역 크기 지정
patch_size = (15, 15)

# getRectSubPix 함수를 사용하여 이미지에서 영역 추출
patch = cv2.getRectSubPix(image, patch_size, center_point)

# 결과 확인
cv2.imshow("Original Image", image)
cv2.imshow("Extracted Patch", patch)
cv2.waitKey(0)
cv2.destroyAllWindows()
