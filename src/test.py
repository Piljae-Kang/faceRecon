import cv2
import numpy as np
from process_raw import DngFile

dng_path = "image1.DNG"

dng = DngFile.read(dng_path)
raw = dng.raw  # np.uint16
raw_8bit = np.uint8(raw >> (dng.bit-8))

print(dng.bit)
print(raw.shape)
print(raw_8bit.shape)

H, W = raw.shape

flag = 0
print("------------------------------------")
for i in range (0, H):
    
    if flag == 5:
        break

    for j in range(0, W):
    

        if raw_8bit[i][j] > 10:
            flag += 1
            print(f'{flag}. 16bit raw : ', np.binary_repr(raw[i][j],width=16))
            print(f'{flag}. 8bit raw :  ', np.binary_repr(raw_8bit[i][j],width=8))
            print("------------------------------------")
            break

cv2.imwrite("raw_8bit.png", raw_8bit)


# rgb1 = dng.postprocess()  # demosaicing by rawpy
# cv2.imwrite("rgb1.jpg", rgb1[:, :, ::-1])
# rgb2 = dng.demosaicing(poww=0.3)  # demosaicing with gamma correction
# cv2.imwrite("rgb2.jpg", rgb2[:, :, ::-1])
# DngFile.save(dng_path + "-save.dng", dng.raw, bit=dng.bit, pattern=dng.pattern)