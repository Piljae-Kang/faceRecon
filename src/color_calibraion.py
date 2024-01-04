import cv2
import numpy as np
from process_raw import DngFile

img = cv2.imread("/home/piljae98/VScode/faceRecon/data/color_calibration/color_chart.jpg")

global_pos = None

def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')

      global global_pos
      global_pos = (x, y)
    
      # put coordinates as text on the image
      cv2.putText(img, f'({x},{y})',(x,y),
      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
      # draw point on the image
      cv2.circle(img, (x,y), 3, (0,255,255), -1)

cv2.namedWindow('Point Coordinates')
cv2.setMouseCallback('Point Coordinates', click_event)

while True:
   cv2.imshow('Point Coordinates',img)
   k = cv2.waitKey(1) & 0xFF
   if k == 27:
      break

dng_path = "/home/piljae98/VScode/faceRecon/data/color_calibration/color_chart.DNG"

dng = DngFile.read(dng_path)
raw = dng.raw  # np.uint16
raw_8bit = np.uint8(raw >> (dng.bit-8))

y = global_pos[0]
x = global_pos[1]

print(x)
print(y)

for i in range (x-1, x+2):
    for j in range(y-1, y+2):

        print(f'16bit raw [{i}][{j}] -> binary form : ', np.binary_repr(raw[i][j],width=16), ' integer form : ', raw[i][j])
        print(f'8bit raw [{i}][{j}] -> binary form : ', np.binary_repr(raw_8bit[i][j],width=8), ' integer form : ', raw_8bit[i][j])
        print("------------------------------------")


def color_demosaicing(x, y, raw):
     
    color_ratio = np.array([0, 0, 0]) # (R G B)
    
    
    if x % 2 == 0 and y % 2 != 0: # red image sensor
       
       print("red")
       
       color_ratio[0] = raw[x][y]
       color_ratio[1] = (raw[x-1][y] + raw[x+1][y] + raw[x][y-1] + raw[x][y+1])/4
       color_ratio[2] = (raw[x-1][y-1] + raw[x+1][y+1] + raw[x-1][y+1] + raw[x+1][y-1])/4
       
    
    
    if (x % 2 == 0 and y % 2 == 0): # green image sensor

        print("green")
      
        color_ratio[0] = (raw[x-1][y] + raw[x+1][y])/2
        color_ratio[1] = raw[x][y]
        color_ratio[2] = (raw[x-1][y] + raw[x+1][y])/2
    
    
    
    if (x % 2 !=0 and y % 2 !=0): # green image sensor
      
      print("green")
      
      color_ratio[0] = (raw[x-1][y] + raw[x+1][y])/2
      color_ratio[1] = raw[x][y]
      color_ratio[2] = (raw[x][y-1] + raw[x][y+1])/2

    if x % 2 != 0 and y % 2 == 0: # blue image sensor
        
        print("blue")

        color_ratio[0] = (raw[x-1][y-1] + raw[x+1][y+1] + raw[x-1][y+1] + raw[x+1][y-1])/4
        color_ratio[1] = (raw[x-1][y] + raw[x+1][y] + raw[x][y-1] + raw[x][y+1])/4
        color_ratio[2] = raw[x][y]
        
        
    return tuple(int(round(x)) for x in color_ratio/np.linalg.norm(color_ratio)*255)


color = color_demosaicing(x, y, raw)

print('color : ', color)

# weight = np.array([52/color[0], 52/color[1], 52/color[2]])

# modified_color = color * weight

# print('modified color : ', modified_color.astype(int))