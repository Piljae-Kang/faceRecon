import numpy as np
import cv2 

pw = 70
grid_size = np.array( [5,6])

image_size = pw * grid_size

print(image_size)

box = np.ones ([pw,pw])

print (box)

row_patches = []

for i in range( grid_size [1] ):

    row_patches.append( box )

    box = 1-box

row_patches = np.concatenate (row_patches, axis = 1 )

rows = []

for j in range( grid_size [0] ):

    rows.append (row_patches)
    row_patches = 1- row_patches

rows = np.concatenate ( rows, axis = 0 )
print (rows)

cv2.imshow ( "checkerboard", rows* 255)
cv2.imwrite("checkerboard_h{}_w{}_pw{}.png".format(grid_size[0],grid_size[1],pw), rows*255 )
cv2.waitKey(0)