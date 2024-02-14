import cv2
import numpy as np
from process_raw import DngFile
import matplotlib.pyplot as plt
import statistics
from sklearn.linear_model import LinearRegression

color_list = [[], [], []]
global_pos = (0, 0)
click_pos_list = []

def color_demosaicing(x, y, raw):
     
    color_ratio = np.array([0, 0, 0], dtype=np.uint64) # (R G B)
    
    if x % 2 == 0 and y % 2 != 0: # red image sensor
       
       color_ratio[0] = raw[x][y]

       color_ratio[1] += raw[x-1][y]
       color_ratio[1] += raw[x+1][y]
       color_ratio[1] += raw[x][y-1]
       color_ratio[1] += raw[x][y+1]
       color_ratio[1] /= 4

       color_ratio[2] += raw[x-1][y-1]
       color_ratio[2] += raw[x+1][y+1]
       color_ratio[2] += raw[x-1][y+1]
       color_ratio[2] += raw[x+1][y-1]
       color_ratio[2] /= 4
       
    if (x % 2 == 0 and y % 2 == 0): # green image sensor
      
        color_ratio[0] += raw[x][y-1]
        color_ratio[0] += raw[x][y+1]
        color_ratio[0] /= 2

        color_ratio[1] += raw[x][y]

        color_ratio[2] += raw[x-1][y]
        color_ratio[2] += raw[x+1][y]
        color_ratio[2] /= 2
    
    if (x % 2 !=0 and y % 2 !=0): # green image sensor
    
      
      color_ratio[0] += raw[x-1][y]
      color_ratio[0] += raw[x+1][y]
      color_ratio[0] /= 2

      color_ratio[1] = raw[x][y]
      
      color_ratio[2] += raw[x][y-1]
      color_ratio[2] += raw[x][y+1]
      color_ratio[2] /= 2

    if x % 2 != 0 and y % 2 == 0: # blue image sensor


        color_ratio[0] += raw[x-1][y-1]
        color_ratio[0] += raw[x+1][y+1]
        color_ratio[0] += raw[x-1][y+1] 
        color_ratio[0] += raw[x+1][y-1]
        color_ratio[0] /= 4

        color_ratio[1] += raw[x-1][y]
        color_ratio[1] += raw[x+1][y]
        color_ratio[1] += raw[x][y-1]
        color_ratio[1] += raw[x][y+1]
        color_ratio[1] /= 4

        color_ratio[2] = raw[x][y]

    return np.array(color_ratio)

def demosaicing(raw):
     
    H, W = raw.shape
    #raw_RGB = np.zeros((H, W, 3), dtype=np.uint64) # (R G B)
    raw_R = np.zeros((H, W), dtype=np.float64)
    raw_G = np.zeros((H, W), dtype=np.float64)
    raw_B = np.zeros((H, W), dtype=np.float64)

    border_top = np.ix_(np.array([0]), np.arange(1, W-1))
    border_bottom = np.ix_(np.array([H-1]), np.arange(1, W-1))
    border_left = np.ix_(np.arange(1, H-1), np.array([0]))
    border_right = np.ix_(np.arange(1, H-1), np.array([W-1]))

    # red sensor part
    red_idx = np.ix_(np.arange(2, H-1, 2), np.arange(1, W-1, 2))

    raw_R[red_idx] += raw[red_idx]
    raw_G[red_idx] += raw[(red_idx[0] - 1, red_idx[1])]
    raw_G[red_idx] += raw[(red_idx[0] + 1, red_idx[1])]
    raw_G[red_idx] += raw[(red_idx[0], red_idx[1] - 1)]
    raw_G[red_idx] += raw[(red_idx[0], red_idx[1] + 1)]
    raw_G[red_idx] /= 4

    raw_B[red_idx] += raw[(red_idx[0] - 1, red_idx[1] - 1)]
    raw_B[red_idx] += raw[(red_idx[0] + 1, red_idx[1] + 1)]
    raw_B[red_idx] += raw[(red_idx[0] + 1, red_idx[1] - 1)]
    raw_B[red_idx] += raw[(red_idx[0] - 1, red_idx[1] + 1)]
    raw_B[red_idx] /= 4

    # green sensor odd part

    green_odd_idx = np.ix_(np.arange(1, H-1, 2), np.arange(1, W-1, 2))
    
    raw_R[green_odd_idx] += raw[(green_odd_idx[0] - 1 , green_odd_idx[1])]
    raw_R[green_odd_idx] += raw[(green_odd_idx[0] + 1, green_odd_idx[1])]
    raw_R[green_odd_idx] /= 2

    raw_G[green_odd_idx] += raw[green_odd_idx]

    raw_B[green_odd_idx] += raw[(green_odd_idx[0], green_odd_idx[1] - 1)]
    raw_B[green_odd_idx] += raw[(green_odd_idx[0], green_odd_idx[1] + 1)]
    raw_B[green_odd_idx] /= 2

    # green sensor even part
    green_even_idx = np.ix_(np.arange(2, H-1, 2), np.arange(2, W-1, 2))
    
    raw_R[green_even_idx] += raw[(green_even_idx[0], green_even_idx[1] -1)]
    raw_R[green_even_idx] += raw[(green_even_idx[0], green_even_idx[1] + 1)]
    raw_R[green_even_idx] /= 2

    raw_G[green_even_idx] += raw[green_even_idx]

    raw_B[green_even_idx] += raw[(green_even_idx[0] - 1, green_even_idx[1])]
    raw_B[green_even_idx] += raw[(green_even_idx[0] + 1, green_even_idx[1])]
    raw_B[green_even_idx] /= 2

    # blue sensor part
    blue_idx = np.ix_(np.arange(1, H-1, 2), np.arange(2, W-1, 2))
    
    raw_R[blue_idx] += raw[(blue_idx[0] - 1, blue_idx[1] - 1)]
    raw_R[blue_idx] += raw[(blue_idx[0] + 1, blue_idx[1] + 1)]
    raw_R[blue_idx] += raw[(blue_idx[0] - 1, blue_idx[1] + 1)]
    raw_R[blue_idx] += raw[(blue_idx[0] + 1, blue_idx[1] - 1)]
    raw_R[blue_idx] /= 4

    raw_G[blue_idx] += raw[(blue_idx[0] - 1, blue_idx[1])]
    raw_G[blue_idx] += raw[(blue_idx[0] + 1, blue_idx[1])]
    raw_G[blue_idx] += raw[(blue_idx[0], blue_idx[1] + 1)]
    raw_G[blue_idx] += raw[(blue_idx[0], blue_idx[1] - 1)]
    raw_G[blue_idx] /= 4

    raw_B[blue_idx] += raw[blue_idx]

    # border top part
    raw_R[border_top] += raw_R[(border_top[0] + 1, border_top[1])]
    raw_G[border_top] += raw_G[(border_top[0] + 1, border_top[1])]
    raw_B[border_top] += raw_B[(border_top[0] + 1, border_top[1])]

    # border bottom part
    raw_R[border_bottom] += raw_R[(border_bottom[0] - 1, border_bottom[1])]
    raw_G[border_bottom] += raw_G[(border_bottom[0] - 1, border_bottom[1])]
    raw_B[border_bottom] += raw_B[(border_bottom[0] - 1, border_bottom[1])]

    # border left part
    raw_R[border_left] += raw_R[(border_left[0], border_left[1] + 1)]
    raw_G[border_left] += raw_G[(border_left[0], border_left[1] + 1)]
    raw_B[border_left] += raw_B[(border_left[0], border_left[1] + 1)]

    # border right part
    raw_R[border_right] += raw_R[(border_right[0], border_right[1] - 1)]
    raw_G[border_right] += raw_G[(border_right[0], border_right[1] - 1)]
    raw_B[border_right] += raw_B[(border_right[0], border_right[1] - 1)]

    # cornor part
    raw_R[0, 0] += raw_R[1, 1]
    raw_G[0, 0] += raw_G[1, 1]
    raw_B[0, 0] += raw_B[1, 1]

    raw_R[0, W-1] += raw_R[1, W-2]
    raw_G[0, W-1] += raw_G[1, W-2]
    raw_B[0, W-1] += raw_B[1, W-2]

    raw_R[H-1, 0] += raw_R[H-2, 1]
    raw_G[H-1, 0] += raw_G[H-2, 1]
    raw_B[H-1, 0] += raw_B[H-2, 1]

    raw_R[H-1, W-1] += raw_R[H-2, W-2]
    raw_G[H-1, W-1] += raw_G[H-2, W-2]
    raw_B[H-1, W-1] += raw_B[H-2, W-2]

    raw_R =raw_R.reshape(H, W, 1)
    raw_G =raw_G.reshape(H, W, 1)
    raw_B =raw_B.reshape(H, W, 1)

    raw_RGB = np.concatenate([raw_R, raw_G, raw_B], axis=-1)

    return raw_RGB

def find_median(patch_loc, raw_patch):
   
    median_color = []
    for x, y in patch_loc:

      c_list = [[],[],[]]

      for i in range(x - 20, x + 21):
         for j in range(y - 20, y + 21):
               color = raw_patch[x, y]
               c_list[0].append(color[0])
               c_list[1].append(color[1])
               c_list[2].append(color[2])
      
      median_red = statistics.median(c_list[0])
      median_green = statistics.median(c_list[1])
      median_blue = statistics.median(c_list[2])
      
      median_color.append([median_red, median_green, median_blue])


    return np.array(median_color)

def gamma_correction(input_values, gamma):
    return np.power(input_values, gamma)

def rawRGB_to_sRGB(raw_RGB, model_red, model_green, model_blue):

    H, W, _ = raw_RGB.shape

    sRGB_R = model_red.predict(raw_RGB[:, :, 0:1].reshape(-1, 1))
    index = sRGB_R < 0
    sRGB_R[index] = 0

    sRGB_R = gamma_correction(sRGB_R, 1/2.4)
    sRGB_R = sRGB_R.astype(int)
    sRGB_R = sRGB_R.reshape(H, W, 1)

    sRGB_G = model_green.predict(raw_RGB[:, :, 1:2].reshape(-1, 1))
    index = sRGB_G < 0
    sRGB_G[index] = 0

    sRGB_G = gamma_correction(sRGB_G, 1/2.4)
    sRGB_G = sRGB_G.astype(int)
    sRGB_G = sRGB_G.reshape(H, W, 1)

    sRGB_B = model_blue.predict(raw_RGB[:, :, 2:3].reshape(-1, 1))
    index = sRGB_B < 0
    sRGB_B[index] = 0

    sRGB_B = gamma_correction(sRGB_B, 1/2.4)
    sRGB_B = sRGB_B.astype(int)
    sRGB_B = sRGB_B.reshape(H, W, 1)


    sRGB_color = np.concatenate([sRGB_R, sRGB_G, sRGB_B], axis=-1)

    sRGB_color = sRGB_color/np.max(sRGB_color) * 255

    return sRGB_color

if __name__ == "__main__":

    img = cv2.imread("/home/piljae98/VScode/faceRecon/data/color_calibration/color_chart.jpg")

    patch_dng_image_path = "/home/piljae98/VScode/faceRecon/data/color_calibration/color_chart.DNG"

    patch_dng = DngFile.read(patch_dng_image_path)
    raw_patch_mosaic = patch_dng.raw  # np.uint16
    raw_patch = demosaicing(raw_patch_mosaic)

    patch_loc = [(228, 866), (352, 873), (477, 870), (717, 872), (834, 872)]
    # get sensor value at each patch

    patch_sensor_value = find_median(patch_loc, raw_patch)

    # Linearity test


    # Image Color Mapping
    # 0. sensor image = s


    dng_folder ="/home/piljae98/VScode/faceRecon/data/raw_data/colmap/dng/video1"

    for i in range(700, 771):
        
        dng_path = dng_folder + f'/test_000{i}.DNG'

        dng = DngFile.read(dng_path)
        raw = dng.raw  # np.uint16
        sensor_image = demosaicing(raw)

        # 1. white balancing ( divide each pixel color by white RGB ) : s = s / s_white (divide values at each channel)

        sensor_image /= patch_sensor_value[0]

        # 2. find the mamximum scalar value M among all RGB values.

        M = np.max(sensor_image)

        # 3. normalize image: s = s / M 

        sensor_image /= M

        # 4. gamma encoding s = s ^ 1/gamma

        sensor_image = np.power(sensor_image, 1/2.4)

        # 5. show gamma-encoded image

        image = sensor_image * 255

        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        output_path = "/home/piljae98/VScode/faceRecon/data/dng2png/myface/"
        cv2.imwrite(output_path + f'image{i-700}.png', image)