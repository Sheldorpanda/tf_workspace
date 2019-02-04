import numpy as np
import pandas as pd
import os
from PIL import Image

# (pixel_0_r, pixel_0_g, pixel_0_b, pixel_1_r, ...)
def convert_image(img_file):
    img = Image.open(img_file)
    img_pixels = img.convert("RGB")
    img_array = np.array(img_pixels.getdata()).flatten()
    return img_array

# shape = (num_imgs, pixels*3)
def convert_all_images(dir):
    imgs = os.listdir(dir)
    result = []
    for file_name in imgs:
        if not file_name.endswith('.png'):
            continue
        print("Converting " + file_name + " ...")
        img_file = dir + file_name
        a = convert_image(img_file)
        result.append(a)
    return np.array(result)

# shape = (num_intervals, step*2, pixels*3) = (tau, 2f, n)
def transform(array, step=20):
    n = array.shape[0] // step
    result = []
    for i in range(n):
        sub_array = array[i : i + step]
        f = np.fft.fft(sub_array, axis=0)
        r = np.real(f)
        i = np.imag(f)
        result.append(np.concatenate((r, i), axis=0))
    return np.array(result)

dir =  "sensor_data/LCA_Real_Life_Highway__511 sec/LCA_Real-Life_Highway_1_Ford_Fiesta_1/"
# img_dir = "CameraSensor_1/"
# r = convert_all_images(dir + img_dir)
# f = transform(r)

# May take very long time on PC, use larger GPUs
# i = 0
# for arr in f:
#     pd.DataFrame(arr).to_csv(dir + "img_processed_" + str(i) + ".csv")
#     i += 1