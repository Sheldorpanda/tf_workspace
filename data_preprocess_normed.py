import numpy as np
import pandas as pd
import os
from PIL import Image

def norm_2d(x):
    new_x = []
    i = 0
    for col in x.T:
        print("Normalizing col ", str(i), "...")
        mean = np.mean(col)
        std = np.std(col)
        new_col = []
        for item in col:
            new_item = item - mean
            if std != 0:
                new_item = new_item / std
            new_col.append(new_item)
        new_x.append(new_col)
        i += 1
    new_x = np.array(new_x).T
    return new_x

# def norm_angle(x):
#     for theta in x:
#         if theta >= 180:
#             theta = theta - 360

# (pixel_0_r, pixel_0_g, pixel_0_b, pixel_1_r, ...)
def convert_image(img_file):
    # print("Converting ", img_file, " ...")
    img = Image.open(img_file)
    img.thumbnail((160, 120), Image.ANTIALIAS)
    img_pixels = img.convert("RGB")
    img_array = np.array(img_pixels.getdata()).flatten()
    return img_array

# shape = (num_imgs, pixels*3)
def convert_all_images(dir):
    imgs = os.listdir(dir)
    imgs.sort()
    result = []
    for file_name in imgs:
        if file_name.endswith('.png'):
            print(file_name)
            img_file = dir + file_name
            a = convert_image(img_file)
            result.append(a)
    return np.array(result)

# shape = (num_intervals, step*2, pixels*3) = (tau, 2f, n)
def transform(array, step=20):
    print("FFT... ")
    n = array.shape[0] // step
    result = []
    for i in range(n):
        sub_array = array[i : i + step]
        f = np.fft.fft(sub_array, axis=0, norm="ortho")
        r = np.real(f)
        i = np.imag(f)
        result.append(np.concatenate((r, i), axis=0))
    return np.array(result)