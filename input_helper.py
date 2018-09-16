import numpy as np
from PIL import Image
import os


# Input the directory of mono-color images
def read_vision_mono(dir_path):
    print(dir_path)
    l = [dir_path + chr(92) + x for x in os.listdir(dir_path)]
    P = []
    for path in l:
        print("Processing ", path, "...")
        arr = np.array(Image.open(path))
        print("shape: ", arr.shape)
        P.append(np.matrix.flatten(arr))
    P = np.array(P)
    print("Result shape: ", P.shape)
    return P


# Input the directory of images, convert to 4 matrices of R, G, B
def read_vision(dir_path):
    print(dir_path)
    l = [dir_path + chr(92) + x for x in os.listdir(dir_path)]
    R = []
    G = []
    B = []
    for path in l:
        print("Processing ", path, "...")
        arr = np.array(Image.open(path))
        print("shape: ", arr.shape)
        R.append(np.matrix.flatten(arr[:,:,0]))
        G.append(np.matrix.flatten(arr[:,:,1]))
        B.append(np.matrix.flatten(arr[:,:,2]))
    R = np.array(R)
    G = np.array(G)
    B = np.array(B)
    print("Result shape of each channel: ", R.shape, ", 3 channels in total")
    return R, G, B


# Non-vision based input, assume in .mat format
# def read_non_vision(dir_path):


# Transform matrices into DeepSense input tensors in frequency domain, return (magnitude, phase)
# Output tensor = (dim, 2f (magnitude first then phase), tau)
def transform(M, stride=10):
    print("Truncating feature map in time domain...")
    n = M.shape[0] // stride
    M = M[0:n * stride, :]
    M_list = np.split(M, n, axis=0)
    print("Truncation done, ", n, " pieces in total.")
    print("Begin discrete fourier transform...")
    F = [np.fft.fft(m) for m in M_list]
    F = np.array(F)
    return np.concatenate((F.real, F.imag), axis=1).transpose((2, 1, 0))

# Calibrate labels and transformed tensors
# def calibrate(labels):



