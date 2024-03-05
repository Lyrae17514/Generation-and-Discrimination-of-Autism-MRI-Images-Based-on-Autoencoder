import numpy as np
import os
import matplotlib
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
# from PIL import Image
# import csv
# import matplotlib.pyplot as plt
# from keras.layers.advanced_activations import LeakyReLU
# from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# from keras.layers import Dense, Flatten, Reshape
# from keras.datasets import mnist
# from keras.models import Model
# from keras.layers import Dense, Input
# import random
import matplotlib.pyplot as plt
from scipy import ndimage
# 检查gpu
import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

preserving_ratio = 0.25

matplotlib.use('TkAgg')

path_G='D:/施雨欣/深度学习/帕金森/T1/'

example_filename = 'D:/施雨欣/深度学习/自闭症/MaxMun/control/control/mri/wm0051332mprage.nii'
img = nib.load(example_filename)

# 打印文件信息
# print(img)
# print(img.dataobj.shape)
width, height, queue = img.dataobj.shape
OrthoSlicer3D(img.dataobj).show()