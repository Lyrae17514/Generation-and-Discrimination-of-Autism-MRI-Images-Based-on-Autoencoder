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
import SimpleITK as sitk
# 检查gpu
import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

preserving_ratio = 0.25


def convert_mgz_to_nii(mgz_file, nii_file):
    # 读取.mgz文件
    mgz_img = nib.load(mgz_file)

    # 获取数据和空间信息
    data = mgz_img.get_fdata()
    affine = mgz_img.affine

    # 创建.nii文件
    nii_img = nib.Nifti1Image(data, affine)

    # 保存为.nii文件
    nib.save(nii_img, nii_file)


# 遍历文件夹中的所有.mgz文件并转换为.nii文件
folder_path = 'D:/施雨欣/深度学习/自闭症/871-control-T1/mgz/'
output_folder = 'D:/施雨欣/深度学习/自闭症/871-control-T1/nii/'

for filename in os.listdir(folder_path):
    if filename.endswith('.mgz'):
        mgz_file = os.path.join(folder_path, filename)
        nii_file = os.path.join(output_folder, filename.replace('.mgz', '.nii'))
        convert_mgz_to_nii(mgz_file, nii_file)


