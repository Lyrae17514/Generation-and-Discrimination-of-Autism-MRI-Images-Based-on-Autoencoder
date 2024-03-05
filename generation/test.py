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

# example_filename = 'D:/施雨欣/深度学习/帕金森/T1/mri/mwp1sub/mwp1sub-patient032077_T1w.nii'
# example_filename = 'D:/施雨欣/深度学习/帕金森/T1/mri/wmsub/wmsub-patient032077_T1w.nii'
# example_filename = 'D:/施雨欣/深度学习/帕金森/T1/sub-patient032077_T1w.nii.gz'
# example_filename = 'D:/施雨欣/深度学习/帕金森/T1/mri/wmsub/'
# example_filename = 'D:/施雨欣/深度学习/Test/1/20220721-12000.nii.gz'
# example_filename = 'D:/施雨欣/深度学习/HCP/MPRAGE_GradWarped_and_Defaced/HCP_mgh_1002_MR_MPRAGE_GradWarped_and_Defaced_Br_20140919135330625_S227329_I444329.nii'
# example_filename = 'D:/施雨欣/深度学习/Results/HCP/10000/20230813-10000.nii.gz'
# example_filename = 'D:/施雨欣/深度学习/Test/GAN/10000.nii.gz'
example_filename = 'D:/施雨欣/深度学习/自闭症/free/SBL_0051580brain.mgz'
# example_filename = 'D:/施雨欣/深度学习/Results/ASD/autoencoder_nii/con_128/20231024-4000.nii.gz'
# example_filename = 'D:/施雨欣/深度学习/Results/871/control/autoencoder_nii/con_128/20231026-4000.nii.gz'
img = nib.load(example_filename)
data = img.get_fdata()

# 打印文件信息
# print(img)
print(img.dataobj.shape)
width, height, queue = img.dataobj.shape
OrthoSlicer3D(img.dataobj).show()
#shape不一定只有三个参数，打印出来看一下
# width, height, queue = img.dataobj.shape

# nifti_file = spm_vol('MNI152_T1_1mm.nii');
# data = spm_read_vols(nifti_file);
#
# % 水平面
# Z_66 = squeeze(data(:,:,66));
# Z_66 = permute(Z_66,[2,1,3]);
#
# figure(1);
# imagesc(mapminmax(Z_66, 0, 255));
# set(gca, 'YDir', 'normal');
# colormap(gray);
# axis off;
#
# % 冠状面
# Y_111 = squeeze(data(:,111,:));
# Y_111 = permute(Y_111,[2,1]);
#
# figure(2);
# imagesc(mapminmax(Y_111, 0, 255));
# set(gca, 'YDir', 'normal');
# colormap(gray);
# axis off;
#
# % 矢状面
# X_99 = squeeze(data(99,:,:));
# X_99 = permute(X_99,[2,1]);
#
# figure(3);
# imagesc(mapminmax(X_99, 0, 255));
# set(gca, 'YDir', 'normal');
# colormap(gray);
# axis off;

# 计算看需要多少个位置来放切片图
# x = int((queue/5) ** 0.5) + 1
# num = 1
# # 按照10的步长，切片，显示2D图像
# for i in range(0, queue, 5):
#     img_arr = img.dataobj[:, :, i]
#     plt.subplot(x, x, num)
#     plt.imshow(img_arr, cmap='gray')
#     num += 1

# plt.show()


#
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# import tensorflow as tf
# import time
#
# begin = time.time()
# n_classes = 10
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(
#     32,(5,5),activation = tf.nn.relu, input_shape = (28,28,1)),
#     tf.keras.layers.MaxPool2D((2,2),(2,2)),
#     tf.keras.layers.Conv2D(64,(3,3),activation = tf.nn.relu),
#     tf.keras.layers.MaxPool2D((2,2),(2,2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(1024,activation=tf.nn.relu),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(n_classes)
# ])
# model.summary()
# mnist = tf.keras.datasets.mnist
# (train_x, train_y),(test_x,test_y) = mnist.load_data()
# train_x = train_x/255. *2 -1
# test_x = test_x/255. *2 -1
# train_x = tf.expand_dims(train_x, -1).numpy()
# test_x = tf.expand_dims(test_x, -1).numpy()
#
# model.compile(
# optimizer=tf.keras.optimizers.Adam(1e-5),
# loss='sparse_categorical_crossentropy',
# metrics=['accuracy'])
#
# model.fit(train_x,train_y,epochs=10,batch_size = 100)
# model.evaluate(test_x, test_y)
# end = time.time()
# print('It cost',end-begin,'s')

# import numpy as np
# import h5py
# import cv2
#
# filename = 'D:/施雨欣/深度学习/Test/MRI_slices_128_isotropic/data_T1_3d_size_128_128_128_res_1.4_1.4_1.4_from_0_to_20.hdf5'
#
# f = h5py.File(filename, 'r')
# # data = f['images'][:]
# # labels = f['labels'][:]
# # Print the keys of groups and datasets under '/'.
# print(f.filename, ":")
# print([key for key in f.keys()], "\n")
# #===================================================
# # Read dataset 'dset' under '/'.
# d = f["images"]
#
# # Print the data of 'dset'.
# print(d.name, ":")
# print(d[:])
#
# # Print the attributes of dataset 'dset'.
# for key in d.attrs.keys():
#     print(key, ":", d.attrs[key])
#
# print()
# num = data.shape[0]
# img1 = data[0]

# for i in range(num):
#     img = data[i]
#     img1 = img[:, :, ::-1]
#     cv2.imwrite(str(i) + '.jprg', img1)
# for i in range(num):
#     label = labels[i] * 255
#
#     cv2.imwrite(str(i) + '.jprg', label)
