import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy import ndimage
from scipy.linalg import sqrtm
import torch
import os
import matplotlib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import pandas as pd
import math
from nibabel.viewers import OrthoSlicer3D
import tensorflow as tf
from imageio import imread
from scipy import linalg
import pathlib
import urllib
import warnings


# path_G='E:/eeg/T11/'
# path_G = 'D:/施雨欣/深度学习/帕金森/T1/mri/wmsub/'
path_G='D:/施雨欣/深度学习/HCP/test_ac/mri/wm/'#数据集路径

# path_GAN='./gan_nii/'
# path_GAN='D:/施雨欣/深度学习/Test/1/'
# path_GAN='D:/施雨欣/深度学习/Test/'
# path_GAN='D:/施雨欣/深度学习/Results/HCP/10000/'
# path_GAN = 'D:/施雨欣/深度学习/Test/GAN/'
# path_GAN='D:/施雨欣/深度学习/Test/GAN/'

path_GAN_list = os.listdir(path_G)
# print(len(path_G_list))
nii = '.nii.gz'
num = "20230813-10000" + nii
# num = "10000" + nii

# img_GAN = nib.load(path_GAN + num)
# img_GAN = img_GAN.get_fdata()
# print(img_GAN.shape)


# OrthoSlicer3D(img_GAN).show()

def read_T1_file_name(path):
    path_file_list = os.listdir(path)
    T1_file_name = []
    for file in path_file_list:
        T1_file_name.append(file)
        # 此处取了一个
    # return T1_file_name
    return T1_file_name


def T1_file_to_array(path, name):
    file_path = path + name
    img = nib.load(file_path)
    img = img.get_fdata()
    img_3d_max = np.amax(img)
    img = img / img_3d_max * 255
    img = img / 127.5 - 1

    data = ndimage.zoom(img,
                        (128 / img.shape[0], 128 / img.shape[1], 128 / img.shape[2]),
                        order=0)
    # data = np.asarray(data).reshape((200*200*200, 1))

    # OrthoSlicer3D(data).show()
    return data


def all_data(path, name):
    img_data = []
    for i in name:
        # print(i)
        img_data.append(T1_file_to_array(path, i))
    img_data = np.asarray(img_data, dtype=np.float32)
    return img_data


T1_file_name = read_T1_file_name(path_G)
all_img_data = all_data(path_G, T1_file_name)
real_data = all_img_data[16]

print(all_img_data.shape)
print(real_data.shape)


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    eps=1e-6
    All_FID = []
    sum_FID = 0
    print(len(act1))
    for i in range(len(act1)):
        mu1, sigma1 = act1[i].mean(axis=0), np.cov(act1[i], rowvar=False)
        mu2, sigma2 = act2[i].mean(axis=0), np.cov(act2[i], rowvar=False)

        # Add a small epsilon to the covariance matrices to avoid singular values
        sigma1 += np.eye(sigma1.shape[0]) * eps
        sigma2 += np.eye(sigma2.shape[0]) * eps
        # calculate sum squared difference between means
        diff = mu1 - mu2
        # calculate sqrt of product between cov
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        # calculate score
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        # All_FID.append(fid)
        sum_FID = sum_FID + fid
        print(fid, sum_FID, i)
    # return All_FID, sum_FID / (len(act1))
    # return fid
    return All_FID, sum_FID / (len(act1))


act1 = random(10 * 2048)
act1 = act1.reshape((10, 2048))
act2 = random(10 * 2048)
act2 = act2.reshape((10, 2048))

a = real_data
# a = gan_data.flatten()
# a = a.reshape(2,100*200*200)

# b = img_GAN
# b = all_img_data[1]
# b = img_GAN.flatten()
# b = b.reshape(2,100*200*200)

# all,fid = calculate_fid(b, b)
# print('FID (same): %.3f' % fid)

# all,fid = calculate_fid(b, a)
# all, fid = calculate_fid(b, a)
# print('FID : %.3f' % fid)
# print('FID_average : %.3f' % (fid / len(b)))
# print('FID (different): %.3f' % fid)
# 准备数据
act1 = real_data  # 真实样本集的特征向量
# folder_path = "D:/施雨欣/深度学习/Results/HCP/4000_128_4/"  # 存放生成样本集特征向量文件的文件夹路径
folder_paths = ["D:/施雨欣/深度学习/Results/HCP/4000_128_3/", "D:/施雨欣/深度学习/Results/HCP/4000_128_4/", "D:/施雨欣/深度学习/Results/HCP/4000_128_5/"]  # 存放生成样本集特征向量文件的文件夹路径列表

# 计算FID并绘制曲线
fid_scores_list = []
for folder_path  in folder_paths:
    # 获取文件夹中的特征向量文件列表
    file_list = os.listdir(folder_path)
    fid_scores = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        act2 = nib.load(file_path)  # 从文件中加载生成样本集的特征向量
        act2 = act2.get_fdata()
        _, fid = calculate_fid(act1, act2)
        fid_scores.append(fid)
    fid_scores_list.append(fid_scores)

# # 绘制FID曲线
# x = np.arange(len(fid_scores))
# plt.plot(x, fid_scores, marker='o')
# plt.xlabel('Generated Sample Index')
# plt.ylabel('FID Score')
# plt.title('FID Scores of Generated Samples')
# plt.xticks(x)
# plt.grid(True)
# plt.show()
# 定义每条曲线的样式
line_styles = ['-', '--', '-.']  # 曲线线型
colors = ['red', 'green', 'blue']  # 曲线颜色
markers = ['o', 's', '^']  # 曲线标记
# 绘制FID曲线
x = np.arange(len(fid_scores))
x_values = x * 100  # 将 x 轴数值扩大一百倍
labels = ["Parameter Variation 1", "Parameter Variation 2", "Parameter Variation 3"]  # 每个样本集的标签

plt.figure(figsize=(8, 5))
for i, fid_scores in enumerate(fid_scores_list):
    plt.plot(x, fid_scores, linestyle=line_styles[i], color=colors[i], marker=markers[i], label=labels[i])


plt.xlabel('Generated Sample Index')
plt.ylabel('FID Score')
plt.title('FID Scores of Generated Samples')
plt.xticks(np.arange(1, len(x_values)+1, 100))  # 设置刻度间隔为 100
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("FID_345_128.jpg")
