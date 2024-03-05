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
path_GAN='D:/施雨欣/深度学习/Results/HCP/10000/'
# path_GAN = 'D:/施雨欣/深度学习/Test/GAN/'
# path_GAN='D:/施雨欣/深度学习/Test/GAN/'

path_GAN_list = os.listdir(path_G)
# print(len(path_G_list))
nii = '.nii.gz'
num = "20230813-10000" + nii
# num = "10000" + nii

img_GAN = nib.load(path_GAN + num)
img_GAN = img_GAN.get_fdata()
print(img_GAN.shape)


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
                        (200 / img.shape[0], 200 / img.shape[1], 200 / img.shape[2]),
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
gan_data = all_img_data[0]

print(all_img_data.shape)
print(gan_data.shape)


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    eps=1e-6
    All_FID = []
    sum_FID = 0
    print(len(act1))
    for i in range(len(act1)):
        mu1, sigma1 = act1[i].mean(axis=0), cov(act1[i], rowvar=False)
        mu2, sigma2 = act2[i].mean(axis=0), cov(act2[i], rowvar=False)
    # for i in range(len(act1)):
    #     mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    #     mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        # mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
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

a = gan_data
# a = gan_data.flatten()
# a = a.reshape(2,100*200*200)

b = img_GAN
# b = all_img_data[1]
# b = img_GAN.flatten()
# b = b.reshape(2,100*200*200)

# all,fid = calculate_fid(b, b)
# print('FID (same): %.3f' % fid)

# all,fid = calculate_fid(b, a)
all, fid = calculate_fid(b, a)
# print('FID : %.3f' % fid)
# print('FID_average : %.3f' % (fid / len(b)))
print('FID (different): %.3f' % fid)
