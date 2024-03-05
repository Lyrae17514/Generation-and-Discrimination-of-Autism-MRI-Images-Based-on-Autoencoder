import numpy as np
import os
import matplotlib
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from PIL import Image
import csv
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LeakyReLU
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
from tensorflow.python.keras.layers import Dense, Flatten, Reshape

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input
import random
import matplotlib.pyplot as plt
from scipy import ndimage
import datetime
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# 检查gpu
import tensorflow as tf

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
config = tf.compat.v1.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
config.gpu_options.allow_growth = True

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

preserving_ratio = 0.25

matplotlib.use('TkAgg')

# path_G = 'D:/施雨欣/深度学习/帕金森/T1/mri/wmsub/'
# path_G = 'D:/施雨欣/深度学习/HCP/test_ac/mri/wm/'
# path_G = 'D:/施雨欣/深度学习/HCP/test_ac/mri/wm/'
path_G = 'D:/施雨欣/深度学习/自闭症/871-ants-nii/control/'

def read_T1_file_name(path):
    path_file_list = os.listdir(path)
    T1_file_name = []
    for file in path_file_list:
        T1_file_name.append(file)
    return T1_file_name


T1_file_name = read_T1_file_name(path_G)
print(T1_file_name)


def T1_file_to_array(path, name):
    file_path = path + name
    # file_path = path
    img = nib.load(file_path)
    img = img.get_fdata()
    img_3d_max = np.amax(img)
    img = img / img_3d_max * 255
    img = img / 127.5 - 1
    data = ndimage.zoom(img,
                        # (200 / img.shape[0], 200 / img.shape[1], 200 / img.shape[2]),
                        # (128 / img.shape[0], 128 / img.shape[1], 128 / img.shape[2]),
                        (64 / img.shape[0], 64 / img.shape[1], 64 / img.shape[2]),
                        order=0)
    data = np.asarray(data).reshape((262144))  # 64
    # data = np.asarray(data).reshape((2097152))#128
    # data = np.asarray(data).reshape((8000000))#200
    return data

# for i in range(0, 19): #
# for i in range(0, 33): #HCP
# for i in range(0, 403):  # ASD
for i in range(0, 468):  #
    bb = []
    cc = T1_file_name[i]
    # cc = T1_file_name[0]
    print(cc)
    aa = T1_file_to_array(path_G, cc)
    # bb.append(aa)
    # bb = np.asarray(bb, dtype=np.float32)
    # print(bb.shape)
    # yield bb,bb
    # gen_imgs = np.array(aa).reshape([2097152]).reshape((128, 128, 128))
    # gen_imgs = np.array(aa).reshape([8000000]).reshape((200, 200, 200))
    gen_imgs = np.array(aa).reshape([262144]).reshape((64, 64, 64))
    # OrthoSlicer3D(gen_imgs).show()
    print(gen_imgs.shape)
    new_image = nib.Nifti1Image(gen_imgs, np.eye(4))
    nib.save(new_image, r'D:/施雨欣/深度学习/自闭症/871-ants-nii/64/control_nii/' + cc )

# def svae_image(generator,num):
#     now = datetime.datetime.now()
#     strtime = now.strftime('%Y%m%d-'+str(num))
#     z = np.random.normal(0, 1, (1, 8000000))
#     gen_imgs = generator.predict(z)
#     gen_imgs = np.array(gen_imgs).reshape([8000000]).reshape((200,200,200))
#     OrthoSlicer3D(gen_imgs).show()
#     # print(gen_imgs.shape)
#     new_image = nib.Nifti1Image(gen_imgs, np.eye(4))
#     nib.save(new_image, r'D:/施雨欣/深度学习/Test/' + strtime + '.nii.gz')
