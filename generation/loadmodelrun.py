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
from tensorflow.keras.optimizers import SGD,Adam
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
from tensorflow.python.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint

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

model = tf.keras.models.load_model('model/auto_128_3')
model.summary()
path_G='D:/施雨欣/深度学习/自闭症/free/SBL_0051580brain.mgz'

# def read_T1_file_name(path) :
#     path_file_list = os.listdir(path)
#     T1_file_name = []
#     for file in path_file_list:
#         T1_file_name.append(file)
#     return T1_file_name
#
#
# T1_file_name = read_T1_file_name(path_G)
# print(len(T1_file_name))


def T1_file_to_array(path):

    # file_path = path + name
    file_path = path
    img = nib.load(file_path)
    img = img.get_fdata()
    img_3d_max = np.amax(img)
    img = img / img_3d_max * 255
    img = img / 127.5 - 1

    data = ndimage.zoom(img,
                        (128 / img.shape[0], 128 / img.shape[1], 128 / img.shape[2]),
                        order=0)
    # data = np.asarray(data).reshape((8000000)) # 200
    data = np.asarray(data).reshape((2097152))  # 128
    # print(data.shape)

    return data


# def randchoose():
#     while 1:
#         bb = []
#         # cc = T1_file_name[random.randint(0, 19)]
#         cc = T1_file_name[random.randint(0, 1)]
#         # cc = T1_file_name[random.randint(0, 467)]
#         # cc = T1_file_name[0]
#         print(cc)
#         aa = T1_file_to_array(path_G, cc)
#         bb.append(aa)
#         bb = np.asarray(bb, dtype=np.float32)
#         # print(bb.shape)
#         yield bb,bb

nii_img = T1_file_to_array(path_G)
print(nii_img.shape)
nii_img = nii_img.reshape((1, 2097152))
reconstructed_nii = model.predict(nii_img)
reconstructed_nii = np.array(reconstructed_nii).reshape([2097152]).reshape((128, 128, 128))
OrthoSlicer3D(reconstructed_nii).show()

def svae_image(generator,num):
    now = datetime.datetime.now()
    strtime = now.strftime('%Y%m%d-'+str(num))
    # z = np.random.normal(0, 1, (1, 8000000))
    z = np.random.normal(0, 1, (1, 2097152))
    gen_imgs = generator.predict(z)
    # gen_imgs = np.array(gen_imgs).reshape([8000000]).reshape((200,200,200))
    gen_imgs = np.array(gen_imgs).reshape([2097152]).reshape((128, 128, 128))
    OrthoSlicer3D(gen_imgs).show()
    # print(gen_imgs.shape)
    new_image = nib.Nifti1Image(gen_imgs, np.eye(4))
    nib.save(new_image, r'D:/施雨欣/深度学习/Results/871/ASD/autoencoder_nii/con_128/' + strtime + '.nii.gz')


# svae_image()