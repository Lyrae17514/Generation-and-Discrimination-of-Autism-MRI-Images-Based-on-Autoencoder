import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,Model,models,utils
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
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

from scipy import ndimage
import datetime
import nibabel as nib
"""
## 数据源准备
"""

path_G = 'D:/施雨欣/深度学习/帕金森/T1/mri/wmsub/'


def read_T1_file_name(path):
    path_file_list = os.listdir(path)
    T1_file_name = []
    for file in path_file_list:
        T1_file_name.append(file)
    return T1_file_name


T1_file_name = read_T1_file_name(path_G)
print(T1_file_name)

def data_in_one(inputdata):
    min = np.nanmin(inputdata)
    max = np.nanmax(inputdata)
    outputdata = (inputdata - min) / (max - min)
    return outputdata

def T1_file_to_array(path, name):
    file_path = path + name
    img = nib.load(file_path)
    img = img.get_fdata()
    img_3d_max = np.amax(img)
    img = img / img_3d_max * 255
    img = img / 127.5 - 1
    img = data_in_one(img)
    data = ndimage.zoom(img,
                        (128 / img.shape[0], 128 / img.shape[1], 128 / img.shape[2]),
                        order=0)
    data = np.asarray(data).reshape((128, 128, 128))
    return data




"""
## 随机抽取函数
"""
import random


def randchoose():
    while 1:
        bb = []
        cc = T1_file_name[random.randint(0, 19)]
        print(cc)
        aa = T1_file_to_array(path_G, cc)
        bb.append(aa)
        bb = np.asarray(bb, dtype=np.float32)
        # print(bb.shape)
        yield bb, bb

import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

def svae_image(generator):
    now = datetime.datetime.now()
    strtime = now.strftime('%Y%m%d')
    z = np.random.normal(0, 1, (1, 128,128,128))
    gen_imgs = generator.predict(z)
    gen_imgs = np.array(gen_imgs).reshape([128*128*128]).reshape((128, 128, 128))
    OrthoSlicer3D(gen_imgs).show()
    # print(gen_imgs.shape)
    new_image = nib.Nifti1Image(gen_imgs, np.eye(4))
    nib.save(new_image, r'D:/施雨欣/深度学习/Test/' + strtime + '.nii.gz')


img_shape = (128,128,128,1)
latent_dim = 256

input_img = layers.Input(shape=img_shape)
# x = layers.Flatten()(input_img)
x = layers.Conv3D(8,3,padding='same',activation='relu')(input_img)

x = layers.Conv3D(8,3,padding='same',activation='relu',strides=2)(x)
x = layers.Conv3D(8,3,padding='same',activation='relu')(x)
x = layers.Conv3D(8,3,padding='same',activation='relu')(x)

x = layers.Conv3D(8,3,padding='same',activation='relu',strides=2)(x)
x = layers.Conv3D(16,3,padding='same',activation='relu')(x)
x = layers.Conv3D(16,3,padding='same',activation='relu',strides=2)(x)

x = layers.Conv3D(16,3,padding='same',activation='relu',strides=2)(x)
x = layers.Conv3D(32,3,padding='same',activation='relu')(x)
x = layers.Conv3D(32,3,padding='same',activation='relu',strides=2)(x)

x = layers.Conv3D(32,3,padding='same',activation='relu',strides=2)(x)
x = layers.Conv3D(64,3,padding='same',activation='relu')(x)
# x = layers.Conv2D(64,3,padding='same',activation='relu',strides=2)(x)

x = layers.Conv3D(64,3,padding='same',activation='relu')(x)
inter_shape = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(256,activation='relu')(x)

encode_mean = layers.Dense(256,name = 'encode_mean')(x)       #分布均值
encode_log_var = layers.Dense(256,name = 'encode_logvar')(x)  #分布对数方差

encoder = Model(input_img,[encode_mean,encode_log_var],name = 'encoder')
encoder.summary()

#%%解码器
input_code = layers.Input(shape=(latent_dim,))
x = layers.Dense(np.prod(inter_shape[1:]),activation='relu')(input_code)
x = layers.Reshape(target_shape=inter_shape[1:])(x)
x = layers.Conv3DTranspose(64,3,padding='same',activation='relu')(x)
x = layers.Conv3DTranspose(64,3,padding='same',activation='relu')(x)
# x = layers.Conv2DTranspose(32,3,padding='same',activation='relu',strides=2)(x)

x = layers.Conv3DTranspose(32,3,padding='same',activation='relu',strides=2)(x)
# x = layers.Conv2D(64,3,padding='same',activation='relu',strides=2)(x)

x = layers.Conv3DTranspose(32,3,padding='same',activation='relu',strides=2)(x)
x = layers.Conv3DTranspose(32,3,padding='same',activation='relu')(x)
x = layers.Conv3DTranspose(16,3,padding='same',activation='relu',strides=2)(x)

x = layers.Conv3DTranspose(16,3,padding='same',activation='relu',strides=2)(x)
x = layers.Conv3DTranspose(16,3,padding='same',activation='relu')(x)
x = layers.Conv3DTranspose(8,3,padding='same',activation='relu',strides=2)(x)

x = layers.Conv3DTranspose(8,3,padding='same',activation='relu')(x)
x = layers.Conv3DTranspose(8,3,padding='same',activation='relu')(x)
x = layers.Conv3DTranspose(8,3,padding='same',activation='relu',strides=2)(x)

x = layers.Conv3DTranspose(8,3,padding='same',activation='relu')(x)

x = layers.Conv3DTranspose(1,3,padding='same',activation='sigmoid')(x)
# x = layers.Conv2D(128,3,padding='same',activation='sigmoid')(x)

decoder = Model(input_code,x,name = 'decoder')
decoder.summary()

# %%整体待训练模型
def sampling(arg):
    mean = arg[0]
    logvar = arg[1]
    epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)  # 从标准正态分布中抽样
    return mean + K.exp(0.5 * logvar) * epsilon  # 获取生成分布的抽样


input_img = layers.Input(shape=img_shape, name='img_input')
code_mean, code_log_var = encoder(input_img)  # 获取生成分布的均值与方差
x = layers.Lambda(sampling, name='sampling')([code_mean, code_log_var])
x = decoder(x)
training_model = Model(input_img, x, name='training_model')

decode_loss = keras.metrics.binary_crossentropy(K.flatten(input_img), K.flatten(x))
kl_loss = -5e-4 * K.mean(1 + code_log_var - K.square(code_mean) - K.exp(code_log_var))
training_model.add_loss(K.mean(decode_loss + kl_loss))  # 新出的方法，方便得很
training_model.compile(optimizer='rmsprop')

# %%读取数据集训练
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.astype('float32') / 255
# x_train = x_train[:, :, :, np.newaxis]

training_model.fit(
    randchoose(),
    steps_per_epoch=1,
    epochs=5000
)
svae_image(training_model)
#%%测试
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
# n = 20
# x = y = norm.ppf(np.linspace(0.01,0.99,n))  #生成标准正态分布数
# X,Y = np.meshgrid(x,y)                      #形成网格
# X = X.reshape([-1,1])                       #数组展平
# Y = Y.reshape([-1,1])
# input_points = np.concatenate([X,Y],axis=-1)#连接为输入
# for i in input_points:
#   plt.scatter(i[0],i[1])
# plt.show()

# img_size = 28

# predict_img = decoder.predict(input_points)
# pic = np.empty([img_size*n,img_size*n,1])
# for i in range(n):
#   for j in range(n):
#     pic[img_size*i:img_size*(i+1), img_size*j:img_size*(j+1)] = predict_img[i*n+j]
# plt.figure(figsize=(10,10))
# plt.axis('off')
# pic = np.squeeze(pic)
# plt.imshow(pic,cmap='bone')
# z = np.random.normal(0, 1, (1,128,128,128))
# gen_imgs = decoder.predict(z)
# gen_imgs = np.array(gen_imgs).reshape([128*128*128]).reshape((128, 128, 128))
# OrthoSlicer3D(gen_imgs).show()
# plt.show()