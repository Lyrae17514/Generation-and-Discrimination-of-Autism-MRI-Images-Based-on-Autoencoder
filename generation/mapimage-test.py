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
import cv2
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
tf.compat.v1.disable_eager_execution()
import imageio
from skimage.transform import resize
from sklearn.decomposition import PCA

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
config = tf.compat.v1.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
config.gpu_options.allow_growth = True


gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:

    tf.config.experimental.set_memory_growth(gpu, True)

preserving_ratio = 0.25

# 设置数据集路径
data_folder = "D:/施雨欣/深度学习/Test/slice/"

# 读取所有图片并进行预处理
image_files = os.listdir(data_folder)
num_images = len(image_files)

# images = []
# for file in image_files:
#     image_path = os.path.join(data_folder, file)
#     image = imageio.imread(image_path)
#     print(image.shape)
#     image = resize(image,(128,128))
#     image = image.astype('float32') / 255.0  # 归一化
#     images.append(image)
image = tf.keras.preprocessing.image.load_img("D:/施雨欣/深度学习/Test/slice/580-x.jpeg",
                                              target_size=(128,128),color_mode='grayscale')
# image = imageio.imread("D:/施雨欣/深度学习/Test/slice/63.jpeg")
# print(image.shape)
# image = resize(image,(128,128))
# print(image.shape)

# image.show()
image = tf.keras.preprocessing.image.img_to_array(image)
print(image.shape)
# 显示原始图像
# plt.figure(figsize=(5, 5))
# plt.imshow(image/255.0)  # 原始图像应该在0到1之间
# plt.title('Original Image1')
# plt.axis('off')
# plt.show()
data = image.astype('float32') / 255.0
data = data.reshape(-1, 128*128)

print(data.shape)

# 显示原始图像
# plt.figure(figsize=(5, 5))
# plt.imshow(data.reshape((128,128,3)))  # 原始图像应该在0到1之间
# plt.title('Original Image2')
# plt.axis('off')
# plt.show()
# (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.
#
# x_train = x_train.reshape(-1, 784)
# x_test = x_test.reshape(-1, 784)

# 生成重构图像
reconstructed_imgs = data
# plt.imshow(reconstructed_imgs[0].reshape((28, 28)), cmap='gray')
# plt.colorbar()  # 添加颜色条以显示强度对应的颜色
# plt.show()
# 显示重构图像
# plt.figure(figsize=(5, 5))
# plt.imshow(reconstructed_imgs.reshape((128, 128)), cmap='gray')  # 使用灰度颜色映射显示重构图像
# plt.title('Reconstructed Image1')
# plt.axis('off')
# plt.show()

cmap = plt.get_cmap('jet')  # 使用jet颜色映射，也可以选择其他颜色映射
colored_image = cmap(reconstructed_imgs.reshape((128, 128)))

# 显示彩色图像
plt.imshow(colored_image)
# plt.colorbar()  # 添加颜色条以显示强度对应的颜色
plt.show()

tf.keras.preprocessing.image.save_img("D:/施雨欣/深度学习/Test/slice/580-x.png",colored_image)