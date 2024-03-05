import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

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
import imageio
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

# 设置数据集路径
data_folder = "D:/871-control-T1/img_1/"


# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# 读取所有图片并进行预处理
image_files = os.listdir(data_folder)
num_images = len(image_files)

images = []
for file in image_files:
    image_path = os.path.join(data_folder, file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图像读取
    image = cv2.resize(image, (128, 128))  # 调整大小为256x256
    image = image.astype('float32') / 255.0  # 归一化
    images.append(image)

    # file_path = path + name
    # # file_path = path
    # img = nib.load(file_path)
    # img = img.get_fdata()
    # img_3d_max = np.amax(img)
    # img = img / img_3d_max * 255
    # img = img / 127.5 - 1
    #
    # data = ndimage.zoom(img,
    #                     (128 / img.shape[0], 128 / img.shape[1]),
    #                     order=0)
    # # data = np.asarray(data).reshape((8000000)) # 200
    # data = np.asarray(data).reshape((16384))  # 128

x_data = np.array(images)
x_data = x_data.reshape(-1, 128*128)

# 创建编码器和解码器模型
encoding_dim = 128

input_img = Input(shape=(128*128,))

encoded = Dense(units=4, activation='relu', )(input_img)
# encoded = Dense(units=4, activation='relu', )(encoded)
encoded = Dense(units=2, activation='relu', )(encoded)

encoded_output = Dense(units=encoding_dim)(encoded)

decoded = Dense(units=2, activation='relu')(encoded_output)
decoded = Dense(units=4, activation='relu')(decoded)
# decoded = Dense(units=6, activation='relu')(decoded)
decoded = Dense(units=128*128, activation='tanh')(decoded)

autoencoder = Model(inputs=input_img, outputs=decoded)
encoder = Model(inputs=input_img, outputs=encoded_output)

# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='mse')
epochs=1001
save_every = 50

# 在每save_every个epoch结束时保存生成的图片
class save_images(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % save_every == 0:
            # encoded_imgs = encoder.predict(x_data)
            # decoded_imgs = autoencoder.predict(x_data)
            # # 随机选择一些图像可视化
            # n = 4
            # plt.figure(figsize=(10, 4))
            # for i in range(n):
            #     plt.subplot(1, n, i + 1)
            #     plt.imshow(decoded_imgs[i].reshape(256, 256), cmap='gray')
            #     plt.axis('off')
            #     plt.savefig(f'D:/施雨欣/深度学习/Results/871/control/img128/epoch_{epoch}.png')
            # 生成图像
            generated_img = autoencoder.predict(x_data)
            # 裁剪生成图像
            # 裁剪和保存生成图像
            # generated_img = np.array(generated_img).reshape([16384]).reshape((128, 128))
            # cropped_img = generated_img[0].reshape(256, 256)

            # # 保存生成图像
            # plt.imshow(generated_img[0].reshape(128, 128), cmap='gray')
            # plt.axis('off')
            # plt.savefig(f'D:/施雨欣/深度学习/Results/871/control/img128/epoch_{epoch}.png',bbox_inches='tight',pad_inches=0)
            # plt.close()

            # 转换图像
            img = (generated_img[0].reshape(128, 128) * 255.0).astype(np.uint8)
            # 保存生成图像
            imageio.imwrite(f'D:/施雨欣/深度学习/Results/871/control/img256/epoch_{epoch}.png', img)
            # img = (generated_img[0].reshape(256, 256) * 255.0).astype(np.uint8)#
            # cv2.imwrite(f'D:/施雨欣/深度学习/Results/871/control/img256/epoch_{epoch}.png', img)


# 训练模型并保存结果
history = autoencoder.fit(
    x_data,
    x_data,
    epochs=epochs,
    batch_size=64,
    validation_split=0.2,
    callbacks=[save_images()]
)

# 保存训练曲线
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(train_loss)
plt.plot(val_loss)
plt.title("Training and Validation Loss")
plt.legend(['train', 'val'])
plt.savefig("D:/施雨欣/深度学习/Results/871/control/img256/training_loss.png")