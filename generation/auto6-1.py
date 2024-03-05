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

# path_G='D:/施雨欣/深度学习/帕金森/T1/mri/wmsub/'#数据集路径
# path_G='D:/施雨欣/深度学习/HCP/test_ac/mri/wm/'#数据集路径
# save_path = "D:/施雨欣/深度学习/Results/HCP/autoencoder_model/"#生成文件存放路径

# path_G='D:/施雨欣/深度学习/自闭症/test/压缩包任务组_20231025_1411/'
# path_G='D:/施雨欣/深度学习/自闭症/871-control-nii/'
path_G='D:/施雨欣/深度学习/自闭症/871-ants-nii/control/'
save_path = 'D:/施雨欣/深度学习/Results/871/control/autoencoder_nii/con_128/'

def read_T1_file_name(path) :
    path_file_list = os.listdir(path)
    T1_file_name = []
    for file in path_file_list:
        T1_file_name.append(file)
    return T1_file_name


T1_file_name = read_T1_file_name(path_G)
print(len(T1_file_name))


def T1_file_to_array(path,name):

    file_path = path + name
    # file_path = path
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


    return data

# bb= []
# for i in range(5):
#     aa = T1_file_to_array(path_G,T1_file_name[random.randint(0,19)])
#     bb.append(aa)
#
# bb = np.asarray(bb, dtype=np.float32)
# OrthoSlicer3D(aa).show()
# data = np.asarray(aa).reshape((1,8000000,1))
# data1 = np.array(data)
# data1 = np.expand_dims(data1, axis=0)

# print(bb.shape)


def randchoose():
    while 1:
        bb = []
        # cc = T1_file_name[random.randint(0, 19)]
        # cc = T1_file_name[random.randint(0, 402)]
        cc = T1_file_name[random.randint(0, 467)]
        # cc = T1_file_name[0]
        print(cc)
        aa = T1_file_to_array(path_G, cc)
        bb.append(aa)
        bb = np.asarray(bb, dtype=np.float32)
        # print(bb.shape)
        yield bb,bb

# f = open('T2.1.csv', 'a', newline='')
# data = np.array(data)
# writer = csv.writer(f)
# writer.writerow(data)
# f.close()

# data = np.asarray(data).reshape((200,200,200))
# OrthoSlicer3D(data).show()
print("data")

# path_G_list = os.listdir(path_G)
# zimulu = len(path_G_list)
#
# print(zimulu)
# print(path_G_list[0])
# T1_file_name = []
# for file in path_G_list:
#     # print(file)
#     T1_file_name.append(file)
# print(T1_file_name)

z_dim = 100

img_rows = 1
img_cols = 8000000
channels = 1

img_shape = (img_rows, img_cols, channels)



losses = []
accuracies = []
iteration_checkpoints = []


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
    nib.save(new_image, r'D:/施雨欣/深度学习/Results/871/control/autoencoder_nii/con_128/' + strtime + '.nii.gz')


def save_image(generator,num):
    now = datetime.datetime.now()
    strtime = now.strftime('%Y%m%d-'+str(num))
    # z = np.random.normal(0, 1, (1, 8000000))
    z = np.random.normal(0, 1, (1, 2097152))
    gen_imgs = generator.predict(z)
    # gen_imgs = np.array(gen_imgs).reshape([8000000]).reshape((200,200,200))
    gen_imgs = np.array(gen_imgs).reshape([2097152]).reshape((128, 128, 128))
    # OrthoSlicer3D(gen_imgs).show()
    # print(gen_imgs.shape)
    new_image = nib.Nifti1Image(gen_imgs, np.eye(4))
    nib.save(new_image, r'D:/施雨欣/深度学习/Results/871/control/autoencoder_nii/con_128/' + strtime + '.nii.gz')

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# X_train = X_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.

# X_train = X_train.reshape(-1, 8000000)
# x_test = x_test.reshape(-1, 784)
# print(X_train.shape)
# print(x_test.shape)
encoding_dim = 128
input_img = Input(shape=(2097152,))

encoded = Dense(units=4, activation='relu', )(input_img)
# encoded = Dense(units=4, activation='relu', )(encoded)
encoded = Dense(units=2, activation='relu', )(encoded)

encoded_output = Dense(units=encoding_dim)(encoded)

decoded = Dense(units=2, activation='relu')(encoded_output)
decoded = Dense(units=4, activation='relu')(decoded)
# decoded = Dense(units=6, activation='relu')(decoded)
decoded = Dense(units=2097152, activation='tanh')(decoded)

autoencoder = Model(inputs=input_img, outputs=decoded)
encoder = Model(inputs=input_img, outputs=encoded_output)

autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.fit(
#     x=bb,
#     y=bb,
#     epochs=10,
#     batch_size=1,
#     shuffle=True,
# )
# autoencoder.fit_generator(
#     randchoose(),
#     epochs=10,
#     steps_per_epoch=1,
#
# )
epochs=84000
save_every = 4000
# filepath = save_path+'model_{epoch:02d}.h5'
# checkpoint_callback =ModelCheckpoint(
#     filepath,
#     monitor='val_loss',
#     verbose=1,
#     save_best_only=False,
#     save_weights_only=False,
#     mode='auto',
#     period=save_every
# )
# 定义回调函数，用于保存每100次迭代后生成器生成的图像
class SaveImageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % save_every == 0:
            save_image(autoencoder, epoch + 1)

#原
autoencoder_model=autoencoder.fit(
    randchoose(),
    epochs=epochs,
    steps_per_epoch=1,
    shuffle=True,
    # callbacks=[checkpoint_callback],
    callbacks=[SaveImageCallback()],  # 添加回调函数
    # validation_split=0.8 # 验证集分割，有问题

)


# encoded_imgs = encoder.predict(x_test)
# print(encoded_imgs[0])
# plt.scatter(x=encoded_imgs[:, 0], y=encoded_imgs[:, 1], c=y_test, s=2)
# plt.show()

# decoded_img = autoencoder.predict(x_test[1].reshape(1, 784))
# encoded_img = encoder.predict(x_test[1].reshape(1, 784))
#
# plt.figure(1)
# plt.imshow(decoded_img[0].reshape(28, 28))
# plt.figure(2)
# plt.imshow(encoded_img[0].reshape(2, 2))
# plt.figure(3)
# plt.imshow(x_test[1].reshape(28, 28))
# plt.show()

# plot the training loss and accuracy

plt.figure()
train_loss = autoencoder_model.history['loss']
# val_loss = autoencoder_model.history['val_loss']


# 创建x轴数据，从1到epochs
x = np.arange(1, epochs + 1)

# 绘制损失曲线和准确率曲线
plt.plot(x, train_loss, label='Training Loss', linestyle='-', color='blue')
# plt.plot(x, val_loss, label='Validation Loss', linestyle='--', color='red')

# plt.plot(np.arange(0, epochs), autoencoder_model.history["loss"], label="train_loss")



plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss_128_{:d}e.jpg".format(epochs))
# 保存生成器生成的图像
svae_image(autoencoder,epochs)

iterations  = 20000
batch_size = 1
sample_interval = 1
save = 100