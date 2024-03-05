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

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input
import random
import matplotlib.pyplot as plt
from scipy import ndimage
import datetime
import os

# adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


# adam = Adam()

preserving_ratio = 0.25

matplotlib.use('TkAgg')

# path_G = 'E:/eeg/T11/'
# path_G = 'D:/施雨欣/深度学习/帕金森/T1/mri/wmsub/'
path_G='D:/施雨欣/深度学习/HCP/test_ac/mri/wm/'#数据集路径

def read_T1_file_name(path) :
    path_file_list = os.listdir(path)
    T1_file_name = []
    for file in path_file_list:
        T1_file_name.append(file)
    return T1_file_name


# T1_file_name = read_T1_file_name(path_G)
# print(T1_file_name)


def T1_file_to_array(path,name):

    file_path = path + name
    img = nib.load(file_path)
    img = img.get_fdata()
    img_3d_max = np.amax(img)
    img = img / img_3d_max * 255

    data = ndimage.zoom(img,
                        (128/img.shape[0],128/img.shape[1],128/img.shape[2]),
                        order=0)
    # data = np.asarray(data).reshape((200*200*200, 1))
    data = data / 127.5 - 1

    # OrthoSlicer3D(data).show()
    return data

    # X_train = []
    # for j in range(img.shape[2]):  # 对切片进行循环
    #     # for j in range(img.shape[3]):
    #     img_2d = img[:, :, j]  # 取出一张图像
    #     img_2d = img_2d / 127.5 - 1
    #     img_2d = Image.fromarray(img_2d)
    #     img_2d = img_2d.resize((200, 200))
    #     # img_2d.show()
    #     img_2d = np.array(img_2d)
    #     X_train.append(img_2d)
    # X_train = np.asarray(X_train, dtype=np.float32)
    # Y_train = []
    # for k in range(X_train.shape[2]):  # 对切片进行循环
    #     # for j in range(img.shape[3]):
    #     img_2d = X_train[:, :, k]  # 取出一张图像
    #     # img_2d = img_2d / 127.5 - 1
    #     img_2d = Image.fromarray(img_2d)
    #     img_2d = img_2d.resize((200, 200))
    #     # img_2d.show()
    #     img_2d = np.array(img_2d)
    #     Y_train.append(img_2d)
    # Y_train = np.asarray(Y_train, dtype=np.float32)
    # Z_train = []
    # for l in range(Y_train.shape[2]):  # 对切片进行循环
    #     # for j in range(img.shape[3]):
    #     img_2d = Y_train[:, :, l]  # 取出一张图像
    #     # img_2d = img_2d / 127.5 - 1
    #     img_2d = Image.fromarray(img_2d)
    #     img_2d = img_2d.resize((200, 200))
    #     # img_2d.show()
    #     img_2d = np.array(img_2d)
    #     Z_train.append(img_2d)
    # Z_train = np.asarray(Z_train, dtype=np.float32)
    # # data = np.asarray(Z_train).reshape((1, 8000000, 1))
    # data = np.asarray(Z_train).reshape((8000000, 1))
    # return data

# aa = T1_file_to_array(path_G,T1_file_name[0])
# OrthoSlicer3D(aa).show()
# data = np.asarray(aa).reshape((1,8000000,1))
# data1 = np.array(data)
# data1 = np.expand_dims(data1, axis=0)
# print(aa.shape)

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

z_dim = 2097152

img_rows = 1
img_cols = 2097152
channels = 1

img_shape = (img_rows, img_cols, channels)

def build_generator(img_shape, z_dim):
    model = Sequential()
    # model.add(Dense(1, input_dim=z_dim))
    # model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(2, activation='relu', input_dim=z_dim))

    model.add(LeakyReLU(alpha=0.0007))
    # model.add(Dense(4, input_dim=z_dim))
    model.add(Dense(4))
    # model.add(LeakyReLU(alpha=0.01))
    # model.add(Dense(8))

    model.add(LeakyReLU(alpha=0.0007))
    # model.add(Dense(1024, activation='tanh'))
    # model.add(LeakyReLU(alpha=0.01))
    # model.add(Dense(4096, activation='tanh'))
    # model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(2097152, activation='tanh'))
    # model.add(LeakyReLU(alpha=0.01))

    model.add(Reshape(img_shape))
    return model

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    # model.add(Dense(8))
    # model.add(LeakyReLU(alpha=0.001))
    model.add(Dense(4))

    model.add(LeakyReLU(alpha=0.0007))
    model.add(Dense(2))
    model.add(LeakyReLU(alpha=0.0007))

    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=adam,metrics=['accuracy'])

generator = build_generator(img_shape, z_dim)
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=adam)

print("discriminator")
discriminator.summary()
print("generator")
generator.summary()
print("gan")
gan.summary()


losses = []
D_loss = []
G_loss = []
accuracies = []
iteration_checkpoints = []

def svae_image(generator ,num):
    z = np.random.normal(0, 1, (1, z_dim))
    gen_imgs = generator.predict(z)
    gen_imgs = np.array(gen_imgs).reshape([2097152]).reshape((128,128,128))
    # if(num%5000==0):
    #     OrthoSlicer3D(gen_imgs).show()
    new_image = nib.Nifti1Image(gen_imgs, np.eye(4))
    nib.save(new_image, r'D:/施雨欣/深度学习/Test/GAN/' + str(num) + '.nii.gz')

    # img = nib.load('gan_nii/'+str(num)+'.nii.gz')
    # img = img.get_fdata()
    # OrthoSlicer3D(img).show()
    # print(gen_imgs.shape)

def all_data(path , name):
    img_data = []
    for i in name:
        img_data.append(T1_file_to_array(path, i))
    img_data = np.asarray(img_data, dtype=np.float32)
    return  img_data

T1_file_name = read_T1_file_name(path_G)
all_img_data = all_data(path_G , T1_file_name)

print(all_img_data.shape)

# gen_imgs = np.array(all_img_data[0]).reshape([8000000]).reshape((200,200,200))
# OrthoSlicer3D(gen_imgs).show()

def train(iterations, batch_size, save_weight, save):

    T1_file_name = read_T1_file_name(path_G)
    # X_train = np.expand_dims(X_train, axis=3)
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    idx = np.random.randint(0, len(T1_file_name) - 1, batch_size)
    imgs = all_img_data[idx]


    for iteration in range(iterations):

        # idx = np.random.randint(0, len(T1_file_name)-1, batch_size)
        # print(str(idx[0])+ " ----- "+str(iteration+1))
        # name = T1_file_name[idx[0]]
        # imgs = all_img_data[idx]
        imgs = np.asarray(imgs).reshape((batch_size, 2097152, 1))
        # print(imgs.shape)
        imgs = np.expand_dims(imgs, axis=3)
        # imgs = T1_file_to_array(path_G,name)
        # imgs = all_img_data
        # imgs = np.expand_dims(imgs, axis=3)

        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 1, (batch_size, z_dim))
        g_loss = gan.train_on_batch(z, real)

        # if (iteration + 1) % sample_interval == 0:
        losses.append((d_loss, g_loss))
        # D_loss.append(d_loss)
        # G_loss.append(g_loss)
        # # accuracies.append(100.0 * accuracy)
        # accuracies.append(accuracy)
        iteration_checkpoints.append(iteration + 1)
        print("%d [D loss: %f, acc.: %.2f%%] [G loss:%f]" % (iteration + 1, d_loss, 100.0 * accuracy, g_loss))




        if (iteration + 1) % save == 0:
            svae_image(generator,iteration + 1)
            # losses=np.array(losses)

            # D_loss, G_loss = losses
            # fig = plt.figure(figsize=(6, 3))  # 设置图大小 figsize=(6,3)
            # plt.plot( np.arange(iteration + 1), D_loss, c='red', label='D-loss')
            # plt.plot( np.arange(iteration + 1), G_loss, c='blue', label='G-loss')
            # plt.plot( np.arange(iteration + 1), accuracies, c='green', label='D-acc')
            # plt.legend(loc='best')
            # plt.savefig('./gan_nii/test2.1.jpg')
            # plt.show()
        if (iteration + 1) % save_weight == 0:
            generator.save_weights("G_model%d.hdf5" % (iteration+1), True)
            discriminator.save_weights("D_model%d.hdf5" % (iteration+1), True)

if not os.path.exists("./gan_nii_w_2"):
    os.makedirs("./gan_nii_w_2")

iterations  = 15000
batch_size = 1
# sample_interval = 1
save = 5000
save_weight = 5000
train(iterations, batch_size, save_weight ,save)




