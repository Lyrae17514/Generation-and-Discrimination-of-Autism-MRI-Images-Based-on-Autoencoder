import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib
import tensorflow as tf

# 导包
from scipy import ndimage
import datetime
import nibabel as nib
# gpu显示
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#
from IPython import display
import pandas as pd
import tensorflow_probability as tfp
ds = tfp.distributions
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
    data = np.asarray(data).reshape((200, 200, 200))
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
        yield bb,bb

"""
## Create a sampling layer
"""


# class Sampling(layers.Layer):
#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# def sampling(inputs):
#     batch = inputs[0]
#     dim = inputs[1]
#     epsilon = tf.keras.backend.random_normal(shape=tf.shape(batch, dim))
#     return z_mean + tf.exp(0.5 * z_log_var) * epsilon
def sampling(inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    # batch = arg[0]
    # dim = arg[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""
## Build the encoder
"""
# latent_dim = 2
# latent_dim = 200
latent_dim = 125
input_shape = (200, 200, 200)
encoder_inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(40, 3, activation="relu", strides=5, padding="same")(encoder_inputs)
x = layers.Conv2D(20, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(10, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(5, 3, activation="relu", strides=2, padding="same")(x)
x = layers.AveragePooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(125, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
# z = Sampling()([z_mean, z_log_var])
z = layers.Lambda(
    sampling,
    output_shape=(latent_dim,))([z_mean, z_log_var])
# z = layers.Lambda(sampling, name='sampling')([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

"""
## Build the decoder
"""

latent_inputs = keras.Input(shape=(latent_dim,))
# x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
# x = layers.Reshape((7, 7, 64))(x)
x = layers.Dense(5 * 5 * 5, activation="relu")(latent_inputs)
x = layers.Reshape((5, 5, 5))(x)
x = layers.Conv2DTranspose(10, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(20, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(40, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(200, 3, activation="relu", strides=5, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(200, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

"""
## Define the VAE as a `Model` with a custom `train_step`
"""


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

"""
## 保存
"""
from nibabel.viewers import OrthoSlicer3D


def svae_image(generator):
    now = datetime.datetime.now()
    strtime = now.strftime('%Y%m%d')
    z = np.random.normal(0, 1, (200,200,200))
    gen_imgs = generator.decoder.predict(z)
    # gen_imgs = generator.predict(z)
    # gen_imgs = np.array(gen_imgs).reshape([8000000]).reshape((200, 200, 200))
    gen_imgs = gen_imgs[0].reshape([8000000]).reshape((200, 200, 200))
    OrthoSlicer3D(gen_imgs).show()
    # print(gen_imgs.shape)
    new_image = nib.Nifti1Image(gen_imgs, np.eye(4))
    nib.save(new_image, r'D:/施雨欣/深度学习/Test/' + strtime + '.nii.gz')


"""
## Train the VAE
"""

# (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
# mnist_digits = np.concatenate([x_train, x_test], axis=0)
# mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
bb= []
for i in range(3):
    aa = T1_file_to_array(path_G,T1_file_name[random.randint(0,19)])
    bb.append(aa)

bb = np.asarray(bb, dtype=np.float32)

datatrain = np.concatenate([bb, bb], axis=0)
datatrain = np.expand_dims(datatrain, -1).astype("float32") / 255
print(datatrain.shape)
# the optimizer for the model
# optimizer = tf.keras.optimizers.Adam(1e-3)
vae = VAE(encoder, decoder)

vae.compile(optimizer=Adam())
vae.fit(
    # datatrain,
    randchoose(),
    epochs=10,
    # steps_per_epoch=1,
    batch_size=1
)


svae_image(vae)
"""
## Display a grid of sampled digits
"""

import matplotlib.pyplot as plt

# def plot_latent_space(vae, n=1, figsize=15):
#     # display a n*n 2D manifold of digits
#     digit_size = 200
#     scale = 1.0
#     figure = np.zeros((digit_size * n, digit_size * n,digit_size * n))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-scale, scale, n)
#     grid_y = np.linspace(-scale, scale, n)[::-1]
#     grid_z = np.linspace(-scale, scale, n)[::-1]
#
#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             for k, zi in enumerate(grid_z):
#                 z_sample = np.array([[xi, yi,zi]])
#                 # x_decoded = vae.decoder.predict(z_sample)
#                 x_decoded = vae.decoder.predict(0, 1, (1,200,200,200))
#                 digit = x_decoded[0].reshape(digit_size, digit_size,digit_size)
#                 figure[
#                     i * digit_size : (i + 1) * digit_size,
#                     j * digit_size : (j + 1) * digit_size,
#                     k * digit_size : (k + 1) * digit_size,
#                 ] = digit
#
#     # plt.figure(figsize=(figsize, figsize))
#     # start_range = digit_size // 2
#     # end_range = n * digit_size + start_range
#     # pixel_range = np.arange(start_range, end_range, digit_size)
#     # sample_range_x = np.round(grid_x, 1)
#     # sample_range_y = np.round(grid_y, 1)
#     # plt.xticks(pixel_range, sample_range_x)
#     # plt.yticks(pixel_range, sample_range_y)
#     # plt.xlabel("z[0]")
#     # plt.ylabel("z[1]")
#     # plt.imshow(figure, cmap="Greys_r")
#     # plt.show()
#     plt.figure(1)
#     plt.imshow(digit)
#     plt.figure(2)
#     plt.imshow(bb[0].reshape(200,200, 200))
#     plt.show()


# plot_latent_space(vae)
#
# """
# ## Display how the latent space clusters different digit classes
# """
#
#
# def plot_label_clusters(vae, data, labels):
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = vae.encoder.predict(data)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.show()
#
#
# (x_train, y_train), _ = keras.datasets.mnist.load_data()
# x_train = np.expand_dims(x_train, -1).astype("float32") / 255
#
# plot_label_clusters(vae, x_train, y_train)
