import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,Model,models,utils
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

img_shape = (28,28,1)
latent_dim = 2

input_img = layers.Input(shape=img_shape)
x = layers.Conv2D(32,3,padding='same',activation='relu')(input_img)
x = layers.Conv2D(64,3,padding='same',activation='relu',strides=2)(x)
x = layers.Conv2D(64,3,padding='same',activation='relu')(x)
x = layers.Conv2D(64,3,padding='same',activation='relu')(x)
inter_shape = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dense(32,activation='relu')(x)

encode_mean = layers.Dense(2,name = 'encode_mean')(x)       #分布均值
encode_log_var = layers.Dense(2,name = 'encode_logvar')(x)  #分布对数方差

encoder = Model(input_img,[encode_mean,encode_log_var],name = 'encoder')

#%%解码器
input_code = layers.Input(shape=[2])
x = layers.Dense(np.prod(inter_shape[1:]),activation='relu')(input_code)
x = layers.Reshape(target_shape=inter_shape[1:])(x)
x = layers.Conv2DTranspose(32,3,padding='same',activation='relu',strides=2)(x)
x = layers.Conv2D(1,3,padding='same',activation='sigmoid')(x)

decoder = Model(input_code,x,name = 'decoder')


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
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_train = x_train[:, :, :, np.newaxis]

training_model.fit(
    x_train,
    batch_size=512,
    epochs=10
)

#%%测试
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
n = 20
x = y = norm.ppf(np.linspace(0.01,0.99,n))  #生成标准正态分布数
X,Y = np.meshgrid(x,y)                      #形成网格
X = X.reshape([-1,1])                       #数组展平
Y = Y.reshape([-1,1])
input_points = np.concatenate([X,Y],axis=-1)#连接为输入
for i in input_points:
  plt.scatter(i[0],i[1])
plt.show()

img_size = 28
predict_img = decoder.predict(input_points)
plt.figure(1)
plt.imshow(predict_img[0].reshape(28, 28))

# pic = np.empty([img_size*n,img_size*n,1])
# for i in range(n):
#   for j in range(n):
#     pic[img_size*i:img_size*(i+1), img_size*j:img_size*(j+1)] = predict_img[i*n+j]
# plt.figure(figsize=(10,10))
# plt.axis('off')
# pic = np.squeeze(pic)
# plt.imshow(pic,cmap='bone')
plt.show()