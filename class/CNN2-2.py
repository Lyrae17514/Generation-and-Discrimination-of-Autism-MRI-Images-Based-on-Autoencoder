import os

import tensorflow.keras as keras
import matplotlib
import numpy as np
import nibabel as nib
from tensorflow.keras import optimizers
from nibabel.viewers import OrthoSlicer3D
from PIL import Image
import csv
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.python.keras.layers import Dense, Flatten, Reshape ,ReLU,BatchNormalization

from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.convolutional import AveragePooling3D
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy import ndimage

from sklearn.model_selection import train_test_split


# path_G = 'G:/eeg/T11/'
# path_G = 'D:/施雨欣/深度学习/帕金森/T11/'
path_G = 'D:/施雨欣/深度学习/HCP/test_ac/mri/wm/'

adam = Adam(lr=0.0001, epsilon=1e-08)
sgd = SGD(lr=0.0001)

def read_T1_name_and_path(path):

    name_and_lable=[]
    for i, j, k in os.walk(path):
        # print(i,j,k)       #路径名，文件夹，文件
        if len(k):           #是文件
            # print(k)
            for name1 in k:
                name = []
                path_and_name = i + "/" + name1      #名字路径拼接
                name.append(path_and_name)
                if name1.startswith("wmsub-control"):
                    name.append(0)
                else:
                    name.append(1)
                name_and_lable.append(name)
    # print(name_and_lable)
    return name_and_lable

T1 = read_T1_name_and_path(path_G)
print(T1)

def T1_file_to_array(file_path):

    # file_path = path + name
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

# T1_file_to_array(T1[1])


def split_date_and_lable(T1_date,
                         test_size_val=0.3,
                         random_state_val=4,
                         shuffle_flag=True):
    T1_date = np.array(T1_date)
    batch_d = T1_date[0:,0]
    batch_l = T1_date[0:,1]
    # 训练集、测试集 划分
    train_x, test_x, train_y, test_y = train_test_split(batch_d, batch_l,
                                                        test_size=test_size_val,
                                                        random_state=random_state_val,
                                                        shuffle=shuffle_flag)
    return train_x, test_x, train_y, test_y;


# train_x, test_x, train_y, test_y = split_date_and_lable(T1,0.2)
# print(len(train_x))
def generate_arrays_from_file( date, batch_size=7 ,is_train=True):
    train_x, test_x, train_y, test_y = split_date_and_lable(date)
    # print(train_x, test_x, train_y, test_y)
    if is_train:
        date_x = train_x
        labe_y = train_y
    else:
        date_x = test_x
        labe_y = test_y
    # print(date_x,labe_y)
    count = 1
    while 1:
        if count * batch_size >=len(date_x):
            batch_x = date_x[(count - 1) * batch_size:len(date_x) ]
            # print(batch_x)
            batch_y = labe_y[(count - 1) * batch_size:len(labe_y) ]
            count = 0
        else:
            batch_x = date_x[(count - 1) * batch_size:count * batch_size]
            batch_y = labe_y[(count - 1) * batch_size:count * batch_size]

        count = count + 1

        batch_x = np.array([T1_file_to_array(img_path) for img_path in batch_x])
        # print(batch_x.shape)
        batch_x = np.expand_dims(batch_x, axis=4)
        batch_y = np.array(batch_y).astype(np.float32)
        batch_y = keras.utils.to_categorical(batch_y)

        # print(batch_x.shape, batch_y)

        yield batch_x, batch_y


# generate_arrays_from_file(T1,5,False)
# print(x,y)
# 定义数据集路径和文件列表
data_folder = "D:/施雨欣/深度学习/帕金森/T11/"
file_list = sorted(os.listdir(data_folder))

# 定义数据生成器
def data_generator(file_list, batch_size):
    while True:
        # 打乱文件列表
        np.random.shuffle(file_list)
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i+batch_size]
            batch_data = []
            batch_labels = []
            for file_name in batch_files:
                file_path = os.path.join(data_folder, file_name)
                # image = nib.load(file_path).get_fdata()
                image = T1_file_to_array(file_path) # 数据预处理，如归一化等
                # image = image.reshape((128, 128, 128, 1))  # 设置图像的形状
                label = 0 if file_name.startswith("wmsub-control") else 1# 根据数据集的标签获取相应的标签值

                # ...
                batch_data.append(image)
                batch_labels.append(label)
            batch_data = np.array(batch_data)
            batch_data = np.expand_dims(batch_data, axis=4)
            # print(batch_data.shape)
            batch_labels = np.array(batch_labels).astype(np.float32)
            batch_labels = keras.utils.to_categorical(batch_labels, num_classes=2)  # 将标签转换为独热编码
            yield batch_data, batch_labels

model = Sequential()
model.add(Conv3D(2, (3, 3, 3), strides=(1, 1, 1),input_shape=( 128, 128, 128, 1), padding='same', activation='relu'))
# model.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu'))
# model.add(Conv3D(4, (3, 3, 3),  padding='same', activation='relu'))
model.add(Conv3D(8, (3, 3, 3),strides=2,padding='same', activation='relu'))
# model.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu'))
model.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu'))

model.add(BatchNormalization())
model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))

# model.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
model.add(Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
model.add(Conv3D(16, (3, 3, 3), strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))

model.add(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
# model.add(Conv3D(32, (3, 3, 3),  padding='same', activation='relu'))
model.add(Conv3D(32, (3, 3, 3), strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))

# model.add(Conv3D(64, (3, 3, 3),  padding='same', activation='relu'))
model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
# model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=1))

# model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))

# model.add(Conv3D(64, (3, 3, 3),strides=(1, 1, 1), padding='same', activation='relu'))
# model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu'))

# model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=1))

model.add(Conv3D(128, 3,strides=(1, 1, 1), padding='same', activation='relu'))
# # model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=1))
model.add(Conv3D(128, 3, padding='same', activation='relu'))
# model.add(Conv3D(256, 3, padding='same', activation='relu'))
# model.add(Conv3D(256, 3, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=1))
model.add(Conv3D(128, 2, padding='same', activation='relu'))
# model.add(Conv3D(200, 2, padding='same', activation='relu'))
model.add(Conv3D(128, 2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling3D(pool_size=(1, 1, 1), strides=1))
# model.add(Conv3D(200, (3, 3, 3), padding='same', activation='relu'))

model.add(Flatten())
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1600, activation='relu'))
# model.add(Dropout(0.8))
# model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1600, activation='relu'))
# model.add(Dropout(0.8))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.8))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.8))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])

# 划分训练集和测试集
train_ratio = 0.8
split_idx = int(train_ratio * len(file_list))
train_file_list = file_list[:split_idx] # 28
test_file_list = file_list[split_idx:]
# 定义训练集和测试集的数据生成器
batch_size = 7
train_data_generator = data_generator(train_file_list, batch_size)
test_data_generator = data_generator(test_file_list, batch_size)
# 训练模型
train_steps = len(train_file_list) // batch_size
test_steps = len(test_file_list) // batch_size
history = model.fit(
    train_data_generator,
    steps_per_epoch=1,#train_steps,
    epochs=100,
    shuffle=True,
    validation_data=test_data_generator,
    validation_steps=1#test_steps
)
# T1 = read_T1_name_and_path(path_G)
# history=model.fit(
#     generate_arrays_from_file(T1),
#     steps_per_epoch=8,
#     epochs=80,
#     # validation_split=0.7
#     validation_data=generate_arrays_from_file(T1,4,False),
#     validation_steps=2
# )
# # history=model.fit(generate_arrays_from_file(T1),steps_per_epoch=8,epochs=50)
# loss,accuracy = model.evaluate(generate_arrays_from_file(T1,7,False), verbose=1, steps=35)
# print(loss,accuracy)

epochs=100

# 曲线图
plt.figure(figsize=(12, 6))
# 绘制训练集损失曲线
plt.plot(np.arange(0, epochs), history.history["loss"], label="Training Loss", color='g')
# 绘制验证集损失曲线
plt.plot(np.arange(0, epochs), history.history["val_loss"], label="Validation Loss", color='r')
# 绘制训练集准确率曲线
plt.plot(np.arange(0, epochs), history.history["accuracy"], label="Training Accuracy", color='b')
# 绘制验证集准确率曲线
plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="Validation Accuracy", color='orange')

plt.title("Loss and Accuracy of the Model")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="lower right")

plt.savefig("loss_accuracy_plot.jpg")  # 保存图片

plt.show()  # 显示图像


