import json
import os
import time

import keras
import matplotlib
import numpy as np
import nibabel as nib
from tensorflow.keras import optimizers
from nibabel.viewers import OrthoSlicer3D
from PIL import Image
import csv
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Flatten, Reshape ,ReLU

from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.convolutional import AveragePooling3D
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy import ndimage

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import load_model

path_G = 'D:/施雨欣/深度学习/帕金森/T11/'
adam = optimizers.Adam(lr=0.0001)

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

# T1 = read_T1_name_and_path(path_G)
# print(T1)

def T1_file_to_array(file_path):

    # file_path = path + name
    img = nib.load(file_path)
    img = img.get_fdata()
    img_3d_max = np.amax(img)
    img = img / img_3d_max * 255

    data = ndimage.zoom(img,
                        (200/img.shape[0],200/img.shape[1],200/img.shape[2]),
                        order=0)
    # data = np.asarray(data).reshape((200*200*200, 1))
    data = data / 127.5 - 1

    # OrthoSlicer3D(data).show()
    return data

# T1_file_to_array(T1[1])


def split_date_and_lable(T1_date,
                         test_size_val=0.8,
                         random_state_val=5,
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
def generate_arrays_from_file( date, batch_size=4 ,is_train=True):
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

def cnn_model_generate(sum = 0,again = False):

    if not again:

        model = Sequential()
        model.add(Conv3D(8, (3, 3, 3), strides=(1, 1, 1), input_shape=( 200, 200, 200, 1), padding='same', activation='relu'))
        model.add(Conv3D(8, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
        model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))

        model.add(Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
        model.add(Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
        model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))

        model.add(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
        model.add(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
        model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))

        model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
        model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
        model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))

        model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
        model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
        model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))

        model.add(Flatten())
        model.add(Dense(4000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        model.summary()

        model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])

    else:
        model = load_model('./model_save/'+str(sum )+'.h5')

    return model,sum + 1

# json文件写入
def save_json(data,sum):

    with open('./model_save/'+str(sum)+'.json', 'w') as json_file:
        json.dump(data, json_file)


# 载入数据
def load_data(sum):

    try:
        with open('./model_save/'+str(sum)+'.json','r') as json_file:
            data = json.load(json_file)
            print('data load success!')
    except:
        print('data load failed!')
    return data

# 读取数据
def get_data(data):
    # 训练集损失数据
    loss = data['loss']
    np_loss = np.array(loss)
    y_loss = np_loss
    x_epoch = np.arange(len(loss))

    # 训练集准确度数据
    acc = data['accuracy']
    print(acc)
    np_acc = np.array(acc)
    y_acc = np_acc
    # x_epoch = np.arange(30)

    # # 测试集损失数据
    # val_loss = data['val_loss']
    # np_val_loss = np.array(val_loss)
    # y_val_loss = np_val_loss
    # x_epoch = np.arange(30)
    #
    # # 测试集准确度数据
    # val_acc = data['val_accuracy']
    # np_val_acc = np.array(val_acc)
    # y_val_acc = np_val_acc
    # x_epoch = np.arange(30)

    return y_loss,y_acc,x_epoch

def main(sum):
    data = load_data(sum)
    y_loss,y_acc,x_epoch = get_data(data)

    plt.figure(figsize=(10,4))
    # loss图像
    plt.subplot(2,2,1)
    plt.plot(x_epoch,y_loss,label='loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # acc图像
    plt.subplot(2,2,2)
    plt.plot(x_epoch,y_acc,label='acc',color='red')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    # plt.ylim(0.9,1)

    # # val_loss图像
    # plt.subplot(2,2,3)
    # plt.plot(x_epoch,y_val_loss,label='val_loss')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('val_loss')
    #
    # # val_acc图像
    # plt.subplot(2,2,4)
    # plt.plot(x_epoch,y_val_acc,label='val_acc',color='red')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('val_acc')
    # plt.ylim(0.9,1)
    plt.savefig('./plt/'+str(sum)+'.jpg')
    # plt.show()

def write_csv(loss,acc,epoch):
    path  = "./model_save/val_acc.csv"
    nowtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [nowtime,loss,acc,epoch]
        csv_write.writerow(data_row)
        f.close()


T1 = read_T1_name_and_path(path_G)

train_sum = 1
sum = 1
while train_sum > 0:

    model, sum = cnn_model_generate(sum ,again = True)

    history=model.fit_generator(generate_arrays_from_file(T1),steps_per_epoch=7,epochs=50)

    # 保存训练结果
    data = history.history
    save_json(data,sum)
    main(sum)

    loss,accuracy = model.evaluate(generate_arrays_from_file(T1,4,False), verbose=1, steps=32)
    print(loss,accuracy)

    # write_csv(loss, accuracy, sum)
    # model.save('./model_save/'+str(sum)+'.h5')

    train_sum = train_sum - 1
