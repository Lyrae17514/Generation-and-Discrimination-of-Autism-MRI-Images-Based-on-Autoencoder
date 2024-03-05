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

from tensorflow.keras.callbacks import ReduceLROnPlateau,LearningRateScheduler
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
from sklearn.metrics import roc_curve, auc ,confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, KFold



# path_G = 'G:/eeg/T11/'
path_G = 'D:/施雨欣/深度学习/帕金森/T11/'
# path_G = 'D:/施雨欣/深度学习/HCP/test_ac/mri/wm/'
# path_G = 'D:/施雨欣/深度学习/Test/mix/'

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
                # path_and_name = i + name1
                name.append(path_and_name)
                if name1.startswith("wmsub-control"):
                    name.append(0)
                else:
                    name.append(1)
                name_and_lable.append(name)
    print(len(name_and_lable))
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
                         test_size_val=0.2,
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


train_x, test_x, train_y, test_y = split_date_and_lable(T1)
# print(len(train_x))
def generate_arrays_from_file(date_x, label_y, batch_size=16):
    count = 0
    while True:
        if count * batch_size >= len(date_x):
            batch_x = date_x[(count - 1) * batch_size:]
            batch_y = label_y[(count - 1) * batch_size:]
            count = 0
        else:
            batch_x = date_x[count * batch_size:(count + 1) * batch_size]
            batch_y = label_y[count * batch_size:(count + 1) * batch_size]

        count += 1

        batch_x = np.array([T1_file_to_array(img_path) for img_path in batch_x])
        batch_x = np.expand_dims(batch_x, axis=4)
        batch_y = np.array(batch_y).astype(np.float32)
        batch_y = keras.utils.to_categorical(batch_y)

        yield batch_x, batch_y
# def generate_arrays_from_file(date, label, batch_size=12):
#     count = 0
#     indices = np.arange(len(date))  # 创建索引数组
#
#     while True:
#         if count * batch_size >= len(date):
#             batch_indices = indices[(count - 1) * batch_size:]
#             count = 0
#         else:
#             batch_indices = indices[count * batch_size:(count + 1) * batch_size]
#
#         count += 1
#
#         batch_x = date[batch_indices]
#         batch_y = label[batch_indices]
#
#         batch_x = np.array([T1_file_to_array(img_path) for img_path in batch_x])
#         batch_x = np.expand_dims(batch_x, axis=4)
#         batch_y = np.array(batch_y).astype(np.float32)
#         batch_y = keras.utils.to_categorical(batch_y)
#
#         yield batch_x, batch_y


# generate_arrays_from_file(T1,5,False)
# print(x,y)



def build_model():
    model = Sequential()
    model.add(Conv3D(2, (3, 3, 3), strides=(1, 1, 1),input_shape=(128, 128, 128, 1), padding='same', activation='relu'))
    model.add(Conv3D(8, (3, 3, 3),strides=2,padding='same', activation='relu'))
    model.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu'))

    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))

    model.add(Conv3D(16, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
    model.add(Conv3D(16, (3, 3, 3), strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))

    model.add(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
    model.add(Conv3D(32, (3, 3, 3), strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))

    model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'))
    model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv3D(128, 3,strides=(1, 1, 1), padding='same', activation='relu'))
    model.add(Conv3D(128, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=1))
    model.add(Conv3D(128, 2, padding='same', activation='relu'))
    model.add(Conv3D(128, 2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling3D(pool_size=(1, 1, 1), strides=1))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])
    return model


# 执行交叉验证并记录性能
mean_val_accuracy = []
val_accuracy_std = []
all_predictions = []
all_true_labels = []


T1 = read_T1_name_and_path(path_G)
train_x, test_x, train_y, test_y = split_date_and_lable(T1)
print(train_x)

# 建立模型
model = build_model()
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.9,patience=5,min_lr=0.0001)
# 设置交叉验证
kf = KFold(n_splits=2, shuffle=True, random_state=42)
fold = 1
for train_index, val_index in kf.split(train_x):
    print(f"Fold: {fold}")

    # 训练集和验证集划分
    train_x_fold, val_x_fold = train_x[train_index], train_x[val_index]
    train_y_fold, val_y_fold = train_y[train_index], train_y[val_index]

    # 将训练集和验证集传入生成器
    train_generator = generate_arrays_from_file(train_x_fold, train_y_fold)
    val_generator = generate_arrays_from_file(val_x_fold, val_y_fold)

    # 训练模型
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=4,
        epochs=10,
        validation_data=val_generator,
        validation_steps=1,
        callbacks=[reduce_lr]
    )

    # 评估模型
    loss, accuracy = model.evaluate(generate_arrays_from_file(val_x_fold, val_y_fold), verbose=1, steps=12)

    # # 模型评估
    # predictions = model.predict(val_x_fold)
    #
    # # 计算准确率和混淆矩阵
    # mean_val_accuracy.append(accuracy)
    # all_predictions.extend(np.argmax(predictions, axis=1))
    # all_true_labels.extend(val_y_fold)

    fold += 1

# 最后，在测试集上评估模型
loss, accuracy = model.evaluate(generate_arrays_from_file(test_x, test_y), verbose=1, steps=12)
print(loss,accuracy)
# 生成预测结果
# predictions = model.predict(generate_arrays_from_file(test_x, test_y), steps=1)


epochs=10

# 计算平均准确率和标准差
mean_val_accuracy = np.mean(mean_val_accuracy)
val_accuracy_std = np.std(mean_val_accuracy)

# 计算整体混淆矩阵
confusion_matrix_all = confusion_matrix(all_true_labels, all_predictions)

# 绘制平均准确率
plt.figure()
plt.bar(range(len(mean_val_accuracy)), mean_val_accuracy)
plt.xlabel('Fold')
plt.ylabel('Mean Validation Accuracy')
plt.title('Mean Validation Accuracy for each Fold')
plt.show()
# 绘制整体混淆矩阵
plt.figure(figsize=(8, 6))
im = plt.imshow(confusion_matrix_all, interpolation='nearest', cmap=plt.cm.Blues)

plt.title('Confusion Matrix')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(confusion_matrix_all)))
plt.yticks(range(len(confusion_matrix_all)))

# 在每个方块中添加概率数值
for i in range(len(confusion_matrix_all)):
    for j in range(len(confusion_matrix_all)):
        plt.text(j, i, str(confusion_matrix_all[i, j]), ha="center", va="center", color="white")

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.savefig("confusion_matrix_plot_test.png")  # 保存图片
plt.show()  # 显示图表

# 计算整体 ROC 曲线
fpr, tpr, _ = roc_curve(all_true_labels, all_predictions)

# 计算 AUC 值
roc_auc = roc_auc_score(all_true_labels, all_predictions)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# if accuracy > 0.7 :
#     # 构建自定义的文件名
#     filename = f'model_{accuracy}.h5'
#
#     # 保存模型
#     model.save(filename)
# model.save('./cnn-5.h5')

