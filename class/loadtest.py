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
from sklearn.metrics import roc_curve, auc ,confusion_matrix, roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

# path_G = 'G:/eeg/T11/'
# path_G = 'D:/施雨欣/深度学习/帕金森/T11/'
# path_G = 'D:/施雨欣/深度学习/HCP/test_ac/mri/wm/'
# path_G = 'D:/施雨欣/深度学习/Test/mix/'
path_G = 'D:/施雨欣/深度学习/自闭症/871-ants-nii/'
train_data_dir = 'D:/施雨欣/深度学习/Test/871_t/'  # 训练集数据目录路径
val_data_dir = 'D:/施雨欣/深度学习/Test/871_v/'  # 验证集数据目录路径
img_height = img_width = img_depth = 128  # 图像尺寸

adam = Adam(lr=0.00001, epsilon=1e-08)
sgd = SGD(lr=0.0001)

# def read_T1_name_and_path(path):
#
#     name_and_lable=[]
#     for i, j, k in os.walk(path):
#         # print(i,j,k)       #路径名，文件夹，文件
#         if len(k):           #是文件
#             # print(k)
#             for name1 in k:
#                 name = []
#                 path_and_name = i + "/" + name1      #名字路径拼接
#                 # path_and_name = i + name1
#                 name.append(path_and_name)
#                 if name1.startswith("wmsub-control"):
#                     name.append(0)
#                 else:
#                     name.append(1)
#                 name_and_lable.append(name)
#     print(len(name_and_lable))
#     return name_and_lable
def read_T1_name_and_path(path):
    name_and_label = []

    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)  # 获取文件路径
            folder_name = os.path.basename(os.path.dirname(file_path))  # 获取文件所在的文件夹名字

            label = 0 if folder_name == "control" else 1  # 如果文件在名为"control"的文件夹中，标签为0，否则为1
            name_and_label.append([file_path, label])  # 将文件路径和对应的标签添加到列表中

    print(len(name_and_label))
    return name_and_label


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
def generate_arrays_from_file(date_x, label_y, batch_size):
    while True:
        data_label = list(zip(date_x, label_y))
        np.random.shuffle(data_label)
        date_x, label_y = zip(*data_label)

        for i in range(len(date_x)//batch_size):
            batch_x = date_x[i*batch_size:(i+1)*batch_size]
            batch_y = label_y[i*batch_size:(i+1)*batch_size]

            batch_x = np.array(batch_x)
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
# 加载并预处理NIfTI图像数据
def load_nifti_images(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    # data = np.transpose(data, (2, 0, 1))  # 调整维度顺序，使得切片数在第一维
    # data = data / np.max(data)  # 归一化到0-1范围
    data_max = np.amax(data)
    data = data / data_max * 255
    data = ndimage.zoom(data,
                        (128 / data.shape[0], 128 / data.shape[1], 128 / data.shape[2]),
                        order=0)
    # data = np.asarray(data).reshape((200*200*200, 1))
    data = data / 127.5 - 1
    # print(data.shape)
    return data

def preprocess_data(images, labels):
    # 对图像数据进行预处理，例如缩放、裁剪等操作
    preprocessed_images = []
    for image in images:
        # 进行预处理操作
        preprocessed_image = keras.preprocess(image)
        preprocessed_images.append(preprocessed_image)
    return np.array(preprocessed_images), np.array(labels)

# 加载训练集和验证集的图像及对应标签
def load_data(data_dir):
    images = []
    labels = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            label = class_dir
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
                    image = load_nifti_images(file_path)
                    # print(image.shape)
                    images.append(image)
                    labels.append(label)
    print("Total images: ", len(images))
    print("Total labels: ", len(labels))
    return images, labels




# 数据增强
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=10,
#     height_shift_range=0.1,  # 随机垂直平移范围
#     zoom_range=0.1,  # 随机缩放范围
#     horizontal_flip=True,  # 随机水平翻转
#     vertical_flip=True  # 随机垂直翻转
# )
# train_generator = train_datagen.flow(train_images, train_labels, batch_size=1)
#
# validation_generator = ImageDataGenerator(rescale=1./255).flow(val_images, val_labels, batch_size=1)



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
    model.add(Dense(16, activation='tanh'))
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
# train_x, test_x, train_y, test_y = split_date_and_lable(T1)
train_images, train_labels = load_data(train_data_dir)
print(len(train_labels))
val_images, val_labels = load_data(val_data_dir)
# 标签编码
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)
# 评估集
edata=val_images
elabel=val_labels
# edata = test_x
# elabel = test_y
filepath = 'model_at_epoch_21_val_acc_0.7500.h5'
# filepath = 'model_0.8333.h5'
# filepath = 'bzcnn_50.h5'
print(filepath)
model = keras.models.load_model(filepath)
# output = autoencoder_model.predict(input)
loss, accuracy = model.evaluate(generate_arrays_from_file(edata, elabel,173), verbose=1,batch_size=2)
print(len(edata))
predictions = model.predict_generator(generate_arrays_from_file(edata, elabel,173), verbose=1,batch_size=2)
print(elabel)
print(predictions)
# 将预测结果转换为类别标签
predicted_labels = np.argmax(predictions, axis=1)
accuracy = accuracy_score(elabel, predicted_labels)
precision = precision_score(elabel, predicted_labels, average='macro')
recall = recall_score(elabel, predicted_labels, average='macro')
f1 = f1_score(elabel, predicted_labels, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
epochs=50

# # 计算平均准确率和标准差
# mean_val_accuracy = np.mean(mean_val_accuracy)
# val_accuracy_std = np.std(mean_val_accuracy)
#
# # 计算整体混淆矩阵
# confusion_matrix_all = confusion_matrix(all_true_labels, all_predictions)
#
# # 绘制平均准确率
# plt.figure()
# plt.bar(range(len(mean_val_accuracy)), mean_val_accuracy)
# plt.xlabel('Fold')
# plt.ylabel('Mean Validation Accuracy')
# plt.title('Mean Validation Accuracy for each Fold')
# plt.show()
# # 绘制整体混淆矩阵
# plt.figure(figsize=(8, 6))
# im = plt.imshow(confusion_matrix_all, interpolation='nearest', cmap=plt.cm.Blues)
#
# plt.title('Confusion Matrix')
# plt.colorbar(im, fraction=0.046, pad=0.04)
# plt.xticks(range(len(confusion_matrix_all)))
# plt.yticks(range(len(confusion_matrix_all)))
#
# # 在每个方块中添加概率数值
# for i in range(len(confusion_matrix_all)):
#     for j in range(len(confusion_matrix_all)):
#         plt.text(j, i, str(confusion_matrix_all[i, j]), ha="center", va="center", color="white")
#
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
#
# plt.savefig("confusion_matrix_plot_test.png")  # 保存图片
# plt.show()  # 显示图表
#
# # 计算整体 ROC 曲线
# fpr, tpr, _ = roc_curve(all_true_labels, all_predictions)
#
# # 计算 AUC 值
# roc_auc = roc_auc_score(all_true_labels, all_predictions)
#
# # 绘制 ROC 曲线
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.show()
# 曲线图
# plt.figure(figsize=(12, 6))
# # 绘制训练集损失曲线
# plt.plot(np.arange(0, epochs), history.history["loss"], label="Training Loss", color='g')
# # 绘制验证集损失曲线
# plt.plot(np.arange(0, epochs), history.history["val_loss"], label="Validation Loss", color='r')
# # 绘制训练集准确率曲线
# plt.plot(np.arange(0, epochs), history.history["accuracy"], label="Training Accuracy", color='b')
# # 绘制验证集准确率曲线
# plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="Validation Accuracy", color='orange')
#
# plt.title("Loss and Accuracy of the Model")
# plt.xlabel("Epochs")
# plt.ylabel("Loss / Accuracy")
# plt.legend(loc="lower right")
#
# plt.savefig("loss_accuracy_plot_test2.jpg")  # 保存图片
#
# plt.show()  # 显示图像

# 将预测结果转换为具体的类别（0或1）
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.array(elabel)
print("True labels data type:", type(true_labels))
print("Predicted labels data type:", type(predicted_labels))

# 转换为数字类型
true_labels = true_labels.astype(float)
predicted_labels = predicted_labels.astype(float)

# 计算每个类别的预测概率
class_probabilities = predictions[:, 1]  # 假设正例是类别1，取出第二列的概率值

# 计算真阳率和假阳率
fpr, tpr, thresholds = roc_curve(true_labels, class_probabilities)

# 计算AUC值
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线图表
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("roc_curve_plot_test2.png")  # 保存图片
plt.show()  # 显示图表


# 计算混淆矩阵
confusion_matrix = confusion_matrix(true_labels, predicted_labels)
confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

# 绘制混淆矩阵图表
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.colorbar()

# 设置标签
class_labels = ['Class 0', 'Class 1']
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

# 添加数值标注(个数)
# thresh = confusion_matrix.max() / 2.
# for i in range(confusion_matrix.shape[0]):
#     for j in range(confusion_matrix.shape[1]):
#         plt.text(j, i, format(confusion_matrix[i, j], 'd'),
#                  ha="center", va="center",
#                  color="white" if confusion_matrix[i, j] > thresh else "black")
# # 添加文本注释
# for i in range(len(confusion_matrix)):
#     for j in range(len(confusion_matrix)):
#         plt.text(j, i, format(confusion_matrix[i, j], '.3f'),
#                 ha="center", va="center", color="white")
# 添加文本注释
for i in range(len(confusion_matrix)):
    total = np.sum(confusion_matrix[i])  # 求当前维度上的总和
    for j in range(len(confusion_matrix)):
        plt.text(j, i, format(confusion_matrix[i, j] / total, '.3f'),
                ha="center", va="center", color="white")

plt.tight_layout()
plt.savefig("confusion_matrix_plot_test2.png")  # 保存图片
plt.show()  # 显示图表

# model.save('./cnn-5.h5')

