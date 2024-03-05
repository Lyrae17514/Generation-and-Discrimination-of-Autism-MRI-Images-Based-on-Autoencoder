import nibabel as nib
import numpy as np
import imageio
import os

def read_niifile(niifile):  # 读取niifile文件
    img = nib.load(niifile)  # 下载niifile文件（其实是提取文件）
    img_fdata = img.get_fdata()  # 获取niifile数据
    img90 = np.rot90(img_fdata) #旋转90度
    #return img_fdata
    return img90

def save_fig(file):  # 保存为图片
    fdata = read_niifile(file)  # 调用上面的函数，获得数据
    (y, x, z) = fdata.shape  # 获得数据shape信息：（长，宽，维度-切片数量）
    # (z,x , y) = fdata.shape
    for k in range(z):
        # silce = fdata[:,k, : ]
        # silce = fdata[:, :, k]
        silce = fdata[k, :, :]  # 三个位置表示三个不同角度的切片
        imageio.imwrite(os.path.join(savepicdir, '{}.jpeg'.format(k)), silce)
        # 将切片信息保存为jpg格式

#print(list(list_nii))
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f


# base ='D:/施雨欣/深度学习/帕金森/T1/1/'
base ='D:/施雨欣/深度学习/Test/c/' # nii文件的路径
# base ='D:/施雨欣/深度学习/帕金森/T1/mri/wmsub/' # nii文件的路径
# output = 'D:/施雨欣/深度学习/帕金森/T1/T1/img/' # 保存png的路径
output = 'D:/施雨欣/深度学习/Test/nii_to_img' # 保存png的路径
for i in findAllFile(base):
    dir = os.path.join(base,i)
    savepicdir = (os.path.join(output,i))
    os.mkdir(savepicdir)
    save_fig(dir)
