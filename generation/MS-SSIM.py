import nibabel as nib
from skimage.metrics import structural_similarity as ssim
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import gzip
import tempfile

# 参考图像路径
ref_path = 'D:/施雨欣/深度学习/帕金森/T1/mri/wmsub/wmsub-patient032077_T1w.nii'

# 待评估图像路径
eval_dir = 'D:/施雨欣/深度学习/Results/autoencoder_nii/'

# 用于保存 MS-SSIM 结果的列表
ms_ssim_list = []

# 加载参考图像
ref_img = nib.load(ref_path).get_fdata()
ref_img = ndimage.zoom(ref_img,
                            (200 / ref_img.shape[0], 200 / ref_img.shape[1], 200 / ref_img.shape[2]),
                            order=0)
ref_img = np.asarray(ref_img).reshape((8000000))
# print(ref_img.dtype)
# 循环遍历所有 NII 文件
for filename in os.listdir(eval_dir):
    if filename.endswith('.nii'):
        # 加载待评估图像
        eval_path = os.path.join(eval_dir, filename)
        # print(eval_path)

        eval_img = nib.load(eval_path).get_fdata()
        img_3d_max = np.amax(eval_img)
        eval_img = eval_img / img_3d_max * 255
        eval_img = eval_img / 127.5 - 1
        data = ndimage.zoom(eval_img,
                            (200 / eval_img.shape[0], 200 / eval_img.shape[1], 200 / eval_img.shape[2]),
                            order=0)
        eval_img = np.asarray(data).reshape((8000000))

        # eval_img=eval_img.astype(np.double)
        # print(eval_img.dtype)
        # 计算 MS-SSIM
        ms_ssim = ssim(ref_img, eval_img, data_range=eval_img.max() - eval_img.min())
        # 将负值转换为非负值
        # ms_ssim = (ms_ssim + 1) / 2
        ms_ssim_list.append(ms_ssim)

        print(filename, 'MS-SSIM:', ms_ssim)

# 绘制折线图
plt.plot(ms_ssim_list)

# 添加标题和轴标签
plt.title('MS-SSIM Curve')
plt.xlabel('Image Index')
plt.ylabel('MS-SSIM')

# 显示图表
plt.show()

print(ms_ssim_list)