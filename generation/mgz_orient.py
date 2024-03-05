import os
import nibabel as nib
import SimpleITK as sitk


# mgz_folder = "D:/871-control-T1/mgz/"
mgz_folder = "D:/871-control-T1/nii/"
output_folder = "D:/871-control-T1/reorient/"


# 遍历文件夹中的所有文件
for filename in os.listdir(mgz_folder):
    # 构建文件的完整路径
    file_path = os.path.join(mgz_folder, filename)

    # 读取文件
    image = sitk.ReadImage(file_path)

    # 获取图像方向
    direction = image.GetDirection()
    origin = image.GetOrigin()
    space = image.GetSpacing()
    # print(direction)

    # 如果需要调整方向，则进行重新定向
    if direction != (1, 0, 0, 0, 1, 0, 0, 0, 1):
        # 将方向设置为标准方向（axial）
        new_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        # resampled = image.SetDirection(new_direction)

        # 构建新的文件路径
        new_file_path = os.path.join(output_folder, "new_" + filename)

        # 保存调整后的文件
        image.SetDirection(new_direction)
        image.SetOrigin(origin)
        image.SetSpacing(space)

        sitk.WriteImage(image,str(new_file_path))
    else:
        print(f"The image {filename} is already in the standard orientation.")


