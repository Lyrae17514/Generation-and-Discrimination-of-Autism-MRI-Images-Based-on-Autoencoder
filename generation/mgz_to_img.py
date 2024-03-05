import nibabel as nib
import numpy as np
import imageio
import os

# 设置输入文件夹路径
mgz_folder = "D:/施雨欣/深度学习/自闭症/871-control-T1/ASD"

# 设置输出文件夹路径
output_folder = "D:/施雨欣/深度学习/自闭症/871-control-T1/img_ASD"

start_slice = 138
end_slice = 141


def save_slices_from_mgz(mgz_folder, start_slice, end_slice, output_folder):
    # Check whether the output_folder exists, otherwise create it
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Loop over all files in the folder
    for filename in os.listdir(mgz_folder):
        if filename.endswith(".mgz"):
            # Load mgz file
            img = nib.load(os.path.join(mgz_folder, filename))
            image_data = img.get_fdata()

            # Loop over slices
            for i in range(start_slice, end_slice + 1):
                # Get slice, normalize and convert to uint8
                slice_data = image_data[:, i, :]
                slice_data = (slice_data / slice_data.max()) * 255.0
                img_slice = slice_data.astype('uint8')

                # Save slice as image
                imageio.imwrite(os.path.join(output_folder, f"{filename}_{i}.png"), img_slice)


save_slices_from_mgz(mgz_folder, start_slice, end_slice, output_folder)