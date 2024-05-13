import os
from PIL import Image
import numpy as np
from tqdm import tqdm  

def load_images(image_directory):
    images = []
    for filename in tqdm(os.listdir(image_directory)):
        if filename.endswith(".jpg"):  # 根据需要调整文件类型
            img = Image.open(os.path.join(image_directory, filename))
            img = img.resize((64, 64))  # 可选：统一图像大小
            img = np.array(img)  # 将 PIL 图像转换为 NumPy 数组
            images.append(img)
    return np.array(images)  # 返回图像数据的 NumPy 数组

# 用实际的图像目录路径替换 'path_to_images'
images = load_images('data/train')

# 计算整体均值
mean = np.mean(images, axis=(0, 1, 2))

# 计算整体方差
std = np.std(images, axis=(0, 1, 2))

print("Mean of the image dataset:", (mean/ 255).round(4))   # [0.4093, 0.4471, 0.3405]
print("Variance of the image dataset:", (std / 255).round(4))   # [0.1914, 0.1762, 0.1936]