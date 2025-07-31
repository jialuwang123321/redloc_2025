# import os
# import shutil
# from PIL import Image, ImageEnhance

# # 原始路径和目标路径
# source_dir = "/home/transposenet/data/7Scenes/heads/seq-01-origional"
# target_dir = "/home/transposenet/data/7Scenes/heads/seq-01"

# # 创建目标目录（如果不存在）
# os.makedirs(target_dir, exist_ok=True)

# # 遍历原始目录中的所有文件
# for filename in os.listdir(source_dir):
#     source_path = os.path.join(source_dir, filename)
#     target_path = os.path.join(target_dir, filename)
    
#     # 检查文件名中是否包含 "color" 且文件扩展名为 ".png"
#     if "color" in filename and filename.endswith(".png"):
#         # 打开图像
#         with Image.open(source_path) as img:
#             # 调暗亮度
#             enhancer = ImageEnhance.Brightness(img)
#             img_darkened = enhancer.enhance(0.3)  # 亮度大幅降低

#             # 降低对比度
#             enhancer = ImageEnhance.Contrast(img_darkened)
#             img_darkened = enhancer.enhance(0.5)  # 对比度降低

#             # 降低饱和度
#             enhancer = ImageEnhance.Color(img_darkened)
#             img_darkened = enhancer.enhance(0.2)  # 饱和度降低
            
#             # 保存图像到目标路径
#             img_darkened.save(target_path)
#     else:
#         # 直接复制不需要处理的文件
#         shutil.copy2(source_path, target_path)

# print("文件复制和夜间效果调整完成。")


import os
import shutil
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# 原始路径和目标路径
# source_dir = "/home/wjl/AnyLoc/demo/data/7Scenes/heads/seq-01-3imgs"
# target_dir = "/home/wjl/AnyLoc/demo/data/7Scenes/heads/seq-01-3imgs-aug"

# source_dir = "/home/wjl/AnyLoc/demo/data/7Scenes/heads/seq-01"
# target_dir = "/home/wjl/AnyLoc/demo/data/7Scenes/heads/seq-01-aug"


source_dir = "/home/transposenet/data/7Scenes/heads/seq-01" #stairs14, chess 35, office 2679 , pumpkin 1,7, redkitchen 3,4,6,12,14)
target_dir = "/home/transposenet/data/7Scenes/heads/seq-01-aug"

print('source_dir = ',source_dir)
print('target_dir = ',target_dir)

# 创建目标目录（如果不存在）
os.makedirs(target_dir, exist_ok=True)

def add_noise(img, mean=0, std=10):
    """添加高斯噪声到图像"""
    # 将图像转换为numpy数组
    np_img = np.array(img)

    # 生成高斯噪声
    noise = np.random.normal(mean, std, np_img.shape).astype(np.uint8)

    # 将噪声添加到图像
    noisy_img = np_img + noise
    
    # 将结果转换回PIL图像
    noisy_img = Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))
    
    return noisy_img

# 遍历原始目录中的所有文件
for filename in os.listdir(source_dir):
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)
    
    # 检查文件名中是否包含 "color" 且文件扩展名为 ".png"
    if "color" in filename and filename.endswith(".png"):
        # 打开图像
        with Image.open(source_path) as img:
            # 调暗亮度
            enhancer = ImageEnhance.Brightness(img)
            img_darkened = enhancer.enhance(0.3)  # 亮度大幅降低

            # 降低对比度
            enhancer = ImageEnhance.Contrast(img_darkened)
            img_darkened = enhancer.enhance(0.5)  # 对比度降低

            # 降低饱和度
            enhancer = ImageEnhance.Color(img_darkened)
            img_darkened = enhancer.enhance(0.2)  # 饱和度降低
            
            # 添加高斯噪点
            img_noisy = add_noise(img_darkened)

            # 添加模糊效果 (可选)
            img_blurred = img_noisy.filter(ImageFilter.GaussianBlur(radius=2))  # 可以调整radius值增加模糊程度
            
            # 保存图像到目标路径
            img_blurred.save(target_path)
    else:
        # 直接复制不需要处理的文件
        shutil.copy2(source_path, target_path)

print("文件复制、夜间效果调整和噪点、模糊处理完成。")
print('source_dir = ',source_dir)
print('target_dir = ',target_dir)