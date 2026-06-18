'''
todo: 绘制mask


'''


import json
import numpy as np
import cv2
from PIL import Image

# 加载 JSON 文件
json_path = './imgs/mask.json'  # 替换为你的 JSON 文件路径
with open(json_path, 'r') as f:
    data = json.load(f)

# 提取图像尺寸
width = data['info']['width']
height = data['info']['height']

# 创建一个空白掩码
mask = np.zeros((height, width), dtype=np.uint8)

# 提取 segmentation 数据
segmentation = data['objects'][0]['segmentation']

# 将 segmentation 数据转换为 NumPy 数组
polygon = np.array(segmentation, dtype=np.int32)

# 在掩码上绘制多边形
cv2.fillPoly(mask, [polygon], color=255)

# 保存掩码
mask_image = Image.fromarray(mask)
mask_image.save('./out/mask.png')

# 显示掩码
#mask_image.show()