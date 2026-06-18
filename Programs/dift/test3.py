'''
TODO: 绘制特征点


'''


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 加载图像
#src_img = np.array(Image.open('./assets/guitar_cat.jpg').convert('RGB'))
#trg_img = np.array(Image.open('./assets/painting_cat.jpg').convert('RGB'))


def img_with_point(src_img,trg_img,src_pts,trg_pts):
    # 可视化源图像的特征点
    src_img = np.array(src_img)
    trg_img = np.array(trg_img)

    src_img_with_points = src_img.copy()
    for (x, y) in src_pts:
        cv2.circle(src_img_with_points, (int(x), int(y)), 5, (0, 255, 0), -1)  # 绘制绿色圆点

    # 可视化目标图像的特征点
    trg_img_with_points = trg_img.copy()
    for (x, y) in trg_pts:
        cv2.circle(trg_img_with_points, (int(x), int(y)), 5, (0, 255, 0), -1)  # 绘制绿色圆点



    # 保存结果
    cv2.imwrite('./out/src_img_with_points.jpg', cv2.cvtColor(src_img_with_points, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./out/trg_img_with_points.jpg', cv2.cvtColor(trg_img_with_points, cv2.COLOR_RGB2BGR))

def f_match(src_img,trg_img,src_pts,trg_pts):
    src_img = np.array(src_img)
    trg_img = np.array(trg_img)

    # 创建一个空白图像，用于显示匹配结果
    h_src, w_src = src_img.shape[:2]
    h_trg, w_trg = trg_img.shape[:2]
    result = np.zeros((max(h_src, h_trg), w_src + w_trg, 3), dtype=np.uint8)
    result[:h_src, :w_src] = src_img
    result[:h_trg, w_src:w_src+w_trg] = trg_img

    # 绘制匹配的特征点对
    for (x1, y1), (x2, y2) in zip(src_pts, trg_pts):
        cv2.circle(result, (int(x1), int(y1)), 5, (0, 255, 0), -1)  # 在源图像上绘制绿色圆点
        cv2.circle(result, (int(x2 + w_src), int(y2)), 5, (0, 255, 0), -1)  # 在目标图像上绘制绿色圆点
        cv2.line(result, (int(x1), int(y1)), (int(x2 + w_src), int(y2)), (0, 0, 255), 2)  # 绘制红色连线

    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # 保存结果
    #cv2.imwrite('./out/feature_matching.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))