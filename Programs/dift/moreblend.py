import numpy as np
import cv2


def gaussian_pyramid(img, levels):
    """
    生成高斯金字塔。

    参数:
        img (np.ndarray): 输入图像。
        levels (int): 金字塔的层数。

    返回:
        pyramid (list): 高斯金字塔。
    """
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def laplacian_pyramid(img, levels):
    """
    生成拉普拉斯金字塔。

    参数:
        img (np.ndarray): 输入图像。
        levels (int): 金字塔的层数。

    返回:
        pyramid (list): 拉普拉斯金字塔。
    """
    gaussian = gaussian_pyramid(img, levels)
    pyramid = []
    for i in range(levels - 1):
        upsampled = cv2.pyrUp(gaussian[i + 1], dstsize=gaussian[i].shape[:2][::-1])
        laplacian = cv2.subtract(gaussian[i], upsampled)
        pyramid.append(laplacian)
    pyramid.append(gaussian[-1])
    return pyramid


def blend_pyramids(pyramid1, pyramid2, mask_pyramid):
    """
    融合拉普拉斯金字塔。

    参数:
        pyramid1 (list): 第一个拉普拉斯金字塔。
        pyramid2 (list): 第二个拉普拉斯金字塔。
        mask_pyramid (list): 掩码金字塔。

    返回:
        blended_pyramid (list): 融合后的拉普拉斯金字塔。
    """
    blended_pyramid = []
    for p1, p2, m in zip(pyramid1, pyramid2, mask_pyramid):
        # 将单通道的掩码扩展为三通道
        m = np.expand_dims(m, axis=-1)  # 形状从 (H, W) 变为 (H, W, 1)
        m = np.tile(m, (1, 1, 3))  # 形状从 (H, W, 1) 变为 (H, W, 3)
        blended = p1 * m + p2 * (1 - m)
        blended_pyramid.append(blended)
    return blended_pyramid


def reconstruct_image(pyramid):
    """
    从拉普拉斯金字塔重建图像。

    参数:
        pyramid (list): 拉普拉斯金字塔。

    返回:
        img (np.ndarray): 重建后的图像。
    """
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        img = cv2.pyrUp(img, dstsize=pyramid[i].shape[:2][::-1])
        img = cv2.add(img, pyramid[i])
    return img


def laplacian_blend(src1, src2,levels=5):
    """
    拉普拉斯金字塔融合。

    参数:
        src1 (np.ndarray): 第一个图像。
        src2 (np.ndarray): 第二个图像。
        mask (np.ndarray): 掩码（单通道，值域 [0, 1]）。
        levels (int): 金字塔的层数。

    返回:
        blended_result (np.ndarray): 融合后的图像。
    """
    # 将 src_img 转换为灰度图像（单通道）
    src_gray = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)

    # 创建掩码（单通道）
    mask = np.zeros(src_gray.shape, dtype=np.uint8)  # 单通道
    mask[src_gray > 0] = 255  # 使用灰度图像的有效区域\
    mask = mask.astype(np.float32) / 255  # 归一化到 [0, 1]

    # 生成拉普拉斯金字塔
    pyramid1 = laplacian_pyramid(src1, levels)
    pyramid2 = laplacian_pyramid(src2, levels)

    # 生成掩码金字塔
    mask_pyramid = gaussian_pyramid(mask, levels)

    # 融合拉普拉斯金字塔
    blended_pyramid = blend_pyramids(pyramid1, pyramid2, mask_pyramid)

    # 重建融合后的图像
    blended_result = reconstruct_image(blended_pyramid)
    return blended_result

'''
# 示例：使用拉普拉斯金字塔融合
src1 = cv2.imread("src1.jpg")
src2 = cv2.imread("src2.jpg")
mask = np.zeros(src1.shape[:2], dtype=np.float32)
mask[50:200, 50:200] = 1  # 假设有效区域是一个矩形

# 使用拉普拉斯金字塔融合
blended_result = laplacian_blend(src1, src2, mask)

# 保存结果
cv2.imwrite("blended_result.jpg", blended_result)
'''