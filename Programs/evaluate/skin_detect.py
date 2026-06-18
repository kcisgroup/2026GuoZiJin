import cv2
import os
from datetime import datetime


def generate_skin_mask(img_path):
    """生成皮肤二值掩膜"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法加载图像: {img_path}")

    # YCrCb颜色空间转换
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    cr = ycrcb[:, :, 1]  # 提取Cr通道

    # 高斯模糊+Otsu阈值
    cr_blur = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin_mask = cv2.threshold(cr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return skin_mask


def save_skin_mask(input_path, output_dir=None):
    """仅生成并保存皮肤二值掩膜
    :param input_path: 输入图片路径
    :param output_dir: 输出目录（默认为输入文件同级目录）
    :return: 保存的mask文件路径
    """
    # 规范化路径处理
    input_path = os.path.normpath(input_path)
    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 安全加载图像
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法加载图像: {input_path}")

    try:
        # YCrCb颜色空间转换
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        cr = ycrcb[:, :, 1]  # 提取Cr通道

        # 高斯模糊+Otsu阈值
        cr_blur = cv2.GaussianBlur(cr, (5, 5), 0)
        _, skin_mask = cv2.threshold(cr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_skinmask.png")

        # 保存二值掩膜（PNG格式无损保存）
        cv2.imwrite(output_path, skin_mask)

        #return skin_mask  #作为函数体传递使用
        return output_path   #作为本方法中实验使用

    except Exception as e:
        raise RuntimeError(f"皮肤掩膜生成失败: {str(e)}")


if __name__ == "__main__":
    # 配置路径（使用原始字符串避免转义问题）
    input_image = r'/home/gzj/test/BrushNetSimple-main/imgs/in/111.png'
    output_folder = r'/home/gzj/test/BrushNetSimple-main/out/skin_mask/'

    try:
        # 前置检查
        if not os.path.exists(input_image):
            raise FileNotFoundError(f"输入文件不存在: {input_image}")
        if not input_image.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError("仅支持.png/.jpg/.jpeg格式")

        # 执行处理
        mask_path = save_skin_mask(input_image, output_folder)
        print(f"皮肤掩膜已保存到: {mask_path}")

    except Exception as e:
        print(f"错误: {str(e)}")