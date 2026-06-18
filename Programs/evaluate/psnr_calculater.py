'''
todo:该代码用于计算图像集PSNR值和LPIPS值

'''


import cv2
import numpy as np
import os
import torchvision.transforms as transforms
import torchvision.models as models
import lpips
from PIL import Image

def calculate_psnr(img1, img2):
    """计算两帧图像的PSNR值"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:  # 完全相同的图像
        return float('inf')
    max_pixel = 255.0
    return 10 * np.log10((max_pixel** 2) / mse)

def calculate_lpips(img1,img2):
    lpips_model=lpips.LPIPS(net='alex')


    preprocess=transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ])
    # OpenCV BGR 转 RGB，再转 PIL
    img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    image1 = preprocess(img1_pil).unsqueeze(0)
    image2 = preprocess(img2_pil).unsqueeze(0)

    # 使用LPIPS模型计算相似性
    similarity_score = lpips_model(image1, image2)

    return similarity_score.item()



def monitor_frame_differences(folder_path):
    """监控文件夹下连续帧的PSNR差异"""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))])
    if len(files) < 2:
        print("需要至少2帧图像进行计算")
        return

    psnr_values = []
    lpips_values = []
    lpips_first_values=[]
    prev_frame = cv2.imread(os.path.join(folder_path, files[0]))
    #H, W = prev_frame.shape[:2]
    first_frame=prev_frame
    for i in range(1, len(files)):
        curr_frame = cv2.imread(os.path.join(folder_path, files[i]))
        #curr_frame.resize(H, W,3)
        #计算PSNR
        psnr = calculate_psnr(prev_frame, curr_frame)
        psnr_values.append(psnr)
        print(f"帧 {files[i - 1]} 与 {files[i]} 的PSNR: {psnr:.2f} dB")

        #计算LPIPS
        lpips_score = calculate_lpips(prev_frame, curr_frame)
        #计算lpips-first
        lpips_first_score = calculate_lpips(first_frame, curr_frame)
        lpips_values.append(lpips_score)
        lpips_first_values.append(lpips_first_score)
        print(f"帧 {files[i - 1]} 与 {files[i]} 的LPIPS: {lpips_score:.2f}")
        print(f"帧 {files[i - 1]} 与 {files[i]} 的lpips-first: {lpips_first_score:.2f}")

        prev_frame = curr_frame

    # 计算整体统计量
    mean_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values)
    print("\n统计结果：")
    print(f"平均PSNR: {mean_psnr:.2f} dB")
    print(f"PSNR标准差: {std_psnr:.2f} dB")
    print(f"最大差异帧: {np.argmin(psnr_values) + 1} (PSNR={min(psnr_values):.2f} dB)")
    print(f"最小差异帧: {np.argmax(psnr_values) + 1} (PSNR={max(psnr_values):.2f} dB)")

    print("\nLPIPS 统计：")
    print(f"平均LPIPS: {np.mean(lpips_values):.4f}")
    print(f"LPIPS标准差: {np.std(lpips_values):.4f}")
    print(f"最大差异帧（最高LPIPS）: 第 {np.argmax(lpips_values) + 1} 帧 (LPIPS={max(lpips_values):.4f})")
    print(f"最小差异帧（最低LPIPS）: 第 {np.argmin(lpips_values) + 1} 帧 (LPIPS={min(lpips_values):.4f})")

    print("\nlpips-first 统计：")
    print(f"平均lpips-first: {np.mean(lpips_first_values):.4f}")
    print(f"lpips-first标准差: {np.std(lpips_first_values):.4f}")
    print(f"最大差异帧（最高lpips-first）: 第 {np.argmax(lpips_first_values) + 1} 帧 (lpips-first={max(lpips_first_values):.4f})")
    print(f"最小差异帧（最低lpips-first）: 第 {np.argmin(lpips_first_values) + 1} 帧 (lpips-first={min(lpips_first_values):.4f})")

    return psnr_values

if __name__ == "__main__":
    # 使用示例
    folder_path = "/home/gzj/test/BrushNetSimple-main/out/20250523_162557/beauty/"  # 替换为你的帧图像文件夹路径
    #folder_path = "/home/gzj/test/BrushNetSimple-main/out/20250523_162557/blendbeauty/"  # 替换为你的帧图像文件夹路径


    #tokenflow
    #folder_path = "/home/gzj/test/others/TokenFlow-master/out/out.mp4/sd_2.0/in/steps_500/nframes_200/frames/"  # 替换为你的帧图像文件夹路径
    psnr_results = monitor_frame_differences(folder_path)