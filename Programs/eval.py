import os

from evaluate.psnr_calculater import monitor_frame_differences

'''
#ours
# 使用示例
folder_path1 = "/home/gzj/test/BrushNetSimple-main/out/blendbeauty/"  # 替换为你的帧图像文件夹路径
psnr_results = monitor_frame_differences(folder_path1)

folder_path2 = "/home/gzj/test/BrushNetSimple-main/out/beauty/"  # 替换为你的帧图像文件夹路径
psnr_results = monitor_frame_differences(folder_path2)
'''

#beautygan
folder_path1 = "/home/gzj/test/others/BeautyGAN/out/2/makeup4/"  # 替换为你的帧图像文件夹路径
psnr_results = monitor_frame_differences(folder_path1)
