'''
function:使用raft光流预测

'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image
import os

import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from torchvision.io import read_video

import cv2

'''
基于光流数据实现动态模糊
一阶段：
通过raft光流预测对连续帧进行光流预测

'''


# 定义一个函数，用于绘制图像, 但是仅仅是用作网格图直接显示出来
def plot(imgs, **imshow_kwargs):
    plt.rcParams["savefig.bbox"] = "tight"
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

from torchvision.io import read_image
# 原版读取视频并显示
def video_read():

    video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
    video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
    _ = urlretrieve(video_url, video_path)


    #使用Torchvision的read_video读取视频

    frames, _, _ = read_video(str(video_path))
    return frames
    # THWC-> TCHW 时间、通道、高、宽
    frames = frames.permute(0, 3, 1, 2)
    #此处用的是100-101和150-151帧的预测，后续应该将其改为函数的形式方便直接调用来检测任意帧
    img1_batch = torch.stack([frames[100], frames[150]])
    img2_batch = torch.stack([frames[101], frames[151]])

    plot(img1_batch)


from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large


# 预处理图片（调整维度大小以能被8整除）
def preprocess(img1_batch,img2_batch):

    weights = Raft_Large_Weights.DEFAULT  #官方预训练权重
    transforms = weights.transforms()  #配套的数据预处理方法
    img1_batch = F.resize(img1_batch, size=[480, 640], antialias=False)
    img2_batch = F.resize(img2_batch, size=[480, 640], antialias=False)
    print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")
    return transforms(img1_batch, img2_batch)


def load_frames_from_folder(folder_path, frame_key=0):
    """
    从文件夹加载图像帧并生成批次张量
    参数：
        folder_path: 包含有序图像帧的文件夹（如frame_0001.jpg, frame_0002.jpg...）
        frame_pairs: 需要配对的帧索引列表，每个元组表示(img1,img2)的帧号
    返回：
        img1_batch: (N, C, H, W) 的参考帧张量
        img2_batch: (N, C, H, W) 的目标帧张量
    """
    # 获取排序后的图像文件列表
    img_files = sorted(Path(folder_path).glob("*.png"))  # 支持.png等格式

    # 加载指定帧对
    img_pairs = []
    img1 = read_image(str(img_files[frame_key]))  # 形状: (C,H,W)
    img2 = read_image(str(img_files[frame_key+1]))
    img3 = read_image(str(img_files[frame_key+2]))
    img4 = read_image(str(img_files[frame_key+3]))
    img_pairs.append((img1, img2))
    img_pairs.append((img3, img4))

    # 堆叠为批次张量
    img1_batch = torch.stack([pair[0] for pair in img_pairs])  # (N,C,H,W)
    img2_batch = torch.stack([pair[1] for pair in img_pairs])  # (N,C,H,W)
    return preprocess(img1_batch,img2_batch)

# 从文件夹加载图像帧并生成批次张量
def load_frames_from_img(src_img,trg_img):
    # 加载指定帧对
    img_pairs = []
    img_pairs.append((src_img, trg_img))
    # 堆叠为批次张量
    img_batch = torch.stack([pair[0] for pair in img_pairs])  # (N,C,H,W)

    return preprocess(img_batch)

#从视频中读取帧生成批次张量
def read_video_with_opencv(video_path):
    """使用OpenCV读取视频"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()  # BGR格式
        if not ret: break
        frames.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    return torch.stack(frames).permute(0,3,1,2)  # (T,C,H,W)

#从视频（下载url和保存路径）中读取帧生成批次张量
def load_frames_from_video(video_url='',video_path=''):
    video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
    video_path = '/home/gzj/test/BrushNetSimple-main/imgs/basketball.mp4'
    #_ = urlretrieve(video_url, video_path)
    #frames, _, _ = read_video(str(video_path))

    # THWC-> TCHW
    #frames = frames.permute(0, 3, 1, 2)
    frames = read_video_with_opencv(video_path)

    img1_batch = torch.stack([frames[100], frames[150]])
    img2_batch = torch.stack([frames[101], frames[151]])
    return preprocess(img1_batch, img2_batch)

#使用raft模型进行光流预测（主预测模型）
def raft_predict(img1_batch,img2_batch):

    #使用raft估计光流
    #直接使用库中的raft_large模型

    # If you can, run this example on a GPU, it will be a lot faster.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #初始化模型
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    #预测光流
    list_of_flows = model(img1_batch.to(device),img2_batch.to(device))
    print(f"type = {type(list_of_flows)}")
    print(f"length = {len(list_of_flows)} = number of iterations of the model")

    return list_of_flows[-1]


from torchvision.utils import flow_to_image
#将预测到的光流可视化
def plot_flow(predicted_flows, save_path):
    """
    将预测的光流可视化并保存

    参数：
        predicted_flows: RAFT模型输出的光流（可能是元组或张量）
        save_path: 保存路径（目录）
    """
    print(f"dtype = {predicted_flows.dtype}")
    print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
    print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 处理RAFT多阶段输出（取最后阶段）
    if isinstance(predicted_flows, (tuple, list)):
        print('111')
        flows = predicted_flows[-1]  # 形状 (N, 2, H, W)
    else:
        print('222')
        flows = predicted_flows

    # 修正方案：添加幅度归一化
    #max_flow = torch.max(torch.abs(flows))
    #flows = predicted_flows / (max_flow + 1e-6)
    # 转换为RGB图像（自动处理批次）
    flow_imgs = flow_to_image(flows)  # 形状 (N, 3, H, W), 范围[-1,1]
    #flow_imgs = (flow_imgs + 1) / 2  # 转换到[0,1]

    # 保存每帧光流
    for i in range(flow_imgs.size(0)):
        # 转换为PIL图像
        img = flow_imgs[i]  # 获取第i个光流图
        img_pil = F.to_pil_image(img.cpu())

        # 保存文件
        img_pil.save(os.path.join(save_path, f'flow_{i:04d}.png'))

'''
基于光流数据实现动态模糊
二阶段：
通过结合光流的方向/幅度信息与动态高斯核生成，可在保持视觉效果的同时，显著提升运动模糊的真实性

'''



def generate_magnitude_and_direction(flow):
    # 光流张量形状为 [B, 2, H, W]，其中第二个维度为 (u, v) 方向分量
    u = flow[0, 0, :, :]  # 水平方向分量
    v = flow[0, 1, :, :]  # 垂直方向分量
    magnitude = torch.sqrt(u**2 + v**2)  # 运动幅度
    direction = torch.atan2(v, u)       # 运动方向（弧度）
    sigma = magnitude.mean() * 0.1  # 幅度越大，σ越大，模糊越强
    return magnitude, direction, sigma

def oriented_gaussian_kernel(size, sigma, theta):
    # 生成旋转后的坐标网格
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y)

    # 坐标旋转（沿光流方向）
    x_rot = x * torch.cos(theta) - y * torch.sin(theta)
    y_rot = x * torch.sin(theta) + y * torch.cos(theta)

    # 计算高斯权重
    kernel = torch.exp(-(x_rot ** 2 + y_rot ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()




def dynamic_gaussian_blur(image, flow, kernel_size=15):
    # 提取光流参数
    magnitude = flow.norm(dim=1, keepdim=True)
    theta = torch.atan2(flow[:, 1], flow[:, 0])

    _,_,sigma=generate_magnitude_and_direction(flow)
    # 生成动态高斯核
    kernel = oriented_gaussian_kernel(kernel_size, sigma, theta)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

    # 分组卷积（每组通道独立处理）
    blurred = F.conv2d(image, kernel, padding=kernel_size // 2, groups=3)
    return blurred

def gaussion_fussion(flows,images):
    # 假设光流序列为flows，图像序列为images
    blurred_frames = []
    for flow, img in zip(flows, images):
        blurred = dynamic_gaussian_blur(img, flow)
        weight = flow.norm(dim=1)  # 按光流幅度分配权重
        blurred_frames.append(blurred * weight)


if __name__ == "__main__":
    img_path='/home/gzj/test/BrushNetSimple-main/imgs/1/'
    save_path='/home/gzj/test/BrushNetSimple-main/out/'
    img1_batch,img2_batch=load_frames_from_folder(img_path)
    #img1_batch,img2_batch=load_frames_from_video()

    predicted_flows=raft_predict(img1_batch,img2_batch)
    plot_flow(predicted_flows,save_path)

