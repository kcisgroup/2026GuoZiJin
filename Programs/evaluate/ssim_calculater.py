'''
1.LMD:用于评估面部图像生成质量


2.SSIM:用于衡量两幅图像之间相似度的指标
SSIM的值范围通常在 -1 到 1 之间，值越接近1表示两幅图像越相似，质量越高

3.FID:用于评估生成模型性能，测量生成图像与真实图像分布之间的差异


'''

import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage import io, img_as_float
import torch
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from tqdm import tqdm
import mediapipe as mp  # 用于自动提取特征点


# ==================== 工具函数 ====================
def extract_landmarks(image):
    """使用MediaPipe提取面部特征点（需提前安装：pip install mediapipe）"""
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = []
    if results.multi_face_landmarks:
        for landmark in results.multi_face_landmarks[0].landmark:
            x = landmark.x * image.shape[1]
            y = landmark.y * image.shape[0]
            landmarks.append((x, y))
    return landmarks


def calculate_lmd(landmarks1, landmarks2):
    """计算两组特征点的平均欧氏距离"""
    assert len(landmarks1) == len(landmarks2), "特征点数量不一致"
    return np.mean([np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    for (x1, y1), (x2, y2) in zip(landmarks1, landmarks2)])


# ==================== 核心评估逻辑 ====================
def evaluate_adjacent_frames(folder_path):
    """评估相邻帧的LMD、SSIM和FID差异"""
    # 1. 准备文件列表
    files = sorted([f for f in os.listdir(folder_path)
                    if f.endswith(('.png', '.jpg', '.jpeg'))])
    if len(files) < 2:
        raise ValueError("至少需要2帧图像")

    # 2. 初始化结果存储
    metrics = {
        'lmd': [],
        'ssim': [],
        'fid': []
    }

    # 3. 初始化FID模型（仅在需要时加载）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fid_model = inception_v3(pretrained=True, aux_logits=True).to(device).eval()

    # 4. 逐帧处理
    prev_img = cv2.imread(os.path.join(folder_path, files[0]))
    prev_landmarks = extract_landmarks(prev_img)

    for i in tqdm(range(1, len(files))):
        curr_img = cv2.imread(os.path.join(folder_path, files[i]))
        curr_landmarks = extract_landmarks(curr_img)

        # 计算LMD
        if prev_landmarks and curr_landmarks:
            metrics['lmd'].append(calculate_lmd(prev_landmarks, curr_landmarks))

        # 计算SSIM
        gray_prev = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        metrics['ssim'].append(ssim(gray_prev, gray_curr))

        # 计算FID（需要GPU）
        if device.type == 'cuda':
            fid = calculate_fid_pair(fid_model, device, prev_img, curr_img)
            metrics['fid'].append(fid)

        prev_img = curr_img
        prev_landmarks = curr_landmarks

    # 5. 汇总结果
    results = {}
    for k, v in metrics.items():
        if v:  # 只返回有数据的指标
            results[f'{k}_mean'] = np.mean(v)
            results[f'{k}_std'] = np.std(v)
    return results


def calculate_fid_pair(model, device, img1, img2):
    """修正后的FID计算函数"""

    def preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 确保RGB格式
        img = cv2.resize(img, (299, 299))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = (img * 255 - 128) / 128  # InceptionV3标准化
        return img.unsqueeze(0).to(device)

    with torch.no_grad():
        # 获取2048维特征向量
        feat1 = model(preprocess(img1)).cpu().numpy().reshape(1, -1)
        feat2 = model(preprocess(img2)).cpu().numpy().reshape(1, -1)

    # 确保特征矩阵形状正确 (1, 2048)
    if feat1.shape[0] != 1 or feat2.shape[0] != 1:
        return float('nan')

    # 计算统计量（添加微小值防止奇异矩阵）
    eps = 1e-6
    mu1, sigma1 = feat1.mean(0), np.cov(feat1, rowvar=False) + eps * np.eye(feat1.shape[1])
    mu2, sigma2 = feat2.mean(0), np.cov(feat2, rowvar=False) + eps * np.eye(feat2.shape[1])

    # 计算FID距离
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)  # 处理复数结果
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 输入帧文件夹路径
    #frame_folder = "/home/gzj/test/others/TokenFlow-master/out/out.mp4/sd_2.0/in/steps_500/nframes_200/frames/"

    #beautygan
    frame_folder = "/home/gzj/test/BrushNetSimple-main/out/20250401_191118/tf/"

    # 运行评估
    results = evaluate_adjacent_frames(frame_folder)

    # 打印结果
    print("\n===== 相邻帧评估结果 =====")
    for k, v in results.items():
        if 'mean' in k:
            print(f"{k.upper():<10}: {v:.4f} ± {results[k.replace('mean', 'std')]:.4f}")