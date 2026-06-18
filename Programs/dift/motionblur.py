import os
import torch
import numpy as np
from pathlib import Path
from torchvision.io import read_image
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from PIL import Image


class MotionBlurGenerator:
    def __init__(self, input_folder, output_folder, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # 初始化RAFT模型
        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_large(weights=self.weights, progress=False).to(self.device)
        self.model = self.model.eval()

    def _preprocess_frames(self, img1_batch, img2_batch):
        """统一预处理为模型输入尺寸"""
        img1_batch = F.interpolate(img1_batch, size=(480, 640), mode='bilinear', align_corners=False)
        img2_batch = F.interpolate(img2_batch, size=(480, 640), mode='bilinear', align_corners=False)
        return self.transforms(img1_batch, img2_batch)

    def _load_frame_sequence(self):
        """加载有序图像序列并生成连续帧对"""
        img_files = sorted(self.input_folder.glob("*.png"))
        frames = [read_image(str(f)) for f in img_files]
        return torch.stack(frames).float().div(255)  # [T, C, H, W]

    def _generate_optical_flows(self, frames):
        """生成连续帧之间的光流"""
        flows = []
        for i in range(len(frames) - 1):
            img1, img2 = frames[i:i + 1], frames[i + 1:i + 2]  # 保持批次维度
            img1, img2 = self._preprocess_frames(img1, img2)
            img1=img1.to(self.device)
            img2=img2.to(self.device)

            with torch.no_grad():
                flow = self.model(img1, img2)[-1]  # 取最终迭代结果
            flows.append(flow.cpu())
        return torch.cat(flows)  # [T-1, 2, H, W]

    def _create_oriented_kernel(self, size, sigma, angle):
        """生成方向性高斯模糊核"""
        kernel = torch.zeros((size, size))
        center = size // 2
        x = torch.arange(size) - center
        y = torch.arange(size) - center
        x, y = torch.meshgrid(x, y, indexing='ij')

        # 坐标旋转变换
        x_rot = x * np.cos(angle) - y * np.sin(angle)
        y_rot = x * np.sin(angle) + y * np.cos(angle)

        kernel = torch.exp(-(x_rot ** 2 + y_rot ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    def _apply_dynamic_blur(self, frames, flows, kernel_size=21, sigma_scale=0.05):
        """应用动态模糊处理"""
        blurred_sequence = []
        for i in range(len(frames) - 1):
            # 获取原始帧和对应光流
            img = frames[i].unsqueeze(0)  # [1, C, H, W]
            flow = flows[i]  # [2, H, W]

            # 计算运动参数
            magnitude = torch.sqrt(flow[0]** 2 + flow[1] ** 2)
            angle = torch.atan2(flow[1], flow[0])
            sigma = magnitude.mean() * sigma_scale

            # 生成动态核
            kernel = self._create_oriented_kernel(kernel_size, sigma.item(), angle.mean().item())
            kernel = kernel.view(1, 1, kernel_size, kernel_size).to(img.device)

            # 分通道卷积
            blurred = F.conv2d(img, kernel.repeat(3, 1, 1, 1), padding=kernel_size // 2, groups=3)
            blurred_sequence.append(blurred.squeeze(0))

        # 处理最后一帧
        blurred_sequence.append(frames[-1])
        return torch.stack(blurred_sequence)

    def process_sequence(self):
        """端到端处理流程"""
        # 1. 加载图像序列
        frames = self._load_frame_sequence()

        # 2. 生成光流序列
        flows = self._generate_optical_flows(frames)

        # 3. 应用动态模糊
        blurred_frames = self._apply_dynamic_blur(frames, flows)

        # 4. 保存结果
        for i, frame in enumerate(blurred_frames):
            img_np = frame.permute(1, 2, 0).clamp(0, 1).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            Image.fromarray(img_np).save(self.output_folder / f"frame_{i:04d}_blurred.png")


if __name__ == "__main__":
    # 使用示例
    processor = MotionBlurGenerator(
        input_folder="/home/gzj/test/BrushNetSimple-main/out/blendbeauty/",
        output_folder="/home/gzj/test/BrushNetSimple-main/out/111/"
    )
    processor.process_sequence()