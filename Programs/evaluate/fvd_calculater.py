import os
import cv2
import torch
import torchvision.transforms as T
from torch_fidelity import calculate_metrics
from PIL import Image
from tempfile import TemporaryDirectory

def save_temp_images(frames, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    for i, img in enumerate(frames):
        path = os.path.join(temp_dir, f"{i:04d}.png")
        cv2.imwrite(path, img)

def extract_frames_from_folder(frame_folder, resize=(256, 256)):
    frame_paths = sorted([
        os.path.join(frame_folder, f) for f in os.listdir(frame_folder)
        if f.lower().endswith(('.png', '.jpg'))
    ])
    frames = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is not None:
            if resize:
                img = cv2.resize(img, resize)
            frames.append(img)
    return frames

def compute_framewise_fid(frames, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    计算连续帧之间的FID，并取平均作为平滑度评分
    """
    if len(frames) < 2:
        raise ValueError("帧数不足，无法评估")

    fids = []
    for i in range(len(frames) - 1):
        with TemporaryDirectory() as tmp1, TemporaryDirectory() as tmp2:
            save_temp_images([frames[i]], tmp1)
            save_temp_images([frames[i + 1]], tmp2)

            metrics = calculate_metrics(
                input1=tmp1,
                input2=tmp2,
                cuda=torch.cuda.is_available(),
                isc=False,
                fid=True,
                kid=False
            )
            fids.append(metrics['frechet_inception_distance'])

    return sum(fids) / len(fids)

if __name__ == "__main__":
    frame_folder = "/home/gzj/test/dataset/refer_result/mvfusion_our/vfhq_8/blendbeauty/"
    frames = extract_frames_from_folder(frame_folder)
    avg_fid = compute_framewise_fid(frames)
    print(f"帧间 FID 平均值（越低越平滑）: {avg_fid:.3f}")

