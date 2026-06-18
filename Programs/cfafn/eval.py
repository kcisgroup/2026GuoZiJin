import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from cfafn.TemporalFeatureFuser import BeautyRefiner
import torch.nn.functional as F
import os
import cv2

def load_model(model_path):
    model = BeautyRefiner().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

#由于模型保存时，使用了 DataParallel，所以保存下的模型参数前缀都带有module
def load_model_modele(model_path):
    model = BeautyRefiner().cuda()  # 你定义的模型结构
    state_dict = torch.load(model_path)

    # 如果是多卡训练保存的模型（带 module.）
    if any(k.startswith("module.") for k in state_dict.keys()):
        # 去除 'module.' 前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()
    return model

def inference_one_sequence(model, image_paths, mask_paths, image_size=(256, 256)):
    """
    image_paths, mask_paths: 长度为 T 的图像路径序列
    """
    assert len(image_paths) == len(mask_paths)
    center_offset = T // 2
    transform_img = transforms.Compose([
        transforms.Resize(image_size), transforms.ToTensor()
    ])
    transform_mask = transforms.Compose([
        transforms.Resize(image_size), transforms.ToTensor()
    ])

    # 预处理全部帧
    all_imgs  = [transform_img(Image.open(p).convert('RGB')) for p in image_paths]
    all_masks = [transform_mask(Image.open(p).convert('L')) for p in mask_paths]

    padded_imgs  = [all_imgs[0]] * center_offset + all_imgs + [all_imgs[-1]] * center_offset
    padded_masks = [all_masks[0]] * center_offset + all_masks + [all_masks[-1]] * center_offset

    output_frames = []

    for i in range(len(image_paths)):
        clip = torch.stack(padded_imgs[i:i+T])   # [T, 3, H, W]
        masks = torch.stack(padded_masks[i:i+T]) # [T, 1, H, W]

        clip  = clip.unsqueeze(0).cuda()
        masks = masks.unsqueeze(0).cuda()

        with torch.no_grad():
            pred = model(clip, masks).squeeze(0).clamp(0, 1).cpu()
        output_frames.append(transforms.ToPILImage()(pred))

    return output_frames  # [PIL.Image, PIL.Image, ...]

#添加高斯权重生成函数
def get_gaussian_weight(patch_size, sigma=1.0):
    ph, pw = patch_size
    y = torch.linspace(-1, 1, ph).view(ph, 1)
    x = torch.linspace(-1, 1, pw).view(1, pw)
    weight = torch.exp(-(x**2 + y**2) / (2 * sigma ** 2))
    weight = weight / weight.max()
    return weight

#膨胀mask
def dilate_soft_mask(mask_tensor, kernel_size=15):
    mask_np = mask_tensor.squeeze(0).numpy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    blurred = cv2.GaussianBlur(dilated.astype(np.float32), (kernel_size, kernel_size), 0)
    return torch.from_numpy(blurred).unsqueeze(0)

def inference_one_sequence_patchwise(model, image_paths, mask_paths, patch_size=(256, 256), stride=(256, 256), T=3):
    assert len(image_paths) == len(mask_paths)
    center_offset = T // 2

    transform_img = transforms.ToTensor()
    transform_mask = transforms.ToTensor()

    output_frames = []
    progress_bar = tqdm(total=600, ncols=200, dynamic_ncols=True)
    for i in range(len(image_paths)):
        # 读取中心帧及其前后 T//2 帧
        clip_paths = get_padded_sequence(image_paths, i, T)
        mask_paths_clip = get_padded_sequence(mask_paths, i, T)

        imgs = [transform_img(Image.open(p).convert("RGB")) for p in clip_paths]
        masks = [transform_mask(Image.open(p).convert("L")) for p in mask_paths_clip]

        _, H, W = imgs[0].shape
        ph, pw = patch_size
        sh, sw = stride

        # 输出图像初始化
        output = torch.zeros(3, H, W)
        counter = torch.zeros(1, H, W)

        # 预生成高斯权重 mask
        gaussian_weight = get_gaussian_weight((ph, pw)).unsqueeze(0)  # [1, ph, pw]

        for top in range(0, H, sh):
            for left in range(0, W, sw):
                bottom = min(top + ph, H)
                right = min(left + pw, W)

                # 裁剪 patch
                patch_imgs = [img[:, top:bottom, left:right] for img in imgs]
                patch_masks = [msk[:, top:bottom, left:right] for msk in masks]

                # 对 patch 填充为 patch_size
                patch_imgs = [F.pad(p, (0, pw - p.shape[2], 0, ph - p.shape[1])) for p in patch_imgs]
                patch_masks = [F.pad(m, (0, pw - m.shape[2], 0, ph - m.shape[1])) for m in patch_masks]

                clip = torch.stack(patch_imgs).unsqueeze(0).cuda()   # [1, T, 3, H, W]
                clip_masks = torch.stack(patch_masks).unsqueeze(0).cuda()  # [1, T, 1, H, W]

                with torch.no_grad():
                    pred_patch = model(clip, clip_masks).squeeze(0).clamp(0, 1).cpu()  # [3, H, W]

                # 剪切 padding 区域
                pred_patch = pred_patch[:, :bottom - top, :right - left]

                weight=gaussian_weight[:, :bottom - top, :right - left]

                output[:, top:bottom, left:right] += pred_patch * weight
                counter[:, top:bottom, left:right] += weight

        # 平均重叠区域
        output = output / counter
        output_img = transforms.ToPILImage()(output)
        output_frames.append(output_img)
        progress_bar.update(1)

    return output_frames

def get_padded_sequence(seq, idx, T):
    """
    给定序列 seq 和索引 idx，返回 idx 为中心的 T 帧（两端补边）
    """
    center_offset = T // 2
    pad_front = max(0, center_offset - idx)
    pad_back = max(0, idx + center_offset + 1 - len(seq))
    idx_range = list(range(idx - center_offset, idx + center_offset + 1))
    idx_range = [max(0, min(i, len(seq) - 1)) for i in idx_range]
    return [seq[i] for i in idx_range]

# ======= 示例推理 =======
# 示例帧输入路径

if __name__ == "__main__":
    T = 3
    base = "./testin/20250401_191118"
    image_dir = os.path.join(base, "beauty")
    mask_dir  = os.path.join(base, "mask")
    out="./testout"
    out_dir = os.path.join(out, "12")
    os.makedirs(out_dir, exist_ok=True)

    # 获取并排序所有文件路径
    image_paths_all = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(('.jpg', '.png'))
    ])

    mask_paths_all = sorted([
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if f.endswith(('.jpg', '.png'))
    ])



    # 加载模型 + 推理
    model = load_model_modele("./models/model_3.pth")
    #result_imgs = inference_one_sequence(model, image_paths_all, mask_paths_all)
    result_imgs = inference_one_sequence_patchwise(model, image_paths_all, mask_paths_all, patch_size=(256, 256), stride=(128, 128), T=T)

    for i, img in enumerate(result_imgs):
        img.save(os.path.join(out_dir,f"infer_result_{i:03d}.jpg"))


