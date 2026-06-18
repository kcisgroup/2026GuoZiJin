import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob

class BeautyDataset(Dataset):
    def __init__(self, root_dir, T=3, transform=None):
        self.T = T
        self.root = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        beauty_frame_dir = os.path.join(root_dir, 'beauty_frames')
        # 提取样本 ID，如 "1_000"
        self.sample_ids = sorted(set(
            "_".join(f.split("_")[:2])
            for f in os.listdir(beauty_frame_dir)
            if f.endswith((".jpg", ".png"))
        ))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        frames = []
        masks = []

        for t in range(self.T):
            # 找图片路径
            img_path = None
            for ext in [".jpg", ".png"]:
                candidate = os.path.join(self.root, 'beauty_frames', f"{sample_id}_0{t}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            assert img_path is not None, f"找不到图像: {sample_id}_0{t}"

            # 找 mask 路径
            mask_path = None
            for ext in [".png"]:
                candidate = os.path.join(self.root, 'masks', f"{sample_id}_0{t}{ext}")
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            #assert mask_path is not None, f"找不到 mask: {sample_id}_0{t}"
            if mask_path is None:
                continue

            img = self.transform(Image.open(img_path).convert('RGB'))      # [3, H, W]
            mask = self.transform(Image.open(mask_path).convert('L'))      # [1, H, W]

            frames.append(img)
            masks.append(mask)

        #检测补帧操作
        # === 检查帧数是否不足，补足到 T ===
        num_valid = len(frames)
        if num_valid < self.T:
            if num_valid == 0:
                raise ValueError(f"样本 {sample_id} 的所有帧都缺失 mask！")
            pad_num = self.T - num_valid
            frames += [frames[0]] * pad_num
            masks += [masks[0]] * pad_num

        # 如果多余的帧就裁掉（极少数情况防御）
        frames = frames[:self.T]
        masks = masks[:self.T]

        frames = torch.stack(frames)   # [T, 3, H, W]
        masks = torch.stack(masks)    # [T, 1, H, W]

        return {
            'beauty_frames': frames,
            'masks': masks,
            'id': sample_id
        }
