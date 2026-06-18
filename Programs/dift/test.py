import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import gc
import imageio
from PIL import Image
from torchvision.transforms import PILToTensor
import os
import json
from PIL import Image, ImageDraw
import torch.nn.functional as F
import cv2
import glob
from torchvision.transforms import PILToTensor
from src.models.dift_sd import SDFeaturizer4Eval
from test3 import img_with_point,f_match

def gen_grid(h, w, device, normalize=False, homogeneous=False):
    if normalize:
        lin_y = torch.linspace(-1., 1., steps=h, device=device)
        lin_x = torch.linspace(-1., 1., steps=w, device=device)
    else:
        lin_y = torch.arange(0, h, device=device)
        lin_x = torch.arange(0, w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
    grid = torch.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
    return grid  # [h, w, 2 or 3]


def normalize_coords(coords, h, w, no_shift=False):
    assert coords.shape[-1] == 2
    if no_shift:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2
    else:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.

def gen_sticker(img,mask):
    full_image = np.array(img)
    mask = np.array(mask)
    # 创建一个 RGBA 图像
    sticker = np.zeros((full_image.shape[0], full_image.shape[1], 4), dtype=np.uint8)

    # 将完整图像的 RGB 部分复制到 sticker
    sticker[..., :3] = full_image

    # 将掩码复制到 sticker 的 Alpha 通道
    sticker[..., 3] = mask
    return sticker

####

cat = 'woman'
dift = SDFeaturizer4Eval(cat_list=['woman'])

####
'''
src_img = Image.open('./assets/guitar_cat.jpg').convert('RGB')
trg_img = Image.open('./assets/painting_cat.jpg').convert('RGB')
sticker = imageio.imread('./assets/cartoon.png')
'''
src_img = Image.open('/home/gzj/test/dift/imgs/1/frame_00023_rgb.png').convert('RGB')
trg_img = Image.open("/home/gzj/test/dift/imgs/1/frame_00096_rgb.png").convert('RGB')
img=Image.open('./out/2.png').convert('RGB')
mask = Image.open("/home/gzj/test/dift/imgs/1/frame_00023_depth_mask.png").convert('L')
sticker = gen_sticker(img,mask)

sticker_color, sticker_mask = sticker[..., :3], sticker[..., 3]

assert np.array(src_img).shape[:2] == sticker.shape[:2]
h_src, w_src = sticker.shape[:2]
h_trg, w_trg = np.array(trg_img).shape[:2]

#使用dift提取源图像和目标图像的特征
sd_feat_src = dift.forward(src_img, cat)
sd_feat_trg = dift.forward(trg_img, cat)

sd_feat_src = F.normalize(sd_feat_src.squeeze(), p=2, dim=0)
sd_feat_trg = F.normalize(sd_feat_trg.squeeze(), p=2, dim=0)
feat_dim = sd_feat_src.shape[0]

grid_src = gen_grid(h_src, w_src, device='cuda')
grid_trg = gen_grid(h_trg, w_trg, device='cuda')

coord_src = grid_src[sticker_mask > 0]
coord_src = coord_src[torch.randperm(len(coord_src))][:10]
coord_src_normed = normalize_coords(coord_src, h_src, w_src)
grid_trg_normed = normalize_coords(grid_trg, h_trg, w_trg)

feat_src = F.grid_sample(sd_feat_src[None], coord_src_normed[None, None], align_corners=True).squeeze().T
feat_trg = F.grid_sample(sd_feat_trg[None], grid_trg_normed[None], align_corners=True).squeeze()
feat_trg_flattened = feat_trg.permute(1, 2, 0).reshape(-1, feat_dim)

distances = torch.cdist(feat_src, feat_trg_flattened)
_, indices = torch.min(distances, dim=1)

src_pts = coord_src.reshape(-1, 2).cpu().numpy()
trg_pts = grid_trg.reshape(-1, 2)[indices].cpu().numpy()
img_with_point(src_img,trg_img,src_pts,trg_pts)
f_match(src_img,trg_img,src_pts,trg_pts)


#计算源图像和目标图像之间的单应性矩阵
M, mask = cv2.findHomography(src_pts, trg_pts, cv2.RANSAC, 5.0)
sticker_out = cv2.warpPerspective(sticker, M, (w_trg, h_trg))

sticker_out_alpha = sticker_out[..., 3:] / 255
sticker_alpha = sticker[..., 3:] / 255

trg_img_with_sticker = sticker_out_alpha * sticker_out[..., :3] + (1 - sticker_out_alpha) * trg_img
src_img_with_sticker = sticker_alpha * sticker[..., :3] + (1 - sticker_alpha) * src_img

# 保存图像到本地文件
cv2.imwrite('out/trg_test.png', trg_img_with_sticker)  # 保存目标图像
cv2.imwrite('out/src_test.png', src_img_with_sticker)  # 保存源图像
####
'''
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(src_img)
axs[0, 0].set_title("Source Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(src_img_with_sticker.astype(np.uint8))
axs[0, 1].set_title("Source Image with Edits")
axs[0, 1].axis('off')

axs[1, 0].imshow(trg_img)
axs[1, 0].set_title("Target Image")
axs[1, 0].axis('off')

axs[1, 1].imshow(trg_img_with_sticker.astype(np.uint8))
axs[1, 1].set_title("Target Image with Propagated Edits")
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()
'''