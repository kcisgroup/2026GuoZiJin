from diffusers import StableDiffusionImg2ImgPipeline,StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
import gc
from dift.controlnetfun import getcanny,get_depth_map
import imageio
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as TF
import cv2
from dift.src.models.dift_sd import SDFeaturizer4Eval
from dift.test3 import img_with_point,f_match
from dift.moreblend import laplacian_blend

# 切换到 GPU 1
#torch.cuda.set_device(1)

# 检查当前 GPU
current_device = torch.cuda.current_device()
print(f"Current GPU: {current_device}")  # 输出 1

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 清理缓存
torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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


def create_pipe_nocontrol():
    model_id = "/home/gzj/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
    pipe = pipe.to(device)
    return pipe

def nocontrol(pipe,img_path,prompt,negative_prompt):

    #加载输入的图片
    input_image = Image.open(img_path).convert("RGB")
    #input_image = input_image.resize((512, 512))  # 调整图像大小

    # 设置固定种子
    generator = torch.Generator(device=device).manual_seed(40)
    '''
    # 加载模型
    model_id = "/home/gzj/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
    pipe = pipe.to(device)
    '''
    #prompt="masterpiece,best quality,high res,photorealistic,red face blush,red lips,extremely detailed"
    #negative_prompt="worst quality, low quality, normal quality, lowres, normal quality,skin spots, acne,acne marks,mole skin blemishes, age spot, watermark,signature water mark"

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        strength=0.4,
        num_inference_steps =50,
        num_images_per_prompt=1,
        generator=generator,
        guidance_scale=7.5,).images[0]
    #print("images count:",len(images))
    #for i in range(len(images)):
    #  images[i].save(f'./out/'+str(i)+".png")
    return images

def singlecontrol_canny(img_path,prompt,negative_prompt):
    #加载输入的图片
    input_image = Image.open(img_path).convert("RGB")
    input_image = input_image.resize((512, 512))  # 调整图像大小

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny",
                                                 torch_dtype=torch.float16)
    model_id = "windwhinny/chilloutmix"
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float16,
        #use_safetensors=True
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 设置固定种子
    generator = torch.Generator(device=device).manual_seed(40)

    #获取canny
    canny_img=getcanny(img_path)

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        strength=0.4,
        control_image=canny_img,
        num_inference_steps =50,
        num_images_per_prompt=1,
        generator=generator,
        guidance_scale=7.5,).images
    print("images count:",len(images))
    #for i in range(len(images)):
    #  images[i].save(f'./out/'+str(i)+".png")
    return images


def singlecontrol_depth(img_path,prompt,negative_prompt):
    #加载输入的图片
    input_image = Image.open(img_path).convert("RGB")
    #input_image = input_image.resize((512, 512))  # 调整图像大小

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",
                                                 torch_dtype=torch.float16)
    model_id = "windwhinny/chilloutmix"
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float16,
        #use_safetensors=True
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 设置固定种子
    generator = torch.Generator(device=device).manual_seed(40)

    #获取depth
    depth_img=get_depth_map(img_path)

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        strength=0.4,
        control_image=depth_img,
        num_inference_steps =50,
        num_images_per_prompt=1,
        generator=generator,
        guidance_scale=7.5,).images
    print("images count:",len(images))
    #for i in range(len(images)):
    #  images[i].save(f'./out/'+str(i)+".png")
    return images

def create_pipe():

    controlnet1 = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny",
                                                 torch_dtype=torch.float16)
    controlnet2 = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",
                                                 torch_dtype=torch.float16)

    controlnets=[controlnet1,controlnet2]
    model_id = "windwhinny/chilloutmix"
    model_id= "/home/gzj/.cache/huggingface/hub/models--windwhinny--chilloutmix/snapshots/e0bf5daebbfb76e757baaf4ce88e0c1cc4df7525"
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id, controlnet=controlnets, torch_dtype=torch.float16,
        use_safetensors=False
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe

def doublecontrol(pipe,img_path,prompt,negative_prompt):
    #加载输入的图片
    input_image = Image.open(img_path).convert("RGB")
    #input_image = input_image.resize((512, 512))  # 调整图像大小
    '''
    controlnet1 = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny",
                                                 torch_dtype=torch.float16)
    controlnet2 = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",
                                                 torch_dtype=torch.float16)

    controlnets=[controlnet1,controlnet2]
    model_id = "windwhinny/chilloutmix"
    model_id= "/home/gzj/.cache/huggingface/hub/models--windwhinny--chilloutmix/snapshots/e0bf5daebbfb76e757baaf4ce88e0c1cc4df7525"
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id, controlnet=controlnets, torch_dtype=torch.float16,
        use_safetensors=False
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    '''
    # 设置固定种子
    generator = torch.Generator(device=device).manual_seed(0)

    #获取canny
    canny_img=getcanny(img_path)
    depth_img=get_depth_map(img_path)
    control_images=[canny_img,depth_img]

    # 检查当前 GPU
    current_device = torch.cuda.current_device()
    print(f"Current GPU: {current_device}")  # 输出 1

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        strength=0.2,
        control_image=control_images,
        num_inference_steps =100,
        num_images_per_prompt=1,
        generator=generator,
        guidance_scale=7.5,).images[0]
    print('generate success')
    """
    print("images count:",len(images))
    for i in range(len(images)):
      images[i].save(f'./out/'+str(i)+".png")
    """
    return images

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

def posisson_blend(src_img,trg_img,trg_pts):
    # 将 src_img 转换为灰度图像（单通道）
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # 创建掩码（单通道）
    mask = np.zeros(src_gray.shape, dtype=np.uint8)  # 单通道
    mask[src_gray > 0] = 255  # 使用灰度图像的有效区域
    cv2.imwrite('./out/mask.jpg',mask)

    # 找到掩码的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大轮廓的边界框
    x, y, w, h = cv2.boundingRect(contours[0])
    # 计算中心点
    center = (x + w // 2, y + h // 2)

    blended_result = cv2.seamlessClone(src_img, trg_img, mask, center, cv2.MIXED_CLONE)
    return blended_result


def post_process_blend(src_img, trg_img):
    """
    对融合结果进行后处理，优化边界的平滑性。

    参数:
        src_img (np.ndarray): 源图像。
        trg_img (np.ndarray): 目标图像。
        mask (np.ndarray): 掩码（单通道）。

    返回:
        blended_result (np.ndarray): 融合结果。
    """
    mask = np.zeros(src_img.shape[:2], dtype=np.uint8)
    mask[50:200, 50:200] = 255  # 假设有效区域是一个矩形
    # 使用 cv2.addWeighted 进行融合
    blended_result = cv2.addWeighted(src_img, 0.5, trg_img, 0.5, 50)

    # 对边界区域进行高斯模糊
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = mask.astype(np.float32) / 255  # 归一化到 [0, 1]

    # 将模糊后的边界区域应用到融合结果中
    blended_result = blended_result * (1 - mask[..., None]) + cv2.GaussianBlur(blended_result, (21, 21), 0) * mask[
        ..., None]
    blended_result = np.clip(blended_result, 0, 255).astype(np.uint8)
    return blended_result

#创建dift函数体
def create_dift(text):
    cat=text
    dift=SDFeaturizer4Eval(cat_list=[text])
    return cat,dift


#原始dift融合函数
def align_and_blend_images(cat,dift,src_img_path, trg_img_path, generated_src, generated_trg,img_mask):
    '''
    cat='woman'
    dift=SDFeaturizer4Eval(cat_list=['woman'])
    '''

    # 加载源图像和目标图像
    src_img = Image.open(src_img_path).convert('RGB')
    trg_img = Image.open(trg_img_path).convert('RGB')
    #mask=Image.open(src_mask_path).convert('L')
    sticker=gen_sticker(generated_src,img_mask)

    sticker_color, sticker_mask = sticker[..., :3], sticker[..., 3]

    # 提取两个视角图像的特征
    sd_feat_src = dift.forward(src_img, cat)
    sd_feat_trg = dift.forward(trg_img, cat)

    # 归一化特征
    sd_feat_src = F.normalize(sd_feat_src.squeeze(), p=2, dim=0)
    sd_feat_trg = F.normalize(sd_feat_trg.squeeze(), p=2, dim=0)
    feat_dim = sd_feat_src.shape[0]

    # 生成网格
    h_src, w_src = np.array(src_img).shape[:2]
    h_trg, w_trg = np.array(trg_img).shape[:2]
    grid_src = gen_grid(h_src, w_src, device='cuda')
    grid_trg = gen_grid(h_trg, w_trg, device='cuda')

    # 随机采样特征点
    coord_src=grid_src[sticker_mask>0]
    coord_src = coord_src[torch.randperm(len(coord_src))][:1000]  # 随机采样 1000 个点
    coord_src_normed = normalize_coords(coord_src, h_src, w_src)
    grid_trg_normed = normalize_coords(grid_trg, h_trg, w_trg)

    # 提取特征
    feat_src = F.grid_sample(sd_feat_src[None], coord_src_normed[None, None], align_corners=True).squeeze().T
    feat_trg = F.grid_sample(sd_feat_trg[None], grid_trg_normed[None], align_corners=True).squeeze()
    feat_trg_flattened = feat_trg.permute(1, 2, 0).reshape(-1, feat_dim)

    # 计算特征距离
    distances = torch.cdist(feat_src, feat_trg_flattened)
    _, indices = torch.min(distances, dim=1)

    # 获取对应点
    src_pts = coord_src.reshape(-1, 2).cpu().numpy()
    trg_pts = grid_trg.reshape(-1, 2)[indices].cpu().numpy()

    match_result=f_match(src_img,trg_img,src_pts,trg_pts)

    # 计算单应性矩阵
    M, mask = cv2.findHomography(src_pts, trg_pts, cv2.RANSAC, 5.0)
    #mask = Image.open("/home/gzj/test/dift/imgs/mask/mask.png").convert('L')
    new_gen=cv2.bitwise_and(np.array(generated_src),np.array(generated_src),mask=np.array(img_mask))
    #cv2.imwrite('./out/new_gen.jpg',new_gen)
    # 对视角 1 的生成结果应用单应性变换
    generated_src_aligned = cv2.warpPerspective(np.array(new_gen), M, (w_trg, h_trg))
    #cv2.imwrite('./out/mask1.jpg',generated_src_aligned)
    # 对视角 2 的生成结果进行融合
    blended_result = cv2.addWeighted(generated_src_aligned, 0.05, np.array(generated_trg), 1, 0)
    #blended_result=posisson_blend(generated_src_aligned,np.array(generated_trg),trg_pts)
    #blended_result=post_process_blend(generated_src_aligned,np.array(generated_trg))
    #blended_result=laplacian_blend(generated_src_aligned,np.array(generated_trg))
    # 保存融合结果
    blended_result = Image.fromarray(blended_result.astype(np.uint8))
    return blended_result,match_result


def resize_half(image):
    """将图像或遮罩的长宽缩小一半"""
    if image.width > 512 or image.height > 512:
        if isinstance(image, np.ndarray):  # 处理numpy数组格式的遮罩
            h, w = image.shape[:2]
            return cv2.resize(image, ((w+1)//2, (h+1)//2), interpolation=cv2.INTER_AREA)
        else:  # 处理PIL图像
            return image.resize(((image.width+1)//2, (image.height+1)//2), Image.Resampling.LANCZOS)
    else:
        return image

#新版dift融合函数
def dift_blend_images(cat,dift,src_img_path, trg_img_path, generated_src, generated_trg,img_mask):
    # 加载图像
    src_img = Image.open(src_img_path).convert('RGB')
    src_img = resize_half(src_img)
    #print(src_img.size)
    trg_img = Image.open(trg_img_path).convert('RGB')
    trg_img = resize_half(trg_img)
    #print(trg_img.size)
    img_mask = resize_half(img_mask)
    generated_src = resize_half(generated_src)
    #print(generated_src.size)
    generated_trg = resize_half(generated_trg)
    #print(generated_trg.size)
    sticker = gen_sticker(generated_src, img_mask)
    sticker_color, sticker_mask = sticker[..., :3], sticker[..., 3]

    # 改进的特征提取（IC-Light）
    sd_feat_src = normalize_with_illumination(dift.forward(src_img, cat), src_img)
    sd_feat_trg = normalize_with_illumination(dift.forward(trg_img, cat), trg_img)

    # 特征匹配（AnyDoor）
    h_src, w_src = np.array(src_img).shape[:2]
    h_trg, w_trg = np.array(trg_img).shape[:2]
    grid_src = gen_grid(h_src, w_src, device='cuda')
    grid_trg = gen_grid(h_trg, w_trg, device='cuda')

    coord_src = grid_src[sticker_mask > 0][torch.randperm(len(grid_src[sticker_mask > 0]))][:1000]
    coord_src_normed = normalize_coords(coord_src, h_src, w_src)
    grid_trg_normed = normalize_coords(grid_trg, h_trg, w_trg)

    feat_src = F.grid_sample(sd_feat_src[None], coord_src_normed[None, None], align_corners=True).squeeze().T
    feat_trg = F.grid_sample(sd_feat_trg[None], grid_trg_normed[None], align_corners=True).squeeze()
    feat_trg_flattened = feat_trg.permute(1, 2, 0).reshape(-1, feat_trg.size(0))

    trg_pts = bidirectional_feature_matching(feat_src, feat_trg_flattened, coord_src, grid_trg)
    src_pts = coord_src.reshape(-1, 2).cpu().numpy()

    # 计算变换矩阵
    M, _ = cv2.findHomography(src_pts, trg_pts, cv2.RANSAC, 5.0)
    new_gen = cv2.bitwise_and(np.array(generated_src), np.array(generated_src), mask=np.array(img_mask))
    generated_src_aligned = cv2.warpPerspective(new_gen, M, (w_trg, h_trg))

    #检查
    #cv2.imwrite('/home/gzj/test/BrushNetSimple-main/out/new_gen.jpg',new_gen)
    #cv2.imwrite('/home/gzj/test/BrushNetSimple-main/out/mask1.jpg',generated_src_aligned)
    #blended_result = cv2.addWeighted(generated_src_aligned, 0.05, np.array(generated_trg), 1, 0)

    # 改进的图像融合
    blended_result = physics_aware_blend(generated_src_aligned, np.array(generated_trg), sticker_mask,np.array(trg_img))
    return Image.fromarray(blended_result), f_match(src_img, trg_img, src_pts, trg_pts)


# 修改特征归一化部分，加入光照感知约束
def normalize_with_illumination(feat, img):
    # IC-Light思想：将光照特征与语义特征解耦
    feat_normalized = F.normalize(feat.squeeze(), p=2, dim=0)

    # 从图像中提取光照特征（模拟IC-Light的轻量版）
    img_tensor = TF.ToTensor()(img).unsqueeze(0).to(feat.device)
    with torch.no_grad():
        # 使用预训练的浅层网络提取光照特征
        light_feat = torch.cat([
            img_tensor.mean(dim=(2, 3)),  # 全局亮度
            img_tensor.std(dim=(2, 3))  # 对比度
        ], dim=1)
    light_feat = light_feat.repeat(1, 8)[:, :48]  # 6 * 8=48
    # 融合语义和光照特征（权重可调）
    return feat_normalized * 0.8 + light_feat.squeeze() * 0.2


# 改进特征匹配流程，加入双向传播
def bidirectional_feature_matching(feat_src, feat_trg, coord_src, grid_trg):
    # 前向匹配（原始方法）
    distances = torch.cdist(feat_src, feat_trg)
    _, fwd_indices = torch.min(distances, dim=1)

    # 反向匹配（AnyDoor思想）
    _, bwd_indices = torch.min(distances, dim=0)

    # 一致性校验
    consistent = (bwd_indices[fwd_indices] == torch.arange(len(fwd_indices)).to(feat_src.device))
    valid_ratio = consistent.float().mean()

    # 动态融合：高置信度时用前向，低置信度时混合
    if valid_ratio > 0.7:
        return grid_trg.reshape(-1, 2)[fwd_indices].cpu().numpy()
    else:
        # 关键修正：对反向结果进行采样点对齐
        sampled_bwd = bwd_indices[fwd_indices]  # 从全图匹配中提取对应采样点的匹配
        blended = 0.7 * grid_trg.reshape(-1, 2)[fwd_indices] + \
                  0.3 * coord_src[sampled_bwd].to(feat_src.device)
        return blended.cpu().numpy()


# 改进融合策略，加入光照补偿
def physics_aware_blend(src_aligned, trg, mask, src_original):
    """基于原图光照参考的物理感知融合
    Args:
        src_aligned: 对齐后的源图像 (RGB, uint8)
        trg: 目标图像 (RGB, uint8)
        mask: 融合区域掩膜 (单通道, 0-255)
        src_original: 未对齐的原始源图像 (RGB, uint8)
    Returns:
        融合结果 (RGB, uint8)
    """
    # 1. 预处理（浮点计算保障精度）
    src = src_aligned.astype(np.float32) / 255.0
    trg = trg.astype(np.float32) / 255.0
    src_orig = src_original.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0

    # 2. 光照分析（IC-Light核心思想）
    def analyze_illumination(img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_mean = np.mean(lab[..., 0])
        l_std = np.std(lab[..., 0])
        ab_mean = np.mean(lab[..., 1:], axis=(0, 1))
        return l_mean, l_std, ab_mean

    # 获取原图/目标图像的光照特征
    src_l_mean, src_l_std, src_ab_mean = analyze_illumination(src_orig)
    trg_l_mean, trg_l_std, trg_ab_mean = analyze_illumination(trg)

    # 3. 光照迁移模型（物理约束）
    lab_src = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
    lab_trg = cv2.cvtColor(trg, cv2.COLOR_RGB2LAB)

    # 亮度迁移：保持原图对比度+匹配目标均值
    lab_mixed = np.zeros_like(lab_src)
    lab_mixed[..., 0] = (lab_src[..., 0] - src_l_mean) * (trg_l_std / src_l_std) + trg_l_mean

    # 色度迁移：混合原图与目标主色
    ab_ratio = 0.7 # 原图色度权重（可调）
    lab_mixed[..., 1:] = ab_ratio * lab_src[..., 1:] + (1 - ab_ratio) * trg_ab_mean

    # 4. 边缘感知融合
    # 生成羽化掩膜（保留高频细节）
    contours, _ = cv2.findContours(
        cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)[1].astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    dist_map = np.zeros_like(mask)
    for cnt in contours:
        cv2.drawContours(dist_map, [cnt], -1, 1.0, thickness=cv2.FILLED)
    dist_map = cv2.distanceTransform(dist_map.astype(np.uint8), cv2.DIST_L2, 5)
    mask_feathered = np.clip(dist_map / 15, 0, 1)

    # 5. 多尺度融合
    blended_lab = lab_mixed * mask_feathered[..., None] * 0.1+ \
                  lab_trg * (1 - mask_feathered[..., None]* 0.1)
    result = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2RGB)

    # 6. 后处理
    result = cv2.GaussianBlur(result, (3, 3), 0)  # 消除色带
    return (np.clip(result * 255, 0, 255).astype(np.uint8))

def get_mask_intersection(mask1, mask2):
    """
    NumPy优化版mask交集计算
    参数:
        mask1: 第一个掩膜(单通道或三通道)
        mask2: 第二个掩膜(相同尺寸)
    返回:
        交集区域的二值掩膜(uint8)
    """
    # 直接转换为布尔型（比>127判断更快）
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    # 计算交集
    intersection = np.logical_and(mask1_bool, mask2_bool)
    return intersection.astype(np.uint8) * 255

#二值融合
def apply_intersection_mask(img_path, modified_img, mask1, mask2):
    """
    结合两个mask的交集进行图像融合
    参数:
        original_img: 原图(BGR格式)
        modified_img: 修改后的图像
        mask1: 第一个应用区域掩膜
        mask2: 第二个限制区域掩膜
    返回:
        融合后的图像
    """
    original_img = cv2.imread(img_path)

    # 转换modified_img为BGR格式
    if isinstance(modified_img, Image.Image):
        mod_arr = cv2.cvtColor(np.array(modified_img), cv2.COLOR_RGB2BGR)
    else:
        mod_arr = np.array(modified_img)

    # 获取Mask
    final_mask = get_mask_intersection(mask1, mask2)

    # 应用Mask
    output = original_img.copy()
    output[final_mask.astype(bool)] = mod_arr[final_mask.astype(bool)]

    return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)),Image.fromarray(final_mask)

#对mask进行膨胀操作
def dilate_mask(mask,iterations=1):
    # 定义膨胀核（控制膨胀程度）
    kernel_size = 5  # 核大小，奇数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 执行膨胀操作
    dilated_mask = cv2.dilate(mask, kernel, iterations)  # iterations控制膨胀次数

    return dilated_mask



#带羽化边缘的融合
def apply_feathered_mask(img_path, modified_img, mask1, mask2, feather_radius=1,blur_sigma=2):
    """
    带羽化边缘的mask融合
    参数:
        img_path: 原图路径
        modified_img: 修改后的图像(PIL/numpy array)
        mask1: 第一个掩膜
        mask2: 第二个掩膜
        feather_radius: 羽化半径(像素)
    返回:
        (融合后的PIL图像, 最终的羽化mask)
    """
    # 读取原图并确保BGR格式
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    '''
    print(f"modified_img type: {type(modified_img)}")
    if hasattr(modified_img, 'shape'):
        print(f"modified_img shape: {modified_img.shape}")
    else:
        print("modified_img is not a NumPy array or PIL Image")
    '''


    # 转换modified_img为BGR格式
    if isinstance(modified_img, Image.Image):
        mod_arr = cv2.cvtColor(np.array(modified_img), cv2.COLOR_RGB2BGR)
    else:
        mod_arr = np.array(modified_img)[:, :, :3]  # 确保3通道

    # 尺寸对齐检查
    if original_img.shape != mod_arr.shape:
        mod_arr = cv2.resize(mod_arr, (original_img.shape[1], original_img.shape[0]))

    # 获取精确交集mask并二值化
    final_mask = get_mask_intersection(mask1, mask2)
    _, binary_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

    # 羽化处理 (核心改进部分)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_radius * 2 + 1, feather_radius * 2 + 1))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    blurred_mask = cv2.GaussianBlur(eroded, (feather_radius * 2 + 1, feather_radius * 2 + 1), blur_sigma)
    blurred_mask=dilate_mask(blurred_mask)
    # 归一化权重
    mask_float = blurred_mask.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask_float, mask_float, mask_float])  # 转为3通道权重

    # 加权融合
    output = original_img * (1 - mask_3ch) + mod_arr * mask_3ch
    output = output.astype(np.uint8)


    # 返回结果 (修正颜色空间转换)
    return (
        Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)),
        Image.fromarray(blurred_mask)
    )

if __name__ == "__main__":
    print(f"Current GPU: {current_device}")  # 输出 1
    print('11111')
    img_path="./imgs/demo1.jpg"
    prompt = "masterpiece, best quality, high res, photorealistic,1girl, Asianbeauty, " \
             "flawless porcelain skin, soft natural makeup," \
             "delicate facial features, bright almond - shaped eyes,glossy lips, subtle blush, smooth skin texture," \
             "radiant glow, elegant and feminine, extremely detailed, cinematic lighting"
    negative_prompt="worst quality, low quality, normal quality, lowres, normal quality,skin spots, acne,acne marks,mole skin blemishes, age spot, watermark,signature water mark"
    """
    i=1
    file_path='./imgs/1/'
    for filename in os.listdir(file_path):
        img_path=os.path.join(file_path,filename)

        result=doublecontrol(img_path,prompt,negative_prompt)
        result.save(f'./out/'+str(i)+".png")
        i=i+1
    """
    pipe=create_pipe()

    # 遍历文件夹中的图像
    img_folder = './assets/2/'
    generated_images = []
    for i, filename in enumerate(os.listdir(img_folder)):
        img_path = os.path.join(img_folder, filename)
        generated_image = doublecontrol(pipe,img_path, prompt, negative_prompt)
        generated_image.save(f'./out/{i+1}.png')
        generated_images.append(generated_image)
    '''
    #直接读取文件夹中的生成图像
    g_folder=''
    for i, filename in enumerate(os.listdir(img_folder)):
        img_path = os.path.join(img_folder, filename)
        generated_image = doublecontrol(img_path, prompt, negative_prompt)
        generated_image.save(f'./out/{i+1}.png')
        generated_images.append(generated_image)
    '''
    mask_path="/home/gzj/test/dift/imgs/mask/frame_00023_depth_mask.png"
    # 对齐和融合生成结果
    if len(generated_images) >= 2:
        blended_result = align_and_blend_images(
            os.path.join(img_folder, os.listdir(img_folder)[0]),
            os.path.join(img_folder, os.listdir(img_folder)[1]),
            generated_images[0],
            generated_images[1],
            mask_path
        )
        blended_result.save('./out/blended_result.png')
