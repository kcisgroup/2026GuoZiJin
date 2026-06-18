import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from PIL import Image
import gc
import imageio
import numpy as np
import torch.nn.functional as F
import cv2
from dift.beautyface import doublecontrol,align_and_blend_images,create_pipe,create_dift,\
    apply_intersection_mask,apply_feathered_mask,dilate_mask,dift_blend_images,\
    nocontrol,singlecontrol_depth,singlecontrol_canny,create_pipe_nocontrol
from Grounded_Segment_Anything.grounded_sam_simple_mask import g_sam
import re
from tqdm import tqdm
from evaluate.skin_detect import generate_skin_mask
from datetime import datetime

from cfafn.eval import inference_one_sequence_patchwise,load_model_modele

#export HF_HOME=/home/gzj/.cache/huggingface
'''
export HUGGINGFACE_HUB_CACHE="/home/gzj/.cache/huggingface/hub" 
export HF_HOME="/home/gzj/.cache/huggingface" 
export XDG_CACHE_HOME="/home/gzj/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export TF_ENABLE_ONEDNN_OPTS=0

cd /home/gzj/test/BrushNetSimple-main/
python main.py

'''

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 检查可用 GPU 数量
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")

import imageio
def video2pic(video_path,output_path):
    os.makedirs(output_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #cv2.imwrite(os.path.join(output_path, f"{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(output_path, f"{i:04d}.png"), frame)
        i += 1
    cap.release()


def resize_large_images(folder_path, min_size=512):
    """
    将文件夹中长宽均大于min_size的图像缩小一半，直接覆盖原文件

    参数:
        folder_path: 图像文件夹路径
        min_size: 触发缩放的最小尺寸（默认512）
    """
    # 支持的图像格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # 获取所有图像文件
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(image_extensions)]

    print(f"找到 {len(image_files)} 张图像，开始检查尺寸...")

    # 处理每张图像
    for filename in tqdm(image_files, desc="处理进度"):
        try:
            filepath = os.path.join(folder_path, filename)
            img = Image.open(filepath)

            # 检查图像长宽是否都大于阈值
            if img.width > min_size and img.height > min_size:
                # 计算新尺寸（原尺寸的一半）
                new_size = (img.width // 2, img.height // 2)

                # 调整尺寸（使用高质量下采样）
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

                # 覆盖保存原文件（保持原始格式）
                resized_img.save(filepath)
                print(f"已缩小: {filename} ({img.width}x{img.height} → {new_size[0]}x{new_size[1]})")
            else:
                print(f"跳过: {filename} (尺寸 {img.width}x{img.height} 未达到阈值)")

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")


def single_blend():

    img_folder='/home/gzj/test/BrushNetSimple-main/imgs/instruct-nerf2nerf/images/'
    out_folder='/home/gzj/test/BrushNetSimple-main/out/20250521_102102/'

    doublecontrol_output_folder =out_folder+ '/beauty/'  # doublecontrol 生成的图像保存文件夹
    blendresult_output_folder = out_folder+'/blendbeauty/'  # 融合结果保存文件夹
    mask_output_folder=out_folder+'/mask/'
    match_output_folder=out_folder+'/match/'

    filenames_sorted = sorted(os.listdir(img_folder), key=lambda x: int(re.search(r'\d+', x).group()))
    # 遍历文件夹中的图像
    pipe=create_pipe()
    generated_images = []
    original_images_path=[]
    masks=[]
    for i, filename in enumerate(filenames_sorted):
        #获取面部mask
        img_path = os.path.join(img_folder, filename)
        original_images_path.append(img_path)

        #mask=255-mask
        #saved_mask = Image.fromarray(mask)

    for filename in sorted(os.listdir(mask_output_folder)):
        if filename.endswith('.png'):
            filepath = os.path.join(mask_output_folder, filename)
            mask = Image.open(filepath)  # 用PIL读取图像
            masks.append(mask)

    for filename in sorted(os.listdir(doublecontrol_output_folder)):
        if filename.endswith('.png'):
            filepath = os.path.join(doublecontrol_output_folder, filename)
            generated_image = Image.open(filepath)  # 用PIL读取图像
            generated_images.append(generated_image)

    cat,dift=create_dift('face')

    progress_bar1 = tqdm(total=600, ncols=200, dynamic_ncols=True)
    # 对齐和融合生成结果
    if len(generated_images) >= 2:
        for i in range(len(generated_images) - 1):
            # 获取当前图像对
            src_img_path=original_images_path[i]
            trg_img_path = original_images_path[i + 1]
            #src_img_path = os.path.join(img_folder, os.listdir(img_folder)[i])
            #trg_img_path = os.path.join(img_folder, os.listdir(img_folder)[i + 1])
            src_img = generated_images[i]
            trg_img = generated_images[i + 1]
            mask=masks[i]

            # 对齐和融合
            blended_result,match_result = dift_blend_images(
                cat, dift,
                src_img_path,
                trg_img_path,
                src_img,
                trg_img,
                mask
            )
            # 保存融合结果
            blended_result.save(os.path.join(blendresult_output_folder, f'blended_result_{i+1:03d}.png'))
            Image.fromarray(match_result).save(os.path.join(match_output_folder, f'match_result_{i+1:03d}.png'))

            progress_bar1.update(1)



import glob


if __name__ == "__main__":
    #torch.cuda.set_device(1)
    #img_path="./imgs/demo1.jpg"
    '''
    prompt = "masterpiece, best quality, high res,1 woman, Asian beauty, " \
             " soft natural makeup," \
             "delicate facial features,glossy lips, smooth skin texture,"\
          "radiant glow, elegant and feminine, extremely detailed"
    '''
    prompt="masterpiece, best quality, high res, 1 man, " \
           "[ethnicity/beauty type], clean-shaven, flawless complexion, " \
           "subtle facial contours, naturally groomed eyebrows, " \
           "smooth poreless skin, moisturized lips, even skin tone, " \
           "soft ambient lighting, refined elegance"

    negative_prompt="beard,worst quality, low quality, normal quality, lowres, normal quality,skin spots, acne,acne marks,mole skin blemishes, age spot, watermark,signature water mark"


    # 输入和输出文件夹
    img_folder = './imgs/1/'  # 输入图像文件夹
    out_folder='./out/'+datetime.now().strftime("%Y%m%d_%H%M%S")
    #out_folder = './out/'
    doublecontrol_output_folder =out_folder+ '/beauty/'  # doublecontrol 生成的图像保存文件夹
    blendresult_output_folder = out_folder+'/blendbeauty/'  # 融合结果保存文件夹
    mask_output_folder=out_folder+'/mask/'
    match_output_folder=out_folder+'/match/'
    # 创建输出文件夹（如果不存在）
    os.makedirs(doublecontrol_output_folder, exist_ok=True)
    os.makedirs(blendresult_output_folder, exist_ok=True)
    os.makedirs(mask_output_folder, exist_ok=True)
    os.makedirs(match_output_folder, exist_ok=True)

    config_path = '/home/gzj/test/BrushNetSimple-main/Grounded_Segment_Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py'
    checkpoint_path = '/home/gzj/test/BrushNetSimple-main/models/groundingdino/groundingdino_swinb_cogcoor.pth'


    #通过视频路径获取图像文件夹img
    video_path='/home/gzj/test/BrushNetSimple-main/imgs/videos/facevid/091071.mp4'
    output_path='/home/gzj/test/BrushNetSimple-main/imgs/videos/facepic/71/'
    #video2pic(video_path,output_path)
    img_folder=output_path

    img_folder='/home/gzj/test/BrushNetSimple-main/imgs/0124-2/'
    #resize_large_images(img_folder)

    #获取整体人mask
    #mask_path="/home/gzj/test/dift/imgs/mask/frame_00023_depth_mask.png"
    #image_path='/home/gzj/test/dift/imgs/1/frame_00023_rgb.png'
    images_path=sorted(glob.glob(os.path.join(img_folder, '*.png'))
                       +glob.glob(os.path.join(img_folder, '*.jpg')))
    image_path=images_path[0]
    origin_mask=g_sam(image_path,'head',config_path,checkpoint_path)
    origin_mask=dilate_mask(origin_mask,3)
    saved_mask = Image.fromarray(origin_mask)
    #saved_mask.save(os.path.join(out_folder, f'mask.png'))

    filenames_sorted = sorted(os.listdir(img_folder), key=lambda x: int(re.search(r'\d+', x).group()))
    # 遍历文件夹中的图像
    #pipe=create_pipe()
    #消融pipe
    pipe=create_pipe_nocontrol()
    generated_images = []
    original_images_path=[]
    masks=[]
    for i, filename in enumerate(filenames_sorted):
        #获取面部mask
        img_path = os.path.join(img_folder, filename)
        original_images_path.append(img_path)
        mask=generate_skin_mask(img_path)
        #mask=255-mask
        #saved_mask = Image.fromarray(mask)



        #进行美颜:正常
        generated_image = doublecontrol(pipe,img_path, prompt, negative_prompt)

        #进行美颜：消融，无controlnet
        #generated_image = nocontrol(pipe,img_path,prompt,negative_prompt)


        #消融处理：无mask覆盖
        #generated_image,final_mask=apply_feathered_mask(img_path,generated_image,origin_mask,mask)
        #final_mask=Image.fromarray(mask)
        #masks.append(final_mask)
        #mask覆盖
        generated_image,final_mask=apply_feathered_mask(img_path,generated_image,origin_mask,mask)
        masks.append(final_mask)

        #消融处理
        #mask_pil=Image.fromarray(final_mask)
        mask_pil=final_mask
        mask_pil.save(os.path.join(mask_output_folder, f'{i + 1:03d}.png'))
        generated_image.save(os.path.join(doublecontrol_output_folder, f'{i+1:03d}.png'))
        generated_images.append(generated_image)
        print(img_path)
        if i>200:
            break

    progress_bar1 = tqdm(total=600, ncols=200, dynamic_ncols=True)

    T=3
    # 获取并排序所有文件路径
    image_paths_all = sorted([
        os.path.join(doublecontrol_output_folder, f)
        for f in os.listdir(doublecontrol_output_folder)
        if f.endswith(('.jpg', '.png'))
    ])

    mask_paths_all = sorted([
        os.path.join(mask_output_folder, f)
        for f in os.listdir(mask_output_folder)
        if f.endswith(('.jpg', '.png'))
    ])



    # 加载模型 + 推理
    model = load_model_modele("./cfafn/models/model_3.pth")
    #result_imgs = inference_one_sequence(model, image_paths_all, mask_paths_all)
    result_imgs = inference_one_sequence_patchwise(model, image_paths_all, mask_paths_all, patch_size=(256, 256), stride=(128, 128), T=T)

    for i, img in enumerate(result_imgs):
        img.save(os.path.join(blendresult_output_folder,f"infer_result_{i:03d}.jpg"))
        progress_bar1.update(1)


    '''
    #老版本融合代码
  
    #torch.cuda.empty_cache()
    cat,dift=create_dift('face')

    progress_bar1 = tqdm(total=600, ncols=200, dynamic_ncols=True)
    # 对齐和融合生成结果
    if len(generated_images) >= 2:
        for i in range(len(generated_images) - 1):
            # 获取当前图像对
            src_img_path=original_images_path[i]
            trg_img_path = original_images_path[i + 1]
            #src_img_path = os.path.join(img_folder, os.listdir(img_folder)[i])
            #trg_img_path = os.path.join(img_folder, os.listdir(img_folder)[i + 1])
            src_img = generated_images[i]
            trg_img = generated_images[i + 1]
            mask=masks[i]

            # 对齐和融合
            blended_result,match_result = dift_blend_images(
                cat, dift,
                src_img_path,
                trg_img_path,
                src_img,
                trg_img,
                mask
            )
            # 保存融合结果
            blended_result.save(os.path.join(blendresult_output_folder, f'blended_result_{i+1:03d}.png'))
            Image.fromarray(match_result).save(os.path.join(match_output_folder, f'match_result_{i+1:03d}.png'))

            progress_bar1.update(1)

if __name__ == "__main__":
    single_blend()
'''