import cv2
import random
import diffusers

import numpy as np

from PIL import Image, ImageOps
from diffusers import UniPCMultistepScheduler

# Diffusers patching

from unet_2d_condition import UNet2DConditionModel as UNet2DConditionModel_patch
from diffusers.models.unets import unet_2d_condition
unet_2d_condition.UNet2DConditionModel = UNet2DConditionModel_patch

import torch
from diffusers import AutoencoderKL
from brushnet import BrushNetModel
from pipeline_brushnet import StableDiffusionBrushNetPipeline
from pipeline_brushnet_sdxl import StableDiffusionXLBrushNetPipeline
print('111')
# input brushnet ckpt path
brushnet_path = f"models/random_mask_brushnet_ckpt/"

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
print('222')
# choose the base model here
base_model_path = f"models/Realistic_Vision_V6.0_NV_B1.safetensors"
base_model_path = "/home/gzj/.cache/huggingface/hub/models--SG161222--RealVisXL_V4.0/snapshots/26dfe44930964cd70d0a817b6d1cc945c130e38d"


pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)
print('333')
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

init_image = Image.open(f'imgs/test_image_cola.png')
# extract mask from alpha channel
mask_image = ImageOps.invert(init_image.getchannel("A"))

print(f'Size = {init_image.size}')

randomize_seed = True
prompt = "Photo of can laying on the beach"
negative_prompt = "blurry, poor quality, distorted, woman, man, people, body, mutated"
num_inference_steps = 20
guidance_scale = 7.5
control_strength = 1.0
blended = True

generator = torch.Generator("cuda").manual_seed(random.randint(0,2147483647) if randomize_seed else seed)

image = pipe(
    [prompt]*2,
    init_image,
    mask_image,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator,
    brushnet_conditioning_scale=float(control_strength),
    negative_prompt=[negative_prompt]*2,
).images

if blended:
    if control_strength<1.0:
        raise gr.Error('Using blurred blending with control strength less than 1.0 is not allowed')
    blended_image=[]
    mask = np.array(mask_image) / 255
    mask = mask[..., np.newaxis]
    original_image = np.array(init_image)[..., :3]
    # blur, you can adjust the parameters for better performance
    mask_blurred = cv2.GaussianBlur(mask*255, (21, 21), 0)/255
    mask_blurred = mask_blurred[:,:,np.newaxis]
    mask = 1-(1-mask) * (1-mask_blurred)
    for image_i in image:
        image_np=np.array(image_i)
        image_pasted=original_image * (1-mask) + image_np*mask

        image_pasted=image_pasted.astype(image_np.dtype)
        blended_image.append(Image.fromarray(image_pasted))
blended_image[0].save(f'out/1.jpg')