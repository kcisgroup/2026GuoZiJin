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

from Grounded_Segment_Anything.grounded_sam_simple_mask import g_sam

print('111')
# input brushnet ckpt path
brushnet_path = "models/random_mask_brushnet_ckpt_sdxl_v0/"

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
print('222')
# choose the base model here
#base_model_path = "models/juggernautXL_v7Rundiffusion.safetensors"
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
base_model_path = "/home/gzj/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"

pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16
)
#    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False, use_safetensors=True
print('333')
# change to sdxl-vae-fp16-fix to avoid nan in VAE encoding when using fp16
pipe.vae = AutoencoderKL.from_pretrained("models/vae/", torch_dtype=torch.float16)
print('444')
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
#pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()


image_path='imgs/demo2.jpg'
#mask_path='imgs/mask1.png'

# extract mask from alpha channel
config_path='/home/gzj/test/BrushNetSimple-main/Grounded_Segment_Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py'
checkpoint_path='/home/gzj/test/BrushNetSimple-main/models/groundingdino/groundingdino_swinb_cogcoor.pth'

#mask_image=Image.open('imgs/mask1.jpg')
#mask_image = ImageOps.invert(init_image.getchannel("A"))
#mask_image = g_sam('Grounded_Segment_Anything/assets/demo4.jpg','black dog',config_path,checkpoint_path)
origin_mask = g_sam(image_path,'mouth',config_path,checkpoint_path)
cv2.imwrite("out/mask_image.png", origin_mask)
#origin_mask.save("out/cola_mask.png")

randomize_seed = True
prompt = "The black mouth on a lady's face"
negative_prompt = "blurry, poor quality, distorted, woman, man, people, body, mutated"
num_inference_steps = 50
guidance_scale = 4
control_strength = 1.0
blended = True

generator = torch.Generator().manual_seed(random.randint(0,2147483647) if randomize_seed else seed)


init_image = cv2.imread(image_path)[:,:,::-1]
#mask_image = 1.*(origin_mask.sum(-1)>255)


mask_image=np.array(origin_mask)[:,:,np.newaxis]/255.0

init_image = init_image * (1-mask_image)

init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")


image = pipe(
    [prompt],
    [prompt],
    init_image,
    mask_image,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator,
    brushnet_conditioning_scale=float(control_strength),
    negative_prompt=[negative_prompt],
    negative_prompt_2=[negative_prompt]
).images[0]
image.save(f'./out/3.png')
'''
if blended:
    image_np = np.array(image)
    init_image_np = cv2.imread(image_path)[:, :, ::-1]
    mask_np = np.array(origin_mask)[:, :, np.newaxis]

    # blur, you can adjust the parameters for better performance
    mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
    mask_blurred = mask_blurred[:, :, np.newaxis]
    mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)

    image_pasted = init_image_np * (1 - mask_np) + image_np * mask_np
    image_pasted = image_pasted.astype(image_np.dtype)
    image = Image.fromarray(image_pasted)

image.save(f'./out/ou1.jpg')
'''