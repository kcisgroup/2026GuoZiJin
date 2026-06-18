from diffusers.utils import load_image, make_image_grid
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import pipeline

def getcanny(img_path):
    original_image = load_image(img_path)
    #original_image = original_image.resize((512, 512))
    image = np.array(original_image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)

    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image.save("./out/canny.jpg")
    return canny_image

def get_depth_map(img_path, depth_estimator=pipeline("depth-estimation")):
    #加载图像
    image=load_image(img_path)
    #image=image.resize((512, 512))

    #用深度估计模型生成深度图
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    #保存深度图
    depth_img=Image.fromarray(image.astype(np.uint8))
    depth_img.save("./out/depth.jpg")

    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map.unsqueeze(0).half().to("cuda")

#depth_map = get_depth_map(image, depth_estimator).unsqueeze(0).half().to("cuda")