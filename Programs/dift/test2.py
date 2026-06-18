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
#from src.models.dift_sd import SDFeaturizer4Eval

import re

if torch.cuda.is_available():
    # 获取 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # 打印每个 GPU 的名称
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print('111')