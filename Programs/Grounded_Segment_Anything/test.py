import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image

from grounded_sam_demo import ground_sam


config=r'GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py'
grounded_checkpoint=r'/home/gzj/test/BrushNetSimple-main/grounded_sam/models/groundingdino/groundingdino_swinb_cogcoor.pth'
sam_hq_checkpoint=r'/home/gzj/test/BrushNetSimple-main/grounded_sam/models/sams/sam_hq_vit_h.pth'
image_path=r'/home/gzj/test/BrushNetSimple-main/imgs/0124-2/127.jpg'
text_prompt='girl'


ground_sam(config,grounded_checkpoint,sam_hq_checkpoint,image_path,text_prompt)