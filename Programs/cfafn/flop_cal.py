from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision
import os
#torch.cuda.set_device(1)
from TemporalFeatureFuser import BeautyRefiner, RefinementLoss
from BDataset import BeautyDataset


from thop import profile

model = BeautyRefiner().cuda()

dummy_clip  = torch.randn(1, 5, 3, 256, 256).cuda()
dummy_mask  = torch.randn(1, 5, 1, 256, 256).cuda()

flops, params = profile(model, inputs=(dummy_clip, dummy_mask))

print(f"FLOPs: {flops/1e9:.2f} G")
print(f"Params: {params/1e6:.2f} M")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = BeautyRefiner().cuda()
params = count_params(model)

print(f"Params: {params/1e6:.2f} M")
