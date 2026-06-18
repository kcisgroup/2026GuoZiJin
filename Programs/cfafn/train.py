from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision
import os
#torch.cuda.set_device(1)
from TemporalFeatureFuser import BeautyRefiner, RefinementLoss
from BDataset import BeautyDataset

# 检查当前 GPU
current_device = torch.cuda.current_device()
print(f"Current GPU: {current_device}")  # 输出 1

torch.autograd.set_detect_anomaly(True)

# 超参数配置
train_root = "./beauty_train_dataset/1"
batch_size = 2
num_epochs = 100
lr = 1e-4
save_path = "./models/model_3.pth"

# 加载数据集
dataset = BeautyDataset(train_root, T=3)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型与优化器
model = BeautyRefiner().cuda()
model = torch.nn.DataParallel(model).cuda()

criterion = RefinementLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_loss = float("inf")

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # 初始化自动混合精度的缩放器

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        frames = batch['beauty_frames'].cuda()  # [B, T, 3, H, W]
        masks  = batch['masks'].cuda()          # [B, T, 1, H, W]
        B, T, _, H, W = frames.shape

        pred_seq = []

        with autocast():  # << 自动混合精度上下文开始
            for i in range(T):
                idx_start = max(0, i - T // 2)
                idx_end   = min(T, i + T // 2 + 1)

                indices = list(range(idx_start, idx_end))
                while len(indices) < T:
                    if idx_start == 0:
                        indices.insert(0, 0)
                    else:
                        indices.append(T - 1)

                # 构造输入 clip
                clip  = torch.stack([frames[:, j] for j in indices], dim=1)  # [B, T, 3, H, W]
                maskc = torch.stack([masks[:, j]  for j in indices], dim=1)  # [B, T, 1, H, W]

                pred = model(clip, maskc)  # [B, 3, H, W]
                pred_seq.append(pred)

            # 拼接预测序列
            pred_seq = torch.stack(pred_seq, dim=1)  # [B, T, 3, H, W]

            # 损失函数：对比原始美颜帧
            loss = criterion(pred_seq, frames)

        optimizer.zero_grad()
        scaler.scale(loss).backward()   # << 自动缩放梯度
        scaler.step(optimizer)          # << 更新权重
        scaler.update()                 # << 更新 scaler 状态

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    # 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print(f"✅ 模型已更新并保存至：{save_path}")
