import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torchvision.models as models
import torchvision.transforms as transforms

class TemporalFeatureFuser(nn.Module):
    def __init__(self, in_channels, num_heads=4, num_offsets=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_offsets = num_offsets

        # 注意力模块
        self.to_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.to_kv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)

        # 可变形 offset 学习
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 2 * num_offsets, 3, 1, 1)
        )

        # 融合输出投影
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, visibility):
        """
        x:          [B, T, C, H, W]
        visibility: [B, T, 1, H, W]
        """
        B, T, C, H, W = x.shape
        center_idx = T // 2
        x_center = x[:, center_idx]             # [B, C, H, W]
        vis_center = visibility[:, center_idx]  # [B, 1, H, W]

        q = self.to_q(x_center)                 # [B, C, H, W]

        fused = torch.zeros_like(x_center)

        # Base grid: [1, H, W, 2]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij"
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # [1, H, W, 2]
        base_grid = base_grid.repeat(B, 1, 1, 1)                         # [B, H, W, 2]

        all_attn = []

        # Step 3: 遍历 T 帧融合
        for t in range(T):
            xt = x[:, t]             # [B, C, H, W]
            vt = visibility[:, t]    # [B, 1, H, W]

            # 使用当前帧 xt 预测 offset（可选：也可以使用 x_center）
            offset = self.offset_predictor(xt)  # [B, 2*num_offsets, H, W]
            offset = offset.view(B, self.num_offsets, 2, H, W)  # [B, N, 2, H, W]

            # deformable sampling
            sampled_feats = []
            for i in range(self.num_offsets):
                dx = offset[:, i, 0]  # [B, H, W]
                dy = offset[:, i, 1]
                flow = torch.stack((dx, dy), dim=-1)  # [B, H, W, 2]
                grid = base_grid + flow               # [B, H, W, 2]
                grid = grid.clamp(-1, 1)

                feat_sampled = F.grid_sample(xt, grid, align_corners=True, mode='bilinear', padding_mode='zeros')  # [B, C, H, W]
                mask_sampled = F.grid_sample(vt, grid, align_corners=True, mode='bilinear', padding_mode='zeros')  # [B, 1, H, W]
                sampled_feats.append(feat_sampled * mask_sampled)

            feat_t = torch.stack(sampled_feats, dim=0).mean(dim=0)  # [B, C, H, W]

            # attention: cosine similarity
            kv = self.to_kv(xt)    # [B, 2C, H, W]
            k, _ = kv.chunk(2, dim=1)
            attn_score = F.cosine_similarity(q, k, dim=1, eps=1e-6).unsqueeze(1)  # [B, 1, H, W]
            attn_score = attn_score * vt  # 加上当前帧 mask

            all_attn.append(attn_score)
            #x[:, t] = feat_t  # 用 deform-sampled 特征覆盖 x[:, t]
            x_new = x.clone()  # 或者用 x.detach().clone()，但clone足够
            x_new[:, t] = feat_t

        # Attention Stack: [B, T, 1, H, W]
        attn_stack = torch.stack(all_attn, dim=1)  # [B, T, 1, H, W]
        attn_weight = F.softmax(attn_stack, dim=1)  # 沿 T softmax

        fused = torch.sum(attn_weight * x_new, dim=1)  # [B, C, H, W]

        out = self.proj(fused)


        return out


class StyleAdaptiveFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x, style_ref):
        style_stat = self.style_fc(style_ref)
        return self.norm(x) * style_stat + x

#Unet Decoder+encoder
class UNetDecoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=3):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Tanh()  # 或者 Sigmoid，依据图像范围
        )

    def forward(self, x):
        return self.dec(x)

#完整模块（整合encoder+fuser+decoder），仅支持单帧输出
class BeautyRefinerforsingle(nn.Module):
    def __init__(self, feat_channels=128):
        super().__init__()
        # 简化Encoder，可替换为ResNet/ViT中间层
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, feat_channels, 3, padding=1),
            nn.ReLU()
        )
        self.fuser = TemporalFeatureFuser(in_channels=feat_channels)
        self.decoder = UNetDecoder(in_channels=feat_channels, out_channels=3)

    def forward(self, frames, masks):
        """
        frames: [B, T, 3, H, W]
        masks: [B, T, 1, H, W]
        """
        B, T, _, H, W = frames.shape
        feats = []
        for t in range(T):
            xt = frames[:, t]  # [B, 3, H, W]
            ft = self.encoder(xt)  # [B, C, H, W]
            feats.append(ft)
        feats = torch.stack(feats, dim=1)  # [B, T, C, H, W]
        fused_feat = self.fuser(feats, masks)  # [B, C, H, W]
        out = self.decoder(fused_feat)  # [B, 3, H, W]
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)
        return out


#完整模块（整合encoder+fuser+decoder），支持多帧输出
class BeautyRefiner(nn.Module):
    def __init__(self, feat_channels=128):
        super().__init__()
        # 简化Encoder，可替换为ResNet/ViT中间层
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, feat_channels, 3, padding=1),
            nn.ReLU()
        )
        self.fuser = TemporalFeatureFuser(in_channels=feat_channels)
        self.decoder = UNetDecoder(in_channels=feat_channels, out_channels=3)

    def forward(self, frames, masks):
        B, T, _, H, W = frames.shape
        feats = torch.stack([self.encoder(frames[:, t]) for t in range(T)], dim=1)
        fused = self.fuser(feats, masks)
        out = self.decoder(fused)
        return F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:9]  # 取到 relu2_2（即第9层）
        self.vgg = vgg.eval()  # 只用于特征提取，eval 模式
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.resize = resize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, x):
        # x: [B, 3, H, W], 归一化和尺寸调整
        if self.resize:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # VGG 期望的是 ImageNet 风格的归一化图像
        x = self.normalize(x)
        return self.vgg(x)


#损失函数
class RefinementLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.vgg = VGGPerceptualLoss()  # 自定义感知网络，可用 torchvision.models.vggXX
        self.vgg.eval()  # 不训练 VGG

    def perceptual(self, x, y):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        y = y.view(B * T, C, H, W)
        return self.l1(self.vgg(x), self.vgg(y))

    def forward(self, pred, target):
        return self.l1(pred, target) + 0.1 * self.perceptual(pred, target)



#推理代码
def inference_clip(model, frames_seq, masks_seq):
    """
    frames_seq: [T, 3, H, W] - 第一阶段输出图像序列
    masks_seq:  [T, 1, H, W]
    """
    model.eval()
    T = frames_seq.shape[0]
    mid = T // 2

    input_frames = frames_seq.unsqueeze(0).cuda()   # [1, T, 3, H, W]
    input_masks = masks_seq.unsqueeze(0).cuda()     # [1, T, 1, H, W]

    with torch.no_grad():
        out = model(input_frames, input_masks)      # [1, 3, H, W]
    return out.squeeze(0)



if __name__ == '__main__':
    model = TemporalFeatureFuser(3)
    x = torch.randn(1, 5, 3, 256, 256)
    visibility = torch.randn(1, 5, 1, 256, 256)
    out = model(x, visibility)
    print(out.shape)