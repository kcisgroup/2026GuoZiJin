import os
import cv2
import torch
import lpips
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms


class MVCSMetric:
    """
    Multi-View Consistency Score (MVCS)

    支持：
    1. Camera Motion（基于深度与相机参数）
    2. Object Motion（基于光流）
    """

    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):

        self.alpha = alpha
        self.beta = beta
        self.device = device

        # LPIPS 感知一致性
        self.lpips_model = lpips.LPIPS(net='alex').to(device)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    # ==========================================================
    # 图像读取
    # ==========================================================

    def load_image(self, path):

        image = Image.open(path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        return image

    def tensor_to_numpy(self, tensor):

        img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)

        return img

    # ==========================================================
    # 光流对齐（Object Motion）
    # ==========================================================

    def optical_flow_alignment(self, img_i, img_j):

        """
        使用 Farneback 光流进行跨视角对齐
        """

        img_i_np = self.tensor_to_numpy(img_i)
        img_j_np = self.tensor_to_numpy(img_j)

        gray_i = cv2.cvtColor(img_i_np, cv2.COLOR_RGB2GRAY)
        gray_j = cv2.cvtColor(img_j_np, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray_i,
            gray_j,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        h, w = gray_i.shape

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)

        aligned = cv2.remap(
            img_j_np,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        aligned_tensor = torch.from_numpy(aligned).float() / 255.0
        aligned_tensor = aligned_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        return aligned_tensor

    # ==========================================================
    # Camera Motion 对齐
    # ==========================================================

    def geometric_projection_alignment(
        self,
        img_j,
        depth_i,
        K_i,
        R_i,
        t_i,
        K_j,
        R_j,
        t_j
    ):

        """
        基于深度与相机参数的几何投影对齐
        """

        img_j_np = self.tensor_to_numpy(img_j)

        h, w, _ = img_j_np.shape

        aligned = np.zeros_like(img_j_np)

        K_i_inv = np.linalg.inv(K_i)

        for v in range(h):
            for u in range(w):

                z = depth_i[v, u]

                p_i = np.array([u, v, 1.0])

                # Equation (5-4)
                P = np.linalg.inv(R_i) @ (
                    z * (K_i_inv @ p_i) - t_i
                )

                # Equation (5-5)
                p_j_homo = K_j @ (R_j @ P + t_j)

                if p_j_homo[2] <= 0:
                    continue

                # Equation (5-6)
                u_j = p_j_homo[0] / p_j_homo[2]
                v_j = p_j_homo[1] / p_j_homo[2]

                if 0 <= u_j < w and 0 <= v_j < h:

                    aligned[v, u] = cv2.getRectSubPix(
                        img_j_np,
                        (1, 1),
                        (float(u_j), float(v_j))
                    )

        aligned_tensor = torch.from_numpy(aligned).float() / 255.0
        aligned_tensor = aligned_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        return aligned_tensor

    # ==========================================================
    # 像素级一致性
    # ==========================================================

    def pixel_consistency(self, img_i, aligned_j):

        """
        Equation (5-10)
        """

        return torch.mean(torch.abs(img_i - aligned_j)).item()

    # ==========================================================
    # 感知级一致性
    # ==========================================================

    def perceptual_consistency(self, img_i, aligned_j):

        """
        Equation (5-11)
        """

        score = self.lpips_model(
            img_i * 2 - 1,
            aligned_j * 2 - 1
        )

        return score.item()

    # ==========================================================
    # 综合差异
    # ==========================================================

    def compute_pair_score(self, img_i, aligned_j):

        """
        Equation (5-14)
        """

        d_pixel = self.pixel_consistency(img_i, aligned_j)

        d_perceptual = self.perceptual_consistency(img_i, aligned_j)

        score = (
            self.alpha * d_pixel +
            self.beta * d_perceptual
        )

        return {
            "pixel": d_pixel,
            "perceptual": d_perceptual,
            "mvcs_pair": score
        }

    # ==========================================================
    # 多视角整体 MVCS
    # ==========================================================

    def compute_mvcs(self, image_paths, mode="object_motion"):

        """
        Equation (5-15)
        """

        images = [self.load_image(p) for p in image_paths]

        N = len(images)

        total_score = 0.0
        count = 0

        print("\nComputing MVCS...\n")

        for i in tqdm(range(N)):

            for j in range(N):

                if i == j:
                    continue

                img_i = images[i]
                img_j = images[j]

                # --------------------------------------------------
                # Object Motion
                # --------------------------------------------------

                if mode == "object_motion":

                    aligned_j = self.optical_flow_alignment(
                        img_i,
                        img_j
                    )

                else:
                    raise NotImplementedError(
                        "Camera motion demo requires depth & camera params."
                    )

                scores = self.compute_pair_score(
                    img_i,
                    aligned_j
                )

                total_score += scores["mvcs_pair"]

                count += 1

        mvcs = total_score / count

        return mvcs


# ==============================================================
# Example
# ==============================================================

if __name__ == "__main__":

    image_dir = "/home/gzj/test/BrushNetSimple-main/imgs/input/test1/"

    image_paths = sorted([
        os.path.join(image_dir, x)
        for x in os.listdir(image_dir)
        if x.endswith((".png", ".jpg", ".jpeg"))
    ])

    metric = MVCSMetric(
        alpha=1.0,
        beta=1.0
    )

    mvcs_score = metric.compute_mvcs(
        image_paths,
        mode="object_motion"
    )

    print("\n===================================")
    print(f"Final MVCS Score: {mvcs_score:.6f}")
    print("===================================\n")