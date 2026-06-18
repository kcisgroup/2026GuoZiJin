import os
import shutil
from pathlib import Path


def build_training_dataset(first_stage_dir, output_dir, T=3, center_indices=None):
    """
    构建训练集：假设图像文件名无 "_"，按数字排序划分滑窗序列。

    参数：
    - first_stage_dir: 包含 beauty/ 和 mask/ 的目录
    - output_dir: 输出训练集的路径
    - T: 每组帧数
    - center_indices: 可选字典，如 {组索引: 中心帧索引}，例如 {0: 1, 1: 2}；
                      若为 None，默认中心帧为 T // 2。
    """

    beauty_dir = Path(first_stage_dir) / "beauty"
    mask_dir = Path(first_stage_dir) / "mask"

    out_dir = Path(output_dir)
    (out_dir / "beauty_frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)
    (out_dir / "gt_frames").mkdir(exist_ok=True)

    # 获取所有图像，支持 jpg 和 png，按文件名数字排序
    all_files = sorted(
        list(beauty_dir.glob("*.jpg")) + list(beauty_dir.glob("*.png")),
        key=lambda x: int(x.stem)
    )

    total_groups = len(all_files) - T + 1
    for i in range(total_groups):
        group = all_files[i:i + T]
        if len(group) < T:
            continue

        new_gid = f"group_{i:04d}"
        center_idx = center_indices[i] if center_indices and i in center_indices else T // 2

        if center_idx >= T:
            print(f"[跳过] 中心帧索引越界：{center_idx} ≥ {T}")
            continue

        for t, img_path in enumerate(group):
            ext = img_path.suffix
            out_img_name = f"{new_gid}_0{t}{ext}"
            out_mask_name = f"{new_gid}_0{t}.png"

            # 拷贝图像帧
            shutil.copy(img_path, out_dir / "beauty_frames" / out_img_name)

            # 拷贝掩码（同名但扩展名为 .png）
            mask_name = img_path.with_suffix(".png").name
            shutil.copy(mask_dir / mask_name, out_dir / "masks" / out_mask_name)

        # 复制中心帧到 gt_frames
        gt_img = group[center_idx]
        shutil.copy(gt_img, out_dir / "gt_frames" / f"{new_gid}_gt{gt_img.suffix}")

    print("✅ 按文件名顺序构建训练数据完成")


def build_dataset_from_multi_folders(input_root_dir, output_dir, T=3, center_frame_dict=None):
    """
    从多个子目录构建训练数据集（每个子目录为一组视频帧，包含 beauty/ 和 mask/）

    参数：
    - input_root_dir: 输入根目录，包含多个子文件夹（如 1/，2/）
    - output_dir: 构建输出目录
    - T: 每组帧数（滑动窗口长度）
    - center_frame_dict: dict，指定每个组使用哪个帧作为中心帧（如 { "1": 3 }），基于组内排序索引
    """

    input_root = Path(input_root_dir)
    out_dir = Path(output_dir)
    (out_dir / "beauty_frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)
    (out_dir / "gt_frames").mkdir(exist_ok=True)

    group_folders = sorted([p for p in input_root.iterdir() if p.is_dir()])
    for group_folder in group_folders:
        gid = group_folder.name  # e.g., "1", "2"

        beauty_dir = group_folder / "beauty"
        mask_dir = group_folder / "mask"

        # 所有图片（支持 .jpg/.png），按文件名数字排序
        all_imgs = sorted(
            list(beauty_dir.glob("*.jpg")) + list(beauty_dir.glob("*.png")),
            key=lambda x: int(x.stem)
        )

        if len(all_imgs) < T:
            print(f"[跳过] {gid} 帧数不足：{len(all_imgs)} < T={T}")
            continue

        center_idx = center_frame_dict.get(gid, T // 2) if center_frame_dict else T // 2
        if center_idx >= T:
            print(f"[跳过] {gid} 中心帧索引越界 center_idx={center_idx} ≥ T={T}")
            continue

        # 滑窗生成多个训练样本
        for i in range(len(all_imgs) - T + 1):
            group = all_imgs[i:i+T]
            sample_id = f"{gid}_{i:03d}"

            for t, img_path in enumerate(group):
                ext = img_path.suffix
                img_out_name = f"{sample_id}_0{t}{ext}"
                mask_out_name = f"{sample_id}_0{t}.png"

                shutil.copy(img_path, out_dir / "beauty_frames" / img_out_name)

                mask_path = mask_dir / (img_path.stem + ".png")
                if mask_path.exists():
                    shutil.copy(mask_path, out_dir / "masks" / mask_out_name)
                else:
                    print(f"[警告] 掩码不存在: {mask_path}")

            # 中心帧 → GT
            gt_img = group[center_idx]
            shutil.copy(gt_img, out_dir / "gt_frames" / f"{sample_id}_gt{gt_img.suffix}")

    print("✅ 多文件夹训练数据构建完成")

#子监督训练数据集
def build_dataset_from_multi_videos(input_root_dir, output_dir, T=3):
    """
    构建平滑美颜训练数据集（每组来自同一个视频子文件夹，使用滑动窗口生成连续帧组）

    参数：
    - input_root_dir: 包含多个子目录，每个子目录表示一个视频，内部有 beauty/ 和 mask/
    - output_dir: 数据构建输出目录
    - T: 滑动窗口帧数（例如 3、5、7）
    """

    input_root = Path(input_root_dir)
    out_dir = Path(output_dir)
    (out_dir / "beauty_frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)

    group_folders = sorted([p for p in input_root.iterdir() if p.is_dir()])

    for group_folder in group_folders:
        gid = group_folder.name  # e.g., "1", "2"

        beauty_dir = group_folder / "beauty"
        mask_dir = group_folder / "mask"

        # 获取所有帧（按数字顺序）
        all_imgs = sorted(
            list(beauty_dir.glob("*.jpg")) + list(beauty_dir.glob("*.png")),
            key=lambda x: int(x.stem)
        )

        if len(all_imgs) < T:
            print(f"[跳过] {gid} 帧数不足：{len(all_imgs)} < T={T}")
            continue

        for i in range(len(all_imgs) - T + 1):
            group = all_imgs[i:i + T]
            sample_id = f"{gid}_{i:03d}"

            for t, img_path in enumerate(group):
                ext = img_path.suffix
                img_out_name = f"{sample_id}_0{t}{ext}"
                mask_out_name = f"{sample_id}_0{t}.png"

                shutil.copy(img_path, out_dir / "beauty_frames" / img_out_name)


                #mask_path = mask_dir / (img_path.stem + ".png")
                mask_name = f"{int(img_path.stem):03d}.png"
                mask_path = mask_dir / mask_name

                if mask_path.exists():
                    shutil.copy(mask_path, out_dir / "masks" / mask_out_name)
                else:
                    print(f"[警告] 掩码不存在: {mask_path}")

    print("✅ 平滑视频训练数据构建完成")


'''
# 示例调用
build_training_dataset(
    first_stage_dir="/mnt/data/first_stage_outputs",
    output_dir="/mnt/data/beauty_train_dataset",
    T=3
)
'''
if __name__ == "__main__":


    build_dataset_from_multi_videos(
        input_root_dir="./first_stage_outputs",
        output_dir="./beauty_train_dataset/1",
        T=5,
    )
