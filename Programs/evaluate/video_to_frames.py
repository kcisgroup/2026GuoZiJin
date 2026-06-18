import cv2
import os
import argparse

def video_to_frames(video_path, output_dir, prefix="frame"):
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频信息: {frame_count} 帧, {fps:.2f} FPS")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 保存帧
        frame_name = f"{prefix}_{idx:05d}.png"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        idx += 1
        if idx % 100 == 0:
            print(f"已保存 {idx}/{frame_count} 帧...")

    cap.release()
    print(f"完成！共保存 {idx} 帧到 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将视频分解为图像帧")
    parser.add_argument("-i", "--input", type=str, required=True, help="输入视频路径 (mp4/avi/mov 等)")
    parser.add_argument("-o", "--output", type=str, default="./frames", help="输出帧保存文件夹")
    parser.add_argument("--prefix", type=str, default="frame", help="帧文件名前缀")
    args = parser.parse_args()

    video_to_frames(args.input, args.output, args.prefix)
