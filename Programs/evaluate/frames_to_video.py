import cv2
import os
import argparse
import shutil

def frames_to_video(frame_dir, output_base_dir, video_name="output.mp4", fps=24):
    """
    将帧文件夹中的图片合成为视频，并组织输出结构
    
    Args:
        frame_dir: 输入帧文件夹路径
        output_base_dir: 输出基础文件夹路径
        video_name: 输出视频文件名
        fps: 帧率
    """
    # 获取所有帧文件
    frames = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not frames:
        raise ValueError("没有在文件夹中找到图片帧！")

    # 读取第一帧来确定分辨率
    first_frame_path = os.path.join(frame_dir, frames[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise ValueError(f"无法读取第一帧: {first_frame_path}")
    height, width, _ = first_frame.shape

    # 创建输出文件夹结构
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    # 在output_base_dir下创建以视频名命名的文件夹
    video_folder_name = os.path.splitext(video_name)[0]  # 去掉扩展名
    output_folder = os.path.join(output_base_dir, video_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    '''
    # 创建frames子文件夹用于存放原始帧
    frames_output_dir = os.path.join(output_folder, "frame")
    os.makedirs(frames_output_dir, exist_ok=True)
    
    # 复制原始帧到frame文件夹
    print("正在复制原始帧...")
    for frame_name in frames:
        src_path = os.path.join(frame_dir, frame_name)
        dst_path = os.path.join(frames_output_dir, frame_name)
        shutil.copy2(src_path, dst_path)
    '''
    # 定义视频输出路径
    video_output_path = os.path.join(output_folder, video_name)
    
    # 定义视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    # 写入所有帧到视频
    print("正在合成视频...")
    for frame_name in frames:
        frame_path = os.path.join(frame_dir, frame_name)
        img = cv2.imread(frame_path)
        if img is None:
            print(f"警告: 跳过无法读取的帧 {frame_path}")
            continue
        img_resized = cv2.resize(img, (width, height))
        out.write(img_resized)

    out.release()
    print(f"视频已保存到 {video_output_path}")
    #print(f"原始帧已复制到 {frames_output_dir}")
    print(f"输出文件夹结构:")
    print(f"  {output_folder}/")
    print(f"  ├── {video_name}")
    print(f"  └── frame/")
    print(f"      └── [原始帧文件]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将文件夹中的帧合成为 mp4 视频")
    parser.add_argument("-i", "--input", type=str, required=True, help="输入帧文件夹路径")
    parser.add_argument("-o", "--output", type=str, default="./output", help="输出基础文件夹路径")
    parser.add_argument("-n", "--name", type=str, default="output.mp4", help="输出视频文件名")
    parser.add_argument("--fps", type=int, default=10, help="帧率 (默认 30)")
    args = parser.parse_args()

    frames_to_video(args.input, args.output, args.name, args.fps)