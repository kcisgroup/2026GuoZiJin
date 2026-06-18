'''
2.	频闪指数FI：由于生成与融合时可能会出现视角间亮度差异，所以通过FI标准化评估
对于实时视频美颜，其阈值范围为<0.03



改进策略：
多尺度频闪分析：划分低频（<80hz）和高频(>80hz)子带，分别计算fl并加权融合
'''
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def load_frames(folder_path):
    """读取文件夹中的连续帧并按文件名排序"""
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in valid_exts]
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    frames = [cv2.imread(os.path.join(folder_path, f)) for f in files]
    return frames


def temporal_flicker_analysis(frames):
    """时域频闪分析：计算亮度波动曲线和频闪指数"""
    # 提取全局亮度时间序列
    brightness = [np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]

    # 频域分析
    n = len(brightness)
    yf = fft(brightness - np.mean(brightness))
    xf = fftfreq(n, 1 / 30)[:n // 2]  # 假设视频为30fps

    # 计算主要频率成分
    main_freq = xf[np.argmax(np.abs(yf[0:n // 2]))]
    amplitude = np.max(np.abs(yf[0:n // 2]))

    # 计算时域FI
    mean_brightness = np.mean(brightness)
    area1 = np.sum([b - mean_brightness for b in brightness if b > mean_brightness])
    area2 = np.sum([mean_brightness - b for b in brightness if b <= mean_brightness])
    fi = (area1 - area2) / (area1 + area2) if (area1 + area2) != 0 else 0

    return {
        'FI': fi,
        'frequency': abs(main_freq),
        'amplitude': amplitude,
        'brightness_curve': brightness
    }


def visualize_results(results):
    """可视化亮度曲线与频谱"""
    plt.figure(figsize=(12, 6))

    # 亮度时域曲线
    plt.subplot(1, 2, 1)
    plt.plot(results['brightness_curve'], 'b-', linewidth=1)
    plt.title(f"Brightness Curve (FI={results['FI']:.4f})")
    plt.xlabel("Frame Index")
    plt.ylabel("Normalized Brightness")
    plt.grid(True)

    # 频域分析
    plt.subplot(1, 2, 2)
    n = len(results['brightness_curve'])
    xf = fftfreq(n, 1 / 30)[:n // 2]
    plt.plot(xf, 2.0 / n * np.abs(results['brightness_curve'][0:n // 2]))
    plt.title(f"Dominant Frequency: {results['frequency']:.2f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 参数设置
    #video_folder = "/home/gzj/test/BrushNetSimple-main/out/111/"  # 包含200帧的文件夹路径
    video_folder = "/home/gzj/test/BrushNetSimple-main/imgs/in/"  # 包含200帧的文件夹路径


    # 1. 加载视频帧
    frames = load_frames(video_folder)
    print(f"成功加载 {len(frames)} 帧")

    # 2. 执行频闪分析
    results = temporal_flicker_analysis(frames)

    # 3. 输出结果
    print(f"""
    频闪分析报告：
    - 平均频闪指数 (FI): {results['FI']:.10f}
    - 主波动频率: {results['frequency']:.2f} Hz
    - 波动幅度: {results['amplitude']:.2f}
    """)

    # 4. 可视化
    #visualize_results(results)