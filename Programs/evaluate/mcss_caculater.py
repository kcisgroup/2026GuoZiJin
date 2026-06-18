import cv2
import os
import numpy as np


def calculate_mcss(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    if not ret or prev_frame is None:
        raise ValueError(f"无法读取视频第一帧：{video_path}")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 初始化光流参数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    mcss_scores = []
    while True:
        ret, curr_frame = cap.read()
        if not ret: break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        # 运动连贯性指标计算
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        angle = np.arctan2(flow[..., 1], flow[..., 0])

        # 计算速度标准差和角度方差
        speed_std = np.std(magnitude)/10
        angle_var = np.var(angle)*100

        # 综合连贯性评分（数值越小越连贯）
        mcss = 0.7 * speed_std + 0.3 * angle_var
        mcss_scores.append(mcss)

        prev_gray = curr_gray

    return np.mean(mcss_scores)


def feature_based_mcss(video_path):
    cap = cv2.VideoCapture(video_path)

    # 初始化ORB检测器
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_kp, prev_des = orb.detectAndCompute(cap.read()[1], None)
    trajectory_consistency = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        curr_kp, curr_des = orb.detectAndCompute(frame, None)
        matches = matcher.match(prev_des, curr_des)

        if len(matches) > 10:
            # 计算匹配点位移向量
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches])

            # 轨迹连贯性分析
            displacements = np.linalg.norm(src_pts - dst_pts, axis=1)
            displacement_var = np.var(displacements)

            # 计算运动方向一致性
            motion_vectors = dst_pts - src_pts
            angles = np.arctan2(motion_vectors[:, 1], motion_vectors[:, 0])
            angle_std = np.std(angles)

            # 综合评分（数值越小越连贯）
            mcss = 0.5 * displacement_var + 0.5 * angle_std
            trajectory_consistency.append(mcss)

        prev_kp, prev_des = curr_kp, curr_des

    return np.median(trajectory_consistency)


def feature_based_mcss_from_frames(frame_dir):
    # 获取帧图像路径列表（按名称排序）
    frame_paths = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    if len(frame_paths) < 2:
        print("帧数不足，无法计算 MCSS")
        return 0.0

    # 初始化 ORB 检测器和匹配器
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 加载第一帧并提取特征
    prev_img = cv2.imread(frame_paths[0])
    if prev_img is None:
        raise ValueError(f"无法读取图像: {frame_paths[0]}")
    prev_kp, prev_des = orb.detectAndCompute(prev_img, None)

    trajectory_consistency = []

    for i in range(1, len(frame_paths)):
        curr_img = cv2.imread(frame_paths[i])
        if curr_img is None:
            print(f"跳过损坏图像: {frame_paths[i]}")
            continue

        curr_kp, curr_des = orb.detectAndCompute(curr_img, None)
        if prev_des is None or curr_des is None:
            print(f"第{i}帧特征提取失败，跳过")
            continue

        matches = matcher.match(prev_des, curr_des)
        if len(matches) > 10:
            # 获取匹配点坐标
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches])

            # 位移向量和幅度
            displacements = np.linalg.norm(src_pts - dst_pts, axis=1)
            displacement_var = np.var(displacements)

            # 运动方向角度变化
            motion_vectors = dst_pts - src_pts
            angles = np.arctan2(motion_vectors[:, 1], motion_vectors[:, 0])
            angle_std = np.std(angles)

            # 综合评分：越小表示跨帧运动越平稳
            mcss = 0.1 * displacement_var + 0.9 * angle_std
            trajectory_consistency.append(mcss)

        prev_kp, prev_des = curr_kp, curr_des

    if not trajectory_consistency:
        print("无有效匹配，返回默认值")
        return 0.0

    return np.median(trajectory_consistency)

if __name__ == "__main__":

    #tokenflow
    #video_path = "/home/gzj/test/others/TokenFlow-master/out/out.mp4/sd_2.0/in/steps_500/nframes_200/inverted.mp4"

    #ours
    #video_path="/home/gzj/test/BrushNetSimple-main/imgs/videos/test5.mp4"

    #beautygan
    video_path="/home/gzj/test/BrushNetSimple-main/out/2.mp4"


    # 光流法测试
    #optical_flow_score = calculate_mcss(video_path)
    #print(f"光流法MCSS评分：{optical_flow_score:.2f}")

    # 特征点法测试
    #feature_score = feature_based_mcss(video_path)
    #print(f"特征点法MCSS评分：{feature_score:.2f}")

    frame_folder = "/home/gzj/test/BrushNetSimple-main/out/20250401_191118/beauty/"
    feature_score_from_frames = feature_based_mcss_from_frames(frame_folder)
    print(f"帧特征点法MCSS评分：{feature_score_from_frames:.2f}")

