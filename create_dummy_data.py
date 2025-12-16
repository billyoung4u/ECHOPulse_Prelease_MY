import os
import cv2
import numpy as np
import pickle as pkl
from pathlib import Path

# 1. 定义数据存放路径（会自动创建）
ROOT_DIR = "MyToyDataset"
VIDEO_DIR = os.path.join(ROOT_DIR, "mp4")
ECG_DIR = os.path.join(ROOT_DIR, "ekg")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(ECG_DIR, exist_ok=True)

# 2. 生成 10 个样本
num_samples = 10
frames = 11  # 代码要求是 11 帧
height, width = 128, 128

print(f"正在生成 {num_samples} 个测试样本...")

# 3. 生成伪造视频和心电图数据？？？test？
for i in range(num_samples):
    # 文件名必须包含数字，因为代码里的 sort_key 依赖数字排序
    # 模拟真实文件名格式
    filename_base = f"patient_sample_{i}"

    # A. 生成伪造视频 (mp4)
    video_path = os.path.join(VIDEO_DIR, f"{filename_base}.mp4")
    # 使用 mp4v 编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 8, (width, height))

    for f in range(frames):
        # 生成噪点背景 + 移动的圆（模拟心脏跳动）
        frame = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        # 画一个圆，位置随帧数变化
        cv2.circle(frame, (64, 64), 20 + f * 2, (255, 255, 255), -1)
        out.write(frame)
    out.release()

    # B. 生成伪造心电图 (.pkl)
    ecg_path = os.path.join(ECG_DIR, f"{filename_base}.pkl")
    # 生成一段随机波形，长度 100，代码会自动把它拉伸到 2250
    fake_ecg = np.random.randn(100).astype(np.float64)  # 使用 float64
    with open(ecg_path, 'wb') as f:
        pkl.dump(fake_ecg, f)

print(f"✅ 数据生成完毕！路径: {os.path.abspath(ROOT_DIR)}")
print("请继续执行下一步修改代码。")