import cv2
import numpy as np

# === グローバル変数 ===
click_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"Point {len(click_points)}: ({x}, {y})")

# === 映像読み込み ===
cap = cv2.VideoCapture("your_video.mp4")
ret, frame = cap.read()

if not ret:
    raise Exception("映像が読み込めません")

# === キャリブレーション ===
cv2.imshow("Click 4 corners of the pool", frame)
cv2.setMouseCallback("Click 4 corners of the pool", mouse_callback)

print("⚠️ プールの4隅をクリックしてください")
cv2.waitKey(0)
cv2.destroyAllWindows()

# === px/m計算（例: 横方向の距離から）===
def calc_scale(points):
    p1, p2 = points[0], points[1]  # 例えば左上・右上
    pixel_dist = np.linalg.norm(np.array(p1) - np.array(p2))
    real_length_m = 25.0  # 例：25mプール
    return real_length_m / pixel_dist  # m/px

scale_m_per_px = calc_scale(click_points)
print(f"Scale: {scale_m_per_px:.5f} m/px")

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture("your_video.mp4")

hand_positions = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # 右手首の座標 (landmark[16])
        wrist = results.pose_landmarks.landmark[16]
        hand_positions.append((wrist.x, wrist.y))  # 0-1の正規化座標

cap.release()
pose.close()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_stroke(timestamps, hand_x, hand_y, scale_m_per_px=1.0):
    # 1. 手部Y位置の極小値を入水として検出（谷を検出）
    inverted_y = -np.array(hand_y)
    stroke_peaks, _ = find_peaks(inverted_y, distance=10)  # 10以上のフレーム間隔
    stroke_times = np.array(timestamps)[stroke_peaks]
    
    if len(stroke_times) < 2:
        raise ValueError("検出されたストロークが2回未満です")

    # 2. ストローク周期（平均秒）
    intervals = np.diff(stroke_times)
    avg_cycle_sec = np.mean(intervals)
    stroke_rate_spm = 60 / avg_cycle_sec

    # 3. X移動距離からストローク長を推定
    x_positions = np.array(hand_x)[stroke_peaks]
    stroke_lengths_px = np.diff(x_positions)
    stroke_lengths_m = stroke_lengths_px * scale_m_per_px
    avg_stroke_length = np.mean(stroke_lengths_m)

    # 4. 速度（1周期あたりの距離 ÷ 時間）
    avg_velocity = avg_stroke_length / avg_cycle_sec

    return {
        "stroke_times": stroke_times,
        "stroke_rate_spm": stroke_rate_spm,
        "avg_stroke_length_m": avg_stroke_length,
        "avg_cycle_sec": avg_cycle_sec,
        "avg_velocity_mps": avg_velocity,
        "stroke_indices": stroke_peaks,
    }
def plot_strokes(timestamps, hand_y, stroke_indices, analysis_result):
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, hand_y, label="Hand Y Position")
    plt.scatter(np.array(timestamps)[stroke_indices], np.array(hand_y)[stroke_indices], color='red', label="Detected Strokes")
    plt.title("Hand Y Position and Stroke Detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Y Position")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

# 使用例
result = analyze_stroke(timestamps, hand_x, hand_y, scale_m_per_px=0.025)
plot_strokes(timestamps, hand_y, result["stroke_indices"], result)    