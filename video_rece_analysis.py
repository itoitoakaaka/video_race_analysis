import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

click_points = []


def mouse_callback(event, x, y, flags, param):
    """Collect calibration clicks in pixel coordinates."""
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"Point {len(click_points)}: ({x}, {y})")


def calc_scale(points, known_distance_m):
    """Return metres per pixel from two calibration points."""
    if len(points) < 2:
        raise ValueError("Two calibration points are required.")
    p1, p2 = np.asarray(points[0], dtype=float), np.asarray(points[1], dtype=float)
    pixel_dist = np.linalg.norm(p1 - p2)
    if pixel_dist <= 0:
        raise ValueError("Calibration points must be different.")
    return known_distance_m / pixel_dist


def calibrate_video(video_path, known_distance_m=25.0):
    """Calibrate a fixed-camera video using two points with known distance."""
    global click_points
    click_points = []

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read video.")

    window = "Click 2 points spanning the known pool distance"
    cv2.imshow(window, frame)
    cv2.setMouseCallback(window, mouse_callback)

    print(
        f"Click two points whose real-world separation is {known_distance_m:.2f} m, "
        "then press any key."
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(click_points) < 2:
        raise RuntimeError("Calibration requires two clicks.")

    return calc_scale(click_points[:2], known_distance_m)


def track_pose(video_path):
    """Track the right wrist in pixel coordinates and preserve video timestamps."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Install mediapipe: pip install mediapipe") from exc

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        pose.close()
        cap.release()
        raise RuntimeError("Could not determine video frame rate.")

    timestamps = []
    hand_x_px = []
    hand_y_px = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            wrist = results.pose_landmarks.landmark[16]
            timestamps.append(frame_index / fps)
            hand_x_px.append(wrist.x * width)
            hand_y_px.append(wrist.y * height)

        frame_index += 1

    cap.release()
    pose.close()

    if len(timestamps) < 3:
        raise RuntimeError("Too few valid wrist detections.")

    return (
        np.asarray(timestamps, dtype=float),
        np.asarray(hand_x_px, dtype=float),
        np.asarray(hand_y_px, dtype=float),
    )


def analyze_stroke(
    timestamps,
    hand_x_px,
    hand_y_px,
    scale_m_per_px,
    min_cycle_interval_s=0.4,
):
    """Detect wrist-position cycles and calculate exploratory movement metrics.

    The horizontal-displacement metric is a wrist-trajectory proxy, not a
    validated estimate of whole-body swimming stroke length.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    hand_x_px = np.asarray(hand_x_px, dtype=float)
    hand_y_px = np.asarray(hand_y_px, dtype=float)

    dt = np.median(np.diff(timestamps))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid timestamps.")

    min_peak_distance = max(1, int(round(min_cycle_interval_s / dt)))
    inverted_y = -hand_y_px
    stroke_peaks, _ = find_peaks(inverted_y, distance=min_peak_distance)
    stroke_times = timestamps[stroke_peaks]

    if len(stroke_times) < 2:
        raise ValueError("Fewer than two cycles were detected.")

    intervals = np.diff(stroke_times)
    avg_cycle_sec = float(np.mean(intervals))
    stroke_rate_cpm = 60.0 / avg_cycle_sec

    x_positions_px = hand_x_px[stroke_peaks]
    cycle_displacement_m = np.abs(np.diff(x_positions_px)) * scale_m_per_px
    avg_wrist_displacement_m = float(np.mean(cycle_displacement_m))
    wrist_displacement_velocity_mps = avg_wrist_displacement_m / avg_cycle_sec

    return {
        "stroke_times": stroke_times,
        "stroke_rate_cpm": stroke_rate_cpm,
        "avg_cycle_sec": avg_cycle_sec,
        "avg_wrist_displacement_m": avg_wrist_displacement_m,
        "wrist_displacement_velocity_mps": wrist_displacement_velocity_mps,
        "stroke_indices": stroke_peaks,
    }


def plot_strokes(timestamps, hand_y_px, stroke_indices):
    """Save detected wrist cycles for visual quality control."""
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, hand_y_px, label="Right wrist Y (px)")
    plt.scatter(
        timestamps[stroke_indices],
        hand_y_px[stroke_indices],
        label="Detected cycles",
    )
    plt.title("Right-wrist trajectory and detected cycles")
    plt.xlabel("Time (s)")
    plt.ylabel("Y position (px)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("stroke_analysis.png", dpi=200)
    print("Saved stroke_analysis.png")


def main():
    if len(sys.argv) < 2:
        print("Usage: python video_rece_analysis.py <video_path> [known_distance_m]")
        raise SystemExit(1)

    video_path = sys.argv[1]
    known_distance_m = float(sys.argv[2]) if len(sys.argv) >= 3 else 25.0

    scale_m_per_px = calibrate_video(video_path, known_distance_m)
    print(f"Calibration: {scale_m_per_px:.6f} m/px")

    timestamps, hand_x_px, hand_y_px = track_pose(video_path)
    result = analyze_stroke(
        timestamps,
        hand_x_px,
        hand_y_px,
        scale_m_per_px=scale_m_per_px,
    )
    plot_strokes(timestamps, hand_y_px, result["stroke_indices"])

    print(f"Cycle rate: {result['stroke_rate_cpm']:.2f} cycles/min")
    print(f"Mean cycle time: {result['avg_cycle_sec']:.3f} s")
    print(
        "Mean horizontal wrist displacement per detected cycle: "
        f"{result['avg_wrist_displacement_m']:.3f} m"
    )


if __name__ == "__main__":
    main()
