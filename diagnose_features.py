"""
Diagnostic: capture one live word and compare its features directly against
your training data for every word, to see exactly why predictions are off.

Setup: same as realtime_inference.py (needs face_landmarker.task in this folder).
Does NOT need final_model.joblib - it reads straight from your data/ folder.

Run:
    python diagnose_features.py
"""

import glob
import math
import os
import sys
import time
import traceback

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH_LANDMARKER = "face_landmarker.task"
DATA_DIR = "data"
WINDOW_NAME = "Diagnostic - SPACE to capture one word, q to quit"
RECORD_SECONDS = 2.0

LEFT_CORNER = 61
RIGHT_CORNER = 291
TOP_INNER_LIP = 13
BOTTOM_INNER_LIP = 14

LIP_LANDMARK_INDICES = [
    61, 291, 0, 17, 269, 405, 181, 91, 146, 61,
    78, 308, 13, 14, 87, 317
]

FEATURE_NAMES = [
    "width_mean", "width_std", "width_min", "width_max", "width_range",
    "height_mean", "height_std", "height_min", "height_max", "height_range",
    "ratio_mean", "ratio_std", "ratio_min", "ratio_max", "ratio_range",
]


def pixel_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def extract_features_from_segment(segment):
    arr = np.array(segment)
    feats = []
    for col in range(3):
        series = arr[:, col]
        feats.extend([
            series.mean(),
            series.std() if len(series) > 1 else 0.0,
            series.min(),
            series.max(),
            series.max() - series.min(),
        ])
    return np.array(feats)


def extract_features_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
    feats = []
    for col in ["width", "height", "ratio"]:
        series = df[col]
        feats.extend([
            series.mean(),
            series.std() if len(series) > 1 else 0.0,
            series.min(),
            series.max(),
            series.max() - series.min(),
        ])
    return np.array(feats)


def load_training_stats():
    stats = {}
    for word in sorted(os.listdir(DATA_DIR)):
        word_dir = os.path.join(DATA_DIR, word)
        if not os.path.isdir(word_dir):
            continue
        vectors = []
        for csv_path in glob.glob(os.path.join(word_dir, "*.csv")):
            feats = extract_features_from_csv(csv_path)
            if feats is not None:
                vectors.append(feats)
        if vectors:
            vectors = np.array(vectors)
            stats[word] = {
                "mean": vectors.mean(axis=0),
                "std": vectors.std(axis=0) + 1e-6,  # avoid div by zero
            }
    return stats


def capture_one_live_segment(landmarker, cap):
    frame_timestamp_ms = 0
    recording = False
    record_start_time = None
    segment = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, 100, 100)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)

    print("Click the video window, then press SPACE to capture one word (2 seconds).")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33

        h, w, _ = frame.shape
        features = None

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            for idx in LIP_LANDMARK_INDICES:
                cx, cy = pixel_point(landmarks, idx, w, h)
                cv2.circle(frame, (int(cx), int(cy)), 2, (0, 255, 0), -1)

            left = pixel_point(landmarks, LEFT_CORNER, w, h)
            right = pixel_point(landmarks, RIGHT_CORNER, w, h)
            top = pixel_point(landmarks, TOP_INNER_LIP, w, h)
            bottom = pixel_point(landmarks, BOTTOM_INNER_LIP, w, h)
            mouth_width = distance(left, right)
            mouth_height = distance(top, bottom)
            aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0.0
            features = (mouth_width, mouth_height, aspect_ratio)

        if recording:
            elapsed = time.time() - record_start_time
            if features is not None:
                segment.append(features)
            cv2.putText(frame, f"CAPTURING {elapsed:.1f}s", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if elapsed >= RECORD_SECONDS:
                cv2.destroyAllWindows()
                return segment
        else:
            cv2.putText(frame, "Ready - SPACE to capture", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return None
        elif key == ord(" ") and not recording:
            recording = True
            record_start_time = time.time()
            segment = []

    return None


def main():
    print("Loading training data stats...")
    stats = load_training_stats()
    print(f"Loaded stats for words: {list(stats.keys())}")

    print("Loading face landmark model...")
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH_LANDMARKER),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
    )
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    with FaceLandmarker.create_from_options(options) as landmarker:
        segment = capture_one_live_segment(landmarker, cap)

    cap.release()
    cv2.destroyAllWindows()

    if not segment or len(segment) < 5:
        print("Capture failed or too short - try again.")
        return

    live_feats = extract_features_from_segment(segment)

    print(f"\nCaptured {len(segment)} frames.\n")
    print("Live feature values vs. each word's training range (mean +/- std):\n")

    distances = {}
    for word, s in stats.items():
        z_scores = (live_feats - s["mean"]) / s["std"]
        euclidean_dist = float(np.linalg.norm(z_scores))
        distances[word] = euclidean_dist

    for word, dist in sorted(distances.items(), key=lambda x: x[1]):
        print(f"  {word}: distance={dist:.2f}  {'<-- closest' if dist == min(distances.values()) else ''}")

    print("\nPer-feature comparison against training data for the CLOSEST word:")
    closest_word = min(distances, key=distances.get)
    s = stats[closest_word]
    print(f"(closest match: {closest_word})\n")
    print(f"{'feature':15s} {'live':>10s} {'train_mean':>12s} {'train_std':>10s} {'z-score':>8s}")
    for name, live_val, mean_val, std_val in zip(FEATURE_NAMES, live_feats, s["mean"], s["std"]):
        z = (live_val - mean_val) / std_val
        flag = "  <-- big drift" if abs(z) > 2 else ""
        print(f"{name:15s} {live_val:10.2f} {mean_val:12.2f} {std_val:10.2f} {z:8.2f}{flag}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nAn exception occurred:")
        traceback.print_exc()
        input("\nPress Enter to close this window...")
        sys.exit(1)