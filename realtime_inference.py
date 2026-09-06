"""
Step 7 (fixed): real-time silent-word recognition with spoken output.

This now matches your training data exactly: press SPACE, get a fixed
2-second recording window (same as record_data.py), and it's classified
right after - no automatic segmentation, no mismatch with training data.

How to use:
    1. Click the video window, confirm you see green dots on your lips.
    2. Press SPACE, then mouth one of your 5 words during the 2-second window.
    3. It classifies the result and speaks it out loud if confident enough.

Setup:
    pip install mediapipe opencv-python scikit-learn pandas joblib pyttsx3 pywin32

Requires (in the same folder):
    face_landmarker.task
    final_model.joblib   (from train_final_model.py)

Run:
    python realtime_inference.py

Press 'q' in the video window to quit.
"""

import math
import sys
import time
import traceback

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH_LANDMARKER = "face_landmarker.task"
MODEL_PATH_CLASSIFIER = "final_model.joblib"
WINDOW_NAME = "Silent Speech Demo - SPACE to capture, q to quit"

LEFT_CORNER = 61
RIGHT_CORNER = 291
TOP_INNER_LIP = 13
BOTTOM_INNER_LIP = 14

LIP_LANDMARK_INDICES = [
    61, 291, 0, 17, 269, 405, 181, 91, 146, 61,
    78, 308, 13, 14, 87, 317
]

RECORD_SECONDS = 2.0             # must match RECORD_SECONDS in record_data.py
CONFIDENCE_THRESHOLD = 0.4       # minimum classifier confidence to speak the result


def pixel_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def extract_features_from_segment(segment):
    # Must match the exact feature order/shape used in train_final_model.py.
    arr = np.array(segment)
    feats = []
    for col in range(3):  # width, height, ratio
        series = arr[:, col]
        feats.extend([
            series.mean(),
            series.std() if len(series) > 1 else 0.0,
            series.min(),
            series.max(),
            series.max() - series.min(),
        ])
    return feats


def main():
    print("Loading classifier...")
    clf = joblib.load(MODEL_PATH_CLASSIFIER)
    print(f"Classes: {list(clf.classes_)}")

    print("Setting up text-to-speech...")
    tts_engine = pyttsx3.init()

    print("Loading face landmark model...")
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH_LANDMARKER),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
    )

    print("Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, 100, 100)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)

    frame_timestamp_ms = 0

    recording = False
    record_start_time = None
    current_segment = []
    last_prediction_text = ""

    with FaceLandmarker.create_from_options(options) as landmarker:
        print("Ready. Click the video window, then press SPACE to capture a word.")

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
            else:
                cv2.putText(frame, "No face detected", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if recording:
                elapsed = time.time() - record_start_time
                if features is not None:
                    current_segment.append(features)

                cv2.putText(frame, f"CAPTURING {elapsed:.1f}s", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if elapsed >= RECORD_SECONDS:
                    recording = False
                    if len(current_segment) >= 5:
                        feats = extract_features_from_segment(current_segment)
                        probs = clf.predict_proba([feats])[0]
                        best_idx = int(np.argmax(probs))
                        best_word = clf.classes_[best_idx]
                        confidence = probs[best_idx]

                        print(f"Captured {len(current_segment)} frames -> "
                              f"{best_word} ({confidence * 100:.0f}% confidence)")

                        if confidence >= CONFIDENCE_THRESHOLD:
                            last_prediction_text = f"{best_word} ({confidence*100:.0f}%)"
                            tts_engine.say(best_word)
                            tts_engine.runAndWait()
                        else:
                            last_prediction_text = f"unsure ({best_word}? {confidence*100:.0f}%)"
                    else:
                        print("Too few frames had a detected face - try again, "
                              "make sure your face is clearly visible.")
                        last_prediction_text = "capture failed - no face"

                    current_segment = []
            else:
                cv2.putText(frame, "Ready - SPACE to capture", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if last_prediction_text:
                cv2.putText(frame, f"Last: {last_prediction_text}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" ") and not recording:
                recording = True
                record_start_time = time.time()
                current_segment = []

    cap.release()
    cv2.destroyAllWindows()
    print("Script finished cleanly.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nAn exception occurred:")
        traceback.print_exc()
        input("\nPress Enter to close this window...")
        sys.exit(1)