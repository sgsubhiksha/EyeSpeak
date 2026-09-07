"""
Real-time recognition using blendshape features.

Usage: click video window, press SPACE, mouth a word for 2 seconds,
it classifies and speaks the result.

Setup:
    pip install mediapipe opencv-python scikit-learn pandas joblib pyttsx3 pywin32

Requires in this folder: face_landmarker.task, final_model_v2.joblib

Run:
    python realtime_inference_v2.py
"""

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
MODEL_PATH_CLASSIFIER = "final_model_v2.joblib"
WINDOW_NAME = "Silent Speech Demo v2 - SPACE to capture, q to quit"
RECORD_SECONDS = 2.0
CONFIDENCE_THRESHOLD = 0.4

BLENDSHAPES_OF_INTEREST = [
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
]


def get_blendshape_scores(result):
    if not result.face_blendshapes:
        return None
    categories = result.face_blendshapes[0]
    scores = {c.category_name: c.score for c in categories}
    return {name: scores.get(name, 0.0) for name in BLENDSHAPES_OF_INTEREST}


def extract_features_from_segment(segment):
    feats = []
    for name in BLENDSHAPES_OF_INTEREST:
        vals = np.array([s[name] for s in segment])
        feats.extend([vals.mean(), vals.max(), vals.max() - vals.min()])
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
        output_face_blendshapes=True,
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
        print("Ready. Click video window, press SPACE to capture a word.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33

            scores = get_blendshape_scores(result)
            if scores is None:
                cv2.putText(frame, "No face detected", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if recording:
                elapsed = time.time() - record_start_time
                if scores is not None:
                    current_segment.append(scores)
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
                        print("Too few frames had a detected face - try again.")
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