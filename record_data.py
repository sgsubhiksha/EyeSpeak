"""
Step 3: record labeled mouth-feature sequences for training data.

Workflow:
    1. Run this script.
    2. Type the word you're about to mouth (e.g. "yes") and press Enter.
    3. Get ready, then press SPACE in the video window to start a ~2 second recording.
    4. Mouth the word silently, consistently, during that window.
    5. It saves the sequence to data/<word>/<timestamp>.csv
    6. Repeat 20-30 times per word (you can keep typing new words without restarting).

Press 'q' in the video window to quit entirely.

Setup is the same as before:
    pip install mediapipe opencv-python
    face_landmarker.task must be in the same folder as this script.
"""

import csv
import math
import os
import sys
import time
import traceback

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "face_landmarker.task"
WINDOW_NAME = "Recording - SPACE to record, q to quit"
DATA_DIR = "data"
RECORD_SECONDS = 2.0

LEFT_CORNER = 61
RIGHT_CORNER = 291
TOP_INNER_LIP = 13
BOTTOM_INNER_LIP = 14

LIP_LANDMARK_INDICES = [
    61, 291, 0, 17, 269, 405, 181, 91, 146, 61,
    78, 308, 13, 14, 87, 317
]


def pixel_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def save_recording(word, sequence):
    word_dir = os.path.join(DATA_DIR, word)
    os.makedirs(word_dir, exist_ok=True)
    filename = os.path.join(word_dir, f"{int(time.time() * 1000)}.csv")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["width", "height", "ratio"])
        writer.writerows(sequence)
    print(f"Saved {len(sequence)} frames to {filename}")


def main():
    print("Script started.")
    print("Loading face landmark model...")
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
    )

    print("Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Close other apps using it and try again.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, 100, 100)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)

    frame_timestamp_ms = 0

    current_word = input("Type the first word to record, then press Enter: ").strip()
    print(f"Ready. Current word: '{current_word}'. Click the video window, then press SPACE to record.")

    recording = False
    record_start_time = None
    current_sequence = []

    with FaceLandmarker.create_from_options(options) as landmarker:
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

            # Recording logic
            if recording:
                elapsed = time.time() - record_start_time
                if features is not None:
                    current_sequence.append(features)

                cv2.putText(frame, f"RECORDING '{current_word}' {elapsed:.1f}s",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if elapsed >= RECORD_SECONDS:
                    recording = False
                    save_recording(current_word, current_sequence)
                    current_sequence = []
                    print(f"Press SPACE to record '{current_word}' again, "
                          f"or press 'n' to type a new word.")
            else:
                cv2.putText(frame, f"Word: '{current_word}' - SPACE to record",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" ") and not recording:
                recording = True
                record_start_time = time.time()
                current_sequence = []
            elif key == ord("n") and not recording:
                # Release window focus briefly to type in the console.
                current_word = input("Type the next word to record, then press Enter: ").strip()
                print(f"Ready. Current word: '{current_word}'. Click the video window, then press SPACE.")

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