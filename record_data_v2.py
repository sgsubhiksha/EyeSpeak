"""
Record labeled data using MediaPipe's built-in face blendshapes -
normalized, camera/distance-robust expression scores, instead of
hand-measured pixel distances.

Workflow: same as before.
    1. Run this script.
    2. Type the word, press Enter.
    3. Click video window, press SPACE to record ~2 seconds.
    4. Repeat 15-20x per word, press 'n' to switch words.

Setup:
    pip install mediapipe opencv-python
    face_landmarker.task must be in this folder.

Run:
    python record_data_v2.py
"""

import csv
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
WINDOW_NAME = "Recording v2 (blendshapes) - SPACE to record, q to quit"
DATA_DIR = "data_v2"
RECORD_SECONDS = 2.0

# Mouth-relevant blendshapes - covers opening, funneling/pursing,
# stretching, and lip curling, which together span most visible mouth shapes.
BLENDSHAPES_OF_INTEREST = [
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
]

# Just for the visual dots - not used in the actual features.
LIP_LANDMARK_INDICES = [
    61, 291, 0, 17, 269, 405, 181, 91, 146, 61,
    78, 308, 13, 14, 87, 317
]


def pixel_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))


def get_blendshape_scores(result):
    """Returns a dict {name: score} for the blendshapes we care about, or None."""
    if not result.face_blendshapes:
        return None
    categories = result.face_blendshapes[0]
    scores = {c.category_name: c.score for c in categories}
    return {name: scores.get(name, 0.0) for name in BLENDSHAPES_OF_INTEREST}


def save_recording(word, sequence, total_frames):
    word_dir = os.path.join(DATA_DIR, word)
    os.makedirs(word_dir, exist_ok=True)
    filename = os.path.join(word_dir, f"{int(time.time() * 1000)}.csv")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(BLENDSHAPES_OF_INTEREST)
        for row in sequence:
            writer.writerow([row[name] for name in BLENDSHAPES_OF_INTEREST])
    print(f"Saved {len(sequence)}/{total_frames} frames had a detected face -> {filename}")
    if len(sequence) == 0:
        print("  WARNING: no face detected during this recording.")


def main():
    print("Loading face landmark model (with blendshapes enabled)...")
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
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
    current_word = input("Type the first word to record, then press Enter: ").strip()
    print(f"Ready. Current word: '{current_word}'. Click the video window, then press SPACE.")

    recording = False
    record_start_time = None
    current_sequence = []
    frames_this_recording = 0

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

            scores = get_blendshape_scores(result)

            if scores is None:
                cv2.putText(frame, "No face detected", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                h, w, _ = frame.shape
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    for idx in LIP_LANDMARK_INDICES:
                        cx, cy = pixel_point(landmarks, idx, w, h)
                        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

                # Show a couple of key values live for sanity-checking.
                cv2.putText(frame, f"jawOpen: {scores['jawOpen']:.2f}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"mouthFunnel: {scores['mouthFunnel']:.2f}", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if recording:
                elapsed = time.time() - record_start_time
                frames_this_recording += 1
                if scores is not None:
                    current_sequence.append(scores)

                cv2.putText(frame, f"RECORDING '{current_word}' {elapsed:.1f}s (do NOT press q)",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if elapsed >= RECORD_SECONDS:
                    recording = False
                    save_recording(current_word, current_sequence, frames_this_recording)
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
                frames_this_recording = 0
            elif key == ord("n") and not recording:
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