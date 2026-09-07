"""
Step 2: extract mouth-shape features (width, height, aspect ratio) live from webcam.

Same setup as before:
    pip install mediapipe opencv-python
    face_landmarker.task must be in the same folder as this script.

Run:
    python lip_features_test.py

Watch the on-screen numbers change as you silently mouth different words —
that's the confirmation that this signal is usable before we build a classifier on it.

Press 'q' (with the video window focused) to quit.
"""

import math
import sys
import traceback

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "face_landmarker.task"
WINDOW_NAME = "Mouth Features - press q to quit"

# Landmark indices used for feature calculation:
LEFT_CORNER = 61
RIGHT_CORNER = 291
TOP_INNER_LIP = 13
BOTTOM_INNER_LIP = 14

# Full lip ring, just for drawing the dots like before.
LIP_LANDMARK_INDICES = [
    61, 291, 0, 17, 269, 405, 181, 91, 146, 61,
    78, 308, 13, 14, 87, 317
]


def pixel_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


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
    frame_count = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        print("Model loaded. Window should appear now — mouth some different words and watch the numbers.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33
            frame_count += 1

            h, w, _ = frame.shape

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

                cv2.putText(frame, f"Width: {mouth_width:.1f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Height: {mouth_height:.1f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Ratio: {aspect_ratio:.3f}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Print to console every ~15 frames so you can see values change
                # in the terminal too, without flooding it every single frame.
                if frame_count % 15 == 0:
                    print(f"width={mouth_width:.1f}  height={mouth_height:.1f}  ratio={aspect_ratio:.3f}")
            else:
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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