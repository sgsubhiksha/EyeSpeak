import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from PIL import Image, ImageTk

try:
    import winsound
except ImportError:
    winsound = None

# ============================== CONFIG ====================================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

LEFT_EYE = [33, 133, 159, 145, 160, 144]
RIGHT_EYE = [362, 263, 386, 374, 387, 373]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# 13-point calibration: 9 grid points + four edge midpoints.
# 25-point calibration grid. The dense grid improves edge and corner mapping.
CAL_VALUES = (0.06, 0.28, 0.50, 0.72, 0.94)
CAL_POINTS = [(x, y) for y in CAL_VALUES for x in CAL_VALUES]

pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = True


# ============================== FILTERS ==================================
class EMAFilter:
    def __init__(self, alpha=0.30):
        self.alpha = alpha
        self.value = None

    def reset(self):
        self.value = None

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.value is None:
            self.value = x.copy()
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value.copy()


class OneEuroFilter2D:
    """Lightweight adaptive filter: stable when still, responsive when moving."""
    def __init__(self, min_cutoff=1.5, beta=0.35, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = np.zeros(2, dtype=float)
        self.t_prev = None

    @staticmethod
    def _alpha(dt, cutoff):
        if dt <= 0:
            return 1.0
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = np.zeros(2, dtype=float)
        self.t_prev = None

    def update(self, x, now=None):
        x = np.asarray(x, dtype=float)
        if now is None:
            now = time.perf_counter()
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x.copy()
            self.t_prev = now
            return x.copy()

        dt = float(np.clip(now - self.t_prev, 1e-3, 0.2))
        raw_dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx = a_d * raw_dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx)
        a = self._alpha(dt, cutoff)
        y = a * x + (1.0 - a) * self.x_prev
        self.x_prev = y.copy()
        self.dx_prev = dx
        self.t_prev = now
        return y.copy()


# ============================== CALIBRATION ===============================
class CalibrationModel:
    """Ridge regression over gaze + head-pose features with quadratic terms."""
    def __init__(self, ridge=2e-3):
        self.coef = None
        self.ridge = ridge
        self.rmse = None
        self.max_error = None
        self.feature_mean = None
        self.feature_std = None

    @staticmethod
    def _features(raw):
        x = np.asarray(raw, dtype=float)
        gx, gy, hx, hy = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        # Head pose terms let the regression compensate for small head motion.
        return np.column_stack([
            gx, gy, hx, hy,
            gx * gx, gy * gy, hx * hx, hy * hy,
            gx * gy, gx * hx, gx * hy, gy * hx, gy * hy, hx * hy,
            np.ones(len(x)),
        ])

    def fit(self, raw_points, screen_points):
        X = np.asarray(raw_points, dtype=float)
        Y = np.asarray(screen_points, dtype=float)
        if len(X) < 9:
            raise ValueError("Not enough calibration points. Please complete at least 9 calibration points.")

        A = self._features(X)
        # Standardize non-constant columns to make ridge well-conditioned.
        self.feature_mean = A[:, :-1].mean(axis=0)
        self.feature_std = A[:, :-1].std(axis=0)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        A2 = A.copy()
        A2[:, :-1] = (A2[:, :-1] - self.feature_mean) / self.feature_std

        reg = self.ridge * np.eye(A2.shape[1])
        reg[-1, -1] = 0.0
        self.coef = np.linalg.solve(A2.T @ A2 + reg, A2.T @ Y)

        pred = A2 @ self.coef
        errors = np.linalg.norm(pred - Y, axis=1)
        self.rmse = float(np.sqrt(np.mean(errors ** 2)))
        self.max_error = float(np.max(errors))

    def predict(self, raw):
        if self.coef is None:
            return None
        x = np.asarray(raw, dtype=float).reshape(1, -1)
        A = self._features(x)
        A[:, :-1] = (A[:, :-1] - self.feature_mean) / self.feature_std
        return (A @ self.coef)[0]


# ============================== BLINK =====================================
class BlinkDetector:
    def __init__(self):
        self.closed_since = None
        self.was_closed = False
        self.last_click_time = 0.0
        self.cooldown = 0.65
        self.left_min = 0.22
        self.right_min = 0.80

    def update(self, eyes_closed, now):
        event = None
        if eyes_closed and not self.was_closed:
            self.closed_since = now
        if not eyes_closed and self.was_closed and self.closed_since is not None:
            duration = now - self.closed_since
            if now - self.last_click_time >= self.cooldown:
                if duration >= self.right_min:
                    event = "right"
                elif duration >= self.left_min:
                    event = "left"
                if event:
                    self.last_click_time = now
            self.closed_since = None
        self.was_closed = eyes_closed
        return event


# ============================== TRACKER ===================================
class EyeTracker:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

    @staticmethod
    def point(lm, idx, w, h):
        p = lm[idx]
        return np.array([p.x * w, p.y * h], dtype=float)

    def iris_center(self, lm, ids, w, h):
        return np.mean([self.point(lm, i, w, h) for i in ids], axis=0)

    def eye_features(self, lm, ids, iris_ids, w, h):
        a = self.point(lm, ids[0], w, h)
        b = self.point(lm, ids[1], w, h)
        iris = self.iris_center(lm, iris_ids, w, h)
        center = (a + b) * 0.5
        vec = b - a
        width = np.linalg.norm(vec) + 1e-9
        ex = vec / width
        ey = np.array([-ex[1], ex[0]])
        delta = iris - center

        # Local eye coordinates are much less sensitive to face distance.
        gx = 0.5 + float(np.dot(delta, ex) / width)
        gy = 0.5 + float(np.dot(delta, ey) / width) * 1.55
        return np.clip([gx, gy], 0.0, 1.0), iris, width

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None, frame

        lm = result.multi_face_landmarks[0].landmark
        lf, li, lw = self.eye_features(lm, LEFT_EYE, LEFT_IRIS, w, h)
        rf, ri, rw = self.eye_features(lm, RIGHT_EYE, RIGHT_IRIS, w, h)

        gaze = (lf + rf) / 2.0
        eye_mid = (self.point(lm, 33, w, h) + self.point(lm, 263, w, h)) * 0.5
        eye_width = (lw + rw) * 0.5
        nose = self.point(lm, 1, w, h)

        # Head-pose proxies. These are deliberately normalized by eye width.
        head_x = float((nose[0] - eye_mid[0]) / (eye_width + 1e-9))
        head_y = float((nose[1] - eye_mid[1]) / (eye_width + 1e-9))

        # A second face-scale signal helps when the user leans toward/away.
        face_scale = float(np.clip(eye_width / max(w, h), 0.01, 0.5))

        def ear(ids):
            p1 = self.point(lm, ids[0], w, h)
            p2 = self.point(lm, ids[1], w, h)
            p3 = self.point(lm, ids[2], w, h)
            p4 = self.point(lm, ids[3], w, h)
            return np.linalg.norm(p3 - p4) / (np.linalg.norm(p1 - p2) + 1e-9)

        le = ear(LEFT_EYE)
        re = ear(RIGHT_EYE)
        eyes_closed = le < 0.18 and re < 0.18

        # Do not feed blink frames into the gaze filter later.
        raw = np.array([gaze[0], gaze[1], head_x, head_y], dtype=float)

        for p in (li, ri):
            cv2.circle(frame, tuple(p.astype(int)), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"Gaze: {gaze[0]:.3f}, {gaze[1]:.3f}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"Head: {head_x:+.3f}, {head_y:+.3f}",
                    (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)
        cv2.putText(frame, f"EAR: {le:.2f}/{re:.2f}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(frame, "BLINK - gaze frozen" if eyes_closed else "TRACKING", 
                    (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if eyes_closed else (0, 255, 0), 2)

        return {
            "gaze": gaze,
            "raw": raw,
            "eyes_closed": eyes_closed,
            "left_ear": le,
            "right_ear": re,
            "face_scale": face_scale,
        }, frame



# ============================== EMERGENCY =================================
class EmergencySystem:
    """
    Automatic accessibility emergency triggers:
      1) three full blink cycles within BLINK_WINDOW seconds;
      2) both eyes continuously closed for EYE_CLOSED_ALARM_TIME seconds.

    The alarm is latched once triggered and continues independently of eye
    tracking until the user presses F8 / Emergency Stop.
    """
    def __init__(self, app):
        self.app = app
        self.BLINK_WINDOW = 4.0
        self.EYE_CLOSED_ALARM_TIME = 3.0

        self.active = False
        self.blink_times = deque(maxlen=3)
        self.last_blink_time = 0.0
        self.eyes_closed_since = None

        self.alarm_thread = None
        self.stop_alarm = threading.Event()

    def update_eye_closure(self, eyes_closed, now):
        if self.active:
            return

        if eyes_closed:
            if self.eyes_closed_since is None:
                self.eyes_closed_since = now

            elapsed = now - self.eyes_closed_since
            remaining = max(0.0, self.EYE_CLOSED_ALARM_TIME - elapsed)

            if elapsed >= self.EYE_CLOSED_ALARM_TIME:
                self.trigger("Both eyes closed continuously for 3 seconds")
            else:
                self.app.emergency_state_var.set(
                    f"EYE-CLOSURE TIMER • {elapsed:.1f}/3.0 s"
                )
                self.app.emergency_progress_var.set(
                    f"Emergency alarm in {remaining:.1f} s if both eyes remain closed"
                )
        else:
            if self.eyes_closed_since is not None:
                self.eyes_closed_since = None
            if not self.active:
                self.app.emergency_state_var.set("EMERGENCY MONITORING ACTIVE")
                self.app.emergency_progress_var.set(
                    "3 full blinks within 4 seconds OR both eyes closed for 3 seconds"
                )

    def blink(self, now):
        """Count every completed blink (eye closure followed by opening)."""
        if self.active:
            return

        # Prevent one noisy frame/release from being counted twice.
        if self.last_blink_time and now - self.last_blink_time < 0.22:
            return

        self.last_blink_time = now
        self.blink_times.append(now)

        while self.blink_times and now - self.blink_times[0] > self.BLINK_WINDOW:
            self.blink_times.popleft()

        count = len(self.blink_times)
        self.app.emergency_state_var.set(f"BLINK EMERGENCY • {count}/3")
        self.app.emergency_progress_var.set(
            f"Completed blinks detected: {count}/3 • window: 4 seconds"
        )

        if count >= 3:
            self.trigger("3 full blinks detected")

    def reset_sequence(self):
        self.blink_times.clear()
        self.last_blink_time = 0.0
        self.eyes_closed_since = None

    def trigger(self, reason="Emergency detected"):
        if self.active:
            return

        self.active = True
        self.stop_alarm.clear()

        self.app.emergency_state_var.set("🚨 EMERGENCY ALARM ACTIVE")
        self.app.emergency_progress_var.set(
            f"LOUD ALARM • {reason} • press F8 / Emergency Stop to silence"
        )
        self.app.status_var.set("🚨 EMERGENCY ALARM ACTIVE")

        self.alarm_thread = threading.Thread(
            target=self._alarm_loop,
            daemon=True
        )
        self.alarm_thread.start()

    @staticmethod
    def _set_windows_volume_max():
        """Raise Windows master volume to maximum using the media-volume key."""
        if not winsound:
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            VK_VOLUME_UP = 0xAF
            KEYEVENTF_KEYUP = 0x0002

            # Repeated volume-up events bring the Windows master volume to 100%.
            for _ in range(60):
                user32.keybd_event(VK_VOLUME_UP, 0, 0, 0)
                user32.keybd_event(VK_VOLUME_UP, 0, KEYEVENTF_KEYUP, 0)
        except Exception:
            pass

    def _alarm_loop(self):
        # IMPORTANT: this thread does not depend on the camera, eyes, or face.
        # Once triggered, it continues until stop_alarm is explicitly set.
        if winsound:
            try:
                self._set_windows_volume_max()

                # Use a repeating Windows system alarm at maximum master volume.
                # SystemHand is a built-in Windows alert sound.
                winsound.PlaySound(
                    "SystemHand",
                    winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_LOOP
                )
                self.stop_alarm.wait()

                winsound.PlaySound(None, 0)
                return
            except Exception:
                try:
                    winsound.PlaySound(None, 0)
                except Exception:
                    pass

        # Fallback: repeating two-tone buzzer.
        while not self.stop_alarm.is_set():
            try:
                if winsound:
                    winsound.Beep(1800, 650)
                    if self.stop_alarm.wait(0.08):
                        break
                    winsound.Beep(900, 650)
                else:
                    print("\a", end="", flush=True)
                    if self.stop_alarm.wait(0.5):
                        break
            except Exception:
                if self.stop_alarm.wait(0.5):
                    break

    def stop(self):
        self.stop_alarm.set()
        try:
            if winsound:
                winsound.PlaySound(None, 0)
        except Exception:
            pass

        self.active = False
        self.reset_sequence()

        self.app.emergency_state_var.set("EMERGENCY MONITORING ACTIVE")
        self.app.emergency_progress_var.set(
            "Ready: 3 full blinks / 4 seconds OR both eyes closed / 3 seconds"
        )
        self.app.status_var.set("Tracking active — emergency monitor reset")


# ============================== APP =======================================
class EyeCursorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EYE Cursor PRO — Advanced Accessibility Controller")
        self.root.geometry("1150x760")
        self.root.minsize(980, 680)
        self.screen_w, self.screen_h = pyautogui.size()

        # Robust Windows webcam initialization: try several backends and camera indices.
        self.cap = self._open_camera()
        if self.cap is None:
            messagebox.showerror(
                "Camera error",
                "Could not open the webcam.\n\n"
                "Close Camera/Zoom/Teams/Meet and check Windows Camera permissions.\n"
                "Then restart EYE Cursor."
            )
            self.root.after(100, self.close)
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # MJPG often prevents black/no-frame capture on integrated Windows webcams.
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        self.tracker = EyeTracker()
        self.calibration = CalibrationModel()
        self.one_euro = OneEuroFilter2D(min_cutoff=1.45, beta=0.42)
        self.ema = EMAFilter(alpha=0.28)
        self.raw_history = deque(maxlen=7)
        self.blink = BlinkDetector()

        self.tracking = False
        self.running = True
        self.current_gaze = None
        self.current_screen = None
        self.last_move = None
        self.last_valid_raw = None
        self.face_lost_since = None
        self.deadzone = 1.5

        self.sensitivity = tk.DoubleVar(value=1.0)
        self.smoothing = tk.DoubleVar(value=0.28)
        self.blink_enabled = tk.BooleanVar(value=False)
        self.dwell_enabled = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready — calibrate before tracking.")
        self.gaze_var = tk.StringVar(value="—")
        self.cursor_var = tk.StringVar(value="—")
        self.quality_var = tk.StringVar(value="Calibration: not done")
        self.dwell_start = None
        self.dwell_last_position = None
        self.dwell_threshold = 45
        self.dwell_time = 1.35

        # Automatic emergency subsystem.
        self.emergency_state_var = tk.StringVar(value="EMERGENCY MONITORING ACTIVE")
        self.emergency_progress_var = tk.StringVar(
            value="3 full blinks within 4 seconds OR both eyes closed for 3 seconds"
        )
        self.emergency = EmergencySystem(self)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(10, self.update_frame)

    def _open_camera(self):
        """Find a working webcam using multiple Windows OpenCV backends."""
        candidates = [
            (0, cv2.CAP_DSHOW), (0, cv2.CAP_MSMF), (0, cv2.CAP_ANY),
            (1, cv2.CAP_DSHOW), (1, cv2.CAP_MSMF), (1, cv2.CAP_ANY),
        ]
        for index, backend in candidates:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            # Verify that we can actually receive a frame, not just open a handle.
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap
            cap.release()
        return None

    def _restart_camera(self):
        """Release and reopen the webcam after fullscreen calibration.
        This fixes webcams that stop delivering frames after the calibration window closes.
        """
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        # Give Windows/DirectShow a moment to release the old camera handle.
        time.sleep(0.15)

        new_cap = self._open_camera()
        if new_cap is None:
            self.status_var.set("Camera reconnect failed — retrying...")
            self.root.after(300, self._retry_camera)
            return False

        self.cap = new_cap
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        # Reset vision filters so stale calibration frames cannot affect tracking.
        self.one_euro.reset()
        self.ema.reset()
        self.raw_history.clear()
        self.last_move = None
        self.face_lost_since = None
        return True

    def _retry_camera(self):
        if not self.running:
            return
        if self._restart_camera():
            self.status_var.set("Camera recovered — ready.")
        else:
            self.root.after(500, self._retry_camera)

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"))
        style.configure("Section.TLabelframe", padding=12)

        header = ttk.Frame(self.root, padding=16)
        header.pack(fill="x")
        ttk.Label(header, text="EYE Cursor PRO", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="  Accessible webcam eye-control", font=("Segoe UI", 11)).pack(side="left", pady=(8, 0))
        ttk.Label(header, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).pack(side="right")

        body = ttk.Frame(self.root, padding=(16, 0, 16, 16))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(body)
        right.pack(side="right", fill="y", padx=(14, 0))

        camera_box = ttk.LabelFrame(left, text="LIVE CAMERA • IRIS TRACKING", style="Section.TLabelframe")
        camera_box.pack(fill="both", expand=True)
        self.preview = ttk.Label(camera_box, anchor="center")
        self.preview.pack(fill="both", expand=True)

        info = ttk.LabelFrame(left, text="LIVE DIAGNOSTICS", style="Section.TLabelframe")
        info.pack(fill="x", pady=(10, 0))
        ttk.Label(info, text="Gaze").grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.gaze_var, font=("Consolas", 11, "bold")).grid(row=0, column=1, sticky="w", padx=(8, 25))
        ttk.Label(info, text="Cursor").grid(row=0, column=2, sticky="w")
        ttk.Label(info, textvariable=self.cursor_var, font=("Consolas", 11, "bold")).grid(row=0, column=3, sticky="w", padx=(8, 25))
        ttk.Label(info, textvariable=self.quality_var).grid(row=0, column=4, sticky="w")

        action = ttk.LabelFrame(right, text="TRACKING", style="Section.TLabelframe")
        action.pack(fill="x")
        ttk.Button(action, text="🎯  Calibrate (25 points)", command=self.start_calibration).pack(fill="x", pady=4)
        self.start_btn = ttk.Button(action, text="▶  Start Tracking", command=self.toggle_tracking)
        self.start_btn.pack(fill="x", pady=4)

        settings = ttk.LabelFrame(right, text="CONTROL SETTINGS", style="Section.TLabelframe")
        settings.pack(fill="x", pady=(10, 0))
        ttk.Label(settings, text="Sensitivity").pack(anchor="w")
        ttk.Scale(settings, from_=0.75, to=1.35, variable=self.sensitivity, orient="horizontal").pack(fill="x")
        ttk.Label(settings, text="Smoothing").pack(anchor="w", pady=(10, 0))
        ttk.Scale(settings, from_=0.10, to=0.65, variable=self.smoothing, orient="horizontal").pack(fill="x")

        emergency_box = ttk.LabelFrame(
            right, text="🚨 EMERGENCY ALARM", style="Section.TLabelframe"
        )
        emergency_box.pack(fill="x", pady=(10, 0), ipady=8)

        ttk.Button(
            emergency_box,
            text="⏹  Stop Emergency Alarm",
            command=self.stop_emergency_alarm
        ).pack(fill="x", pady=4)

        self.root.bind("<F8>", lambda e: self.stop_tracking())
        self.root.bind("<Escape>", lambda e: self.stop_tracking())

    # -------------------------- tracking ---------------------------------
    def toggle_tracking(self):
        if not self.tracking:
            if self.calibration.coef is None:
                messagebox.showwarning("Calibration required", "Complete calibration first.")
                return
            self.tracking = True
            self.one_euro.reset()
            self.ema.reset()
            self.raw_history.clear()
            self.last_move = None
            self.start_btn.config(text="⏸  Pause Tracking")
            self.status_var.set("TRACKING ACTIVE")
        else:
            self.tracking = False
            self.emergency.reset_sequence()
            self.emergency_state_var.set("LOOK AT BUTTON TO ARM")
            self.start_btn.config(text="▶  Start Tracking")
            self.status_var.set("Paused")

    def stop_tracking(self):
        self.tracking = False
        self.emergency.reset_sequence()
        self.emergency_state_var.set("EMERGENCY MONITORING ACTIVE")
        self.emergency_progress_var.set(
            "3 full blinks within 4 seconds OR both eyes closed for 3 seconds"
        )
        self.one_euro.reset()
        self.ema.reset()
        self.raw_history.clear()
        self.last_move = None
        self.start_btn.config(text="▶  Start Tracking")
        self.status_var.set("Paused / Safe")

    def stop_emergency_alarm(self):
        self.emergency.stop()
        self.status_var.set("Emergency alarm stopped")

    def recenter(self):
        self.one_euro.reset()
        self.ema.reset()
        self.raw_history.clear()
        self.last_move = None
        self.status_var.set("Filters recentered")

    # -------------------------- calibration ------------------------------
    def start_calibration(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open the webcam.")
            return
        self.tracking = False
        self.emergency.reset_sequence()
        self.emergency_state_var.set("EMERGENCY MONITORING ACTIVE")
        self.emergency_progress_var.set(
            "3 full blinks within 4 seconds OR both eyes closed for 3 seconds"
        )
        self.start_btn.config(text="▶  Start Tracking")
        self.cal_points_raw = []
        self.cal_points_screen = []
        self.cal_index = 0
        self.calibration_window = tk.Toplevel(self.root)
        self.calibration_window.title("EYE Cursor PRO Calibration — 25 Points")
        self.calibration_window.attributes("-fullscreen", True)
        self.calibration_window.configure(bg="black")
        self.calibration_window.protocol("WM_DELETE_WINDOW", self.finish_calibration_cancel)
        self.cal_canvas = tk.Canvas(self.calibration_window, bg="black", highlightthickness=0)
        self.cal_canvas.pack(fill="both", expand=True)
        self.cal_instruction = self.cal_canvas.create_text(
            self.screen_w // 2, 45, text="Look at the dot with your eyes. Keep your head still.",
            fill="white", font=("Segoe UI", 20, "bold"))
        self.cal_hint = self.cal_canvas.create_text(
            self.screen_w // 2, self.screen_h - 45,
            text="Move only your eyes • keep face centered • press Esc to cancel",
            fill="#bbbbbb", font=("Segoe UI", 13))
        self.cal_dot = None
        self.cal_samples = []
        self.cal_phase_start = time.perf_counter()
        self.calibration_window.bind("<Escape>", lambda e: self.finish_calibration_cancel())
        self._show_calibration_point()

    def _show_calibration_point(self):
        if self.cal_index >= len(CAL_POINTS):
            self._finish_calibration()
            return
        self.cal_canvas.delete("dot")
        x = int(CAL_POINTS[self.cal_index][0] * self.screen_w)
        y = int(CAL_POINTS[self.cal_index][1] * self.screen_h)
        r = 16
        self.cal_dot = self.cal_canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="gray", width=2, tags="dot")
        # Smaller inner dot makes it easier to maintain exact fixation.
        self.cal_canvas.create_oval(x-5, y-5, x+5, y+5, fill="#666666", outline="", tags="dot")
        self.cal_phase_start = time.perf_counter()
        self.cal_samples = []
        self.cal_canvas.itemconfig(self.cal_instruction, text=f"Calibration {self.cal_index + 1}/{len(CAL_POINTS)} — look at the dot")

    def calibration_tick(self, data):
        if not hasattr(self, "calibration_window") or not self.calibration_window.winfo_exists():
            return
        elapsed = time.perf_counter() - self.cal_phase_start
        # 0–0.8 s: eye movement; 0.8–2.0 s: stable sample collection.
        if 0.65 < elapsed < 2.35 and data is not None and not data["eyes_closed"]:
            self.cal_samples.append(data["raw"].copy())
        if elapsed >= 2.35:
            if len(self.cal_samples) >= 5:
                arr = np.asarray(self.cal_samples)
                med = np.median(arr, axis=0)
                mad = np.median(np.abs(arr - med), axis=0) + 1e-6
                z = np.abs(arr - med) / (1.4826 * mad)
                good = np.all(z < 3.5, axis=1)
                clean = arr[good]
                if len(clean) >= 5:
                    # Mean after robust outlier removal gives a precise center.
                    center = np.mean(clean, axis=0)
                    sx = CAL_POINTS[self.cal_index][0] * self.screen_w
                    sy = CAL_POINTS[self.cal_index][1] * self.screen_h
                    self.cal_points_raw.append(center)
                    self.cal_points_screen.append([sx, sy])
                    self.cal_index += 1
                    self._show_calibration_point()
                    return
            self.cal_phase_start = time.perf_counter()
            self.cal_samples = []
            self.cal_canvas.itemconfig(self.cal_instruction, text="Hold your head still — collecting stable samples...")

    def _finish_calibration(self):
        try:
            self.calibration.fit(self.cal_points_raw, self.cal_points_screen)
            self.calibration_window.destroy()

            # IMPORTANT: fullscreen calibration can leave some Windows webcams
            # with a stale DirectShow/Media Foundation handle. Reopen the camera
            # before returning to the live preview.
            self.status_var.set("Calibration complete — reconnecting camera...")
            self.root.update_idletasks()
            self._restart_camera()

            rmse = self.calibration.rmse
            self.quality_var.set(f"Calibration error: {rmse:.0f}px RMSE")
            self.status_var.set("Calibration complete — camera ready.")
            messagebox.showinfo(
                "Calibration complete",
                f"{len(CAL_POINTS)}-point calibration finished.\n\nEstimated calibration RMSE: {rmse:.0f} px\n\nLower is better. Press Start Tracking to test."
            )
        except Exception as e:
            if hasattr(self, "calibration_window") and self.calibration_window.winfo_exists():
                self.calibration_window.destroy()
            messagebox.showerror("Calibration failed", str(e))

    def finish_calibration_cancel(self):
        if hasattr(self, "calibration_window") and self.calibration_window.winfo_exists():
            self.calibration_window.destroy()
        self.status_var.set("Calibration cancelled.")

    # -------------------------- cursor -----------------------------------
    def _move_cursor(self, raw):
        self.raw_history.append(np.asarray(raw, dtype=float))
        robust = np.median(np.asarray(self.raw_history), axis=0)

        # One-Euro filtering on gaze only. Head terms are kept from the robust sample.
        gaze_filtered = self.one_euro.update(robust[:2], time.perf_counter())
        blend = float(self.smoothing.get())
        # Small EMA after One-Euro; higher slider means more responsive.
        self.ema.alpha = float(np.clip(0.18 + blend * 0.75, 0.18, 0.68))
        gaze_final = self.ema.update(gaze_filtered)
        model_input = np.array([gaze_final[0], gaze_final[1], robust[2], robust[3]])

        predicted = self.calibration.predict(model_input)
        if predicted is None:
            return None

        # Sensitivity is intentionally limited; calibration already spans the screen.
        s = float(self.sensitivity.get())
        cx, cy = self.screen_w / 2.0, self.screen_h / 2.0
        x = cx + (float(predicted[0]) - cx) * s
        y = cy + (float(predicted[1]) - cy) * s

        x = float(np.clip(x, 2, self.screen_w - 2))
        y = float(np.clip(y, 2, self.screen_h - 2))
        p = np.array([x, y])
        if self.last_move is None or np.linalg.norm(p - np.asarray(self.last_move)) >= self.deadzone:
            pyautogui.moveTo(int(round(x)), int(round(y)), duration=0)
            self.last_move = (x, y)
        return x, y

    # -------------------------- clicks -----------------------------------
    def _process_clicks(self, eyes_closed, now, emergency_only=False):
        # Emergency detection is always active.
        # Any completed blink (left-click blink or long/right-click blink) counts
        # as one full blink for the 3-blink emergency trigger.
        emergency_event = self.blink.update(eyes_closed, now)
        if emergency_event in ("left", "right"):
            self.emergency.blink(now)

        if emergency_only or not self.blink_enabled.get():
            return
        if emergency_event == "left":
            pyautogui.click(button="left")
        elif emergency_event == "right":
            pyautogui.click(button="right")

    def _process_dwell(self, cursor):
        if not self.dwell_enabled.get() or cursor is None:
            self.dwell_start = None
            self.dwell_last_position = None
            return
        p = np.asarray(cursor, dtype=float)
        if self.dwell_last_position is None:
            self.dwell_last_position = p
            self.dwell_start = time.perf_counter()
            return
        if np.linalg.norm(p - self.dwell_last_position) > self.dwell_threshold:
            self.dwell_last_position = p
            self.dwell_start = time.perf_counter()
            return
        if self.dwell_start is not None and time.perf_counter() - self.dwell_start >= self.dwell_time:
            pyautogui.click(button="left")
            self.dwell_start = time.perf_counter() + 0.7

    # -------------------------- loop -------------------------------------
    def update_frame(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            data, annotated = self.tracker.process(frame)
            if data is not None:
                self.face_lost_since = None
                self.current_gaze = data["gaze"]
                self.gaze_var.set(f"{data['gaze'][0]:.3f}, {data['gaze'][1]:.3f}")

                if hasattr(self, "calibration_window") and self.calibration_window.winfo_exists():
                    self.calibration_tick(data)

                now = time.perf_counter()
                # 3-second continuous-eye-closure alarm works independently
                # of gaze direction and independently of cursor tracking.
                self.emergency.update_eye_closure(data["eyes_closed"], now)

                # Emergency blink detection is always active, even when cursor
                # tracking is paused. This keeps the emergency trigger independent
                # of the normal mouse-control state.
                if self.tracking:
                    # Never move the cursor from blink frames.
                    if not data["eyes_closed"]:
                        cursor = self._move_cursor(data["raw"])
                        self.current_screen = cursor
                        if cursor:
                            self.cursor_var.set(f"{int(cursor[0])}, {int(cursor[1])}")
                        self._process_dwell(cursor)
                    else:
                        cursor = self.current_screen
                        self._process_dwell(None)
                    self._process_clicks(data["eyes_closed"], now)
                else:
                    self._process_clicks(data["eyes_closed"], now, emergency_only=True)
            else:
                self.gaze_var.set("Face not detected")
                if self.face_lost_since is None:
                    self.face_lost_since = time.perf_counter()
                # Freeze briefly, then reset filters after sustained loss.
                if time.perf_counter() - self.face_lost_since > 0.35:
                    self.one_euro.reset()
                    self.ema.reset()
                    self.raw_history.clear()
                    self.last_move = None
                self.blink.update(False, time.perf_counter())

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.thumbnail((760, 560))
            photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=photo)
            self.preview.image = photo

        self.root.after(15, self.update_frame)

    def close(self):
        try:
            self.emergency.stop()
        except Exception:
            pass
        self.running = False
        self.tracking = False
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.tracker.mesh.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    EyeCursorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
