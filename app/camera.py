import cv2
import os
import time
import numpy as np


def open_camera_source(camera_source):
    """
    Open webcam, video file, RTSP, HTTP/IP camera, or other OpenCV-compatible source.
    """
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print(f"Error: Could not open camera source: {camera_source}")
        return None

    return cap


def get_camera_source_label(camera_source):
    if isinstance(camera_source, int):
        if camera_source == 0:
            return "Laptop Webcam"
        return f"Camera Device {camera_source}"

    if isinstance(camera_source, str):
        if camera_source.startswith(("http://", "https://")):
            return "IP / Phone Camera Stream"

        if camera_source.startswith("rtsp://"):
            return "RTSP / CCTV Stream"

        if camera_source.startswith("rtmp://"):
            return "RTMP Stream"

        if is_video_file(camera_source):
            return "Video File"

        return "Custom Camera Source"

    return "Unknown Source"


def is_video_file(camera_source):
    if not isinstance(camera_source, str):
        return False

    video_extensions = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    )

    return camera_source.lower().endswith(video_extensions)


def video_file_exists(camera_source):
    if not is_video_file(camera_source):
        return True

    return os.path.exists(camera_source)


def restart_video_if_needed(cap, camera_source, video_loop):
    """
    Loop video files smoothly when VIDEO_LOOP=True.
    This prevents uploaded demo videos from stopping during presentations.
    """
    if is_video_file(camera_source) and video_loop:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return True

    return False


def safe_round(value, decimals=2, fallback=0):
    try:
        return round(float(value), decimals)
    except Exception:
        return fallback


class CameraHealthAnalyzer:
    """
    Lightweight camera health and tamper detector.

    It detects practical CCTV problems:
    - dark / covered lens
    - overexposed glare
    - blurry lens
    - frozen frame
    - low visual signal

    It does not identify faces or people. It only reads frame quality.
    """

    def __init__(
        self,
        blur_threshold=55.0,
        dark_threshold=28.0,
        bright_threshold=238.0,
        frozen_seconds=8.0,
        min_frame_change=1.35,
        warmup_frames=8,
        alert_hold_seconds=6.0
    ):
        self.blur_threshold = float(blur_threshold)
        self.dark_threshold = float(dark_threshold)
        self.bright_threshold = float(bright_threshold)
        self.frozen_seconds = float(frozen_seconds)
        self.min_frame_change = float(min_frame_change)
        self.warmup_frames = int(warmup_frames)
        self.alert_hold_seconds = float(alert_hold_seconds)

        self.previous_gray_small = None
        self.last_motion_time = time.time()
        self.frame_count = 0
        self.last_alert_time = 0
        self.last_health = self.default_health()

    def default_health(self):
        return {
            "enabled": True,
            "status": "Initializing",
            "severity": "Info",
            "tamper_detected": False,
            "tamper_type": "None",
            "message": "Camera health analyzer is warming up.",
            "brightness": 0,
            "blur_score": 0,
            "frame_change": 0,
            "frozen_seconds": 0,
            "covered_lens": False,
            "too_dark": False,
            "too_bright": False,
            "blurry": False,
            "frozen_frame": False,
            "signal_quality": "Warming Up",
            "last_alert_at": None,
            "alert_ready": False
        }

    def _small_gray(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)

    def _quality_label(self, blur_score, brightness):
        if brightness <= self.dark_threshold:
            return "Poor"
        if brightness >= self.bright_threshold:
            return "Poor"
        if blur_score < self.blur_threshold:
            return "Weak"
        if blur_score >= self.blur_threshold * 2:
            return "Strong"
        return "Good"

    def update(self, frame, current_time=None):
        if current_time is None:
            current_time = time.time()

        if frame is None or frame.size == 0:
            self.last_health = {
                **self.default_health(),
                "status": "No Signal",
                "severity": "Critical",
                "tamper_detected": True,
                "tamper_type": "No Signal",
                "message": "Camera is not returning usable frames.",
                "signal_quality": "No Signal",
                "alert_ready": True
            }
            return self.last_health

        self.frame_count += 1

        gray_small = self._small_gray(frame)
        brightness = float(np.mean(gray_small))
        blur_score = float(cv2.Laplacian(gray_small, cv2.CV_64F).var())

        frame_change = 0.0
        if self.previous_gray_small is not None:
            diff = cv2.absdiff(gray_small, self.previous_gray_small)
            frame_change = float(np.mean(diff))

            if frame_change >= self.min_frame_change:
                self.last_motion_time = current_time
        else:
            self.last_motion_time = current_time

        self.previous_gray_small = gray_small

        frozen_seconds = max(0.0, current_time - self.last_motion_time)

        too_dark = brightness <= self.dark_threshold
        too_bright = brightness >= self.bright_threshold
        blurry = blur_score < self.blur_threshold and self.frame_count > self.warmup_frames
        frozen_frame = (
            frozen_seconds >= self.frozen_seconds and
            self.frame_count > self.warmup_frames
        )
        covered_lens = too_dark and blur_score < (self.blur_threshold * 0.65)

        tamper_type = "None"
        severity = "Normal"
        status = "Healthy"
        message = "Camera signal is healthy."

        if covered_lens:
            tamper_type = "Covered Lens / Blackout"
            severity = "Critical"
            status = "Tamper Suspected"
            message = "Camera image is very dark and low-detail. Lens may be covered or the area may be blacked out."
        elif frozen_frame:
            tamper_type = "Frozen Frame"
            severity = "Critical"
            status = "Tamper Suspected"
            message = "Camera image appears frozen. Check network stream, cable, or camera source."
        elif too_dark:
            tamper_type = "Low Light / Possible Obstruction"
            severity = "Warning"
            status = "Weak Signal"
            message = "Camera image is too dark. Improve lighting or check for obstruction."
        elif too_bright:
            tamper_type = "Overexposure / Glare"
            severity = "Warning"
            status = "Weak Signal"
            message = "Camera image is overexposed. Check glare, lighting, or camera angle."
        elif blurry:
            tamper_type = "Blurred Lens"
            severity = "Warning"
            status = "Weak Signal"
            message = "Camera image is blurry. Clean lens or adjust focus."

        tamper_detected = severity in ["Warning", "Critical"]

        alert_ready = False
        if tamper_detected and current_time - self.last_alert_time >= self.alert_hold_seconds:
            self.last_alert_time = current_time
            alert_ready = True

        health = {
            "enabled": True,
            "status": status,
            "severity": severity,
            "tamper_detected": tamper_detected,
            "tamper_type": tamper_type,
            "message": message,
            "brightness": safe_round(brightness, 2),
            "blur_score": safe_round(blur_score, 2),
            "frame_change": safe_round(frame_change, 2),
            "frozen_seconds": safe_round(frozen_seconds, 2),
            "covered_lens": bool(covered_lens),
            "too_dark": bool(too_dark),
            "too_bright": bool(too_bright),
            "blurry": bool(blurry),
            "frozen_frame": bool(frozen_frame),
            "signal_quality": self._quality_label(blur_score, brightness),
            "last_alert_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_alert_time)) if self.last_alert_time else None,
            "alert_ready": bool(alert_ready)
        }

        self.last_health = health
        return health

    def reset(self):
        self.previous_gray_small = None
        self.last_motion_time = time.time()
        self.frame_count = 0
        self.last_alert_time = 0
        self.last_health = self.default_health()
        return self.last_health