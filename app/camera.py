import cv2
import os


def open_camera_source(camera_source):
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
        if camera_source.startswith("http"):
            return "IP / Phone Camera Stream"

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
    if is_video_file(camera_source) and video_loop:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return True

    return False