"""
CrowdVision AI Optional Edge Worker v41

Purpose
-------
Run this script on another laptop, mini PC, CCTV machine, or remote site.
It reads one camera/video source locally and sends lightweight metadata to the
main CrowdVision FastAPI backend. It also sends a lightweight live JPEG preview so the central dashboard can show camera movement.

This does not replace app/main.py. It is optional and safe.

Example
-------
python3 app/edge_worker.py \
  --edge-id gate_01 \
  --name "Gate 01 Camera" \
  --source 0 \
  --api http://127.0.0.1:8000

With optional YOLO people counting:
python3 app/edge_worker.py \
  --edge-id gate_01 \
  --name "Gate 01 Camera" \
  --source data/sample_videos/test.mp4 \
  --api http://127.0.0.1:8000 \
  --detect \
  --loop
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import requests

try:
    from app.security_config import get_edge_api_key
except Exception:
    try:
        from security_config import get_edge_api_key
    except Exception:
        def get_edge_api_key():
            return os.environ.get("CROWDVISION_EDGE_API_KEY", "")

try:
    cv2.setNumThreads(1)
except Exception:
    pass

try:
    from camera import open_camera_source, get_camera_source_label, restart_video_if_needed, CameraHealthAnalyzer
except Exception:
    # Allows running from project root or from inside app/.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from app.camera import open_camera_source, get_camera_source_label, restart_video_if_needed, CameraHealthAnalyzer


def parse_source(raw_source):
    if isinstance(raw_source, int):
        return raw_source

    source_text = str(raw_source).strip()

    if source_text.isdigit():
        return int(source_text)

    return source_text


def density_from_people(people_count):
    people_count = int(people_count or 0)

    if people_count >= 14:
        return "Critical"
    if people_count >= 7:
        return "High"
    if people_count >= 3:
        return "Medium"
    return "Low"


def resize_for_detection(frame, width=416):
    if frame is None or frame.size == 0:
        return frame

    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame

    scale = width / float(current_width)
    return cv2.resize(frame, (width, int(height * scale)), interpolation=cv2.INTER_AREA)


class OptionalYoloCounter:
    def __init__(self, enabled=False, model_path="yolov8n.pt", confidence=0.35, frame_width=416):
        self.enabled = bool(enabled)
        self.model_path = model_path
        self.confidence = float(confidence)
        self.frame_width = int(frame_width)
        self.model = None
        self.ready = False
        self.last_error = None

        if self.enabled:
            self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            self.ready = True
            print(f"YOLO edge counting enabled: {self.model_path}")
        except Exception as error:
            self.ready = False
            self.last_error = str(error)
            print(f"YOLO edge counting disabled: {error}")

    def count_people(self, frame):
        if not self.enabled or not self.ready or frame is None or frame.size == 0:
            return 0, 0.0

        try:
            inference_start = time.time()
            detection_frame = resize_for_detection(frame, self.frame_width)

            results = self.model(
                detection_frame,
                conf=self.confidence,
                classes=[0],
                verbose=False
            )

            people_count = 0
            if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None:
                people_count = len(results[0].boxes)

            elapsed = max(time.time() - inference_start, 0.0001)
            ai_fps = round(1.0 / elapsed, 2)
            return int(people_count), ai_fps
        except Exception as error:
            self.last_error = str(error)
            return 0, 0.0


def build_security_headers(api_key=""):
    key = str(api_key or "").strip()
    if not key:
        return {}
    return {"X-CrowdVision-Edge-Key": key}


def post_telemetry(api_base_url, payload, timeout=4, api_key=""):
    url = api_base_url.rstrip("/") + "/api/edge/telemetry"
    response = requests.post(url, json=payload, headers=build_security_headers(api_key), timeout=timeout)
    response.raise_for_status()
    return response.json()


def resize_for_preview(frame, width=640):
    if frame is None or frame.size == 0:
        return frame

    height, current_width = frame.shape[:2]

    if current_width <= width:
        return frame

    scale = width / float(current_width)
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)


def post_preview(api_base_url, edge_id, frame, width=640, quality=68, timeout=4, api_key=""):
    if frame is None or frame.size == 0:
        return None

    preview_frame = resize_for_preview(frame, width=max(320, int(width)))
    success, encoded = cv2.imencode(
        ".jpg",
        preview_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(int(quality), 90))]
    )

    if not success:
        return None

    url = api_base_url.rstrip("/") + f"/api/edge/preview/{edge_id}"
    files = {
        "file": (f"{edge_id}.jpg", encoded.tobytes(), "image/jpeg")
    }
    response = requests.post(url, files=files, headers=build_security_headers(api_key), timeout=timeout)
    response.raise_for_status()
    return response.json()


def save_local_cache(edge_id, payload):
    os.makedirs("data/edge", exist_ok=True)
    cache_path = os.path.join("data", "edge", f"{edge_id}_offline_cache.jsonl")

    with open(cache_path, "a") as file:
        file.write(json.dumps(payload) + "\n")


def run_edge_worker(args):
    source = parse_source(args.source)
    api_base_url = args.api.rstrip("/")
    api_key = str(args.api_key or get_edge_api_key() or "").strip()
    source_label = get_camera_source_label(source)

    cap = open_camera_source(source)
    if cap is None:
        print(f"Could not open edge source: {source}")
        return 1

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, args.target_fps)
    except Exception:
        pass

    health = CameraHealthAnalyzer(
        blur_threshold=args.blur_threshold,
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
        frozen_seconds=args.frozen_seconds,
        min_frame_change=args.min_frame_change,
        warmup_frames=args.warmup_frames,
        alert_hold_seconds=args.alert_hold_seconds,
    )

    counter = OptionalYoloCounter(
        enabled=args.detect,
        model_path=args.model,
        confidence=args.confidence,
        frame_width=args.frame_width,
    )

    print("CrowdVision Edge Worker started.")
    print(f"Edge ID: {args.edge_id}")
    print(f"Name: {args.name}")
    print(f"Source: {source}")
    print(f"Source Label: {source_label}")
    print(f"Central API: {api_base_url}")
    print(f"Security Key: {'Enabled' if api_key else 'Not configured'}")
    preview_enabled = not args.no_preview

    print(f"YOLO Detection: {args.detect}")
    print(f"Dashboard Preview: {preview_enabled}")
    print("Press Ctrl+C to stop.")

    last_post_time = 0.0
    last_preview_time = 0.0
    last_preview_error_time = 0.0
    last_fps_time = time.time()
    frame_count = 0
    detection_frame_count = 0
    fps = 0.0
    people = 0
    ai_fps = 0.0

    while True:
        success, frame = cap.read()
        current_time = time.time()

        if not success or frame is None:
            if restart_video_if_needed(cap, source, args.loop):
                time.sleep(0.05)
                continue

            camera_health = health.update(None, current_time)
            payload = {
                "edge_id": args.edge_id,
                "name": args.name,
                "source": str(source),
                "source_type": source_label,
                "location": args.location,
                "online": False,
                "status": "No Signal",
                "people": 0,
                "zone_count": 0,
                "occupancy": 0,
                "density": "Unknown",
                "fps": 0,
                "ai_fps": 0,
                "camera_health": camera_health,
                "privacy_mode": "Anonymous edge telemetry",
                "message": "Edge source is not returning frames.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp_epoch": current_time,
            }

            try:
                post_telemetry(api_base_url, payload, timeout=args.timeout, api_key=api_key)
                print("No-signal telemetry sent.")
            except Exception as error:
                print(f"Telemetry failed: {error}")
                save_local_cache(args.edge_id, payload)

            time.sleep(args.interval)
            continue

        frame_count += 1
        detection_frame_count += 1

        if current_time - last_fps_time >= 1.0:
            fps = round(frame_count / max(current_time - last_fps_time, 0.0001), 2)
            frame_count = 0
            last_fps_time = current_time

        camera_health = health.update(frame, current_time)

        if args.detect and detection_frame_count % max(args.process_every, 1) == 0:
            people, ai_fps = counter.count_people(frame)

        if preview_enabled and current_time - last_preview_time >= max(args.preview_interval, 0.5):
            try:
                post_preview(
                    api_base_url=api_base_url,
                    edge_id=args.edge_id,
                    frame=frame,
                    width=args.preview_width,
                    quality=args.preview_quality,
                    timeout=min(max(args.timeout, 2.0), 6.0),
                    api_key=api_key
                )
            except Exception as error:
                if current_time - last_preview_error_time >= 10:
                    print(f"Preview upload warning: {error}")
                    last_preview_error_time = current_time

            last_preview_time = current_time

        if current_time - last_post_time >= args.interval:
            density = density_from_people(people)

            payload = {
                "edge_id": args.edge_id,
                "name": args.name,
                "source": str(source),
                "source_type": source_label,
                "location": args.location,
                "online": True,
                "status": "Online",
                "people": people,
                "zone_count": people,
                "occupancy": people,
                "density": density,
                "fps": fps,
                "ai_fps": ai_fps,
                "camera_health": camera_health,
                "tracking": {
                    "session_unique_people": people,
                    "active_tracks": people,
                    "memory_tracks": people,
                    "reidentified_people": 0,
                    "duplicates_prevented": 0,
                    "tracking_quality": "Edge telemetry active",
                    "privacy_mode": "Anonymous edge telemetry"
                },
                "privacy_mode": "Anonymous edge telemetry",
                "message": camera_health.get("message", "Edge telemetry active."),
                "metadata": {
                    "detect_enabled": bool(args.detect),
                    "model": args.model if args.detect else "disabled",
                    "edge_version": "v41",
                    "process_every": args.process_every,
                    "preview_enabled": preview_enabled,
                    "preview_interval": args.preview_interval,
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp_epoch": current_time,
            }

            try:
                response = post_telemetry(api_base_url, payload, timeout=args.timeout, api_key=api_key)
                print(
                    f"Sent edge telemetry | people={people} density={density} "
                    f"fps={fps} ai_fps={ai_fps} response={response.get('message')}"
                )
            except Exception as error:
                print(f"Telemetry failed, saved locally: {error}")
                save_local_cache(args.edge_id, payload)

            last_post_time = current_time

        if args.display:
            preview = frame.copy()
            status_text = f"{args.name} | People: {people} | FPS: {fps} | {camera_health.get('status', 'Health')}"
            cv2.putText(preview, status_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 180), 2)
            cv2.imshow("CrowdVision Edge Worker", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        time.sleep(max(args.loop_delay, 0.0))

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="CrowdVision AI optional multi-camera edge worker")

    parser.add_argument("--edge-id", default="edge_01", help="Unique ID for this edge worker")
    parser.add_argument("--name", default="Edge Camera 01", help="Human-readable camera name")
    parser.add_argument("--source", default="0", help="Camera source: 0, video path, HTTP, RTSP, etc.")
    parser.add_argument("--location", default="Unassigned", help="Physical location label")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="Central CrowdVision API base URL")
    parser.add_argument("--api-key", default="", help="Optional CrowdVision remote camera security key")

    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between telemetry posts")
    parser.add_argument("--timeout", type=float, default=4.0, help="HTTP timeout in seconds")
    parser.add_argument("--target-fps", type=int, default=12, help="Camera capture FPS hint")
    parser.add_argument("--loop-delay", type=float, default=0.02, help="Small sleep in capture loop")
    parser.add_argument("--loop", action="store_true", help="Loop video files when they end")
    parser.add_argument("--display", action="store_true", help="Show local OpenCV preview window")
    parser.add_argument("--no-preview", action="store_true", help="Disable dashboard JPEG preview uploads")
    parser.add_argument("--preview-interval", type=float, default=1.0, help="Seconds between dashboard preview uploads")
    parser.add_argument("--preview-width", type=int, default=640, help="Maximum dashboard preview width")
    parser.add_argument("--preview-quality", type=int, default=68, help="Dashboard JPEG preview quality")

    parser.add_argument("--detect", action="store_true", help="Enable optional YOLO people counting on the edge worker")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path for edge counting")
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--frame-width", type=int, default=416, help="Detection frame width")
    parser.add_argument("--process-every", type=int, default=6, help="Run detection every N frames")

    parser.add_argument("--blur-threshold", type=float, default=55.0)
    parser.add_argument("--dark-threshold", type=float, default=28.0)
    parser.add_argument("--bright-threshold", type=float, default=238.0)
    parser.add_argument("--frozen-seconds", type=float, default=8.0)
    parser.add_argument("--min-frame-change", type=float, default=1.35)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--alert-hold-seconds", type=float, default=6.0)

    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run_edge_worker(build_parser().parse_args()))
    except KeyboardInterrupt:
        print("\nEdge worker stopped by operator.")
        raise SystemExit(0)