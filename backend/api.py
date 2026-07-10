from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, PlainTextResponse
from io import StringIO, BytesIO
from datetime import datetime
import pandas as pd
import os
import time
import subprocess
import signal
import sys
import json

from app.report_generator import txt_report_to_pdf
from app.config import (
    get_camera_profiles as load_camera_profiles,
    get_active_camera_profile,
    set_active_camera_profile,
    add_camera_source_profile,
    delete_camera_source_profile,
    get_performance_status,
    set_saved_performance_mode,
    resolve_performance_mode,
    LIVE_STREAM_SLEEP,
    CAMERA_STATUS_FILE,
    CAMERA_OFFLINE_AFTER_SECONDS,
    EVIDENCE_DIRS,
    NOTIFICATION_LIMIT
)

app = FastAPI(title="CrowdVision AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EVENTS_FILE = "data/events.csv"
LIVE_FRAME_PATH = "data/live/latest_frame.jpg"
REPORTS_DIR = "data/reports"
UPLOADS_DIR = "data/uploads"
NOTIFICATION_FILE = "data/notifications.json"
EDGE_TELEMETRY_FILE = "data/edge_telemetry.json"
EDGE_WORKER_OFFLINE_AFTER_SECONDS = 30
EDGE_TELEMETRY_LIMIT = 500
EDGE_PID_DIR = "data/edge/pids"
EDGE_LOG_DIR = "data/edge/logs"
EDGE_META_DIR = "data/edge/meta"
EDGE_PREVIEW_DIR = "data/edge/previews"
EDGE_PREVIEW_MAX_BYTES = 2_500_000
EDGE_PREVIEW_STALE_AFTER_SECONDS = 5
LEGACY_SOURCE_PRESET_KEYS = {"webcam", "crowd_video"}

ENGINE_PROCESS = None
PROJECT_ROOT = os.getcwd()
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "app", "main.py")
EDGE_WORKER_SCRIPT = os.path.join(PROJECT_ROOT, "app", "edge_worker.py")
EDGE_WORKER_PROCESSES = {}
ENGINE_PID_FILE = "data/engine.pid"
ENGINE_LOG_FILE = "data/engine.log"

os.makedirs("data/live", exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs("data/edge", exist_ok=True)
os.makedirs(EDGE_PID_DIR, exist_ok=True)
os.makedirs(EDGE_LOG_DIR, exist_ok=True)
os.makedirs(EDGE_META_DIR, exist_ok=True)
os.makedirs(EDGE_PREVIEW_DIR, exist_ok=True)

app.mount("/live", StaticFiles(directory="data/live"), name="live")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/edge-previews", StaticFiles(directory=EDGE_PREVIEW_DIR), name="edge_previews")

for _evidence_dir in ["data/evidence", "data/snapshots", "data/alerts"]:
    os.makedirs(_evidence_dir, exist_ok=True)

    try:
        app.mount(
            f"/{_evidence_dir.replace('data/', '')}",
            StaticFiles(directory=_evidence_dir),
            name=_evidence_dir.replace("/", "_")
        )
    except RuntimeError:
        pass


def load_events():
    if not os.path.exists(EVENTS_FILE):
        return pd.DataFrame()

    df = pd.read_csv(EVENTS_FILE)

    if df.empty:
        return df

    return df.fillna(0)


def as_bool(value):
    """Convert CSV / JSON boolean-like values without truthiness mistakes."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def event_epoch(value):
    """Return a Unix timestamp for CrowdVision event timestamps."""
    if value in [None, "", 0]:
        return 0.0

    try:
        return float(value)
    except Exception:
        pass

    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return 0.0


def get_runtime_context(max_event_age_seconds=45):
    """
    Separate live state from historical records.

    A CSV row is considered current only when:
    - the monitoring engine is running,
    - camera_status.json is fresh and online,
    - the latest CSV event is recent enough.

    This prevents old incidents/anomalies from appearing ACTIVE after monitoring stops.
    """
    status = read_camera_status_file() or {}
    running = is_engine_running()
    now = time.time()

    try:
        status_epoch = float(status.get("updated_at_epoch", 0) or 0)
    except Exception:
        status_epoch = 0.0

    status_age = round(now - status_epoch, 2) if status_epoch else None
    status_fresh = bool(
        running
        and status.get("online")
        and status_age is not None
        and status_age <= CAMERA_OFFLINE_AFTER_SECONDS
    )

    latest_event = get_latest_event_record()
    latest_event_age = None
    event_fresh = False

    if latest_event:
        latest_event_epoch = event_epoch(latest_event.get("timestamp"))
        latest_event_age = round(now - latest_event_epoch, 2) if latest_event_epoch else None
        event_fresh = bool(
            status_fresh
            and latest_event_age is not None
            and latest_event_age <= max_event_age_seconds
        )

    return {
        "running": running,
        "status": status,
        "status_fresh": status_fresh,
        "status_age_seconds": status_age,
        "latest_event": latest_event,
        "latest_event_fresh": event_fresh,
        "latest_event_age_seconds": latest_event_age,
        "current_event": latest_event if event_fresh else None,
    }


def get_last_recorded_incident():
    df = load_events()
    if df.empty or "incident_active" not in df.columns:
        return None

    incidents_df = df[df["incident_active"].astype(str).str.lower() == "true"]
    if incidents_df.empty:
        return None

    return incidents_df.tail(1).to_dict(orient="records")[0]


def get_last_recorded_anomaly():
    df = load_events()
    if df.empty or "anomaly_detected" not in df.columns:
        return None

    anomalies_df = df[df["anomaly_detected"].astype(str).str.lower() == "true"]
    if anomalies_df.empty:
        return None

    return anomalies_df.tail(1).to_dict(orient="records")[0]


def default_tracking_summary():
    return {
        "session_unique_people": 0,
        "active_tracks": 0,
        "memory_tracks": 0,
        "reidentified_people": 0,
        "duplicates_prevented": 0,
        "raw_tracker_continuity": 0,
        "tracking_quality": "Waiting",
        "memory_seconds": 0,
        "privacy_mode": "Anonymous body tracking"
    }


def normalize_tracking_summary(status):
    if not status or not isinstance(status, dict):
        return default_tracking_summary()

    tracking = status.get("tracking")

    if not isinstance(tracking, dict):
        return default_tracking_summary()

    normalized = default_tracking_summary()

    for key in normalized.keys():
        if key in tracking:
            normalized[key] = tracking.get(key)

    integer_keys = [
        "session_unique_people",
        "active_tracks",
        "memory_tracks",
        "reidentified_people",
        "duplicates_prevented",
        "raw_tracker_continuity",
        "memory_seconds"
    ]

    for key in integer_keys:
        try:
            normalized[key] = int(float(normalized.get(key, 0)))
        except Exception:
            normalized[key] = 0

    normalized["privacy_mode"] = tracking.get("privacy_mode", "Anonymous body tracking")
    normalized["tracking_quality"] = tracking.get("tracking_quality", "Tracking active")

    return normalized


def default_camera_health():
    return {
        "enabled": True,
        "status": "Waiting",
        "severity": "Info",
        "tamper_detected": False,
        "tamper_type": "None",
        "message": "Waiting for camera frames.",
        "brightness": 0,
        "blur_score": 0,
        "frame_change": 0,
        "frozen_seconds": 0,
        "covered_lens": False,
        "too_dark": False,
        "too_bright": False,
        "blurry": False,
        "frozen_frame": False,
        "signal_quality": "Waiting",
        "last_alert_at": None
    }


def normalize_camera_health(status):
    if not status or not isinstance(status, dict):
        return default_camera_health()

    camera_health = status.get("camera_health")

    if not isinstance(camera_health, dict):
        return default_camera_health()

    normalized = default_camera_health()

    for key in normalized.keys():
        if key in camera_health:
            normalized[key] = camera_health.get(key)

    numeric_keys = ["brightness", "blur_score", "frame_change", "frozen_seconds"]

    for key in numeric_keys:
        try:
            normalized[key] = round(float(normalized.get(key, 0)), 2)
        except Exception:
            normalized[key] = 0

    bool_keys = [
        "enabled",
        "tamper_detected",
        "covered_lens",
        "too_dark",
        "too_bright",
        "blurry",
        "frozen_frame"
    ]

    for key in bool_keys:
        normalized[key] = bool(normalized.get(key, False))

    return normalized


def read_camera_status_file():
    if not os.path.exists(CAMERA_STATUS_FILE):
        return None

    try:
        with open(CAMERA_STATUS_FILE, "r") as file:
            return json.load(file)
    except Exception:
        return None


def write_camera_status_file(status):
    try:
        os.makedirs(os.path.dirname(CAMERA_STATUS_FILE), exist_ok=True)

        with open(CAMERA_STATUS_FILE, "w") as file:
            json.dump(status, file, indent=4)

        return True
    except Exception as error:
        print(f"Unable to write camera status file: {error}")
        return False


def mark_camera_status_stopped(reason="Monitoring engine stopped by operator"):
    previous_status = read_camera_status_file() or {}
    performance = get_performance_status()
    now = time.time()

    stopped_status = {
        **previous_status,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at_epoch": now,
        "active_profile": get_active_camera_profile(),
        "source": previous_status.get("source", ""),
        "online": False,
        "status": "Stopped",
        "reason": reason,
        "fps": 0,
        "ai_fps": 0,
        "mode": performance.get("resolved_mode", "LOW"),
        "tracking": normalize_tracking_summary(previous_status),
        "camera_health": {
            **normalize_camera_health(previous_status),
            "status": "Stopped",
            "severity": "Info",
            "tamper_detected": False,
            "tamper_type": "None",
            "message": reason
        }
    }

    write_camera_status_file(stopped_status)


def generate_live_stream():
    last_frame = None

    while True:
        try:
            if (
                os.path.exists(LIVE_FRAME_PATH)
                and os.path.getsize(LIVE_FRAME_PATH) > 0
            ):
                with open(LIVE_FRAME_PATH, "rb") as image_file:
                    frame = image_file.read()

                if frame:
                    last_frame = frame

            if last_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                    b"Pragma: no-cache\r\n"
                    b"Expires: 0\r\n\r\n"
                    + last_frame
                    + b"\r\n"
                )

        except Exception as error:
            print(f"Live stream error: {error}")

        time.sleep(LIVE_STREAM_SLEEP)


def get_forecast_message(predicted_occupancy, risk_trend):
    if predicted_occupancy >= 10 and risk_trend == "Increasing":
        return "Critical congestion likely soon. Prepare immediate response."

    if predicted_occupancy >= 8:
        return "High crowd pressure expected. Increase monitoring."

    if risk_trend == "Increasing":
        return "Crowd level is rising. Continue close observation."

    if risk_trend == "Decreasing":
        return "Crowd level is reducing. Maintain monitoring."

    return "Crowd level appears stable."


def get_saved_engine_pid():
    if not os.path.exists(ENGINE_PID_FILE):
        return None

    try:
        with open(ENGINE_PID_FILE, "r") as file:
            pid = int(file.read().strip())

        return pid
    except Exception:
        return None


def save_engine_pid(pid):
    os.makedirs("data", exist_ok=True)

    with open(ENGINE_PID_FILE, "w") as file:
        file.write(str(pid))


def clear_engine_pid():
    if os.path.exists(ENGINE_PID_FILE):
        try:
            os.remove(ENGINE_PID_FILE)
        except Exception:
            pass


def is_pid_running(pid):
    if not pid:
        return False

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def is_engine_running():
    global ENGINE_PROCESS

    if ENGINE_PROCESS is not None and ENGINE_PROCESS.poll() is None:
        return True

    saved_pid = get_saved_engine_pid()

    if saved_pid and is_pid_running(saved_pid):
        return True

    clear_engine_pid()
    ENGINE_PROCESS = None
    return False


def get_engine_pid():
    if ENGINE_PROCESS is not None and ENGINE_PROCESS.poll() is None:
        return ENGINE_PROCESS.pid

    saved_pid = get_saved_engine_pid()

    if saved_pid and is_pid_running(saved_pid):
        return saved_pid

    return None


def stop_pid(pid):
    if not pid:
        return

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    for _ in range(20):
        if not is_pid_running(pid):
            return

        time.sleep(0.2)

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def safe_edge_id(edge_id):
    text = str(edge_id or "").strip().lower()
    safe = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in text).strip("_")
    return safe or "edge_worker"


def edge_pid_file(edge_id):
    return os.path.join(EDGE_PID_DIR, f"{safe_edge_id(edge_id)}.pid")


def edge_meta_file(edge_id):
    return os.path.join(EDGE_META_DIR, f"{safe_edge_id(edge_id)}.json")


def edge_log_file(edge_id):
    return os.path.join(EDGE_LOG_DIR, f"{safe_edge_id(edge_id)}.log")


def edge_preview_file(edge_id):
    return os.path.join(EDGE_PREVIEW_DIR, f"{safe_edge_id(edge_id)}.jpg")


def get_edge_preview_info(edge_id, online=False):
    safe_id = safe_edge_id(edge_id)
    preview_path = edge_preview_file(safe_id)

    if not os.path.exists(preview_path):
        return {
            "preview_available": False,
            "preview_url": None,
            "preview_age_seconds": None,
            "preview_live": False
        }

    try:
        age = round(max(0.0, time.time() - os.path.getmtime(preview_path)), 2)
    except Exception:
        age = None

    return {
        "preview_available": True,
        "preview_url": f"/edge-previews/{safe_id}.jpg",
        "preview_age_seconds": age,
        "preview_live": bool(online and age is not None and age <= EDGE_PREVIEW_STALE_AFTER_SECONDS)
    }


def save_edge_pid(edge_id, pid):
    os.makedirs(EDGE_PID_DIR, exist_ok=True)
    with open(edge_pid_file(edge_id), "w") as file:
        file.write(str(pid))


def get_saved_edge_pid(edge_id):
    path = edge_pid_file(edge_id)

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as file:
            pid = int(file.read().strip())

        return pid
    except Exception:
        return None


def clear_edge_pid(edge_id):
    path = edge_pid_file(edge_id)

    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def save_edge_meta(edge_id, meta):
    try:
        os.makedirs(EDGE_META_DIR, exist_ok=True)
        with open(edge_meta_file(edge_id), "w") as file:
            json.dump(meta, file, indent=2)
    except Exception as error:
        print(f"Unable to save edge meta: {error}")


def read_edge_meta(edge_id):
    path = edge_meta_file(edge_id)

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def list_managed_edge_ids():
    ids = set()

    if os.path.exists(EDGE_PID_DIR):
        for filename in os.listdir(EDGE_PID_DIR):
            if filename.endswith(".pid"):
                ids.add(filename[:-4])

    if os.path.exists(EDGE_META_DIR):
        for filename in os.listdir(EDGE_META_DIR):
            if filename.endswith(".json"):
                ids.add(filename[:-5])

    for edge_id in EDGE_WORKER_PROCESSES.keys():
        ids.add(safe_edge_id(edge_id))

    return sorted(ids)


def get_edge_worker_pid(edge_id):
    safe_id = safe_edge_id(edge_id)

    process = EDGE_WORKER_PROCESSES.get(safe_id)

    if process is not None and process.poll() is None:
        return process.pid

    saved_pid = get_saved_edge_pid(safe_id)

    if saved_pid and is_pid_running(saved_pid):
        return saved_pid

    clear_edge_pid(safe_id)
    EDGE_WORKER_PROCESSES.pop(safe_id, None)
    return None


def is_edge_worker_running(edge_id):
    return get_edge_worker_pid(edge_id) is not None


def stop_edge_worker_process(edge_id):
    safe_id = safe_edge_id(edge_id)
    meta = read_edge_meta(safe_id)
    camera_name = str(meta.get("name") or safe_id.replace("_", " ").title())
    pid = get_edge_worker_pid(safe_id)

    if not pid:
        EDGE_WORKER_PROCESSES.pop(safe_id, None)
        clear_edge_pid(safe_id)
        return False

    stop_pid(pid)
    EDGE_WORKER_PROCESSES.pop(safe_id, None)
    clear_edge_pid(safe_id)

    append_edge_notification({
        "category": "edge_control",
        "severity": "normal",
        "title": "Remote Camera Disconnected",
        "message": f"{camera_name} is no longer connected to the command center.",
        "source": camera_name,
        "action": "Monitor the remaining connected cameras.",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": time.time(),
        "dedupe_key": f"camera:control:stopped:{safe_id}",
        "metadata": {"edge_id": safe_id}
    })

    return True


def ensure_local_source_available(source, requested_edge_id=None):
    source_text = str(source).strip()

    if source_text != "0":
        return

    if is_engine_running():
        active_profile = get_active_camera_profile()
        active_profile_data = load_camera_profiles().get(active_profile, {})
        active_source = str(active_profile_data.get("source", "")).strip()

        if active_source == "0":
            raise HTTPException(
                status_code=409,
                detail=(
                    "The laptop camera is already being used by Main Monitoring. "
                    "Stop Main Monitoring first, then start the Laptop Camera here. "
                    "One physical camera can only be used by one monitoring session at a time."
                )
            )

    requested_id = safe_edge_id(requested_edge_id or "edge_worker")

    for edge_id in list_managed_edge_ids():
        if safe_edge_id(edge_id) == requested_id:
            continue

        meta = read_edge_meta(edge_id)
        camera_name = str(meta.get("name") or safe_edge_id(edge_id).replace("_", " ").title())
        if str(meta.get("source", "")).strip() == "0" and is_edge_worker_running(edge_id):
            raise HTTPException(
                status_code=409,
                detail=f"The laptop camera is already being used by {camera_name}. Stop that camera first."
            )


def build_edge_command(
    edge_id,
    name,
    source,
    api_url="http://127.0.0.1:8000",
    detect=False,
    model="yolov8n.pt",
    loop=True,
    timeout=15,
    interval=10,
    process_every=10,
    location="Unassigned",
    display=False,
    preview=True,
    preview_interval=1.0,
    preview_width=640,
    preview_quality=68
):
    command = [
        sys.executable,
        EDGE_WORKER_SCRIPT,
        "--edge-id", safe_edge_id(edge_id),
        "--name", str(name or edge_id),
        "--source", str(source),
        "--api", str(api_url or "http://127.0.0.1:8000"),
        "--timeout", str(timeout),
        "--interval", str(interval),
        "--process-every", str(process_every),
        "--location", str(location or "Unassigned"),
        "--preview-interval", str(preview_interval),
        "--preview-width", str(preview_width),
        "--preview-quality", str(preview_quality),
    ]

    if not preview:
        command.append("--no-preview")

    if detect:
        command.extend(["--detect", "--model", str(model or "yolov8n.pt")])

    if loop:
        command.append("--loop")

    if display:
        command.append("--display")

    return command


def start_edge_worker_process(config):
    if not os.path.exists(EDGE_WORKER_SCRIPT):
        raise HTTPException(status_code=404, detail="The remote camera service is unavailable. Contact the system administrator.")

    edge_id = safe_edge_id(config.get("edge_id") or "edge_worker")
    name = str(config.get("name") or edge_id)
    source = str(config.get("source") or "0")

    ensure_local_source_available(source, requested_edge_id=edge_id)

    if is_edge_worker_running(edge_id):
        return {
            "message": "Camera is already connected",
            "edge_id": edge_id,
            "running": True,
            "pid": get_edge_worker_pid(edge_id),
            "config": read_edge_meta(edge_id)
        }

    detect = bool(config.get("detect", False))
    model = str(config.get("model") or "yolov8n.pt")
    loop = bool(config.get("loop", True))
    timeout = float(config.get("timeout", 15))
    interval = float(config.get("interval", 10))
    process_every = int(config.get("process_every", 10))
    location = str(config.get("location") or "Unassigned")
    display = bool(config.get("display", False))
    preview = bool(config.get("preview", True))
    preview_interval = max(0.5, float(config.get("preview_interval", 1.0)))
    preview_width = max(320, min(int(config.get("preview_width", 640)), 1280))
    preview_quality = max(40, min(int(config.get("preview_quality", 68)), 90))
    api_url = str(config.get("api") or "http://127.0.0.1:8000")

    command = build_edge_command(
        edge_id=edge_id,
        name=name,
        source=source,
        api_url=api_url,
        detect=detect,
        model=model,
        loop=loop,
        timeout=timeout,
        interval=interval,
        process_every=process_every,
        location=location,
        display=display,
        preview=preview,
        preview_interval=preview_interval,
        preview_width=preview_width,
        preview_quality=preview_quality
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    os.makedirs(EDGE_LOG_DIR, exist_ok=True)
    log_file = open(edge_log_file(edge_id), "a")

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
        env=env
    )

    EDGE_WORKER_PROCESSES[edge_id] = process
    save_edge_pid(edge_id, process.pid)

    meta = {
        "edge_id": edge_id,
        "name": name,
        "source": source,
        "api": api_url,
        "detect": detect,
        "model": model if detect else "disabled",
        "loop": loop,
        "timeout": timeout,
        "interval": interval,
        "process_every": process_every,
        "location": location,
        "display": display,
        "preview": preview,
        "preview_interval": preview_interval,
        "preview_width": preview_width,
        "preview_quality": preview_quality,
        "pid": process.pid,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "started_at_epoch": time.time(),
        "managed_by_api": True
    }
    save_edge_meta(edge_id, meta)

    append_edge_notification({
        "category": "edge_control",
        "severity": "normal",
        "title": "Remote Camera Connected",
        "message": f"{name} is now connected to the command center.",
        "source": name,
        "action": "Monitor the camera view, crowd activity, and camera health.",
        "time": meta["started_at"],
        "epoch": meta["started_at_epoch"],
        "dedupe_key": f"camera:control:started:{edge_id}",
        "metadata": {"edge_id": edge_id, "pid": process.pid, "detect": detect}
    })

    return {
        "message": "Camera connected. The live view will appear in the dashboard within a few seconds.",
        "edge_id": edge_id,
        "running": True,
        "pid": process.pid,
        "config": meta
    }


def build_managed_edge_placeholder(edge_id):
    meta = read_edge_meta(edge_id)
    running = is_edge_worker_running(edge_id)

    if not meta and not running:
        return None

    now = time.time()
    started_epoch = safe_epoch_from_time(meta.get("started_at_epoch"))
    age = round(now - started_epoch, 2) if started_epoch else None

    return {
        "edge_id": safe_edge_id(edge_id),
        "name": meta.get("name", safe_edge_id(edge_id)),
        "source": str(meta.get("source", "Unknown")),
        "source_type": "Remote Camera",
        "location": meta.get("location", "Unassigned"),
        "online": False,
        "status": "Starting" if running else "Stopped",
        "people": 0,
        "total_people": 0,
        "zone_count": 0,
        "occupancy": 0,
        "density": "Waiting" if running else "Offline",
        "fps": 0,
        "ai_fps": 0,
        "camera_health": default_camera_health(),
        "tracking": default_tracking_summary(),
        "privacy_mode": "Anonymous monitoring",
        "message": "Camera connection is starting; waiting for the first status update." if running else "Camera is offline.",
        "metadata": {
            "detect_enabled": bool(meta.get("detect", False)),
            "model": meta.get("model", "disabled"),
            "edge_version": "v40-control",
            "process_every": meta.get("process_every", 10),
            "managed_by_api": True
        },
        "received_at": meta.get("started_at"),
        "received_at_epoch": started_epoch,
        "updated_at": meta.get("started_at"),
        "updated_at_epoch": started_epoch,
        "age_seconds": age,
        "health": "Starting" if running else "Stopped",
        "offline_after_seconds": EDGE_WORKER_OFFLINE_AFTER_SECONDS,
        "process_running": running,
        "process_pid": get_edge_worker_pid(edge_id),
        "managed_by_api": True,
        **get_edge_preview_info(edge_id, online=running)
    }


def merge_managed_edge_workers(workers):
    existing_ids = {safe_edge_id(worker.get("edge_id")) for worker in workers if isinstance(worker, dict)}

    for edge_id in list_managed_edge_ids():
        if edge_id in existing_ids:
            continue

        placeholder = build_managed_edge_placeholder(edge_id)
        if placeholder:
            workers.append(placeholder)

    workers.sort(key=lambda item: safe_epoch_from_time(item.get("received_at_epoch")) if isinstance(item, dict) else 0, reverse=True)
    return workers


@app.get("/api/system/performance")
def get_system_performance():
    return get_performance_status()


@app.post("/api/system/performance/{mode}")
def update_performance_mode(mode: str):
    mode = mode.upper()

    if not set_saved_performance_mode(mode):
        raise HTTPException(status_code=400, detail="Invalid performance mode")

    status = get_performance_status()

    return {
        "message": f"Performance mode updated to {mode}",
        "performance": status,
        "restart_required": True
    }


@app.get("/api/engine/status")
def get_engine_status():
    running = is_engine_running()
    performance = get_performance_status()

    return {
        "running": running,
        "pid": get_engine_pid() if running else None,
        "active_profile": get_active_camera_profile(),
        "mode": performance["resolved_mode"],
        "selected_mode": performance["selected_mode"],
        "recommended_mode": performance["recommended_mode"],
        "system": performance["system"]
    }


@app.post("/api/engine/start")
def start_engine():
    global ENGINE_PROCESS

    performance = get_performance_status()

    if is_engine_running():
        return {
            "message": "AI monitoring engine is already running",
            "running": True,
            "pid": get_engine_pid(),
            "active_profile": get_active_camera_profile(),
            "mode": performance["resolved_mode"],
            "selected_mode": performance["selected_mode"],
            "recommended_mode": performance["recommended_mode"]
        }

    if not os.path.exists(MAIN_SCRIPT):
        raise HTTPException(status_code=404, detail="app/main.py not found")

    os.makedirs("data", exist_ok=True)

    env = os.environ.copy()
    env["CROWDVISION_ENGINE_MODE"] = "web"
    env["CROWDVISION_HEADLESS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDVISION_PERFORMANCE_MODE"] = performance["selected_mode"]
    env["CROWDVISION_RESOLVED_PERFORMANCE_MODE"] = performance["resolved_mode"]

    if performance["resolved_mode"] in ["LOW", "BALANCED"]:
        env["CROWDVISION_LOW_RESOURCE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
    else:
        env["CROWDVISION_LOW_RESOURCE"] = "0"

    log_file = open(ENGINE_LOG_FILE, "a")

    ENGINE_PROCESS = subprocess.Popen(
        [sys.executable, MAIN_SCRIPT],
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
        env=env
    )

    save_engine_pid(ENGINE_PROCESS.pid)

    return {
        "message": f"AI monitoring engine started in {performance['resolved_mode']} mode",
        "running": True,
        "pid": ENGINE_PROCESS.pid,
        "active_profile": get_active_camera_profile(),
        "mode": performance["resolved_mode"],
        "selected_mode": performance["selected_mode"],
        "recommended_mode": performance["recommended_mode"]
    }


@app.post("/api/engine/stop")
def stop_engine():
    global ENGINE_PROCESS

    pid = get_engine_pid()

    if not pid:
        ENGINE_PROCESS = None
        clear_engine_pid()
        mark_camera_status_stopped("Monitoring engine is not running")

        return {
            "message": "AI monitoring engine is not running",
            "running": False
        }

    stop_pid(pid)

    ENGINE_PROCESS = None
    clear_engine_pid()
    mark_camera_status_stopped("Monitoring engine stopped by operator")

    return {
        "message": "AI monitoring engine stopped",
        "running": False
    }


@app.post("/api/engine/restart")
def restart_engine():
    stop_engine()
    return start_engine()


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "CrowdVision AI API"
    }


@app.get("/api/video-feed")
def video_feed():
    return StreamingResponse(
        generate_live_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/live-image")
def get_live_image():
    if not os.path.exists(LIVE_FRAME_PATH):
        return {
            "image": None,
            "live": False,
            "frame_age_seconds": None
        }

    status = read_camera_status_file()
    running = is_engine_running()
    now = time.time()

    frame_age = round(now - os.path.getmtime(LIVE_FRAME_PATH), 2)

    active_status_age = None

    if status and status.get("updated_at_epoch"):
        active_status_age = round(now - float(status.get("updated_at_epoch", now)), 2)

    live = bool(
        running and
        status and
        status.get("online") and
        active_status_age is not None and
        active_status_age <= CAMERA_OFFLINE_AFTER_SECONDS
    )

    return {
        "image": "http://127.0.0.1:8000/live/latest_frame.jpg",
        "live": live,
        "frame_age_seconds": frame_age
    }


def detect_report_type(filename, file_path=None):
    """Classify reports from filename and Report Type content."""
    lower_name = str(filename or "").lower()

    if lower_name.startswith("incident"):
        return "Incident"
    if lower_name.startswith("anomaly"):
        return "Anomaly"
    if lower_name.startswith(("camera_health", "tamper", "camera_integrity")):
        return "Camera Integrity"

    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r") as file:
                for _ in range(16):
                    line = file.readline()
                    if not line:
                        break
                    if line.lower().startswith("report type:"):
                        value = line.split(":", 1)[1].strip().lower()
                        if "camera" in value or "tamper" in value or "integrity" in value:
                            return "Camera Integrity"
                        if "incident" in value:
                            return "Incident"
                        if "anomaly" in value:
                            return "Anomaly"
        except Exception:
            pass

    return "General"


@app.get("/api/reports")
def get_reports():
    reports = []

    if not os.path.exists(REPORTS_DIR):
        return {"reports": []}

    for filename in os.listdir(REPORTS_DIR):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(REPORTS_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        pdf_filename = filename.replace(".txt", ".pdf")
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

        if not os.path.exists(pdf_path):
            txt_report_to_pdf(file_path)

        report_type = detect_report_type(filename, file_path)

        reports.append({
            "filename": filename,
            "pdf_filename": pdf_filename,
            "type": report_type,
            "type_key": report_type.lower().replace(" ", "_"),
            "size": os.path.getsize(file_path),
            "pdf_size": os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0,
            "created": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(os.path.getmtime(file_path))
            ),
            "url": f"http://127.0.0.1:8000/reports/{filename}",
            "pdf_url": f"http://127.0.0.1:8000/reports/{pdf_filename}"
        })

    reports.sort(key=lambda report: report["created"], reverse=True)
    return {"reports": reports}


@app.get("/api/reports/{filename}")
def read_report(filename: str):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(REPORTS_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")

    with open(file_path, "r") as file:
        content = file.read()

    return PlainTextResponse(content)


@app.get("/api/events")
def get_events():
    df = load_events()

    if df.empty:
        return {"events": []}

    return {"events": df.tail(100).to_dict(orient="records")}


@app.get("/api/latest-event")
def get_latest_event():
    df = load_events()

    if df.empty:
        return {"latest_event": None}

    return {"latest_event": df.tail(1).to_dict(orient="records")[0]}


@app.get("/api/summary")
def get_summary():
    df = load_events()

    if df.empty:
        return {
            "total_events": 0,
            "peak_people": 0,
            "peak_occupancy": 0,
            "critical_alerts": 0,
            "high_alerts": 0,
            "latest_density": "Unknown"
        }

    return {
        "total_events": len(df),
        "peak_people": int(df["total_people"].max()),
        "peak_occupancy": int(df["peak_occupancy"].max()),
        "critical_alerts": int((df["density_level"] == "Critical").sum()),
        "high_alerts": int((df["density_level"] == "High").sum()),
        "latest_density": str(df.tail(1)["density_level"].values[0])
    }


@app.get("/api/intelligence")
def get_intelligence():
    df = load_events()

    if df.empty:
        return {
            "average_people": 0,
            "average_occupancy": 0,
            "average_zone_count": 0,
            "highest_risk_time": "None",
            "highest_risk_density": "Unknown",
            "latest_camera_profile": "None",
            "alert_rate": 0
        }

    risk_order = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Critical": 4
    }

    df = df.copy()
    df["risk_score"] = df["density_level"].map(risk_order).fillna(0)

    highest_risk_row = df.sort_values(
        by=["risk_score", "total_people", "occupancy"],
        ascending=False
    ).head(1)

    alert_count = int(df["density_level"].isin(["High", "Critical"]).sum())

    return {
        "average_people": round(float(df["total_people"].mean()), 2),
        "average_occupancy": round(float(df["occupancy"].mean()), 2),
        "average_zone_count": round(float(df["zone_count"].mean()), 2),
        "highest_risk_time": str(highest_risk_row["timestamp"].values[0]),
        "highest_risk_density": str(highest_risk_row["density_level"].values[0]),
        "latest_camera_profile": str(df.tail(1)["camera_profile"].values[0]),
        "alert_rate": round((alert_count / len(df)) * 100, 2)
    }


@app.get("/api/prediction")
def get_prediction():
    df = load_events()

    if df.empty or "predicted_people" not in df.columns:
        return {
            "current_people": 0,
            "predicted_people": 0,
            "current_occupancy": 0,
            "predicted_occupancy": 0,
            "risk_trend": "Stable",
            "forecast_message": "No prediction data available yet."
        }

    latest = df.tail(1).to_dict(orient="records")[0]

    predicted_occupancy = int(latest.get("predicted_occupancy", 0))
    risk_trend_value = str(latest.get("risk_trend", "Stable"))

    return {
        "current_people": int(latest.get("total_people", 0)),
        "predicted_people": int(latest.get("predicted_people", 0)),
        "current_occupancy": int(latest.get("occupancy", 0)),
        "predicted_occupancy": predicted_occupancy,
        "risk_trend": risk_trend_value,
        "forecast_message": get_forecast_message(predicted_occupancy, risk_trend_value)
    }


@app.get("/api/anomalies")
def get_anomalies():
    df = load_events()

    if df.empty or "anomaly_detected" not in df.columns:
        return {"anomalies": []}

    anomalies_df = df[df["anomaly_detected"].astype(str).str.lower() == "true"]
    if anomalies_df.empty:
        return {"anomalies": []}

    return {"anomalies": anomalies_df.tail(50).to_dict(orient="records")}


@app.get("/api/latest-anomaly")
def get_latest_anomaly():
    runtime = get_runtime_context()
    current_event = runtime.get("current_event")
    last_recorded = get_last_recorded_anomaly()

    active_anomaly = current_event if current_event and as_bool(current_event.get("anomaly_detected")) else None

    return {
        "latest_anomaly": active_anomaly,
        "last_recorded_anomaly": last_recorded,
        "state": "active" if active_anomaly else ("historical" if last_recorded else "none"),
        "monitoring_running": runtime.get("running", False),
        "live_data": runtime.get("latest_event_fresh", False)
    }


@app.get("/api/anomaly-summary")
def get_anomaly_summary():
    df = load_events()
    runtime = get_runtime_context()
    current_event = runtime.get("current_event")
    active_anomaly = current_event if current_event and as_bool(current_event.get("anomaly_detected")) else None

    empty = {
        "active_anomalies": 1 if active_anomaly else 0,
        "total_anomalies": 0,
        "highest_anomaly_score": 0,
        "latest_anomaly_type": "Normal",
        "latest_anomaly_severity": "Normal",
        "latest_anomaly_zone": "None",
        "latest_anomaly_recommendation": "Monitor",
        "latest_recorded_at": None,
        "monitoring_running": runtime.get("running", False),
    }

    if df.empty or "anomaly_detected" not in df.columns:
        return empty

    anomalies_df = df[df["anomaly_detected"].astype(str).str.lower() == "true"]
    if anomalies_df.empty:
        return empty

    latest = anomalies_df.tail(1).to_dict(orient="records")[0]
    return {
        **empty,
        "total_anomalies": len(anomalies_df),
        "highest_anomaly_score": int(pd.to_numeric(anomalies_df["anomaly_score"], errors="coerce").fillna(0).max()),
        "latest_anomaly_type": latest.get("anomaly_type", "Normal"),
        "latest_anomaly_severity": latest.get("anomaly_severity", "Normal"),
        "latest_anomaly_zone": latest.get("anomaly_zone", "None"),
        "latest_anomaly_recommendation": latest.get("anomaly_recommendation", "Monitor"),
        "latest_recorded_at": latest.get("timestamp"),
    }


@app.get("/api/incidents")
def get_incidents(
    page: int = 1,
    page_size: int = 25,
    density: str = "all",
    source: str = "all"
):
    df = load_events()

    try:
        page = max(1, int(page))
        page_size = max(5, min(int(page_size), 100))
    except Exception:
        page, page_size = 1, 25

    if df.empty or "incident_active" not in df.columns:
        return {
            "incidents": [],
            "pagination": {"page": page, "page_size": page_size, "total": 0, "pages": 1}
        }

    incidents_df = df[df["incident_active"].astype(str).str.lower() == "true"].copy()

    if density.lower() != "all" and "density_level" in incidents_df.columns:
        incidents_df = incidents_df[incidents_df["density_level"].astype(str).str.lower() == density.lower()]

    if source.lower() != "all" and "camera_profile" in incidents_df.columns:
        incidents_df = incidents_df[incidents_df["camera_profile"].astype(str).str.lower() == source.lower()]

    incidents_df = incidents_df.iloc[::-1]
    total = len(incidents_df)
    pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, pages)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "incidents": incidents_df.iloc[start:end].to_dict(orient="records"),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": pages,
            "has_previous": page > 1,
            "has_next": page < pages,
        }
    }


@app.get("/api/latest-incident")
def get_latest_incident():
    runtime = get_runtime_context()
    current_event = runtime.get("current_event")
    last_recorded = get_last_recorded_incident()

    active_incident = current_event if current_event and as_bool(current_event.get("incident_active")) else None

    return {
        "latest_incident": active_incident,
        "last_recorded_incident": last_recorded,
        "state": "active" if active_incident else ("historical" if last_recorded else "none"),
        "monitoring_running": runtime.get("running", False),
        "live_data": runtime.get("latest_event_fresh", False)
    }


@app.get("/api/incident-summary")
def get_incident_summary():
    df = load_events()
    runtime = get_runtime_context()
    current_event = runtime.get("current_event")
    active_incident = current_event if current_event and as_bool(current_event.get("incident_active")) else None

    empty = {
        "active_incidents": 1 if active_incident else 0,
        "total_incident_events": 0,
        "longest_incident_duration": 0,
        "latest_danger_zone": "None",
        "latest_recommendation": "Monitor",
        "latest_recorded_at": None,
        "monitoring_running": runtime.get("running", False),
    }

    if df.empty or "incident_active" not in df.columns:
        return empty

    incidents_df = df[df["incident_active"].astype(str).str.lower() == "true"]
    if incidents_df.empty:
        return empty

    latest_incident = incidents_df.tail(1).to_dict(orient="records")[0]
    return {
        **empty,
        "total_incident_events": len(incidents_df),
        "longest_incident_duration": int(pd.to_numeric(incidents_df["incident_duration"], errors="coerce").fillna(0).max()),
        "latest_danger_zone": latest_incident.get("danger_zone", "None"),
        "latest_recommendation": latest_incident.get("recommendation", "Monitor"),
        "latest_recorded_at": latest_incident.get("timestamp"),
    }


@app.get("/api/chart-data")
def get_chart_data():
    df = load_events()

    if df.empty:
        return {"chart_data": []}

    chart_df = df.tail(50).copy()

    columns = [
        "timestamp",
        "total_people",
        "zone_count",
        "occupancy",
        "peak_occupancy",
        "line_entries",
        "line_exits",
        "density_level"
    ]

    optional_columns = [
        "predicted_people",
        "predicted_occupancy",
        "risk_trend",
        "anomaly_score",
        "anomaly_type",
        "anomaly_severity"
    ]

    for column in optional_columns:
        if column in chart_df.columns:
            columns.append(column)

    return {"chart_data": chart_df[columns].to_dict(orient="records")}


@app.get("/api/alert-distribution")
def get_alert_distribution():
    df = load_events()

    if df.empty:
        return {"alerts": []}

    counts = df["density_level"].value_counts().reset_index()
    counts.columns = ["density_level", "count"]

    return {"alerts": counts.to_dict(orient="records")}


@app.get("/api/tracking-summary")
def get_tracking_summary_endpoint():
    status = read_camera_status_file()

    return {
        "tracking": normalize_tracking_summary(status),
        "status": status
    }


def get_export_dataframe(export_type):
    df = load_events()

    if df.empty:
        return pd.DataFrame()

    if export_type == "events":
        return df

    if export_type == "incidents":
        if "incident_active" not in df.columns:
            return pd.DataFrame()

        return df[df["incident_active"].astype(str) == "True"]

    if export_type == "anomalies":
        if "anomaly_detected" not in df.columns:
            return pd.DataFrame()

        return df[df["anomaly_detected"].astype(str) == "True"]

    if export_type == "summary":
        summary_data = {
            "total_events": [len(df)],
            "peak_people": [int(df["total_people"].max())],
            "peak_occupancy": [int(df["peak_occupancy"].max())],
            "critical_alerts": [int((df["density_level"] == "Critical").sum())],
            "high_alerts": [int((df["density_level"] == "High").sum())],
            "average_people": [round(float(df["total_people"].mean()), 2)],
            "average_occupancy": [round(float(df["occupancy"].mean()), 2)],
            "exported_at": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }

        return pd.DataFrame(summary_data)

    return pd.DataFrame()


@app.get("/api/export/{export_type}/csv")
def export_csv(export_type: str):
    export_type = export_type.lower()

    if export_type not in ["events", "incidents", "anomalies", "summary"]:
        raise HTTPException(status_code=400, detail="Invalid export type")

    export_df = get_export_dataframe(export_type)

    if export_df.empty:
        raise HTTPException(status_code=404, detail="No data available for export")

    csv_buffer = StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    filename = f"crowdvision_{export_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@app.get("/api/export/{export_type}/excel")
def export_excel(export_type: str):
    export_type = export_type.lower()

    if export_type not in ["events", "incidents", "anomalies", "summary"]:
        raise HTTPException(status_code=400, detail="Invalid export type")

    export_df = get_export_dataframe(export_type)

    if export_df.empty:
        raise HTTPException(status_code=404, detail="No data available for export")

    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name=export_type.title())

    excel_buffer.seek(0)

    filename = f"crowdvision_{export_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    return StreamingResponse(
        excel_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


def get_latest_event_record():
    df = load_events()

    if df.empty:
        return None

    return df.tail(1).to_dict(orient="records")[0]


def list_evidence_files():
    evidence = []

    for directory in EVIDENCE_DIRS:
        if not os.path.exists(directory):
            continue

        for filename in os.listdir(directory):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            file_path = os.path.join(directory, filename)

            if not os.path.isfile(file_path):
                continue

            evidence.append({
                "filename": filename,
                "directory": directory,
                "type": "Snapshot",
                "size": os.path.getsize(file_path),
                "created": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(os.path.getmtime(file_path))
                ),
                "url": f"http://127.0.0.1:8000/{directory.replace('data/', '')}/{filename}" if directory.startswith("data/") else ""
            })

    evidence.sort(key=lambda item: item["created"], reverse=True)

    return evidence


@app.get("/api/camera-health")
def get_camera_health():
    active_profile = get_active_camera_profile()
    all_profiles = load_camera_profiles()
    presented_profiles = present_camera_profiles(all_profiles, active_profile)
    profiles = {item["key"]: all_profiles[item["key"]] for item in presented_profiles}
    status = read_camera_status_file()
    latest_event = get_latest_event_record()
    running = is_engine_running()
    now = time.time()

    frame_age = None
    preview_url = None

    if os.path.exists(LIVE_FRAME_PATH):
        frame_age = round(now - os.path.getmtime(LIVE_FRAME_PATH), 2)
        preview_url = "http://127.0.0.1:8000/live/latest_frame.jpg"

    active_status_age = None

    if status and status.get("updated_at_epoch"):
        active_status_age = round(now - float(status.get("updated_at_epoch", now)), 2)

    active_online = bool(
        running and
        status and
        status.get("online") and
        active_status_age is not None and
        active_status_age <= CAMERA_OFFLINE_AFTER_SECONDS
    )

    active_tracking = normalize_tracking_summary(status)
    active_camera_health = normalize_camera_health(status)
    cameras = []

    for key, profile in profiles.items():
        is_active = key == active_profile

        if is_active:
            health = "Online" if active_online else ("Starting" if running else "Stopped")

            cameras.append({
                "key": key,
                "name": profile.get("name", key),
                "source": str(profile.get("source", "")),
                "source_type": profile.get("source_type", "camera"),
                "active": True,
                "online": active_online,
                "health": health,
                "last_seen": status.get("updated_at") if status else None,
                "age_seconds": active_status_age,
                "frame_age_seconds": frame_age,
                "preview_url": preview_url,
                "people": status.get("people", latest_event.get("total_people", 0) if latest_event else 0) if status else 0,
                "density": status.get("density", latest_event.get("density_level", "Unknown") if latest_event else "Unknown") if status else "Unknown",
                "fps": status.get("fps", 0) if status else 0,
                "ai_fps": status.get("ai_fps", 0) if status else 0,
                "reason": status.get("reason", "") if status else "",
                "camera_health": active_camera_health,
                "tracking": active_tracking
            })
        else:
            cameras.append({
                "key": key,
                "name": profile.get("name", key),
                "source": str(profile.get("source", "")),
                "source_type": profile.get("source_type", "camera"),
                "active": False,
                "online": False,
                "health": "Standby",
                "last_seen": None,
                "age_seconds": None,
                "frame_age_seconds": None,
                "preview_url": None,
                "people": 0,
                "density": "Standby",
                "fps": 0,
                "ai_fps": 0,
                "reason": "Not active",
                "camera_health": default_camera_health(),
                "tracking": default_tracking_summary()
            })

    return {
        "active_profile": active_profile,
        "engine_running": running,
        "active_online": active_online,
        "stream_live": active_online,
        "latest_frame_age_seconds": frame_age,
        "live_frame_url": preview_url,
        "camera_health": active_camera_health,
        "tracking": active_tracking,
        "status": status,
        "cameras": cameras
    }


def read_notification_file():
    """
    Read stored professional notifications written by app/alerts.py.

    Your monitoring engine can now create persistent alerts for:
    - high / critical crowd density
    - active incidents
    - anomalies
    - camera integrity / tamper warnings
    - engine status events

    This function is intentionally safe. If the file does not exist yet,
    the API still works and simply returns the live fallback notifications.
    """
    if not os.path.exists(NOTIFICATION_FILE):
        return []

    try:
        with open(NOTIFICATION_FILE, "r") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []
    except Exception as error:
        print(f"Unable to read notification file: {error}")
        return []


def safe_epoch_from_time(value):
    if value in [None, "", 0]:
        return 0.0

    try:
        return float(value)
    except Exception:
        pass

    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return 0.0


def operator_friendly_notification_text(title, message, action, source, metadata=None):
    """Translate older technical camera-network wording for operator-facing screens."""
    title = str(title or "CrowdVision Notification")
    message = str(message or "System event recorded.")
    action = str(action or "Monitor")
    source = str(source or "CrowdVision AI")
    metadata = metadata if isinstance(metadata, dict) else {}

    exact_titles = {
        "Edge Worker Started": "Remote Camera Connected",
        "Edge Worker Stopped": "Remote Camera Disconnected",
        "Edge Worker Alert": "Remote Camera Alert",
    }
    title = exact_titles.get(title, title)

    if title.startswith("Edge Camera Integrity Alert:"):
        title = title.replace("Edge Camera Integrity Alert:", "Camera Integrity Alert:", 1)

    if title.startswith("Edge Worker ") and title.endswith(" Density"):
        density = title.replace("Edge Worker ", "", 1).replace(" Density", "").strip()
        title = f"{density} Crowd Level Detected"

    message_replacements = {
        "Edge worker ": "Remote camera ",
        "edge worker ": "remote camera ",
        "edge workers": "remote cameras",
        "Edge workers": "Remote cameras",
        "edge telemetry": "camera status",
        "Edge telemetry": "Camera status",
        "edge camera": "remote camera",
        "Edge camera": "Remote camera",
        "edge nodes": "connected cameras",
        "Edge nodes": "Connected cameras",
    }
    for old_text, new_text in message_replacements.items():
        message = message.replace(old_text, new_text)
        action = action.replace(old_text, new_text)

    edge_id = str(metadata.get("edge_id") or "").strip()
    if edge_id and source in {edge_id, safe_edge_id(edge_id)}:
        meta = read_edge_meta(edge_id)
        camera_name = str(meta.get("name") or "").strip()
        if camera_name:
            source = camera_name

    return title, message, action, source


def normalize_notification_record(notification, fallback_index=0):
    if not isinstance(notification, dict):
        notification = {}

    category = str(
        notification.get("category") or
        notification.get("type") or
        "system"
    ).lower()

    severity = str(notification.get("severity") or "normal").lower()

    # Keep the frontend classes predictable.
    severity_aliases = {
        "info": "normal",
        "success": "normal",
        "healthy": "normal",
        "medium": "warning",
        "high": "warning",
        "critical": "critical",
        "danger": "critical",
        "warning": "warning",
        "normal": "normal"
    }
    severity = severity_aliases.get(severity, severity)

    time_text = str(
        notification.get("time") or
        notification.get("created") or
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    epoch = safe_epoch_from_time(notification.get("epoch"))
    if epoch <= 0:
        epoch = safe_epoch_from_time(time_text)

    title = str(notification.get("title") or "CrowdVision Notification")
    message = str(notification.get("message") or "System event recorded.")
    source = str(notification.get("source") or get_active_camera_profile() or "CrowdVision AI")
    action = str(notification.get("action") or "Monitor")
    metadata = notification.get("metadata") if isinstance(notification.get("metadata"), dict) else {}

    title, message, action, source = operator_friendly_notification_text(
        title=title,
        message=message,
        action=action,
        source=source,
        metadata=metadata
    )

    dedupe_key = str(
        notification.get("dedupe_key") or
        f"{category}:{title}:{message}:{source}"
    )

    return {
        "id": str(notification.get("id") or f"notification_{fallback_index}_{int(epoch)}"),
        "type": category,
        "category": category,
        "severity": severity,
        "title": title,
        "message": message,
        "source": source,
        "action": action,
        "time": time_text,
        "epoch": epoch,
        "dedupe_key": dedupe_key,
        "metadata": metadata
    }


def build_live_notifications():
    """Build only conditions that are true right now."""
    notifications = []
    runtime = get_runtime_context()
    current_event = runtime.get("current_event")
    camera_health = get_camera_health()
    active_camera = next((camera for camera in camera_health.get("cameras", []) if camera.get("active")), None)
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now_epoch = time.time()

    if not runtime.get("running"):
        notifications.append({
            "type": "engine",
            "category": "engine",
            "severity": "warning",
            "title": "Monitoring Paused",
            "message": "The AI monitoring engine is not currently running.",
            "source": get_active_camera_profile(),
            "action": "Start monitoring when ready.",
            "time": now_text,
            "epoch": now_epoch,
            "dedupe_key": "live:engine:stopped"
        })

    if runtime.get("running") and active_camera and not active_camera.get("online"):
        notifications.append({
            "type": "camera",
            "category": "camera",
            "severity": "critical",
            "title": "Camera Feed Interrupted",
            "message": f"{active_camera.get('name')} is not producing fresh frames.",
            "source": active_camera.get("name") or get_active_camera_profile(),
            "action": "Check camera source, cable, stream URL, or video file.",
            "time": now_text,
            "epoch": now_epoch,
            "dedupe_key": f"live:camera:offline:{active_camera.get('key')}"
        })

    active_camera_integrity = active_camera.get("camera_health", {}) if active_camera else {}
    if runtime.get("running") and active_camera_integrity.get("tamper_detected"):
        severity = "critical" if active_camera_integrity.get("severity") == "Critical" else "warning"
        tamper_type = active_camera_integrity.get("tamper_type", "Camera Integrity Warning")
        notifications.append({
            "type": "camera_health",
            "category": "camera_health",
            "severity": severity,
            "title": f"Camera Integrity Alert: {tamper_type}",
            "message": active_camera_integrity.get("message", "Camera signal integrity warning."),
            "source": active_camera.get("name") if active_camera else get_active_camera_profile(),
            "action": "Inspect the camera lens, lighting, signal, cable, or network.",
            "time": active_camera_integrity.get("last_alert_at") or now_text,
            "epoch": safe_epoch_from_time(active_camera_integrity.get("last_alert_at")) or now_epoch,
            "dedupe_key": f"live:camera_health:{tamper_type}"
        })

    if current_event:
        density = str(current_event.get("density_level", "Unknown"))
        event_time = current_event.get("timestamp", now_text)
        event_epoch_value = safe_epoch_from_time(event_time) or now_epoch
        event_source = current_event.get("camera_profile", get_active_camera_profile())

        if density in ["High", "Critical"]:
            notifications.append({
                "type": "density",
                "category": "density",
                "severity": "critical" if density == "Critical" else "warning",
                "title": f"{density} Crowd Density",
                "message": current_event.get("alert_message", "Crowd alert detected."),
                "source": event_source,
                "action": current_event.get("recommendation", "Increase monitoring and prepare response."),
                "time": event_time,
                "epoch": event_epoch_value,
                "dedupe_key": f"live:density:{event_source}:{density}"
            })

        if as_bool(current_event.get("incident_active")):
            notifications.append({
                "type": "incident",
                "category": "incident",
                "severity": "critical",
                "title": "Active Crowd Incident",
                "message": f"Danger zone: {current_event.get('danger_zone', 'Unknown')}. {current_event.get('recommendation', 'Monitor')}",
                "source": event_source,
                "action": current_event.get("recommendation", "Respond to active incident."),
                "time": event_time,
                "epoch": event_epoch_value + 0.01,
                "dedupe_key": f"live:incident:{event_source}:{current_event.get('danger_zone', 'Unknown')}"
            })

        if as_bool(current_event.get("anomaly_detected")):
            anomaly_severity = str(current_event.get("anomaly_severity", "warning")).lower()
            if anomaly_severity == "high":
                anomaly_severity = "warning"
            notifications.append({
                "type": "anomaly",
                "category": "anomaly",
                "severity": anomaly_severity,
                "title": f"Anomaly: {current_event.get('anomaly_type', 'Detected')}",
                "message": f"Score: {current_event.get('anomaly_score', 0)}%. Zone: {current_event.get('anomaly_zone', 'Unknown')}",
                "source": event_source,
                "action": current_event.get("anomaly_recommendation", "Investigate anomaly."),
                "time": event_time,
                "epoch": event_epoch_value + 0.02,
                "dedupe_key": f"live:anomaly:{event_source}:{current_event.get('anomaly_type', 'Detected')}"
            })

    # Current remote-camera conditions are also active alerts while the camera is online.
    try:
        for remote_camera in latest_edge_records():
            if not remote_camera.get("online"):
                continue

            edge_id = str(remote_camera.get("edge_id") or "")
            if edge_id == "manual_test":
                continue

            edge_name = remote_camera.get("name") or "Remote Camera"
            edge_health = remote_camera.get("camera_health") or {}
            edge_density = str(remote_camera.get("density") or "Unknown")
            edge_time = remote_camera.get("received_at") or now_text
            edge_epoch = safe_epoch_from_time(remote_camera.get("received_at_epoch")) or now_epoch

            if edge_health.get("tamper_detected"):
                tamper_type = edge_health.get("tamper_type") or "Camera Integrity Warning"
                notifications.append({
                    "type": "remote_camera_health",
                    "category": "remote_camera_health",
                    "severity": "critical" if edge_health.get("severity") == "Critical" else "warning",
                    "title": f"Camera Integrity Alert: {tamper_type}",
                    "message": edge_health.get("message") or "Remote camera integrity warning detected.",
                    "source": edge_name,
                    "action": "Inspect the camera lens, lighting, cable, stream, or network connection.",
                    "time": edge_time,
                    "epoch": edge_epoch,
                    "dedupe_key": f"live:remote-camera-health:{edge_id}:{tamper_type}"
                })

            if edge_density in ["High", "Critical"]:
                notifications.append({
                    "type": "remote_density",
                    "category": "remote_density",
                    "severity": "critical" if edge_density == "Critical" else "warning",
                    "title": f"{edge_density} Crowd Level Detected",
                    "message": f"{edge_name} currently reports {remote_camera.get('people', 0)} people and a {edge_density.lower()} crowd level.",
                    "source": edge_name,
                    "action": "Review the camera view and prepare a response if crowd pressure continues.",
                    "time": edge_time,
                    "epoch": edge_epoch + 0.03,
                    "dedupe_key": f"live:remote-density:{edge_id}:{edge_density}"
                })
    except Exception as error:
        print(f"Unable to evaluate current remote-camera alerts: {error}")

    active_items = [normalize_notification_record(item, index) for index, item in enumerate(notifications)]
    for item in active_items:
        item["is_active"] = True
    return active_items


def get_recent_notification_activity(stored_notifications, active_notifications, limit=6):
    active_keys = {item.get("dedupe_key") for item in active_notifications}
    normalized = [
        normalize_notification_record(item, index)
        for index, item in enumerate(stored_notifications or [])
    ]
    normalized.sort(key=lambda item: float(item.get("epoch", 0)), reverse=True)

    recent = []
    seen = set()
    for item in normalized:
        key = item.get("dedupe_key") or item.get("id")
        if key in active_keys or key in seen:
            continue
        seen.add(key)
        item["is_active"] = False
        recent.append(item)
        if len(recent) >= int(limit):
            break
    return recent


@app.get("/api/notifications")
def get_notifications():
    stored_notifications = read_notification_file()
    active_notifications = build_live_notifications()
    recent_activity = get_recent_notification_activity(
        stored_notifications=stored_notifications,
        active_notifications=active_notifications,
        limit=6
    )

    combined = active_notifications + recent_activity
    return {
        "notifications": combined,
        "active_notifications": active_notifications,
        "recent_activity": recent_activity,
        "active_count": len(active_notifications),
        "recent_count": len(recent_activity),
    }


# -----------------------------------------------------------------------------
# Optional Multi-Camera Edge Worker Architecture
# -----------------------------------------------------------------------------

def read_edge_telemetry_file():
    """
    Read telemetry sent by optional edge workers.

    Edge workers are separate camera processors that can run on another laptop,
    mini PC, CCTV machine, or remote site. They send lightweight metadata to the
    central CrowdVision API instead of forcing every camera through one process.
    """
    if not os.path.exists(EDGE_TELEMETRY_FILE):
        return []

    try:
        with open(EDGE_TELEMETRY_FILE, "r") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []
    except Exception as error:
        print(f"Unable to read edge telemetry file: {error}")
        return []


def write_edge_telemetry_file(records):
    try:
        os.makedirs(os.path.dirname(EDGE_TELEMETRY_FILE), exist_ok=True)
        temp_path = f"{EDGE_TELEMETRY_FILE}.tmp"

        with open(temp_path, "w") as file:
            json.dump(records[-EDGE_TELEMETRY_LIMIT:], file, indent=2)

        os.replace(temp_path, EDGE_TELEMETRY_FILE)
        return True
    except Exception as error:
        print(f"Unable to write edge telemetry file: {error}")
        return False


def append_edge_notification(notification):
    """
    Add important edge-worker events into the same notification store used by
    the dashboard. This keeps the frontend unchanged.
    """
    if not isinstance(notification, dict):
        return False

    try:
        os.makedirs(os.path.dirname(NOTIFICATION_FILE), exist_ok=True)
        existing = read_notification_file()

        existing.append({
            "id": notification.get("id") or f"edge_{int(time.time())}",
            "category": notification.get("category", "edge"),
            "type": notification.get("type", notification.get("category", "edge")),
            "severity": notification.get("severity", "normal"),
            "title": notification.get("title", "Remote Camera Alert"),
            "message": notification.get("message", "Remote camera event recorded."),
            "source": notification.get("source", "Remote Camera"),
            "action": notification.get("action", "Monitor camera status."),
            "time": notification.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "epoch": notification.get("epoch", time.time()),
            "dedupe_key": notification.get("dedupe_key", f"edge:{notification.get('title', 'event')}"),
            "metadata": notification.get("metadata", {}) if isinstance(notification.get("metadata", {}), dict) else {}
        })

        # Keep recent records only. NotificationCenter needs the newest and most useful items.
        existing = sorted(
            existing,
            key=lambda item: safe_epoch_from_time(item.get("epoch")) or safe_epoch_from_time(item.get("time")),
            reverse=True
        )[:200]

        temp_path = f"{NOTIFICATION_FILE}.tmp"
        with open(temp_path, "w") as file:
            json.dump(existing, file, indent=2)

        os.replace(temp_path, NOTIFICATION_FILE)
        return True
    except Exception as error:
        print(f"Unable to append edge notification: {error}")
        return False


def normalize_edge_health(raw_health):
    if not isinstance(raw_health, dict):
        return default_camera_health()

    normalized = default_camera_health()

    for key in normalized.keys():
        if key in raw_health:
            normalized[key] = raw_health.get(key)

    for key in ["brightness", "blur_score", "frame_change", "frozen_seconds"]:
        try:
            normalized[key] = round(float(normalized.get(key, 0)), 2)
        except Exception:
            normalized[key] = 0

    for key in ["enabled", "tamper_detected", "covered_lens", "too_dark", "too_bright", "blurry", "frozen_frame"]:
        normalized[key] = bool(normalized.get(key, False))

    return normalized


def normalize_edge_record(payload):
    if not isinstance(payload, dict):
        payload = {}

    edge_id = str(payload.get("edge_id") or payload.get("id") or "edge_unknown").strip()
    if not edge_id:
        edge_id = "edge_unknown"

    now = time.time()
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    people = payload.get("people", payload.get("total_people", 0))
    try:
        people = int(float(people))
    except Exception:
        people = 0

    zone_count = payload.get("zone_count", 0)
    try:
        zone_count = int(float(zone_count))
    except Exception:
        zone_count = 0

    occupancy = payload.get("occupancy", people)
    try:
        occupancy = int(float(occupancy))
    except Exception:
        occupancy = people

    fps = payload.get("fps", 0)
    ai_fps = payload.get("ai_fps", 0)

    try:
        fps = round(float(fps), 2)
    except Exception:
        fps = 0

    try:
        ai_fps = round(float(ai_fps), 2)
    except Exception:
        ai_fps = 0

    health = normalize_edge_health(payload.get("camera_health", {}))

    record = {
        "edge_id": edge_id,
        "name": str(payload.get("name") or payload.get("edge_name") or edge_id),
        "source": str(payload.get("source") or "Unknown"),
        "source_type": str(payload.get("source_type") or "edge_camera"),
        "location": str(payload.get("location") or "Unassigned"),
        "online": bool(payload.get("online", True)),
        "status": str(payload.get("status") or "Online"),
        "people": people,
        "total_people": people,
        "zone_count": zone_count,
        "occupancy": occupancy,
        "density": str(payload.get("density") or payload.get("density_level") or "Unknown"),
        "fps": fps,
        "ai_fps": ai_fps,
        "camera_health": health,
        "tracking": payload.get("tracking") if isinstance(payload.get("tracking"), dict) else default_tracking_summary(),
        "privacy_mode": str(payload.get("privacy_mode") or "Anonymous monitoring"),
        "message": str(payload.get("message") or health.get("message") or "Camera status received."),
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        "received_at": now_text,
        "received_at_epoch": now,
        "updated_at": str(payload.get("updated_at") or payload.get("timestamp") or now_text),
        "updated_at_epoch": safe_epoch_from_time(payload.get("updated_at_epoch")) or safe_epoch_from_time(payload.get("timestamp_epoch")) or now,
    }

    return record


def latest_edge_records():
    records = read_edge_telemetry_file()
    latest = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        edge_id = str(record.get("edge_id") or "edge_unknown")
        previous = latest.get(edge_id)

        if not previous:
            latest[edge_id] = record
            continue

        current_epoch = safe_epoch_from_time(record.get("received_at_epoch"))
        previous_epoch = safe_epoch_from_time(previous.get("received_at_epoch"))

        if current_epoch >= previous_epoch:
            latest[edge_id] = record

    now = time.time()
    workers = []

    for record in latest.values():
        received_epoch = safe_epoch_from_time(record.get("received_at_epoch")) or 0
        age = round(now - received_epoch, 2) if received_epoch else None
        online = bool(record.get("online")) and age is not None and age <= EDGE_WORKER_OFFLINE_AFTER_SECONDS

        edge_id = safe_edge_id(record.get("edge_id"))
        process_running = is_edge_worker_running(edge_id)
        process_pid = get_edge_worker_pid(edge_id)
        meta = read_edge_meta(edge_id)
        managed_by_api = bool(meta.get("managed_by_api", False))

        if managed_by_api and not process_running:
            online = False

        workers.append({
            **record,
            "age_seconds": age,
            "online": online,
            "health": "Online" if online else ("Process Running" if process_running else "Offline"),
            "offline_after_seconds": EDGE_WORKER_OFFLINE_AFTER_SECONDS,
            "process_running": process_running,
            "process_pid": process_pid,
            "managed_by_api": managed_by_api,
            "control": meta,
            **get_edge_preview_info(edge_id, online=online)
        })

    workers.sort(key=lambda item: safe_epoch_from_time(item.get("received_at_epoch")), reverse=True)
    return workers


@app.post("/api/edge/telemetry")
def receive_edge_telemetry(payload: dict = Body(...)):
    """
    Receive lightweight metadata from an optional edge worker.

    This endpoint does not replace the existing monitoring engine. It lets you
    add remote cameras later without changing your current dashboard workflow.
    """
    record = normalize_edge_record(payload)
    records = read_edge_telemetry_file()
    records.append(record)

    if not write_edge_telemetry_file(records):
        raise HTTPException(status_code=500, detail="Unable to save edge telemetry")

    density = str(record.get("density", "Unknown"))
    health = record.get("camera_health", {})
    edge_name = record.get("name", record.get("edge_id"))

    if health.get("tamper_detected"):
        tamper_type = health.get("tamper_type", "Camera Integrity Warning")
        append_edge_notification({
            "category": "edge_camera_health",
            "severity": "critical" if health.get("severity") == "Critical" else "warning",
            "title": f"Camera Integrity Alert: {tamper_type}",
            "message": health.get("message", "Camera integrity warning detected."),
            "source": edge_name,
            "action": "Inspect the camera lens, lighting, cable, or network connection.",
            "time": record.get("received_at"),
            "epoch": record.get("received_at_epoch"),
            "dedupe_key": f"edge:tamper:{record.get('edge_id')}:{tamper_type}",
            "metadata": {"edge_id": record.get("edge_id"), "source": record.get("source")}
        })

    if density in ["High", "Critical"]:
        append_edge_notification({
            "category": "edge_density",
            "severity": "critical" if density == "Critical" else "warning",
            "title": f"{density} Crowd Level Detected",
            "message": f"{edge_name} detected {record.get('people', 0)} people and a {density.lower()} crowd level.",
            "source": edge_name,
            "action": "Review the camera view and prepare a response if crowd pressure continues.",
            "time": record.get("received_at"),
            "epoch": record.get("received_at_epoch"),
            "dedupe_key": f"edge:density:{record.get('edge_id')}:{density}",
            "metadata": {"edge_id": record.get("edge_id"), "people": record.get("people", 0)}
        })

    return {
        "message": "Camera status received",
        "edge_id": record.get("edge_id"),
        "received_at": record.get("received_at"),
        "online": record.get("online"),
        "density": record.get("density"),
        "people": record.get("people")
    }




@app.post("/api/edge/preview/{edge_id}")
async def receive_edge_preview(edge_id: str, file: UploadFile = File(...)):
    """Receive a lightweight JPEG preview from a local or remote edge worker."""
    safe_id = safe_edge_id(edge_id)
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty edge preview")

    if len(content) > EDGE_PREVIEW_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Edge preview is too large")

    if not content.startswith(b"\xff\xd8"):
        raise HTTPException(status_code=400, detail="Edge preview must be a JPEG image")

    os.makedirs(EDGE_PREVIEW_DIR, exist_ok=True)
    preview_path = edge_preview_file(safe_id)
    temp_path = f"{preview_path}.tmp"

    try:
        with open(temp_path, "wb") as preview_file:
            preview_file.write(content)

        os.replace(temp_path, preview_path)
    except Exception as error:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

        raise HTTPException(status_code=500, detail=f"Unable to save edge preview: {error}")

    return {
        "message": "Camera preview received",
        "edge_id": safe_id,
        **get_edge_preview_info(safe_id, online=True)
    }


@app.get("/api/edge/processes")
def get_edge_processes():
    processes = []

    for edge_id in list_managed_edge_ids():
        meta = read_edge_meta(edge_id)
        pid = get_edge_worker_pid(edge_id)
        processes.append({
            "edge_id": safe_edge_id(edge_id),
            "running": pid is not None,
            "pid": pid,
            "name": meta.get("name", safe_edge_id(edge_id)),
            "source": meta.get("source", "Unknown"),
            "detect": bool(meta.get("detect", False)),
            "model": meta.get("model", "disabled"),
            "started_at": meta.get("started_at"),
            "log_file": edge_log_file(edge_id),
        })

    return {
        "processes": processes,
        "running": len([process for process in processes if process.get("running")])
    }


@app.post("/api/edge/workers/start-demo")
def start_demo_edge_worker(payload: dict = Body(default={})):
    config = {
        "edge_id": payload.get("edge_id", "demo_video_01"),
        "name": payload.get("name", "Demo Edge Video"),
        "source": payload.get("source", "data/sample_videos/test.mp4"),
        "api": payload.get("api", "http://127.0.0.1:8000"),
        "detect": bool(payload.get("detect", True)),
        "model": payload.get("model", "yolov8n.pt"),
        "loop": bool(payload.get("loop", True)),
        "timeout": payload.get("timeout", 15),
        "interval": payload.get("interval", 10),
        "process_every": payload.get("process_every", 10),
        "location": payload.get("location", "Demo Video Source"),
        "display": bool(payload.get("display", False)),
        "preview": bool(payload.get("preview", True)),
        "preview_interval": payload.get("preview_interval", 1.0),
        "preview_width": payload.get("preview_width", 640),
        "preview_quality": payload.get("preview_quality", 68),
    }

    return start_edge_worker_process(config)


@app.post("/api/edge/workers/start-webcam")
def start_webcam_edge_worker(payload: dict = Body(default={})):
    config = {
        "edge_id": payload.get("edge_id", "gate_01"),
        "name": payload.get("name", "Gate 01 Camera"),
        "source": payload.get("source", "0"),
        "api": payload.get("api", "http://127.0.0.1:8000"),
        "detect": bool(payload.get("detect", False)),
        "model": payload.get("model", "yolov8n.pt"),
        "loop": bool(payload.get("loop", False)),
        "timeout": payload.get("timeout", 15),
        "interval": payload.get("interval", 8),
        "process_every": payload.get("process_every", 10),
        "location": payload.get("location", "Gate 01"),
        "display": bool(payload.get("display", False)),
        "preview": bool(payload.get("preview", True)),
        "preview_interval": payload.get("preview_interval", 1.0),
        "preview_width": payload.get("preview_width", 640),
        "preview_quality": payload.get("preview_quality", 68),
    }

    return start_edge_worker_process(config)


@app.post("/api/edge/workers/start")
def start_custom_edge_worker(payload: dict = Body(default={})):
    required = ["edge_id", "name", "source"]

    for field in required:
        if not payload.get(field):
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    config = {
        "edge_id": payload.get("edge_id"),
        "name": payload.get("name"),
        "source": payload.get("source"),
        "api": payload.get("api", "http://127.0.0.1:8000"),
        "detect": bool(payload.get("detect", False)),
        "model": payload.get("model", "yolov8n.pt"),
        "loop": bool(payload.get("loop", True)),
        "timeout": payload.get("timeout", 15),
        "interval": payload.get("interval", 10),
        "process_every": payload.get("process_every", 10),
        "location": payload.get("location", "Unassigned"),
        "display": bool(payload.get("display", False)),
        "preview": bool(payload.get("preview", True)),
        "preview_interval": payload.get("preview_interval", 1.0),
        "preview_width": payload.get("preview_width", 640),
        "preview_quality": payload.get("preview_quality", 68),
    }

    return start_edge_worker_process(config)


@app.post("/api/edge/workers/{edge_id}/stop")
def stop_edge_worker_endpoint(edge_id: str):
    stopped = stop_edge_worker_process(edge_id)

    return {
        "message": "Camera disconnected" if stopped else "Camera was already offline",
        "edge_id": safe_edge_id(edge_id),
        "running": is_edge_worker_running(edge_id),
        "stopped": stopped
    }


@app.post("/api/edge/workers/stop-all")
def stop_all_edge_workers():
    stopped = []

    for edge_id in list_managed_edge_ids():
        if stop_edge_worker_process(edge_id):
            stopped.append(safe_edge_id(edge_id))

    return {
        "message": "All dashboard-managed cameras stopped",
        "stopped": stopped,
        "count": len(stopped)
    }

@app.get("/api/edge/status")
def get_edge_status():
    workers = merge_managed_edge_workers(latest_edge_records())

    return {
        "enabled": True,
        "total_workers": len(workers),
        "online_workers": len([worker for worker in workers if worker.get("online")]),
        "offline_workers": len([worker for worker in workers if not worker.get("online")]),
        "offline_after_seconds": EDGE_WORKER_OFFLINE_AFTER_SECONDS,
        "workers": workers
    }


@app.get("/api/edge/telemetry")
def get_edge_telemetry(limit: int = 100):
    try:
        limit = max(1, min(int(limit), EDGE_TELEMETRY_LIMIT))
    except Exception:
        limit = 100

    records = read_edge_telemetry_file()
    records = sorted(
        records,
        key=lambda item: safe_epoch_from_time(item.get("received_at_epoch")) if isinstance(item, dict) else 0,
        reverse=True
    )

    return {
        "telemetry": records[:limit],
        "count": len(records[:limit]),
        "total_saved": len(records)
    }


@app.delete("/api/edge/telemetry")
def clear_edge_telemetry():
    write_edge_telemetry_file([])
    return {"message": "Camera status history cleared"}


@app.get("/api/evidence")
def get_evidence():
    return {"evidence": list_evidence_files()}


def camera_profile_signature(profile):
    source = str(profile.get("source", "")).strip()
    source_type = str(profile.get("source_type", "camera")).lower()

    # Numeric OpenCV device indexes refer to the same physical device even when
    # older profiles used different labels such as webcam / USB camera.
    if source.lstrip("-").isdigit() or source_type in {"webcam", "usb_camera"}:
        return f"device:{source}"
    if source_type in {"phone_camera", "ip_camera", "rtsp_camera", "drone_stream"}:
        return f"stream:{source.rstrip('/').lower()}"
    return f"source:{source.lower()}"


def present_camera_profiles(camera_profiles, active_profile):
    """
    Return only operator-facing camera sources.

    Older CrowdVision versions always exposed the built-in ``webcam`` and
    ``crowd_video`` presets. Those presets are now treated as internal
    fallbacks: they stay available to the engine, but disappear from the
    operator interface once a saved source replaces them.

    Duplicate physical devices and duplicate stream URLs are collapsed
    without deleting the user's configuration. The active source always wins.
    """

    groups = {}

    for key, profile in camera_profiles.items():
        signature = camera_profile_signature(profile)
        groups.setdefault(signature, []).append((key, profile))

    presented = []

    for items in groups.values():
        items.sort(
            key=lambda item: (
                item[0] == active_profile,
                item[0] not in LEGACY_SOURCE_PRESET_KEYS,
                str(item[1].get("created_at", "")),
            ),
            reverse=True,
        )

        key, profile = items[0]
        is_active = key == active_profile

        # Hide unused built-in presets from operators. They remain available
        # internally as engine fallbacks, so no configuration is deleted.
        if key in LEGACY_SOURCE_PRESET_KEYS and not is_active:
            continue

        duplicate_keys = [item_key for item_key, _ in items[1:]]

        presented.append({
            "key": key,
            "name": profile.get("name", key),
            "source": str(profile.get("source", "")),
            "source_type": profile.get("source_type", "camera"),
            "zones": len(profile.get("zones", [])),
            "active": is_active,
            "duplicate_count": len(duplicate_keys),
            "hidden_duplicate_keys": duplicate_keys,
            "legacy_preset": key in LEGACY_SOURCE_PRESET_KEYS,
        })

    presented.sort(
        key=lambda profile: (
            not profile.get("active"),
            profile.get("name", "").lower(),
        )
    )

    return presented

@app.get("/api/camera-profiles")
def get_camera_profiles_endpoint():
    active_profile = get_active_camera_profile()
    camera_profiles = load_camera_profiles()

    return {
        "active_profile": active_profile,
        "profiles": present_camera_profiles(camera_profiles, active_profile)
    }


@app.post("/api/camera-profiles/{profile_key}/activate")
def activate_camera_profile(profile_key: str):
    if profile_key not in load_camera_profiles():
        raise HTTPException(status_code=404, detail="Camera profile not found")

    updated = set_active_camera_profile(profile_key)

    if not updated:
        raise HTTPException(status_code=500, detail="Unable to update camera profile")

    return {
        "message": "Camera profile updated successfully",
        "active_profile": profile_key,
        "restart_required": True
    }


def create_source_key(prefix):
    safe_prefix = "".join(
        ch.lower() if ch.isalnum() else "_"
        for ch in prefix
    ).strip("_")

    return f"{safe_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def default_source_profile(name, source, source_type):
    return {
        "name": name,
        "source": source,
        "source_type": source_type,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zone": {
            "name": "Main Monitoring Zone",
            "x1": 40,
            "y1": 80,
            "x2": 560,
            "y2": 520
        },
        "zones": [
            {
                "name": "Main Monitoring Zone",
                "x1": 40,
                "y1": 80,
                "x2": 560,
                "y2": 520
            }
        ],
        "line": {
            "name": "Entry / Exit Line",
            "y": 300
        }
    }


@app.get("/api/sources")
def get_sources():
    return get_camera_profiles_endpoint()


@app.post("/api/sources/device")
def create_device_source(
    name: str = Form(...),
    device_index: int = Form(0),
    source_type: str = Form("webcam")
):
    requested_signature = f"device:{device_index}"
    existing_profiles = load_camera_profiles()

    for existing_key, existing_profile in existing_profiles.items():
        # Built-in fallback presets do not count as operator-created camera
        # records. This lets the user replace the old default webcam card with
        # a named saved source while still preventing real duplicates.
        if existing_key in LEGACY_SOURCE_PRESET_KEYS:
            continue

        if camera_profile_signature(existing_profile) == requested_signature:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Camera device {device_index} is already saved as "
                    f"{existing_profile.get('name', existing_key)}. Use the existing source instead."
                )
            )

    key = create_source_key(source_type)
    profile = default_source_profile(name, device_index, source_type)
    add_camera_source_profile(key, profile)
    set_active_camera_profile(key)

    return {
        "message": "Camera device source created and selected",
        "key": key,
        "active_profile": key,
        "profile": profile,
        "restart_required": True
    }


@app.post("/api/sources/stream")
def create_stream_source(
    name: str = Form(...),
    stream_url: str = Form(...),
    source_type: str = Form("ip_camera")
):
    if not stream_url.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        raise HTTPException(
            status_code=400,
            detail="Stream URL must start with http, https, rtsp, or rtmp"
        )

    key = create_source_key(source_type)
    profile = default_source_profile(name, stream_url, source_type)

    add_camera_source_profile(key, profile)
    set_active_camera_profile(key)

    return {
        "message": "Stream source created and selected",
        "key": key,
        "active_profile": key,
        "profile": profile,
        "restart_required": True
    }


@app.post("/api/sources/video-upload")
async def upload_video_source(
    name: str = Form("Uploaded Video"),
    file: UploadFile = File(...)
):
    allowed_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    original_name = file.filename or "uploaded_video.mp4"
    extension = os.path.splitext(original_name)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    os.makedirs(UPLOADS_DIR, exist_ok=True)

    safe_name = "".join(
        ch if ch.isalnum() or ch in ["-", "_", "."] else "_"
        for ch in original_name
    )

    stored_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    file_path = os.path.join(UPLOADS_DIR, stored_name)

    with open(file_path, "wb") as output_file:
        while True:
            chunk = await file.read(1024 * 1024)

            if not chunk:
                break

            output_file.write(chunk)

    key = create_source_key("uploaded_video")
    profile = default_source_profile(name, file_path, "uploaded_video")

    add_camera_source_profile(key, profile)
    set_active_camera_profile(key)

    return {
        "message": "Video uploaded and selected successfully",
        "key": key,
        "active_profile": key,
        "filename": stored_name,
        "url": f"http://127.0.0.1:8000/uploads/{stored_name}",
        "profile": profile,
        "restart_required": True
    }


@app.delete("/api/sources/{profile_key}")
def delete_source(profile_key: str):
    active_profile = get_active_camera_profile()

    if profile_key in ["webcam", "crowd_video"]:
        raise HTTPException(status_code=400, detail="Built-in sources cannot be deleted")

    deleted = delete_camera_source_profile(profile_key)

    if not deleted:
        raise HTTPException(status_code=404, detail="Source not found")

    if active_profile == profile_key:
        set_active_camera_profile("webcam")

    return {
        "message": "Source deleted successfully",
        "deleted": profile_key,
        "active_profile": get_active_camera_profile()
    }