MODEL_PATH = "yolov8m.pt"

CONFIDENCE_THRESHOLD = 0.40

# AUTO lets CrowdVision choose the safest mode for the current device.
# You can still force LOW, BALANCED, HIGH, or ULTRA.
PERFORMANCE_MODE = "AUTO"

PERFORMANCE_SETTINGS = {
    "LOW": {
        "frame_width": 416,
        "process_every_n_frames": 6,
        "loop_delay": 0.03,
        "description": "Optimized low-resource mode"
    },
    "BALANCED": {
        "frame_width": 640,
        "process_every_n_frames": 4,
        "loop_delay": 0.02,
        "description": "Optimized balanced mode"
    },
    "HIGH": {
        "frame_width": 640,
        "process_every_n_frames": 3,
        "loop_delay": 0.02,
        "description": "Best for 16GB RAM systems or strong CPUs"
    },
    "ULTRA": {
        "frame_width": 960,
        "process_every_n_frames": 1,
        "loop_delay": 0.01,
        "description": "Best for workstation/GPU systems"
    }
}

LOG_INTERVAL = 8
ALERT_CAPTURE_INTERVAL = 30
LIVE_FRAME_SAVE_INTERVAL = 2
LIVE_STREAM_SLEEP = 0.05
LIVE_JPEG_QUALITY = 72
LIVE_PREVIEW_MAX_WIDTH = 960
LIVE_CAMERA_TARGET_FPS = 15

# Enterprise monitoring settings
CAMERA_STATUS_FILE = "data/camera_status.json"
NOTIFICATION_LIMIT = 50
NOTIFICATION_FILE = "data/notifications.json"
NOTIFICATION_COOLDOWN_SECONDS = 20
CAMERA_OFFLINE_AFTER_SECONDS = 20
EVIDENCE_DIRS = ["data/evidence", "data/snapshots", "data/alerts"]

# Optional future multi-camera edge worker telemetry.
# This does not change your current desktop/web monitoring flow.
EDGE_WORKER_STATUS_FILE = "data/edge_workers.json"
ENABLE_EDGE_WORKER_TELEMETRY = True

# Camera health / tamper intelligence.
# These checks catch common CCTV problems such as covered lens, blackout,
# blur, frozen frame, glare, and weak signal. No face recognition is used.
ENABLE_CAMERA_HEALTH_INTELLIGENCE = True
CAMERA_TAMPER_BLUR_THRESHOLD = 55.0
CAMERA_TAMPER_DARK_THRESHOLD = 28.0
CAMERA_TAMPER_BRIGHT_THRESHOLD = 238.0
CAMERA_TAMPER_FROZEN_SECONDS = 8.0
CAMERA_TAMPER_MIN_FRAME_CHANGE = 1.35
CAMERA_TAMPER_WARMUP_FRAMES = 8
CAMERA_TAMPER_ALERT_HOLD_SECONDS = 6.0
CAMERA_TAMPER_SAVE_SNAPSHOT = True


VIDEO_LOOP = True

ENABLE_ZONE_MONITORING = True
ENABLE_LINE_CROSSING = True

# Duplicate-safe line crossing.
# Increased slightly so jitter around the line does not create fake crossings.
LINE_CROSSING_COOLDOWN = 30
LINE_CROSSING_HYSTERESIS = 14

# Heatmap is beautiful but expensive on 4GB RAM systems.
# Keep it False for smooth dashboard monitoring.
ENABLE_HEATMAP = False
HEATMAP_ONLY_INSIDE_ZONE = True
HEATMAP_DECAY_RATE = 0.985
HEATMAP_RADIUS = 16
HEATMAP_OPACITY = 0.10
HEATMAP_BLUR_SIZE = 31

ENABLE_MOVEMENT_ANALYTICS = True
MOVEMENT_THRESHOLD = 5

ENABLE_SMART_INTELLIGENCE = True
ENABLE_MULTI_ZONE_ANALYTICS = True

ENABLE_PROFESSIONAL_UI = True
SIDE_PANEL_WIDTH = 380

# v38.1 anonymous tracking and duplicate-count prevention.
# This is privacy-friendly tracking. It does not use face recognition.
#
# Important:
# This improves re-identification stability when a person briefly disappears,
# moves out of view, changes position, or returns to the camera view.
ENABLE_ANONYMOUS_REID = True
REID_MEMORY_SECONDS = 45
REID_MAX_CENTER_DISTANCE = 320
REID_MIN_MATCH_SCORE = 0.48
REID_MIN_COLOR_SCORE = 0.14
REID_IOU_WEIGHT = 0.15
REID_DISTANCE_WEIGHT = 0.35
REID_COLOR_WEIGHT = 0.50
REID_HIST_BINS = 16

# Keep track memory longer so webcam movement does not create a new person too quickly.
TRACK_LOST_TTL_FRAMES = 300

ACTIVE_CAMERA_PROFILE = "crowd_video"

CAMERA_PROFILES = {
    "webcam": {
        "name": "Laptop Webcam",
        "source": 0,
        "zone": {
            "name": "Webcam Monitoring Zone",
            "x1": 40,
            "y1": 80,
            "x2": 440,
            "y2": 420
        },
        "zones": [
            {
                "name": "Webcam Main Zone",
                "x1": 40,
                "y1": 80,
                "x2": 440,
                "y2": 420
            }
        ],
        "line": {
            "name": "Webcam Entry / Exit Line",
            "y": 300
        }
    },

    "crowd_video": {
        "name": "Crowd Video Test",
        "source": "data/sample_videos/test.mp4",
        "zone": {
            "name": "Main Monitoring Zone",
            "x1": 80,
            "y1": 120,
            "x2": 560,
            "y2": 700
        },
        "zones": [
            {
                "name": "Entrance Area",
                "x1": 0,
                "y1": 120,
                "x2": 180,
                "y2": 700
            },
            {
                "name": "Table Area",
                "x1": 180,
                "y1": 220,
                "x2": 430,
                "y2": 700
            },
            {
                "name": "Service Area",
                "x1": 300,
                "y1": 120,
                "x2": 560,
                "y2": 520
            }
        ],
        "line": {
            "name": "Entry / Exit Line",
            "y": 420
        }
    }
}


PERFORMANCE_MODE_FILE = "data/performance_mode.txt"


def get_system_profile():
    """
    Detect system resources without crashing if psutil is missing.
    Returns RAM, CPU, GPU hint, and whether psutil is available.
    """
    profile = {
        "psutil_available": False,
        "cpu_cores": 1,
        "ram_gb": 0,
        "available_ram_gb": 0,
        "gpu_available": False,
        "gpu_name": "Not detected"
    }

    try:
        import psutil

        memory = psutil.virtual_memory()
        profile["psutil_available"] = True
        profile["cpu_cores"] = psutil.cpu_count(logical=True) or 1
        profile["ram_gb"] = round(memory.total / (1024 ** 3), 2)
        profile["available_ram_gb"] = round(memory.available / (1024 ** 3), 2)
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            profile["gpu_available"] = True
            profile["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return profile


def recommend_performance_mode():
    system = get_system_profile()
    ram_gb = system["ram_gb"]
    cpu_cores = system["cpu_cores"]
    gpu_available = system["gpu_available"]

    if gpu_available and ram_gb >= 16:
        return "ULTRA"

    if ram_gb >= 16 and cpu_cores >= 8:
        return "HIGH"

    if ram_gb >= 8 and cpu_cores >= 4:
        return "BALANCED"

    return "LOW"


def get_saved_performance_mode():
    try:
        with open(PERFORMANCE_MODE_FILE, "r") as file:
            mode = file.read().strip().upper()

        if mode in ["AUTO", "LOW", "BALANCED", "HIGH", "ULTRA"]:
            return mode

    except FileNotFoundError:
        pass

    return PERFORMANCE_MODE


def set_saved_performance_mode(mode):
    mode = mode.upper()

    if mode not in ["AUTO", "LOW", "BALANCED", "HIGH", "ULTRA"]:
        return False

    import os
    os.makedirs("data", exist_ok=True)

    with open(PERFORMANCE_MODE_FILE, "w") as file:
        file.write(mode)

    return True


def resolve_performance_mode(mode=None):
    selected_mode = (mode or get_saved_performance_mode() or PERFORMANCE_MODE).upper()

    if selected_mode == "AUTO":
        return recommend_performance_mode()

    if selected_mode in PERFORMANCE_SETTINGS:
        return selected_mode

    return "LOW"


def get_performance_status():
    selected_mode = get_saved_performance_mode()
    recommended_mode = recommend_performance_mode()
    resolved_mode = resolve_performance_mode(selected_mode)

    return {
        "selected_mode": selected_mode,
        "recommended_mode": recommended_mode,
        "resolved_mode": resolved_mode,
        "system": get_system_profile(),
        "settings": PERFORMANCE_SETTINGS,
        "auto_enabled": selected_mode == "AUTO"
    }


SOURCE_PROFILES_FILE = "data/source_profiles.json"
ACTIVE_PROFILE_FILE = "data/active_camera_profile.txt"


def _default_zone():
    return {
        "name": "Main Monitoring Zone",
        "x1": 40,
        "y1": 80,
        "x2": 560,
        "y2": 520
    }


def _default_line():
    return {
        "name": "Entry / Exit Line",
        "y": 300
    }


def normalize_camera_profile(profile):
    zone = profile.get("zone") or _default_zone()
    zones = profile.get("zones") or [zone]
    line = profile.get("line") or _default_line()

    return {
        "name": profile.get("name", "Custom Source"),
        "source": profile.get("source", 0),
        "source_type": profile.get("source_type", "custom"),
        "created_at": profile.get("created_at", ""),
        "zone": zone,
        "zones": zones,
        "line": line
    }


def load_source_profiles():
    import json
    import os

    if not os.path.exists(SOURCE_PROFILES_FILE):
        return {}

    try:
        with open(SOURCE_PROFILES_FILE, "r") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return {}

        return {
            key: normalize_camera_profile(value)
            for key, value in data.items()
            if isinstance(value, dict)
        }
    except Exception:
        return {}


def save_source_profiles(profiles):
    import json
    import os

    os.makedirs("data", exist_ok=True)

    with open(SOURCE_PROFILES_FILE, "w") as file:
        json.dump(profiles, file, indent=2)


def get_camera_profiles():
    profiles = {
        key: normalize_camera_profile(value)
        for key, value in CAMERA_PROFILES.items()
    }

    profiles.update(load_source_profiles())
    return profiles


def add_camera_source_profile(key, profile):
    profiles = load_source_profiles()
    profiles[key] = normalize_camera_profile(profile)
    save_source_profiles(profiles)
    return key


def delete_camera_source_profile(key):
    profiles = load_source_profiles()

    if key not in profiles:
        return False

    profiles.pop(key)
    save_source_profiles(profiles)
    return True


def get_active_camera_profile():
    profiles = get_camera_profiles()

    try:
        with open(ACTIVE_PROFILE_FILE, "r") as file:
            profile = file.read().strip()

        if profile in profiles:
            return profile

    except FileNotFoundError:
        pass

    if ACTIVE_CAMERA_PROFILE in profiles:
        return ACTIVE_CAMERA_PROFILE

    return list(profiles.keys())[0]


def set_active_camera_profile(profile):
    if profile not in get_camera_profiles():
        return False

    import os
    os.makedirs("data", exist_ok=True)

    with open(ACTIVE_PROFILE_FILE, "w") as file:
        file.write(profile)

    return True


WINDOW_NAME = "CrowdVision AI v41 - Camera Health Intelligence"