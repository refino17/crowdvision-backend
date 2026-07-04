import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import time
import json

try:
    cv2.setNumThreads(1)
except Exception:
    pass

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    PERFORMANCE_MODE,
    PERFORMANCE_SETTINGS,
    resolve_performance_mode,
    get_performance_status,
    LOG_INTERVAL,
    ALERT_CAPTURE_INTERVAL,
    LIVE_FRAME_SAVE_INTERVAL,
    LIVE_JPEG_QUALITY,
    LIVE_PREVIEW_MAX_WIDTH,
    LIVE_CAMERA_TARGET_FPS,
    CAMERA_STATUS_FILE,
    ENABLE_CAMERA_HEALTH_INTELLIGENCE,
    CAMERA_TAMPER_BLUR_THRESHOLD,
    CAMERA_TAMPER_DARK_THRESHOLD,
    CAMERA_TAMPER_BRIGHT_THRESHOLD,
    CAMERA_TAMPER_FROZEN_SECONDS,
    CAMERA_TAMPER_MIN_FRAME_CHANGE,
    CAMERA_TAMPER_WARMUP_FRAMES,
    CAMERA_TAMPER_ALERT_HOLD_SECONDS,
    CAMERA_TAMPER_SAVE_SNAPSHOT,
    VIDEO_LOOP,
    ENABLE_ZONE_MONITORING,
    ENABLE_LINE_CROSSING,
    LINE_CROSSING_COOLDOWN,
    LINE_CROSSING_HYSTERESIS,
    TRACK_LOST_TTL_FRAMES,
    ENABLE_HEATMAP,
    HEATMAP_ONLY_INSIDE_ZONE,
    HEATMAP_DECAY_RATE,
    HEATMAP_RADIUS,
    HEATMAP_OPACITY,
    HEATMAP_BLUR_SIZE,
    ENABLE_MOVEMENT_ANALYTICS,
    MOVEMENT_THRESHOLD,
    ENABLE_PROFESSIONAL_UI,
    SIDE_PANEL_WIDTH,
    WINDOW_NAME,
    ENABLE_ANONYMOUS_REID,
    REID_MEMORY_SECONDS,
    REID_MAX_CENTER_DISTANCE,
    REID_MIN_MATCH_SCORE,
    REID_MIN_COLOR_SCORE,
    REID_IOU_WEIGHT,
    REID_DISTANCE_WEIGHT,
    REID_COLOR_WEIGHT,
    REID_HIST_BINS,
    get_active_camera_profile,
    get_camera_profiles
)

from camera import open_camera_source, get_camera_source_label, video_file_exists, restart_video_if_needed, CameraHealthAnalyzer
from detector import PeopleDetector
from statistics import SessionStats
from density import get_density_level
from alerts import (
    get_alert_message,
    record_density_notification,
    record_incident_notification,
    record_anomaly_notification,
    record_camera_health_notification,
    record_engine_notification,
)
from logger import save_event
from evidence import save_alert_snapshot
from display import (
    draw_detections,
    draw_counting_line,
    draw_multi_zones,
    create_professional_dashboard_panel
)
from zones import (
    filter_detections_inside_zone,
    analyze_zones,
    get_most_dangerous_zone
)
from occupancy import OccupancyTracker
from line_counter import LineCrossingCounter
from heatmap import CrowdHeatmap
from movement import MovementAnalyzer
from intelligence import calculate_congestion_score, get_congestion_level
from incident import analyze_incident
from predictor import (
    update_history,
    predict_people,
    predict_occupancy,
    risk_trend
)
from anomaly import analyze_anomaly
from report_generator import generate_incident_report, generate_anomaly_report, generate_camera_health_report


def resize_frame_for_live(frame, max_width=960):
    """
    Resize browser preview only.
    This does not force the IP camera stream itself.
    """
    if frame is None or frame.size == 0:
        return frame

    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / float(width)
    new_height = int(height * scale)

    return cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)


def save_latest_live_frame(frame):
    """
    Atomic live-frame save for the dashboard.
    Prevents browser/API from reading a half-written JPEG.
    """
    live_path = "data/live/latest_frame.jpg"
    temp_path = "data/live/latest_frame_tmp.jpg"

    try:
        if frame is None or frame.size == 0:
            return

        preview_frame = resize_frame_for_live(frame, LIVE_PREVIEW_MAX_WIDTH)

        success, encoded_frame = cv2.imencode(
            ".jpg",
            preview_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(LIVE_JPEG_QUALITY)]
        )

        if not success:
            return

        with open(temp_path, "wb") as file:
            file.write(encoded_frame.tobytes())

        os.replace(temp_path, live_path)

    except Exception as error:
        print(f"Live frame save error: {error}")


def write_camera_status(data):
    """
    Camera health heartbeat.
    The backend reads this file to know whether the active camera is alive.
    """
    try:
        os.makedirs(os.path.dirname(CAMERA_STATUS_FILE), exist_ok=True)
        temp_path = f"{CAMERA_STATUS_FILE}.tmp"

        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at_epoch": time.time(),
            **data
        }

        with open(temp_path, "w") as file:
            json.dump(payload, file, indent=2)

        os.replace(temp_path, CAMERA_STATUS_FILE)

    except Exception as error:
        print(f"Camera status write error: {error}")


def empty_tracking_summary():
    return {
        "session_unique_people": 0,
        "active_tracks": 0,
        "memory_tracks": 0,
        "reidentified_people": 0,
        "duplicates_prevented": 0,
        "privacy_mode": "Anonymous body tracking"
    }


def empty_camera_health():
    return {
        "enabled": bool(ENABLE_CAMERA_HEALTH_INTELLIGENCE),
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
        "last_alert_at": None,
        "alert_ready": False
    }


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def clean_tracking_summary(raw_summary, visible_people=0):
    """
    v38.1 tracking summary cleanup.

    Why this exists:
    Some detector-side duplicate counters can rise every normal frame,
    which makes duplicates_prevented look artificially high.

    This function exposes a cleaner operational report:

    - Current people = visible people now.
    - Active tracks = anonymous IDs currently visible.
    - Unique people seen = corrected session unique count.
    - Re-identified = successful anonymous body re-identification.
    - Duplicate counts prevented = real re-ID based prevention, not frame spam.

    No face recognition is used.
    """
    if not isinstance(raw_summary, dict):
        raw_summary = {}

    visible_people = safe_int(visible_people, 0)

    raw_unique_people = safe_int(raw_summary.get("session_unique_people", 0), 0)
    active_tracks = safe_int(raw_summary.get("active_tracks", 0), 0)
    memory_tracks = safe_int(raw_summary.get("memory_tracks", 0), 0)
    reidentified_people = safe_int(raw_summary.get("reidentified_people", 0), 0)
    raw_duplicates_prevented = safe_int(raw_summary.get("duplicates_prevented", 0), 0)

    active_tracks = max(active_tracks, visible_people)
    raw_unique_people = max(raw_unique_people, active_tracks)

    corrected_unique_people = max(
        active_tracks,
        raw_unique_people - reidentified_people
    )

    corrected_duplicates_prevented = min(
        raw_duplicates_prevented,
        reidentified_people
    )

    corrected_duplicates_prevented = max(corrected_duplicates_prevented, 0)

    return {
        "session_unique_people": int(corrected_unique_people),
        "active_tracks": int(active_tracks),
        "memory_tracks": int(memory_tracks),
        "reidentified_people": int(reidentified_people),
        "duplicates_prevented": int(corrected_duplicates_prevented),
        "privacy_mode": raw_summary.get("privacy_mode", "Anonymous body tracking")
    }


def get_detection_tracking_id(detection):
    return detection.get(
        "stable_tracking_id",
        detection.get("tracking_id", None)
    )


def dedupe_detections_by_tracking_id(detections):
    """
    Prevents one anonymous tracking ID from appearing twice in the same frame.

    This protects:
    - people count
    - zone count
    - line crossing count
    - occupancy calculations
    """
    clean_detections = []
    seen_tracking_ids = set()

    for detection in detections or []:
        tracking_id = get_detection_tracking_id(detection)

        if tracking_id is None or tracking_id == "N/A":
            clean_detections.append(detection)
            continue

        if tracking_id in seen_tracking_ids:
            continue

        seen_tracking_ids.add(tracking_id)
        clean_detections.append(detection)

    return clean_detections


def count_unique_people(detections):
    tracking_ids = set()
    unknown_count = 0

    for detection in detections or []:
        tracking_id = get_detection_tracking_id(detection)

        if tracking_id is None or tracking_id == "N/A":
            unknown_count += 1
        else:
            tracking_ids.add(tracking_id)

    return len(tracking_ids) + unknown_count


def mark_camera_offline(active_camera_profile, camera_source, reason):
    offline_health = {
        **empty_camera_health(),
        "status": "Offline",
        "severity": "Critical",
        "tamper_detected": True,
        "tamper_type": "No Signal",
        "message": reason,
        "signal_quality": "No Signal",
        "alert_ready": True
    }

    try:
        record_engine_notification(
            "Camera Source Offline",
            f"{active_camera_profile} is offline. Reason: {reason}",
            severity="critical",
            source=str(camera_source)
        )
        record_camera_health_notification(active_camera_profile, offline_health)
    except Exception as error:
        print(f"Notification write error: {error}")

    write_camera_status({
        "active_profile": active_camera_profile,
        "source": str(camera_source),
        "online": False,
        "status": "Offline",
        "reason": reason,
        "fps": 0,
        "ai_fps": 0,
        "people": 0,
        "zone_count": 0,
        "occupancy": 0,
        "density": "Unknown",
        "tracking": empty_tracking_summary(),
        "camera_health": offline_health,
        "line_tracking": {
            "entries": 0,
            "exits": 0,
            "duplicate_crossings_blocked": 0
        }
    })


def build_detector(frame_width):
    return PeopleDetector(
        MODEL_PATH,
        CONFIDENCE_THRESHOLD,
        frame_width,
        enable_reid=ENABLE_ANONYMOUS_REID,
        reid_memory_seconds=REID_MEMORY_SECONDS,
        reid_max_center_distance=REID_MAX_CENTER_DISTANCE,
        reid_min_match_score=REID_MIN_MATCH_SCORE,
        reid_min_color_score=REID_MIN_COLOR_SCORE,
        reid_iou_weight=REID_IOU_WEIGHT,
        reid_distance_weight=REID_DISTANCE_WEIGHT,
        reid_color_weight=REID_COLOR_WEIGHT,
        reid_hist_bins=REID_HIST_BINS
    )


def main():
    os.makedirs("data/live", exist_ok=True)

    active_camera_profile = get_active_camera_profile()
    camera_profiles = get_camera_profiles()

    if active_camera_profile not in camera_profiles:
        print(f"Invalid camera profile: {active_camera_profile}")
        mark_camera_offline(active_camera_profile, "unknown", "Invalid camera profile")
        return

    camera_profile = camera_profiles[active_camera_profile]
    camera_source = camera_profile["source"]
    monitoring_zone = camera_profile["zone"]
    monitoring_zones = camera_profile.get("zones", [monitoring_zone])
    counting_line = camera_profile["line"]

    if not video_file_exists(camera_source):
        print(f"Video file not found: {camera_source}")
        mark_camera_offline(active_camera_profile, camera_source, "Video file not found")
        return

    selected_performance_mode = os.getenv("CROWDVISION_PERFORMANCE_MODE", PERFORMANCE_MODE).upper()
    resolved_performance_mode = resolve_performance_mode(selected_performance_mode)
    performance_status = get_performance_status()

    performance_config = PERFORMANCE_SETTINGS[resolved_performance_mode]
    frame_width = performance_config["frame_width"]
    process_every_n_frames = performance_config["process_every_n_frames"]
    loop_delay = performance_config.get("loop_delay", 0.05)

    engine_mode = os.getenv("CROWDVISION_ENGINE_MODE", "desktop").lower()
    headless_mode = os.getenv("CROWDVISION_HEADLESS", "0") == "1" or engine_mode == "web"
    low_resource_mode = os.getenv("CROWDVISION_LOW_RESOURCE", "0") == "1"

    detector = build_detector(frame_width)
    stats = SessionStats()
    occupancy_tracker = OccupancyTracker()

    heatmap = CrowdHeatmap(
        decay_rate=HEATMAP_DECAY_RATE,
        radius=HEATMAP_RADIUS,
        opacity=HEATMAP_OPACITY,
        blur_size=HEATMAP_BLUR_SIZE
    )

    line_counter = LineCrossingCounter(
        counting_line["y"],
        LINE_CROSSING_COOLDOWN,
        line_buffer=LINE_CROSSING_HYSTERESIS,
        lost_ttl_frames=TRACK_LOST_TTL_FRAMES
    )

    movement_analyzer = MovementAnalyzer(
        movement_threshold=MOVEMENT_THRESHOLD,
        lost_ttl_frames=TRACK_LOST_TTL_FRAMES
    )

    cap = open_camera_source(camera_source)

    if cap is None:
        mark_camera_offline(active_camera_profile, camera_source, "Unable to open camera source")
        return

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, LIVE_CAMERA_TARGET_FPS)
    except Exception:
        pass

    camera_label = get_camera_source_label(camera_source)

    camera_health_analyzer = CameraHealthAnalyzer(
        blur_threshold=CAMERA_TAMPER_BLUR_THRESHOLD,
        dark_threshold=CAMERA_TAMPER_DARK_THRESHOLD,
        bright_threshold=CAMERA_TAMPER_BRIGHT_THRESHOLD,
        frozen_seconds=CAMERA_TAMPER_FROZEN_SECONDS,
        min_frame_change=CAMERA_TAMPER_MIN_FRAME_CHANGE,
        warmup_frames=CAMERA_TAMPER_WARMUP_FRAMES,
        alert_hold_seconds=CAMERA_TAMPER_ALERT_HOLD_SECONDS
    )

    print("CrowdVision AI v42 started.")
    print("Feature: Notification Engine + Camera Health Intelligence + Anonymous Tracking")
    print("Privacy: No face recognition. No identity database. Anonymous body/track memory only.")
    print("Duplicate handling: Cleaned operational duplicate reporting enabled.")
    print("Camera health: Tamper, blur, blackout, glare, and frozen-frame detection enabled.")
    print(f"Active Profile: {active_camera_profile}")
    print(f"Camera Source: {camera_source}")
    print(f"Zones: {len(monitoring_zones)}")
    print(f"Engine Mode: {engine_mode}")
    print(f"Headless Mode: {headless_mode}")
    print(f"Selected Performance Mode: {selected_performance_mode}")
    print(f"Resolved Performance Mode: {resolved_performance_mode}")
    print(f"Recommended Mode: {performance_status['recommended_mode']}")
    print(f"Frame Width: {frame_width}")
    print(f"Process Every N Frames: {process_every_n_frames}")
    print(f"Anonymous Re-ID: {ENABLE_ANONYMOUS_REID}")
    print(f"Re-ID Memory Seconds: {REID_MEMORY_SECONDS}")
    print("Press q to quit only in desktop mode.")

    try:
        record_engine_notification(
            "Monitoring Engine Started",
            f"CrowdVision AI started on profile {active_camera_profile} using {camera_label} in {resolved_performance_mode} mode.",
            severity="normal",
            source=active_camera_profile
        )
    except Exception as error:
        print(f"Notification write error: {error}")

    last_log_time = 0
    last_capture_time = 0
    last_incident_report_time = 0
    last_anomaly_report_time = 0
    last_camera_health_report_time = 0
    frame_counter = 0
    latest_detections = []

    occupancy_data = {"occupancy": 0, "peak": 0}
    line_data = {"entries": 0, "exits": 0, "line_y": counting_line["y"]}
    movement_data = {"left": 0, "right": 0, "up": 0, "down": 0, "stationary": 0}

    raw_tracking_summary = empty_tracking_summary()
    tracking_summary = empty_tracking_summary()
    camera_health_data = empty_camera_health()

    smart_data = {
        "visible_people": 0,
        "tracked_people": 0,
        "congestion_score": 0,
        "congestion_level": "Low",
        **tracking_summary
    }

    zone_analytics = []
    most_dangerous_zone = {
        "name": "None",
        "count": 0,
        "congestion_score": 0,
        "congestion_level": "Low"
    }

    heatmap_enabled = ENABLE_HEATMAP and not (headless_mode and low_resource_mode)
    professional_ui_enabled = ENABLE_PROFESSIONAL_UI

    while True:
        ret, frame = cap.read()

        if not ret:
            restarted = restart_video_if_needed(cap, camera_source, VIDEO_LOOP)

            if restarted:
                latest_detections = []
                occupancy_tracker = OccupancyTracker()

                line_counter = LineCrossingCounter(
                    counting_line["y"],
                    LINE_CROSSING_COOLDOWN,
                    line_buffer=LINE_CROSSING_HYSTERESIS,
                    lost_ttl_frames=TRACK_LOST_TTL_FRAMES
                )

                heatmap = CrowdHeatmap(
                    decay_rate=HEATMAP_DECAY_RATE,
                    radius=HEATMAP_RADIUS,
                    opacity=HEATMAP_OPACITY,
                    blur_size=HEATMAP_BLUR_SIZE
                )

                movement_analyzer = MovementAnalyzer(
                    movement_threshold=MOVEMENT_THRESHOLD,
                    lost_ttl_frames=TRACK_LOST_TTL_FRAMES
                )

                raw_tracking_summary = empty_tracking_summary()
                tracking_summary = empty_tracking_summary()

                try:
                    detector.reset_tracking()
                except Exception:
                    pass

                try:
                    camera_health_analyzer.reset()
                except Exception:
                    pass

                camera_health_data = empty_camera_health()

                continue

            print("End of stream or unable to read frame.")
            mark_camera_offline(active_camera_profile, camera_source, "End of stream or unable to read frame")
            break

        frame_counter += 1
        current_time = time.time()

        if ENABLE_CAMERA_HEALTH_INTELLIGENCE:
            camera_health_data = camera_health_analyzer.update(frame, current_time)
        else:
            camera_health_data = {
                **empty_camera_health(),
                "enabled": False,
                "status": "Disabled",
                "severity": "Info",
                "message": "Camera health intelligence is disabled.",
                "signal_quality": "Disabled"
            }

        display_fps = stats.calculate_display_fps()
        runtime = stats.get_runtime_seconds()

        should_process_frame = frame_counter % process_every_n_frames == 0

        if should_process_frame:
            latest_detections = detector.detect_and_track(frame)
            raw_tracking_summary = detector.get_tracking_summary()
            ai_fps = stats.update_ai_fps()
        else:
            stats.add_skipped_frame()
            ai_fps = stats.get_ai_fps()

        detections = dedupe_detections_by_tracking_id(latest_detections)
        person_count = count_unique_people(detections)

        tracking_summary = clean_tracking_summary(
            raw_tracking_summary,
            visible_people=person_count
        )

        if ENABLE_ZONE_MONITORING:
            zone_detections = filter_detections_inside_zone(
                detections,
                monitoring_zone
            )

            zone_detections = dedupe_detections_by_tracking_id(zone_detections)
            zone_count = count_unique_people(zone_detections)

            zone_analytics = analyze_zones(
                detections,
                monitoring_zones
            )

            most_dangerous_zone = get_most_dangerous_zone(
                zone_analytics
            )
        else:
            zone_detections = detections
            zone_count = person_count
            zone_analytics = []
            most_dangerous_zone = {
                "name": "None",
                "count": 0,
                "congestion_score": 0,
                "congestion_level": "Low"
            }

        occupancy_data = occupancy_tracker.update(zone_detections)

        update_history(
            person_count,
            occupancy_data["occupancy"]
        )

        predicted_people = predict_people()
        predicted_occupancy = predict_occupancy()
        future_risk = risk_trend()

        if ENABLE_LINE_CROSSING:
            line_data = line_counter.update(detections)

        if ENABLE_MOVEMENT_ANALYTICS:
            movement_data = movement_analyzer.analyze(detections)

        if HEATMAP_ONLY_INSIDE_ZONE:
            heatmap_source_detections = zone_detections
        else:
            heatmap_source_detections = detections

        if heatmap_enabled:
            heatmap.update(frame, heatmap_source_detections)
            display_frame = heatmap.draw(frame)
        else:
            display_frame = frame.copy()

        active_tracking_ids = draw_detections(display_frame, detections)
        tracked_people = len(active_tracking_ids)

        if tracked_people <= 0:
            tracked_people = person_count

        congestion_score = calculate_congestion_score(
            zone_count,
            occupancy_data["occupancy"],
            tracked_people
        )

        congestion_level = get_congestion_level(congestion_score)

        smart_data = {
            "visible_people": person_count,
            "tracked_people": tracked_people,
            "congestion_score": congestion_score,
            "congestion_level": congestion_level,
            **tracking_summary
        }

        if ENABLE_ZONE_MONITORING:
            draw_multi_zones(display_frame, zone_analytics)

        if ENABLE_LINE_CROSSING:
            draw_counting_line(display_frame, counting_line)

        density_level = get_density_level(zone_count)
        alert_message = get_alert_message(density_level)

        if frame_counter % LIVE_FRAME_SAVE_INTERVAL == 0:
            write_camera_status({
                "active_profile": active_camera_profile,
                "source": str(camera_source),
                "online": True,
                "status": "Online",
                "reason": "Receiving frames",
                "fps": round(float(display_fps), 2),
                "ai_fps": round(float(ai_fps), 2),
                "people": int(person_count),
                "zone_count": int(zone_count),
                "occupancy": int(occupancy_data["occupancy"]),
                "density": density_level,
                "alert": alert_message,
                "mode": resolved_performance_mode,
                "camera_health": camera_health_data,
                "tracking": tracking_summary,
                "line_tracking": {
                    "entries": int(line_data.get("entries", 0)),
                    "exits": int(line_data.get("exits", 0)),
                    "duplicate_crossings_blocked": int(line_data.get("duplicate_crossings_blocked", 0))
                }
            })

        incident_data = analyze_incident(
            density_level,
            most_dangerous_zone["name"],
            congestion_score
        )

        anomaly_data = analyze_anomaly(
            person_count,
            occupancy_data["occupancy"],
            line_data["entries"],
            density_level,
            most_dangerous_zone["name"]
        )

        try:
            record_density_notification(
                density_level,
                active_camera_profile,
                zone_count,
                occupancy_data["occupancy"],
                most_dangerous_zone["name"],
                alert_message
            )
            record_incident_notification(
                active_camera_profile,
                density_level,
                incident_data
            )
            record_anomaly_notification(
                active_camera_profile,
                anomaly_data
            )
            record_camera_health_notification(
                active_camera_profile,
                camera_health_data
            )
        except Exception as error:
            print(f"Notification write error: {error}")

        if current_time - last_log_time >= LOG_INTERVAL:
            save_event(
                person_count,
                zone_count,
                occupancy_data["occupancy"],
                occupancy_data["peak"],
                line_data["entries"],
                line_data["exits"],
                density_level,
                alert_message,
                active_camera_profile,
                incident_data["active"],
                incident_data["duration"],
                incident_data["danger_zone"],
                incident_data["recommendation"],
                predicted_people,
                predicted_occupancy,
                future_risk,
                anomaly_data["anomaly_detected"],
                anomaly_data["anomaly_type"],
                anomaly_data["anomaly_score"],
                anomaly_data["anomaly_severity"],
                anomaly_data["anomaly_zone"],
                anomaly_data["anomaly_recommendation"]
            )
            last_log_time = current_time

        if incident_data["active"]:
            if current_time - last_incident_report_time >= ALERT_CAPTURE_INTERVAL:
                incident_report = generate_incident_report(
                    active_camera_profile,
                    person_count,
                    zone_count,
                    occupancy_data["occupancy"],
                    density_level,
                    incident_data["danger_zone"],
                    incident_data["duration"],
                    incident_data["recommendation"]
                )
                print(f"Incident report generated: {incident_report}")
                last_incident_report_time = current_time

        if anomaly_data["anomaly_detected"]:
            if current_time - last_anomaly_report_time >= ALERT_CAPTURE_INTERVAL:
                anomaly_report = generate_anomaly_report(
                    active_camera_profile,
                    person_count,
                    occupancy_data["occupancy"],
                    anomaly_data["anomaly_type"],
                    anomaly_data["anomaly_score"],
                    anomaly_data["anomaly_severity"],
                    anomaly_data["anomaly_zone"],
                    anomaly_data["anomaly_recommendation"]
                )
                print(f"Anomaly report generated: {anomaly_report}")
                last_anomaly_report_time = current_time

        if (
            camera_health_data.get("tamper_detected") and
            camera_health_data.get("severity") in ["Warning", "Critical"]
        ):
            if current_time - last_camera_health_report_time >= ALERT_CAPTURE_INTERVAL:
                camera_report = generate_camera_health_report(
                    active_camera_profile,
                    camera_health_data
                )
                print(f"Camera health report generated: {camera_report}")
                last_camera_health_report_time = current_time

        if (
            CAMERA_TAMPER_SAVE_SNAPSHOT and
            camera_health_data.get("tamper_detected") and
            camera_health_data.get("severity") == "Critical"
        ):
            if current_time - last_capture_time >= ALERT_CAPTURE_INTERVAL:
                file_path = save_alert_snapshot(display_frame, "camera_tamper", prefix="tamper_snapshot")
                if file_path:
                    stats.add_alert_snapshot()
                    print(f"Camera tamper evidence saved: {file_path}")
                last_capture_time = current_time

        if density_level in ["High", "Critical"]:
            if current_time - last_capture_time >= ALERT_CAPTURE_INTERVAL:
                file_path = save_alert_snapshot(display_frame, density_level)
                if file_path:
                    stats.add_alert_snapshot()
                    print(f"Evidence saved: {file_path}")
                last_capture_time = current_time

        output_frame = display_frame

        if professional_ui_enabled:
            output_frame = create_professional_dashboard_panel(
                display_frame,
                person_count,
                zone_count,
                occupancy_data,
                line_data,
                movement_data,
                smart_data,
                zone_analytics,
                most_dangerous_zone,
                density_level,
                alert_message,
                display_fps,
                ai_fps,
                tracked_people,
                runtime,
                stats.total_alert_snapshots,
                stats.processed_ai_frames,
                stats.skipped_frames,
                camera_label,
                active_camera_profile,
                resolved_performance_mode,
                SIDE_PANEL_WIDTH,
                camera_health_data
            )

        if frame_counter % LIVE_FRAME_SAVE_INTERVAL == 0:
            save_latest_live_frame(output_frame)

        if not headless_mode:
            cv2.imshow(WINDOW_NAME, output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if loop_delay > 0:
            time.sleep(loop_delay)

    cap.release()

    if not headless_mode:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()