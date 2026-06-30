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
    VIDEO_LOOP,
    ENABLE_ZONE_MONITORING,
    ENABLE_LINE_CROSSING,
    LINE_CROSSING_COOLDOWN,
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
    get_active_camera_profile,
    get_camera_profiles
)

from camera import open_camera_source, get_camera_source_label, video_file_exists, restart_video_if_needed
from detector import PeopleDetector
from statistics import SessionStats
from density import get_density_level
from alerts import get_alert_message
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
from report_generator import generate_incident_report, generate_anomaly_report



def resize_frame_for_live(frame, max_width=960):
    """
    Resize the browser preview frame only.
    This does not force the phone/IP camera stream itself.
    It only reduces the image written to data/live/latest_frame.jpg.
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

    v19 improvement:
    - Encode JPEG in memory first.
    - Write to a temp file.
    - Atomically replace latest_frame.jpg.
    This prevents the browser/API from reading a half-written JPEG.
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
    v20 camera health heartbeat.
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


def mark_camera_offline(active_camera_profile, camera_source, reason):
    write_camera_status({
        "active_profile": active_camera_profile,
        "source": str(camera_source),
        "online": False,
        "status": "Offline",
        "reason": reason,
        "fps": 0,
        "ai_fps": 0,
        "people": 0,
        "density": "Unknown"
    })


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

    detector = PeopleDetector(MODEL_PATH, CONFIDENCE_THRESHOLD, frame_width)
    stats = SessionStats()
    occupancy_tracker = OccupancyTracker()

    heatmap = CrowdHeatmap(
        decay_rate=HEATMAP_DECAY_RATE,
        radius=HEATMAP_RADIUS,
        opacity=HEATMAP_OPACITY,
        blur_size=HEATMAP_BLUR_SIZE
    )

    line_counter = LineCrossingCounter(counting_line["y"], LINE_CROSSING_COOLDOWN)
    movement_analyzer = MovementAnalyzer(movement_threshold=MOVEMENT_THRESHOLD)

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

    print("CrowdVision AI v20.0 started.")
    print("Feature: Enterprise Monitoring Upgrade")
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
    print("Press q to quit only in desktop mode.")

    last_log_time = 0
    last_capture_time = 0
    last_incident_report_time = 0
    last_anomaly_report_time = 0
    frame_counter = 0
    latest_detections = []

    occupancy_data = {"occupancy": 0, "peak": 0}
    line_data = {"entries": 0, "exits": 0, "line_y": counting_line["y"]}
    movement_data = {"left": 0, "right": 0, "up": 0, "down": 0, "stationary": 0}

    smart_data = {
        "visible_people": 0,
        "tracked_people": 0,
        "congestion_score": 0,
        "congestion_level": "Low"
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
                line_counter = LineCrossingCounter(counting_line["y"], LINE_CROSSING_COOLDOWN)
                heatmap = CrowdHeatmap(
                    decay_rate=HEATMAP_DECAY_RATE,
                    radius=HEATMAP_RADIUS,
                    opacity=HEATMAP_OPACITY,
                    blur_size=HEATMAP_BLUR_SIZE
                )
                movement_analyzer = MovementAnalyzer(movement_threshold=MOVEMENT_THRESHOLD)
                continue

            print("End of stream or unable to read frame.")
            mark_camera_offline(active_camera_profile, camera_source, "End of stream or unable to read frame")
            break

        frame_counter += 1
        current_time = time.time()

        display_fps = stats.calculate_display_fps()
        runtime = stats.get_runtime_seconds()

        should_process_frame = frame_counter % process_every_n_frames == 0

        if should_process_frame:
            latest_detections = detector.detect_and_track(frame)
            ai_fps = stats.update_ai_fps()
        else:
            stats.add_skipped_frame()
            ai_fps = stats.get_ai_fps()

        detections = latest_detections
        person_count = len(detections)

        if ENABLE_ZONE_MONITORING:
            zone_detections = filter_detections_inside_zone(
                detections,
                monitoring_zone
            )

            zone_analytics = analyze_zones(
                detections,
                monitoring_zones
            )

            most_dangerous_zone = get_most_dangerous_zone(
                zone_analytics
            )
        else:
            zone_detections = detections
            zone_analytics = []
            most_dangerous_zone = {
                "name": "None",
                "count": 0,
                "congestion_score": 0,
                "congestion_level": "Low"
            }

        zone_count = len(zone_detections)
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
            "congestion_level": congestion_level
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
                "mode": resolved_performance_mode
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

        if density_level in ["High", "Critical"]:
            if current_time - last_capture_time >= ALERT_CAPTURE_INTERVAL:
                file_path = save_alert_snapshot(display_frame, density_level)
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
                SIDE_PANEL_WIDTH
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