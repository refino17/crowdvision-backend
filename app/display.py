import cv2
import numpy as np


def draw_detections(frame, detections):
    active_tracking_ids = set()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = detection["confidence"]
        tracking_id = detection["tracking_id"]

        if tracking_id != "N/A":
            active_tracking_ids.add(tracking_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"ID {tracking_id} | {confidence:.2f}"

        cv2.putText(frame, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)
        cv2.circle(frame, (foot_x, foot_y), 5, (0, 255, 255), -1)

    return active_tracking_ids


def draw_zone(frame, zone, zone_count):
    x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
    name = zone["name"]

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(frame, f"{name}: {zone_count}", (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


def draw_multi_zones(frame, zone_analytics):
    for zone_data in zone_analytics:
        zone = zone_data["zone"]
        count = zone_data["count"]
        level = zone_data["congestion_level"]
        score = zone_data["congestion_score"]

        x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]

        if level == "Critical":
            color = (0, 0, 255)
        elif level == "High":
            color = (0, 165, 255)
        elif level == "Medium":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"{zone_data['name']}: {count} | {level} {score}%"

        cv2.putText(
            frame,
            label,
            (x1 + 5, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )


def draw_counting_line(frame, line_config):
    line_y = line_config["y"]
    line_name = line_config["name"]
    height, width = frame.shape[:2]

    cv2.line(frame, (0, line_y), (width, line_y), (0, 165, 255), 3)
    cv2.putText(frame, line_name, (20, max(line_y - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)


def create_professional_dashboard_panel(
    frame,
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
    active_tracking_count,
    runtime,
    total_alert_snapshots,
    processed_ai_frames,
    skipped_frames,
    camera_label,
    camera_profile,
    performance_mode,
    side_panel_width
):
    height = frame.shape[0]
    panel = np.zeros((height, side_panel_width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    y = 28

    def write(text, value=None, color=(255, 255, 255), scale=0.48, gap=24):
        nonlocal y
        display_text = f"{text}: {value}" if value is not None else text
        cv2.putText(panel, display_text, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += gap

    write("CrowdVision AI", color=(0, 255, 255), scale=0.75, gap=32)
    write("Multi-Zone Danger Analytics", color=(180, 180, 180), scale=0.45, gap=28)

    write("Visible People", smart_data["visible_people"], (0, 255, 255))
    write("Tracked People", smart_data["tracked_people"], (0, 255, 255))
    write("Main Zone Count", zone_count, (0, 255, 255))
    write("Smoothed Occupancy", occupancy_data["occupancy"], (0, 255, 255))
    write("Peak Occupancy", occupancy_data["peak"], (0, 255, 255))

    write("Overall Score", f"{smart_data['congestion_score']}%", (0, 165, 255))
    write("Overall Level", smart_data["congestion_level"], (0, 165, 255))

    write("Danger Zone", most_dangerous_zone["name"], (0, 0, 255))
    write("Danger Score", f"{most_dangerous_zone['congestion_score']}%", (0, 0, 255))

    write("Line Entries", line_data["entries"], (0, 165, 255))
    write("Line Exits", line_data["exits"], (0, 165, 255))

    write("Movement Flow", color=(255, 255, 0), scale=0.52)
    write("L/R/U/D", f"{movement_data['left']}/{movement_data['right']}/{movement_data['up']}/{movement_data['down']}", (255, 255, 0))
    write("Stationary", movement_data["stationary"], (255, 255, 0))

    write("Zones", color=(200, 200, 200), scale=0.52)

    for zone_data in zone_analytics[:4]:
        zone_text = (
            f"{zone_data['name']}: "
            f"{zone_data['count']} | "
            f"{zone_data['congestion_level']} "
            f"{zone_data['congestion_score']}%"
        )

        level = zone_data["congestion_level"]

        if level == "Critical":
            color = (0, 0, 255)
        elif level == "High":
            color = (0, 165, 255)
        elif level == "Medium":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        write(zone_text, color=color, scale=0.42, gap=22)

    if density_level == "Critical":
        density_color = (0, 0, 255)
    elif density_level == "High":
        density_color = (0, 165, 255)
    elif density_level == "Medium":
        density_color = (0, 255, 255)
    else:
        density_color = (0, 255, 0)

    write("Density", density_level, density_color)
    write(alert_message, color=density_color, scale=0.40)

    write("FPS AI/Display", f"{ai_fps:.2f}/{display_fps:.2f}", (255, 255, 255))
    write("Active IDs", active_tracking_count, (255, 255, 255))
    write("Runtime", f"{runtime}s", (255, 255, 255))

    write("Alert Shots", total_alert_snapshots, (200, 200, 200))
    write("AI Frames", processed_ai_frames, (200, 200, 200))
    write("Skipped Frames", skipped_frames, (200, 200, 200))

    write("Profile", camera_profile, (180, 180, 180), scale=0.42)
    write("Source", camera_label, (180, 180, 180), scale=0.42)
    write("Mode", performance_mode, (180, 180, 180), scale=0.42)

    return np.hstack((frame, panel))