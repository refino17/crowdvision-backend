import csv
import os
from datetime import datetime

LOG_FILE = "data/events.csv"

HEADERS = [
    "timestamp",
    "camera_profile",
    "total_people",
    "zone_count",
    "occupancy",
    "peak_occupancy",
    "line_entries",
    "line_exits",
    "density_level",
    "alert_message",
    "incident_active",
    "incident_duration",
    "danger_zone",
    "recommendation",
    "predicted_people",
    "predicted_occupancy",
    "risk_trend",
    "anomaly_detected",
    "anomaly_type",
    "anomaly_score",
    "anomaly_severity",
    "anomaly_zone",
    "anomaly_recommendation"
]


def ensure_log_file():
    os.makedirs("data", exist_ok=True)

    if not os.path.isfile(LOG_FILE):
        return False

    with open(LOG_FILE, mode="r", newline="") as file:
        reader = csv.reader(file)
        existing_headers = next(reader, [])

    if existing_headers != HEADERS:
        backup_file = f"data/events_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.rename(LOG_FILE, backup_file)
        print(f"Old events.csv backed up as: {backup_file}")
        return False

    return True


def save_event(
    total_people,
    zone_count,
    occupancy,
    peak_occupancy,
    line_entries,
    line_exits,
    density_level,
    alert_message,
    camera_profile,
    incident_active=False,
    incident_duration=0,
    danger_zone="None",
    recommendation="Monitor",
    predicted_people=0,
    predicted_occupancy=0,
    risk_trend="Stable",
    anomaly_detected=False,
    anomaly_type="Normal",
    anomaly_score=0,
    anomaly_severity="Normal",
    anomaly_zone="None",
    anomaly_recommendation="Monitor"
):
    file_exists = ensure_log_file()

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(HEADERS)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            camera_profile,
            total_people,
            zone_count,
            occupancy,
            peak_occupancy,
            line_entries,
            line_exits,
            density_level,
            alert_message,
            incident_active,
            incident_duration,
            danger_zone,
            recommendation,
            predicted_people,
            predicted_occupancy,
            risk_trend,
            anomaly_detected,
            anomaly_type,
            anomaly_score,
            anomaly_severity,
            anomaly_zone,
            anomaly_recommendation
        ])